//! Absorbed latent attention for the full-attention layers of MiniCPM-V-4.7.
//!
//! Each of those layers originally caches a key and a value per KV head. The
//! fit replaces the non-rotary key rows and the whole value projection with one
//! latent of rank `r` shared by every head. The rotary key is still cached
//! explicitly, because a position rotation does not factor through that latent.
//! Decode then dots the query against the latent and up-projects the value
//! afterwards, so the cache row is the latent, the rotated rope key, and the
//! per-head RMS used to scale the key.

use burn::tensor::{Bool, FloatDType, Tensor, TensorData, activation, backend::Backend};
use omni_r2t2::Error;
use safetensors::{Dtype, SafeTensors};

/// Projections for one full-attention layer. `down` and `rope` are `[out, hidden]`.
pub struct LatentFactors<B: Backend> {
    pub down: Tensor<B, 2>,
    pub rope: Tensor<B, 2>,
    pub k_up: Tensor<B, 3>,
    pub v_up: Tensor<B, 3>,
    /// `(1 + k_norm)` on the non-rotary key dimensions.
    pub w_nope: Tensor<B, 1>,
    pub rank: usize,
}

impl<B: Backend> LatentFactors<B> {
    /// Rank stored for one full-attention layer in a factors file.
    pub fn stored_rank(bytes: &[u8], layer: usize) -> Result<usize, Error> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|err| Error::Checkpoint(format!("latent factors: {err}")))?;
        let name = format!("down.{layer}");
        let view = tensors
            .tensor(&name)
            .map_err(|err| Error::Checkpoint(format!("{name}: {err}")))?;
        view.shape()
            .first()
            .copied()
            .ok_or_else(|| Error::Checkpoint(format!("{name}: missing rank")))
    }

    pub fn load(
        bytes: &[u8],
        layers: &[usize],
        hidden: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        max_rank: usize,
        dtype: FloatDType,
        device: &B::Device,
    ) -> Result<Vec<Self>, Error> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|err| Error::Checkpoint(format!("latent factors: {err}")))?;
        layers
            .iter()
            .map(|&layer| {
                let nope = head_dim - rope_dim;
                let down = read_f32(&tensors, &format!("down.{layer}"), [max_rank, hidden])?;
                let rope = read_f32(
                    &tensors,
                    &format!("rope.{layer}"),
                    [kv_heads * rope_dim, hidden],
                )?;
                let k_up = read_f32(
                    &tensors,
                    &format!("k_up.{layer}"),
                    [kv_heads, nope, max_rank],
                )?;
                let v_up = read_f32(
                    &tensors,
                    &format!("v_up.{layer}"),
                    [kv_heads, head_dim, max_rank],
                )?;
                let w_nope = read_f32(&tensors, &format!("w_nope.{layer}"), [nope])?;
                Ok(Self {
                    down: tensor2(&down, [max_rank, hidden], dtype, device),
                    rope: tensor2(&rope, [kv_heads * rope_dim, hidden], dtype, device),
                    k_up: tensor3(&k_up, [kv_heads, nope, max_rank], dtype, device),
                    v_up: tensor3(&v_up, [kv_heads, head_dim, max_rank], dtype, device),
                    w_nope: tensor1(&w_nope, dtype, device),
                    rank: max_rank,
                })
            })
            .collect()
    }

    /// Keep the leading singular components. The fit stores them in that order.
    pub fn truncated(&self, rank: usize) -> Self {
        let [kv, nope, full] = self.k_up.dims();
        let [kv_v, head, _] = self.v_up.dims();
        let [_, hidden] = self.down.dims();
        assert!(rank > 0 && rank <= full && kv == kv_v);
        Self {
            down: self.down.clone().slice([0..rank, 0..hidden]),
            rope: self.rope.clone(),
            k_up: self.k_up.clone().slice([0..kv, 0..nope, 0..rank]),
            v_up: self.v_up.clone().slice([0..kv, 0..head, 0..rank]),
            w_nope: self.w_nope.clone(),
            rank,
        }
    }
}

/// Latent history for one full-attention layer.
pub struct LatentCache<B: Backend> {
    c: Option<Tensor<B, 2>>,
    rope: Option<Tensor<B, 3>>,
    rms: Option<Tensor<B, 2>>,
    rank: usize,
    kv_heads: usize,
    rope_dim: usize,
}

impl<B: Backend> LatentCache<B> {
    pub fn new(rank: usize, kv_heads: usize, rope_dim: usize) -> Self {
        Self {
            c: None,
            rope: None,
            rms: None,
            rank,
            kv_heads,
            rope_dim,
        }
    }

    pub fn len(&self) -> usize {
        self.c.as_ref().map_or(0, |c| c.dims()[0])
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Activation bytes for the latent and the rope key, plus f32 bytes for RMS.
    pub fn nbytes(&self, activation_bytes: usize) -> usize {
        let tokens = self.len();
        tokens * (self.rank + self.kv_heads * self.rope_dim) * activation_bytes
            + tokens * self.kv_heads * std::mem::size_of::<f32>()
    }

    fn append(&mut self, c: Tensor<B, 2>, rope: Tensor<B, 3>, rms: Tensor<B, 2>) {
        self.c = Some(cat2(self.c.clone(), c));
        self.rope = Some(cat3(self.rope.clone(), rope));
        self.rms = Some(cat2(self.rms.clone(), rms));
    }
}

/// Attend `q` (already normalized and partially rotated) against the latent history.
///
/// `c_new` is `[seq, rank]`, `rope_new` is `[seq, kv, rope]` after key-norm and
/// RoPE, and `rms_new` is `[seq, kv]` in f32. Returns `[1, heads, seq, head]`.
pub fn attend<B: Backend>(
    q: Tensor<B, 4>,
    c_new: Tensor<B, 2>,
    rope_new: Tensor<B, 3>,
    rms_new: Tensor<B, 2>,
    k_up: &Tensor<B, 3>,
    v_up: &Tensor<B, 3>,
    w_nope: &Tensor<B, 1>,
    cache: &mut LatentCache<B>,
    scale: f64,
) -> Tensor<B, 4> {
    let [_, heads, seq, head] = q.dims();
    let [kv, nope, rank] = k_up.dims();
    let rope_dim = head - nope;
    assert_eq!(heads % kv, 0);
    assert_eq!(rope_new.dims(), [seq, kv, rope_dim]);
    assert_eq!(c_new.dims(), [seq, rank]);
    let offset = cache.len();
    cache.append(c_new, rope_new, rms_new);
    let c = cache.c.clone().expect("latent");
    let rope = cache.rope.clone().expect("latent rope");
    let rms = cache.rms.clone().expect("latent rms");
    let total = c.dims()[0];
    let group = heads / kv;
    let dtype = q.dtype();
    let q_rope = q.clone().slice([0..1, 0..heads, 0..seq, 0..rope_dim]);
    let q_nope = q.slice([0..1, 0..heads, 0..seq, rope_dim..head]);
    let mut head_out = Vec::with_capacity(kv);

    for h in 0..kv {
        let qn = q_nope
            .clone()
            .slice([0..1, h * group..(h + 1) * group, 0..seq, 0..nope])
            .reshape([group * seq, nope]);
        let up_k = k_up
            .clone()
            .slice([h..h + 1, 0..nope, 0..rank])
            .reshape([nope, rank]);
        let weight = w_nope.clone().reshape([1, nope]).cast(dtype);
        let q_abs = (qn * weight).matmul(up_k);
        let latent = q_abs.matmul(c.clone().swap_dims(0, 1));
        let qr = q_rope
            .clone()
            .slice([0..1, h * group..(h + 1) * group, 0..seq, 0..rope_dim])
            .reshape([group * seq, rope_dim]);
        let kr = rope
            .clone()
            .slice([0..total, h..h + 1, 0..rope_dim])
            .reshape([total, rope_dim])
            .swap_dims(0, 1);
        // The cached rope key is already key-normalized. The latent still has to
        // be scaled by that token's RMS, which is different for every KV head.
        let rms_h = rms
            .clone()
            .slice([0..total, h..h + 1])
            .reshape([1, 1, 1, total]);
        let latent = latent.reshape([1, group, seq, total]).cast(FloatDType::F32) / rms_h;
        let rope_scores = qr
            .matmul(kr)
            .reshape([1, group, seq, total])
            .cast(FloatDType::F32);
        let mut scores = (latent + rope_scores) * scale;
        if seq > 1 || offset + 1 < total {
            scores = mask_future(scores, offset);
        }
        let attn = activation::softmax(scores, 3)
            .cast(dtype)
            .reshape([group * seq, total]);
        let up_v = v_up
            .clone()
            .slice([h..h + 1, 0..head, 0..rank])
            .reshape([head, rank])
            .swap_dims(0, 1);
        let ctx = attn.matmul(c.clone());
        head_out.push(ctx.matmul(up_v).reshape([1, group, seq, head]));
    }
    Tensor::cat(head_out, 1)
}

fn mask_future<B: Backend>(scores: Tensor<B, 4>, offset: usize) -> Tensor<B, 4> {
    let [_, _, seq, total] = scores.dims();
    let hide: Vec<bool> = (0..seq)
        .flat_map(|q| (0..total).map(move |k| k > q + offset))
        .collect();
    let mask = Tensor::<B, 4, Bool>::from_data(
        TensorData::new(hide, [1, 1, seq, total]),
        &scores.device(),
    );
    scores.mask_fill(mask, f32::NEG_INFINITY)
}

fn cat2<B: Backend>(previous: Option<Tensor<B, 2>>, next: Tensor<B, 2>) -> Tensor<B, 2> {
    match previous {
        Some(previous) => Tensor::cat(vec![previous, next], 0),
        None => next,
    }
}

fn cat3<B: Backend>(previous: Option<Tensor<B, 3>>, next: Tensor<B, 3>) -> Tensor<B, 3> {
    match previous {
        Some(previous) => Tensor::cat(vec![previous, next], 0),
        None => next,
    }
}

fn read_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: impl AsRef<[usize]>,
) -> Result<Vec<f32>, Error> {
    let view = tensors
        .tensor(name)
        .map_err(|err| Error::Checkpoint(format!("{name}: {err}")))?;
    let shape = shape.as_ref();
    if view.shape() != shape {
        return Err(Error::Checkpoint(format!(
            "{name}: shape {:?} != {shape:?}",
            view.shape()
        )));
    }
    if view.dtype() != Dtype::F32 {
        return Err(Error::Checkpoint(format!("{name}: expected f32")));
    }
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32")))
        .collect())
}

fn tensor1<B: Backend>(values: &[f32], dtype: FloatDType, device: &B::Device) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [values.len()]), device).cast(dtype)
}

fn tensor2<B: Backend>(
    values: &[f32],
    shape: [usize; 2],
    dtype: FloatDType,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(values.to_vec(), shape), device).cast(dtype)
}

fn tensor3<B: Backend>(
    values: &[f32],
    shape: [usize; 3],
    dtype: FloatDType,
    device: &B::Device,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(values.to_vec(), shape), device).cast(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    fn data(values: Vec<f32>, shape: [usize; 4]) -> Tensor<NdArray, 4> {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        Tensor::<NdArray, 4>::from_data(TensorData::new(values, shape), &device)
    }

    fn vec4(tensor: Tensor<NdArray, 4>) -> Vec<f32> {
        tensor.to_data().to_vec::<f32>().unwrap()
    }

    /// Scalar attention using the same split the kernel claims to compute.
    fn reference(
        q: &[f32],
        c: &[f32],
        rope: &[f32],
        rms: &[f32],
        k_up: &[f32],
        v_up: &[f32],
        w: &[f32],
        heads: usize,
        seq: usize,
        head: usize,
        rope_dim: usize,
        rank: usize,
        offset: usize,
        scale: f32,
    ) -> Vec<f32> {
        let nope = head - rope_dim;
        let total = offset + seq;
        let mut out = vec![0f32; heads * seq * head];
        for h in 0..heads {
            let mut scores = vec![0f32; seq * total];
            for q_i in 0..seq {
                for t in 0..total {
                    if t > q_i + offset {
                        scores[q_i * total + t] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut score = 0f32;
                    for d in 0..rope_dim {
                        let qv = q[((h * seq + q_i) * head) + d];
                        let kv = rope[(t * rope_dim) + d];
                        score += qv * kv;
                    }
                    for d in 0..nope {
                        let qv = q[((h * seq + q_i) * head) + rope_dim + d];
                        let mut raw = 0f32;
                        for r in 0..rank {
                            raw += c[t * rank + r] * k_up[d * rank + r];
                        }
                        score += qv * raw * w[d] / rms[t];
                    }
                    scores[q_i * total + t] = score * scale;
                }
            }
            for q_i in 0..seq {
                let row = &scores[q_i * total..(q_i + 1) * total];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut weights = vec![0f32; total];
                let mut sum = 0f32;
                for t in 0..total {
                    let e = if row[t].is_finite() {
                        (row[t] - max).exp()
                    } else {
                        0.0
                    };
                    weights[t] = e;
                    sum += e;
                }
                for d in 0..head {
                    let mut acc = 0f32;
                    for t in 0..total {
                        let mut value = 0f32;
                        for r in 0..rank {
                            value += c[t * rank + r] * v_up[d * rank + r];
                        }
                        acc += weights[t] / sum * value;
                    }
                    out[(h * seq + q_i) * head + d] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn absorbed_attention_matches_the_scalar_definition() {
        let (heads, seq, head, rope_dim, rank) = (2, 2, 8, 4, 3);
        let kv = 1;
        let nope = head - rope_dim;
        let mut state = 7u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / 16_777_216.0 - 0.5
        };
        let q: Vec<f32> = (0..heads * seq * head).map(|_| next()).collect();
        let c: Vec<f32> = (0..seq * rank).map(|_| next()).collect();
        let rope: Vec<f32> = (0..seq * kv * rope_dim).map(|_| next()).collect();
        let rms: Vec<f32> = (0..seq * kv).map(|_| next().abs() + 0.4).collect();
        let k_up: Vec<f32> = (0..kv * nope * rank).map(|_| next()).collect();
        let v_up: Vec<f32> = (0..kv * head * rank).map(|_| next()).collect();
        let w: Vec<f32> = (0..nope).map(|_| next() + 1.2).collect();
        let scale = 1.0 / (head as f32).sqrt();
        let expected = reference(
            &q, &c, &rope, &rms, &k_up, &v_up, &w, heads, seq, head, rope_dim, rank, 0, scale,
        );

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let q_t = data(q, [1, heads, seq, head]);
        let c_t = Tensor::<NdArray, 2>::from_data(TensorData::new(c, [seq, rank]), &device);
        let rope_t =
            Tensor::<NdArray, 3>::from_data(TensorData::new(rope, [seq, kv, rope_dim]), &device);
        let rms_t = Tensor::<NdArray, 2>::from_data(TensorData::new(rms, [seq, kv]), &device);
        let k_up_t =
            Tensor::<NdArray, 3>::from_data(TensorData::new(k_up, [kv, nope, rank]), &device);
        let v_up_t =
            Tensor::<NdArray, 3>::from_data(TensorData::new(v_up, [kv, head, rank]), &device);
        let w_t = Tensor::<NdArray, 1>::from_data(TensorData::new(w, [nope]), &device);
        let mut cache = LatentCache::new(rank, kv, rope_dim);
        let got = attend(
            q_t,
            c_t,
            rope_t,
            rms_t,
            &k_up_t,
            &v_up_t,
            &w_t,
            &mut cache,
            scale as f64,
        );
        let got = vec4(got);
        let mut worst = 0f32;
        for (a, b) in got.iter().zip(&expected) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst < 1e-4,
            "absorbed attention drifted by {worst}: {got:?} vs {expected:?}"
        );
        assert_eq!(cache.len(), seq);
    }
}
