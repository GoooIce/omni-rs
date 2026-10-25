//! One launch for the gated-delta recurrence over a verify block.
//!
//! The host loop issues a kernel per token per tensor op. This kernel keeps
//! that FP32 update order and writes each token's state for a later rollback.
use burn::tensor::{DType, Shape, Tensor as BurnTensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use std::mem::{forget, size_of};

#[cube(launch_unchecked)]
fn gdn_scan_kernel(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    g: &Tensor<f32>,
    beta: &Tensor<f32>,
    state: &mut Tensor<f32>,
    out: &mut Tensor<f32>,
    steps: &mut Tensor<f32>,
    t_len: u32,
    heads: u32,
    k_dim: u32,
    v_dim: u32,
    record: u32,
) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let width = CUBE_DIM_X;
    if h < heads {
        let chunks = (v_dim + width - 1) / width;
        for t in 0..t_len {
            let head = t * heads + h;
            let decay = g[head as usize];
            let b = beta[head as usize];
            for chunk in 0..chunks {
                let v_i = chunk * width + lane;
                if v_i < v_dim {
                    // Read-only kv pass, then a single write pass: the old
                    // form stored the decayed state inside the reduction,
                    // paying a store per element and making the second pass
                    // wait on those stores.
                    let mut kv = 0.0f32;
                    for kk in 0..k_dim {
                        let s_index = ((h * k_dim + kk) * v_dim + v_i) as usize;
                        kv += state[s_index] * decay * k[((head * k_dim) + kk) as usize];
                    }
                    let delta = (v[((head * v_dim) + v_i) as usize] - kv) * b;
                    let mut y = 0.0f32;
                    for kk in 0..k_dim {
                        let s_index = ((h * k_dim + kk) * v_dim + v_i) as usize;
                        let updated =
                            state[s_index] * decay + k[((head * k_dim) + kk) as usize] * delta;
                        state[s_index] = updated;
                        y += updated * q[((head * k_dim) + kk) as usize];
                    }
                    out[((head * v_dim) + v_i) as usize] = y;
                    if record != 0 {
                        for kk in 0..k_dim {
                            let s_index = ((h * k_dim + kk) * v_dim + v_i) as usize;
                            let step = (((t * heads + h) * k_dim + kk) * v_dim + v_i) as usize;
                            steps[step] = state[s_index];
                        }
                    }
                }
            }
        }
    }
}

/// Split-kk variant: `KSPLIT` units share one v column, each reducing its
/// slice of the k dimension through cube address space, so the per-unit
/// serial chain shrinks KSPLIT-fold and the block runs
/// heads x v_dim x KSPLIT units. Requires k_dim divisible by KSPLIT and
/// v_dim x KSPLIT at most 1024 (Metal's cube cap); otherwise the scalar
/// kernel above stays the path.
#[cube(launch_unchecked)]
fn gdn_scan_split_kernel(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    g: &Tensor<f32>,
    beta: &Tensor<f32>,
    state: &mut Tensor<f32>,
    out: &mut Tensor<f32>,
    steps: &mut Tensor<f32>,
    t_len: u32,
    heads: u32,
    k_dim: u32,
    v_dim: u32,
    record: u32,
    #[comptime] ksplit: u32,
) {
    let h = CUBE_POS_X;
    let unit = UNIT_POS_X;
    if h < heads {
        let v_i = unit % v_dim;
        let ks = unit / v_dim;
        let slice = k_dim / ksplit;
        let k_base = ks * slice;
        // v_dim x KSPLIT partials; sized for the largest supported v_dim.
        let mut kv_parts = SharedMemory::<f32>::new(1024usize);
        let mut y_parts = SharedMemory::<f32>::new(1024usize);
        let kv_slot = (v_i * ksplit + ks) as usize;
        for t in 0..t_len {
            let head = t * heads + h;
            let decay = g[head as usize];
            let b = beta[head as usize];
            let mut kv = 0.0f32;
            for i in 0..slice {
                let kk = k_base + i;
                let s_index = ((h * k_dim + kk) * v_dim + v_i) as usize;
                kv += state[s_index] * decay * k[((head * k_dim) + kk) as usize];
            }
            kv_parts[kv_slot] = kv;
            sync_cube();
            // Every unit redundantly totals its column's partials, so the
            // delta needs no second barrier.
            let mut kv_total = 0.0f32;
            for part in 0..ksplit {
                kv_total += kv_parts[(v_i * ksplit + part) as usize];
            }
            let delta = (v[((head * v_dim) + v_i) as usize] - kv_total) * b;
            let mut y = 0.0f32;
            for i in 0..slice {
                let kk = k_base + i;
                let s_index = ((h * k_dim + kk) * v_dim + v_i) as usize;
                let updated = state[s_index] * decay + k[((head * k_dim) + kk) as usize] * delta;
                state[s_index] = updated;
                y += updated * q[((head * k_dim) + kk) as usize];
            }
            y_parts[kv_slot] = y;
            sync_cube();
            let mut y_total = 0.0f32;
            for part in 0..ksplit {
                y_total += y_parts[(v_i * ksplit + part) as usize];
            }
            if ks == 0 {
                out[((head * v_dim) + v_i) as usize] = y_total;
            }
            if record != 0 {
                for i in 0..slice {
                    let kk = k_base + i;
                    let s_index = ((h * k_dim + kk) * v_dim + v_i) as usize;
                    let step = (((t * heads + h) * k_dim + kk) * v_dim + v_i) as usize;
                    steps[step] = state[s_index];
                }
            }
        }
    }
}

pub(crate) fn try_scan<B: Backend>(
    q: BurnTensor<B, 4>,
    k: BurnTensor<B, 4>,
    v: BurnTensor<B, 4>,
    g: BurnTensor<B, 3>,
    beta: BurnTensor<B, 3>,
    state: BurnTensor<B, 4>,
    record: bool,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>, Vec<BurnTensor<B, 4>>)> {
    use burn::backend::wgpu::Metal;
    if std::any::TypeId::of::<B>() != std::any::TypeId::of::<Metal>()
        || size_of::<BurnTensor<B, 4>>() != size_of::<BurnTensor<Metal, 4>>()
    {
        return None;
    }
    let q: BurnTensor<Metal, 4> = unsafe { take_as(q) };
    let k: BurnTensor<Metal, 4> = unsafe { take_as(k) };
    let v: BurnTensor<Metal, 4> = unsafe { take_as(v) };
    let g: BurnTensor<Metal, 3> = unsafe { take_as(g) };
    let beta: BurnTensor<Metal, 3> = unsafe { take_as(beta) };
    let state: BurnTensor<Metal, 4> = unsafe { take_as(state) };
    let (out, state, steps) = scan_metal(q, k, v, g, beta, state, record)?;
    Some((
        unsafe { take_as(out) },
        unsafe { take_as(state) },
        steps
            .into_iter()
            .map(|step| unsafe { take_as(step) })
            .collect(),
    ))
}

/// # Safety
/// `T` and `U` are the same type: the caller checked `TypeId` and size.
unsafe fn take_as<T, U>(value: T) -> U {
    let out = unsafe { std::ptr::read((&value) as *const T as *const U) };
    forget(value);
    out
}

fn scan_metal(
    q: BurnTensor<burn::backend::wgpu::Metal, 4>,
    k: BurnTensor<burn::backend::wgpu::Metal, 4>,
    v: BurnTensor<burn::backend::wgpu::Metal, 4>,
    g: BurnTensor<burn::backend::wgpu::Metal, 3>,
    beta: BurnTensor<burn::backend::wgpu::Metal, 3>,
    state: BurnTensor<burn::backend::wgpu::Metal, 4>,
    record: bool,
) -> Option<(
    BurnTensor<burn::backend::wgpu::Metal, 4>,
    BurnTensor<burn::backend::wgpu::Metal, 4>,
    Vec<BurnTensor<burn::backend::wgpu::Metal, 4>>,
)> {
    let [qb, t_len, heads, k_dim] = q.dims();
    let [kb, kt, kh, kk] = k.dims();
    let [vb, vt, vh, v_dim] = v.dims();
    let [gb, gt, gh] = g.dims();
    let [bb, bt, bh] = beta.dims();
    let [sb, sh, sk, sv] = state.dims();
    if qb != 1
        || kb != 1
        || vb != 1
        || gb != 1
        || bb != 1
        || sb != 1
        || t_len == 0
        || heads == 0
        || k_dim == 0
        || v_dim == 0
        || v_dim > 256
        || (kt, kh, kk) != (t_len, heads, k_dim)
        || (vt, vh, v_dim) != (t_len, heads, v_dim)
        || (gt, gh) != (t_len, heads)
        || (bt, bh) != (t_len, heads)
        || (sh, sk, sv) != (heads, k_dim, v_dim)
        || q.dtype() != DType::F32
    {
        return None;
    }
    // The kernel writes the state buffer. A Burn clone shares that allocation
    // with the caller, so copy before the launch.
    let state_shape = state.dims();
    let state_device = state.device();
    let state = state + BurnTensor::zeros(state_shape, &state_device);
    let g = g.exp();
    let q = cube(q)?;
    let k = cube(k)?;
    let v = cube(v)?;
    let g = cube(g)?;
    let beta = cube(beta)?;
    let mut state = cube(state)?;
    state = burn_cubecl::kernel::into_contiguous(state);
    let client = state.client.clone();
    let device = state.device.clone();
    let out = alloc(&client, &device, [1, t_len, heads, v_dim]);
    let steps = if record {
        alloc(&client, &device, [t_len, heads, k_dim, v_dim])
    } else {
        alloc(&client, &device, [1, 1, 1, 1])
    };
    // Split-kk path: k_dim sliced four ways per column. Falls back to the
    // scalar kernel when the shapes do not divide.
    let split = k_dim % 4 == 0 && v_dim % 4 == 0 && v_dim * 4 <= 1024 && v_dim >= 4;
    unsafe {
        if split {
            gdn_scan_split_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(heads as u32, 1, 1),
                CubeDim::new_1d((v_dim * 4) as u32),
                q.into_tensor_arg(),
                k.into_tensor_arg(),
                v.into_tensor_arg(),
                g.into_tensor_arg(),
                beta.into_tensor_arg(),
                state.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                steps.clone().into_tensor_arg(),
                t_len as u32,
                heads as u32,
                k_dim as u32,
                v_dim as u32,
                u32::from(record),
                4,
            );
        } else {
            gdn_scan_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(heads as u32, 1, 1),
                // One unit per v_dim column: the v-chunk loop degenerates and the
                // whole block runs heads x v_dim threads instead of heads x 32.
                CubeDim::new_1d(v_dim.min(128) as u32),
                q.into_tensor_arg(),
                k.into_tensor_arg(),
                v.into_tensor_arg(),
                g.into_tensor_arg(),
                beta.into_tensor_arg(),
                state.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                steps.clone().into_tensor_arg(),
                t_len as u32,
                heads as u32,
                k_dim as u32,
                v_dim as u32,
                u32::from(record),
            );
        }
    }
    let out = wrap4(out, [1, t_len, heads, v_dim]);
    let state = wrap4(state, [1, heads, k_dim, v_dim]);
    let steps = if record {
        let steps = wrap4(steps, [t_len, heads, k_dim, v_dim]);
        (0..t_len)
            .map(|t| {
                steps
                    .clone()
                    .slice([t..t + 1, 0..heads, 0..k_dim, 0..v_dim])
            })
            .collect()
    } else {
        Vec::new()
    };
    Some((out, state, steps))
}

fn cube<const D: usize>(
    tensor: BurnTensor<burn::backend::wgpu::Metal, D>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let primitive = tensor.into_primitive();
    let TensorPrimitive::Float(tensor) = primitive else {
        return None;
    };
    Some(burn_cubecl::kernel::into_contiguous(tensor))
}

fn alloc(
    client: &cubecl::client::ComputeClient<WgpuRuntime>,
    device: &cubecl::wgpu::WgpuDevice,
    shape: [usize; 4],
) -> CubeTensor<WgpuRuntime> {
    let bytes = shape.iter().product::<usize>() * size_of::<f32>();
    CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from(shape),
        client.empty(bytes),
        DType::F32,
    )
}

#[cfg(test)]
mod tests {
    use super::scan_metal;
    use burn::backend::wgpu::{Metal, WgpuDevice};
    use burn::tensor::{Tensor, TensorData};

    fn tensor4(values: Vec<f32>, shape: [usize; 4]) -> Tensor<Metal, 4> {
        let device = WgpuDevice::default();
        Tensor::from_data(TensorData::new(values, shape), &device)
    }

    fn tensor3(values: Vec<f32>, shape: [usize; 3]) -> Tensor<Metal, 3> {
        let device = WgpuDevice::default();
        Tensor::from_data(TensorData::new(values, shape), &device)
    }

    #[test]
    fn scan_matches_sequential_sum() {
        let (t_len, heads, k_dim, v_dim) = (2, 4, 128, 128);
        let qv: Vec<f32> = (0..t_len * heads * k_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
            .collect();
        let kv: Vec<f32> = (0..t_len * heads * k_dim)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.03)
            .collect();
        let vv: Vec<f32> = (0..t_len * heads * v_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.02)
            .collect();
        let gv: Vec<f32> = (0..t_len * heads)
            .map(|i| -0.05 * (i as f32 + 1.0))
            .collect();
        let bv: Vec<f32> = (0..t_len * heads).map(|i| 0.15 + 0.01 * i as f32).collect();
        let sv: Vec<f32> = (0..heads * k_dim * v_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
            .collect();
        let (out, state, steps) = scan_metal(
            tensor4(qv.clone(), [1, t_len, heads, k_dim]),
            tensor4(kv.clone(), [1, t_len, heads, k_dim]),
            tensor4(vv.clone(), [1, t_len, heads, v_dim]),
            tensor3(gv.clone(), [1, t_len, heads]),
            tensor3(bv.clone(), [1, t_len, heads]),
            tensor4(sv.clone(), [1, heads, k_dim, v_dim]),
            true,
        )
        .expect("scan");
        let mut cpu_state = sv;
        let mut cpu_out = vec![0f32; t_len * heads * v_dim];
        for t in 0..t_len {
            for h in 0..heads {
                let decay = gv[t * heads + h].exp();
                let b = bv[t * heads + h];
                let mut acc = vec![0f32; v_dim];
                for kk in 0..k_dim {
                    for v_i in 0..v_dim {
                        let s = &mut cpu_state[(h * k_dim + kk) * v_dim + v_i];
                        *s *= decay;
                        acc[v_i] += *s * kv[((t * heads + h) * k_dim) + kk];
                    }
                }
                for v_i in 0..v_dim {
                    let delta = (vv[(t * heads + h) * v_dim + v_i] - acc[v_i]) * b;
                    let mut y = 0f32;
                    for kk in 0..k_dim {
                        let s = &mut cpu_state[(h * k_dim + kk) * v_dim + v_i];
                        *s += kv[((t * heads + h) * k_dim) + kk] * delta;
                        y += *s * qv[((t * heads + h) * k_dim) + kk];
                    }
                    cpu_out[(t * heads + h) * v_dim + v_i] = y;
                }
            }
        }
        let got = out.into_data().convert::<f32>();
        let got = got.as_slice::<f32>().unwrap();
        let mut max = 0f32;
        for (a, b) in got.iter().zip(&cpu_out) {
            max = max.max((a - b).abs());
        }
        let got_state = state.into_data().convert::<f32>();
        let got_state = got_state.as_slice::<f32>().unwrap();
        for (a, b) in got_state.iter().zip(&cpu_state) {
            max = max.max((a - b).abs());
        }
        assert_eq!(steps.len(), t_len);
        assert!(max < 1e-4, "scan diff {max}");
    }

    #[test]
    fn burn_sum_versus_sequential() {
        let device = WgpuDevice::default();
        let values: Vec<f32> = (0..128)
            .map(|i| ((i * 17) % 50) as f32 * 0.02 - 0.3)
            .collect();
        let tensor = Tensor::<Metal, 1>::from_data(TensorData::new(values.clone(), [128]), &device);
        let burned = tensor.sum_dim(0).into_data().convert::<f32>();
        let burned = burned.as_slice::<f32>().unwrap()[0];
        let mut sequential = 0f32;
        for value in &values {
            sequential += value;
        }
        assert!((burned - sequential).abs() < 1e-3);
    }
}

fn wrap4(
    tensor: CubeTensor<WgpuRuntime>,
    shape: [usize; 4],
) -> BurnTensor<burn::backend::wgpu::Metal, 4> {
    let _ = shape;
    BurnTensor::from_primitive(TensorPrimitive::Float(tensor))
}
