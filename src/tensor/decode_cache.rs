//! Append/rewind attention history with optional Metal page storage.
//! The page atlas never escapes this module. A dependency on the preceding
//! reader orders in-place writes even when callers defer GPU readback.
use super::{AttentionOptions, KvCache, attention};
use burn::tensor::{Tensor, backend::Backend};

pub struct DecodeCache<B: Backend> {
    dense: KvCache<B>,
    paged: bool,
    len: usize,
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    store: Option<Store<B>>,
}
impl<B: Backend> DecodeCache<B> {
    pub fn new(paged: bool) -> Self {
        Self {
            dense: KvCache::default(),
            paged,
            len: 0,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            store: None,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn clear(&mut self) {
        *self = Self::new(self.paged);
    }
    pub fn forward(
        &mut self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        options: AttentionOptions,
    ) -> Tensor<B, 4> {
        assert!(
            options.offset <= self.len,
            "cannot skip uninitialized history"
        );
        let [b, h, n, d] = q.dims();
        assert!(b > 0 && n > 0 && d > 0);
        assert_eq!(k.dims(), v.dims());
        assert_eq!(q.dtype(), k.dtype());
        assert_eq!(k.dtype(), v.dtype());
        assert_eq!(q.device(), k.device());
        assert_eq!(k.device(), v.device());
        assert_eq!((k.dims()[0], k.dims()[2], k.dims()[3]), (b, n, d));
        assert!(h.is_multiple_of(k.dims()[1]));
        self.len = options.offset + n;
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.paged
            && (self.store.is_some() || options.offset == 0)
            && options.causal
            && options.window.is_none()
            && omni_kernel::paged::supported::<B>(&q.device(), q.dtype())
        {
            return self
                .store
                .get_or_insert_with(Store::default)
                .forward(q, k, v, options);
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        assert!(
            self.store.is_none(),
            "clear a paged history before changing its attention policy"
        );
        let (k, v) = self.dense.append(k, v, options.offset);
        attention(q, k, v, options)
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
struct Store<B: Backend> {
    atlas: Option<Tensor<B, 5>>,
    reader: Option<Tensor<B, 3>>,
    pages: usize,
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
impl<B: Backend> Default for Store<B> {
    fn default() -> Self {
        Self {
            atlas: None,
            reader: None,
            pages: 0,
        }
    }
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
impl<B: Backend> Store<B> {
    fn forward(
        &mut self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        options: AttentionOptions,
    ) -> Tensor<B, 4> {
        const PAGE: usize = 16;
        let [b, heads, n, d] = q.dims();
        let kv = k.dims()[1];
        let len = options.offset + n;
        let needed = len.div_ceil(PAGE);
        // Geometric growth bounds allocation/copy overhead. Rewinds overwrite
        // a private suffix; no stale position is included in the causal range.
        if needed > self.pages {
            let pages = needed.next_power_of_two();
            let mut next = Tensor::empty([b * pages, 2, PAGE, kv, d], (&q.device(), q.dtype()));
            if let Some(previous) = self.atlas.take() {
                assert_eq!(previous.dims(), [b * self.pages, 2, PAGE, kv, d]);
                for i in 0..b {
                    next = next.slice_assign(
                        [
                            i * pages..i * pages + self.pages,
                            0..2,
                            0..PAGE,
                            0..kv,
                            0..d,
                        ],
                        previous.clone().slice([
                            i * self.pages..(i + 1) * self.pages,
                            0..2,
                            0..PAGE,
                            0..kv,
                            0..d,
                        ]),
                    );
                }
            }
            self.atlas = Some(next);
            self.pages = pages;
        }
        let atlas = self.atlas.take().unwrap();
        assert_eq!(atlas.dims(), [b * self.pages, 2, PAGE, kv, d]);
        assert_eq!(atlas.dtype(), q.dtype());
        assert_eq!(atlas.device(), q.device());
        let table: Vec<u32> = (0..b * self.pages).map(|v| v as u32).collect();
        let pages = self.pages;
        let rows: Vec<u32> = (0..b)
            .flat_map(|i| {
                (options.offset..len).flat_map(move |p| {
                    [p as u32, (i * pages) as u32, (i * pages * PAGE + p) as u32]
                })
            })
            .collect();
        let queries = q.clone().swap_dims(1, 2).reshape([b * n, heads, d]);
        // SAFETY: this private atlas has no external aliases; the old reader is
        // an explicit dependency, and each disjoint slot is within its capacity.
        let atlas = unsafe {
            omni_kernel::paged::write(
                atlas,
                k.swap_dims(1, 2).reshape([b * n, kv, d]),
                v.swap_dims(1, 2).reshape([b * n, kv, d]),
                rows.clone(),
                Vec::new(),
                self.reader.take(),
            )
        };
        let output = if n <= 8 {
            // SAFETY: all causal positions were written, including any rewind.
            unsafe { omni_kernel::paged::attend(queries, &atlas, rows, table) }
        } else {
            // Dense prefill uses the optimized matrix attention path. Only the
            // bounded prefill reads gather history; token decoding reads pages.
            let histories = |which| {
                Tensor::cat(
                    (0..b)
                        .map(|i| {
                            atlas
                                .clone()
                                .slice([
                                    i * self.pages..i * self.pages + needed,
                                    which..which + 1,
                                    0..PAGE,
                                    0..kv,
                                    0..d,
                                ])
                                .reshape([1, needed * PAGE, kv, d])
                                .slice([0..1, 0..len, 0..kv, 0..d])
                                .swap_dims(1, 2)
                        })
                        .collect(),
                    0,
                )
            };
            attention(q, histories(0), histories(1), options)
                .swap_dims(1, 2)
                .reshape([b * n, heads, d])
        };
        self.reader = Some(output.clone());
        self.atlas = Some(atlas);
        output.reshape([b, n, heads, d]).swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{FloatDType, TensorData};
    fn run<B: Backend>(device: &B::Device, precision: FloatDType, tolerance: f32) {
        type Cpu = burn::backend::NdArray;
        for (b, d) in [(1, 8), (2, 128)] {
            let mut tested = DecodeCache::<B>::new(true);
            let mut reference = DecodeCache::<Cpu>::new(false);
            let mut outputs = Vec::new();
            for (step, (offset, n)) in [
                (0, 17),
                (17, 1),
                (18, 15),
                (33, 1),
                (7, 3),
                (10, 70),
                (80, 1),
            ]
            .into_iter()
            .enumerate()
            {
                let data = |h: usize, seed: usize| {
                    TensorData::new(
                        (0..b * h * n * d)
                            .map(|i| ((i * 13 + seed + step * 149) as f32 * 0.071).sin() * 0.7)
                            .collect::<Vec<_>>(),
                        [b, h, n, d],
                    )
                };
                let q = data(4, 11);
                let k = data(2, 23);
                let v = data(2, 37);
                let options = AttentionOptions {
                    causal: true,
                    offset,
                    ..Default::default()
                };
                let expected = reference.forward(
                    Tensor::from_data(q.clone(), &Default::default()),
                    Tensor::from_data(k.clone(), &Default::default()),
                    Tensor::from_data(v.clone(), &Default::default()),
                    options,
                );
                let actual = tested.forward(
                    Tensor::<B, 4>::from_data(q, device).cast(precision),
                    Tensor::<B, 4>::from_data(k, device).cast(precision),
                    Tensor::<B, 4>::from_data(v, device).cast(precision),
                    options,
                );
                // Deliberately defer every readback until after rewinding and
                // overwriting the atlas, exercising the production lazy seam.
                outputs.push((actual, expected));
            }
            for (actual, expected) in outputs {
                let actual = actual.into_data().convert::<f32>().to_vec::<f32>().unwrap();
                let expected = expected.into_data().to_vec::<f32>().unwrap();
                let error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    actual.iter().all(|x| x.is_finite()) && error < tolerance,
                    "error={error}"
                );
            }
            tested.clear();
            assert!(tested.is_empty());
        }
    }
    #[test]
    fn paged_cache_cpu_fallback_rewinds_and_grows() {
        run::<burn::backend::NdArray>(&Default::default(), FloatDType::F32, 1e-6);
    }
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn paged_cache_metal_orders_lazy_readers_before_rewind_and_growth() {
        use burn::backend::wgpu::Metal;
        let device = crate::tensor::tests::metal_device();
        run::<Metal>(&device, FloatDType::F32, 3e-5);
        run::<Metal>(&device, FloatDType::F16, 2e-3);
    }
}
