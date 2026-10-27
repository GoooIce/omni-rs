use burn::tensor::{Tensor, backend::Backend};

/// Resident key/value history. Models choose append, retained suffix or immutable
/// copied prefix; storage and offset truncation use the same implementation.
#[derive(Clone)]
pub struct KvCache<B: Backend> {
    tensors: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
}
impl<B: Backend> Default for KvCache<B> {
    fn default() -> Self {
        Self { tensors: None }
    }
}
impl<B: Backend> KvCache<B> {
    pub fn len(&self) -> usize {
        self.tensors.as_ref().map_or(0, |(k, _)| k.dims()[2])
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn clear(&mut self) {
        self.tensors = None;
    }
    /// Independent storage for a sequence range. Shared prefix caches use this
    /// when splitting an edge so a short prefix cannot pin an entire request.
    pub fn copy_range(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.len());
        if start == end {
            return Self::default();
        }
        let (k, v) = self.tensors.as_ref().expect("nonempty cache");
        let [b, h, _, d] = k.dims();
        let source = [0..b, 0..h, start..end, 0..d];
        let destination = [0..b, 0..h, 0..end - start, 0..d];
        let copy = |x: &Tensor<B, 4>| {
            Tensor::empty([b, h, end - start, d], (&x.device(), x.dtype()))
                .slice_assign(destination.clone(), x.clone().slice(source.clone()))
        };
        Self {
            tensors: Some((copy(k), copy(v))),
        }
    }
    /// Independent storage for the selected slots, in the given order. A
    /// speculative tree verify compacts its accepted path with this.
    pub fn copy_selected(&self, slots: &[usize]) -> Self {
        assert!(slots.iter().all(|&slot| slot < self.len()));
        if slots.is_empty() {
            return Self::default();
        }
        let (k, v) = self.tensors.as_ref().expect("nonempty cache");
        let device = k.device();
        let ids = Tensor::<B, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(
                slots.iter().map(|&slot| slot as i64).collect(),
                [slots.len()],
            ),
            &device,
        );
        let select = |x: &Tensor<B, 4>| x.clone().select(2, ids.clone());
        Self {
            tensors: Some((select(k), select(v))),
        }
    }
    /// Concatenate a logical prefix without changing the retained cache.
    pub fn concat(
        &self,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        assert_eq!(k.dims(), v.dims());
        if offset == 0 {
            return (k, v);
        }
        let (previous_k, previous_v) = self.tensors.as_ref().expect("KV prefix must exist");
        assert!(offset <= self.len());
        let [b, h, _, d] = k.dims();
        let range = [0..b, 0..h, 0..offset, 0..d];
        (
            Tensor::cat(vec![previous_k.clone().slice(range.clone()), k], 2),
            Tensor::cat(vec![previous_v.clone().slice(range), v], 2),
        )
    }
    pub fn append(
        &mut self,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (k, v) = self.concat(k, v, offset);
        self.tensors = Some((k.clone(), v.clone()));
        (k, v)
    }
    pub fn retain_last(&mut self, keep: usize) {
        if keep == 0 {
            self.clear();
            return;
        }
        if let Some((k, v)) = self.tensors.take() {
            let [b, h, n, d] = k.dims();
            let range = [0..b, 0..h, n.saturating_sub(keep)..n, 0..d];
            self.tensors = Some((k.slice(range.clone()), v.slice(range)));
        }
    }
    /// A narrow allocation prevents a static image prefix from pinning target KV.
    pub fn copy_prefix(k: &Tensor<B, 4>, v: &Tensor<B, 4>, count: usize) -> Self {
        assert_eq!(k.dims(), v.dims());
        let [b, h, n, d] = k.dims();
        assert!(count <= n);
        if count == 0 {
            return Self::default();
        }
        let range = [0..b, 0..h, 0..count, 0..d];
        let copy = |x: &Tensor<B, 4>| {
            Tensor::empty([b, h, count, d], (&x.device(), x.dtype()))
                .slice_assign(range.clone(), x.clone().slice(range.clone()))
        };
        Self {
            tensors: Some((copy(k), copy(v))),
        }
    }
}
