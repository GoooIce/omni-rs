use crate::{Error, Precision, Result, weights::WeightSource};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

/// Bounded resident vocabulary shards shared by all text towers.
#[derive(Clone)]
pub struct Embedding<B: Backend> {
    shards: Vec<Tensor<B, 2>>,
    width: usize,
    size: usize,
    device: B::Device,
}
impl<B: Backend> Embedding<B> {
    pub const SHARD_ROWS: usize = 8192;
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        size: usize,
        width: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Self::load_with(w, name, size, width, p, d, || Ok(()))
    }
    /// Called before each bounded upload, allowing model loading to be cancelled.
    pub fn load_with<E: From<Error>>(
        w: &impl WeightSource<B>,
        name: &str,
        size: usize,
        width: usize,
        p: Precision,
        d: &B::Device,
        mut before_shard: impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<Self, E> {
        if size == 0 || width == 0 {
            return Err(Error::Checkpoint("empty embedding shape".into()).into());
        }
        let mut shards = Vec::new();
        for start in (0..size).step_by(Self::SHARD_ROWS) {
            before_shard()?;
            shards.push(w.rows(
                name,
                [size, width],
                start,
                (start + Self::SHARD_ROWS).min(size),
                p,
                d,
            )?);
        }
        Ok(Self {
            shards,
            width,
            size,
            device: d.clone(),
        })
    }
    pub fn embed(&self, ids: &[u32]) -> Result<Tensor<B, 2>> {
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= self.size) {
            return Err(Error::Inference(
                "empty or invalid embedding token IDs".into(),
            ));
        }
        let mut output = Vec::new();
        let mut start = 0;
        while start < ids.len() {
            let shard = ids[start] as usize / Self::SHARD_ROWS;
            let mut end = start + 1;
            while end < ids.len() && ids[end] as usize / Self::SHARD_ROWS == shard {
                end += 1;
            }
            let indices: Vec<i32> = ids[start..end]
                .iter()
                .map(|&id| (id as usize % Self::SHARD_ROWS) as i32)
                .collect();
            let indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices, [end - start]),
                &self.device,
            );
            output.push(self.shards[shard].clone().select(0, indices));
            start = end;
        }
        Ok(Tensor::cat(output, 0))
    }
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor<B, 3>> {
        Ok(self.embed(ids)?.reshape([1, ids.len(), self.width]))
    }
    pub fn logits(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        assert_eq!(hidden.dims()[1], self.width);
        assert!(
            hidden.dims()[0] > 0,
            "logits require at least one query row"
        );
        Tensor::cat(
            self.shards
                .iter()
                .map(|w| hidden.clone().matmul(w.clone().transpose()))
                .collect(),
            1,
        )
    }
}
