use crate::{Error, Precision, Result};
pub(crate) mod layout;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use half::{bf16, f16};
pub use layout::{CheckpointLayout, CheckpointStage, TensorSpec};
use safetensors::{Dtype, SafeTensors};
#[cfg(feature = "native")]
use std::collections::BTreeSet;
use std::collections::HashMap;

pub trait WeightSource<B: Backend> {
    fn contains(&self, name: &str) -> bool;
    /// Raw safetensors payload of one tensor (packed int4 bytes, bf16 scales).
    /// Only pack-quantized loading needs it; other sources keep the default.
    fn raw_bytes(&self, _name: &str) -> Result<Vec<u8>> {
        Err(Error::Checkpoint("raw byte access unavailable".into()))
    }
    fn tensor<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, D>>;
    fn rows(
        &self,
        name: &str,
        shape: [usize; 2],
        start: usize,
        end: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>>;
}

impl<B: Backend> WeightSource<B> for Weights {
    fn contains(&self, name: &str) -> bool {
        Weights::contains(self, name)
    }
    fn raw_bytes(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.view(name)?.data().to_vec())
    }
    fn tensor<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        Weights::tensor(self, name, shape, precision, device)
    }
    fn rows(
        &self,
        name: &str,
        shape: [usize; 2],
        start: usize,
        end: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        Weights::rows(self, name, shape, start, end, precision, device)
    }
}

enum Storage {
    Bytes(Vec<u8>),
    #[cfg(feature = "native")]
    Mapped(memmap2::Mmap),
}

struct PreparedTensor<B: Backend> {
    shape: Vec<usize>,
    chunks: Vec<Tensor<B, 1>>,
    elements: usize,
}

/// Browser loader: upload bounded pieces immediately, then discard their CPU bytes.
/// The complete 4 GB checkpoint is never held in WebAssembly linear memory.
pub struct DeviceWeights<B: Backend> {
    tensors: HashMap<String, PreparedTensor<B>>,
    precision: Precision,
    device: B::Device,
}

impl<B: Backend> DeviceWeights<B> {
    pub fn new(precision: Precision, device: &B::Device) -> Self {
        Self {
            tensors: HashMap::new(),
            precision,
            device: device.clone(),
        }
    }

    pub fn add_chunk(
        &mut self,
        name: String,
        shape: Vec<usize>,
        dtype: Dtype,
        offset: usize,
        bytes: &[u8],
    ) -> Result<()> {
        let width = match dtype {
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F32 => 4,
            _ => return Err(Error::Checkpoint("unsupported chunk dtype".into())),
        };
        let total = shape
            .iter()
            .try_fold(1_usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Checkpoint("tensor size overflow".into()))?;
        if shape.is_empty()
            || shape.contains(&0)
            || bytes.is_empty()
            || bytes.len() % width != 0
            || bytes.len() > 64 * 1024 * 1024
        {
            return Err(Error::Checkpoint(
                "invalid tensor chunk; pieces must be nonempty and at most 64 MiB".into(),
            ));
        }
        let elements = bytes.len() / width;
        let entry = self.tensors.entry(name).or_insert_with(|| PreparedTensor {
            shape: shape.clone(),
            chunks: Vec::new(),
            elements: 0,
        });
        if entry.shape != shape
            || entry.elements != offset
            || offset.checked_add(elements).is_none_or(|e| e > total)
        {
            return Err(Error::Checkpoint(
                "tensor chunks must have consistent shapes and contiguous offsets".into(),
            ));
        }
        entry.chunks.push(Weights::upload(
            bytes,
            dtype,
            [elements],
            self.precision,
            &self.device,
        )?);
        entry.elements += elements;
        Ok(())
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<&PreparedTensor<B>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Checkpoint(format!("missing tensor {name}")))?;
        if tensor.shape != shape || tensor.elements != shape.iter().product::<usize>() {
            return Err(Error::Checkpoint(format!(
                "incomplete tensor or mismatched shape: {name}"
            )));
        }
        Ok(tensor)
    }
}

impl<B: Backend> WeightSource<B> for DeviceWeights<B> {
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    fn raw_bytes(&self, _name: &str) -> Result<Vec<u8>> {
        Err(Error::Checkpoint(
            "raw byte access is unavailable for chunked device weights".into(),
        ))
    }
    fn tensor<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
        precision: Precision,
        _: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let tensor = self.get(name, &shape)?;
        Ok(Tensor::cat(tensor.chunks.clone(), 0)
            .reshape(shape)
            .cast(precision.dtype()))
    }
    fn rows(
        &self,
        name: &str,
        shape: [usize; 2],
        start: usize,
        end: usize,
        precision: Precision,
        _: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let tensor = self.get(name, &shape)?;
        if start >= end || end > shape[0] {
            return Err(Error::Checkpoint(format!("invalid row slice: {name}")));
        }
        let start = start * shape[1];
        let end = end * shape[1];
        let mut offset = 0;
        let mut pieces = Vec::new();
        for chunk in &tensor.chunks {
            let len = chunk.dims()[0];
            let low = start.max(offset);
            let high = end.min(offset + len);
            if low < high {
                pieces.push(chunk.clone().slice([low - offset..high - offset]));
            }
            offset += len;
        }
        Ok(Tensor::cat(pieces, 0)
            .reshape([(end - start) / shape[1], shape[1]])
            .cast(precision.dtype()))
    }
}

impl AsRef<[u8]> for Storage {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            #[cfg(feature = "native")]
            Self::Mapped(mapping) => mapping,
        }
    }
}

/// Validated Safetensors files. Native loading maps files read-only and converts
/// one parameter at a time, avoiding a second full FP32 copy of the model.
#[derive(Default)]
pub struct Weights {
    shards: Vec<Storage>,
    names: HashMap<String, usize>,
    /// Parsed headers cached per shard: large checkpoints hold ~100k tensors
    /// and re-parsing a shard header per tensor lookup is quadratic.
    headers: std::cell::RefCell<Vec<Option<std::rc::Rc<safetensors::SafeTensors<'static>>>>>,
}

impl Weights {
    pub fn from_bytes(shards: Vec<Vec<u8>>) -> Result<Self> {
        let mut weights = Self::default();
        for bytes in shards {
            weights.add(Storage::Bytes(bytes))?;
        }
        *weights.headers.get_mut() = vec![None; weights.shards.len()];
        Ok(weights)
    }

    fn add(&mut self, storage: Storage) -> Result<()> {
        let tensors = SafeTensors::deserialize(storage.as_ref())
            .map_err(|e| Error::Checkpoint(e.to_string()))?;
        let shard = self.shards.len();
        for name in tensors.names() {
            if self.names.contains_key(name) {
                return Err(Error::Checkpoint(format!("duplicate tensor {name}")));
            }
            self.names.insert(name.to_string(), shard);
        }
        self.shards.push(storage);
        Ok(())
    }

    #[cfg(feature = "native")]
    pub fn open(directory: &std::path::Path) -> Result<Self> {
        Self::open_named(directory, "model")
    }

    /// Read a Transformers or Diffusers checkpoint without renaming its files.
    #[cfg(feature = "native")]
    pub fn open_named(directory: &std::path::Path, stem: &str) -> Result<Self> {
        if stem.is_empty() || !stem.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            return Err(Error::Checkpoint("invalid checkpoint stem".into()));
        }
        let index = directory.join(format!("{stem}.safetensors.index.json"));
        let files: BTreeSet<String> = if index.is_file() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: HashMap<String, String>,
            }
            let index: Index = serde_json::from_slice(&std::fs::read(index)?)?;
            index.weight_map.into_values().collect()
        } else {
            BTreeSet::from([format!("{stem}.safetensors")])
        };
        let mut result = Self::default();
        for name in files {
            let relative = std::path::Path::new(&name);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(Error::Checkpoint(format!("invalid shard path: {name}")));
            }
            let file = std::fs::File::open(directory.join(relative))?;
            // SAFETY: read-only mapping. The checkpoint must not be modified while loading.
            let mapping = unsafe { memmap2::MmapOptions::new().map(&file)? };
            result.add(Storage::Mapped(mapping))?;
        }
        *result.headers.get_mut() = vec![None; result.shards.len()];
        Ok(result)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains_key(name)
    }

    fn view(&self, name: &str) -> Result<safetensors::tensor::TensorView<'_>> {
        let index = self
            .names
            .get(name)
            .ok_or_else(|| Error::Checkpoint(format!("missing tensor {name}")))?;
        // Lazily parse and cache each shard header (see `headers`).
        let tensors = {
            let mut headers = self.headers.borrow_mut();
            if headers[*index].is_none() {
                let parsed = SafeTensors::deserialize(self.shards[*index].as_ref())
                    .map_err(|e| Error::Checkpoint(e.to_string()))?;
                // SAFETY: the parsed header borrows the mmap in `self.shards`,
                // which is never mutated after open; the Rc outlives views.
                let parsed: SafeTensors<'static> = unsafe { std::mem::transmute(parsed) };
                headers[*index] = Some(std::rc::Rc::new(parsed));
            }
            headers[*index].as_ref().expect("header cached").clone()
        };
        tensors
            .tensor(name)
            .map_err(|e| Error::Checkpoint(e.to_string()))
    }

    pub fn tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let view = self.view(name)?;
        if view.shape() != shape {
            return Err(Error::Checkpoint(format!(
                "{name}: expected {shape:?}, got {:?}",
                view.shape()
            )));
        }
        Self::upload(view.data(), view.dtype(), shape, precision, device)
    }

    /// Slice on the host before uploading: even the embedding table fits WebGPU's
    /// storage-buffer limits, and loading never allocates a whole FP32 vocabulary.
    pub fn rows<B: Backend>(
        &self,
        name: &str,
        shape: [usize; 2],
        start: usize,
        end: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let view = self.view(name)?;
        if view.shape() != shape || start >= end || end > shape[0] {
            return Err(Error::Checkpoint(format!(
                "invalid row slice or shape for {name}"
            )));
        }
        let width = match view.dtype() {
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F32 => 4,
            dtype => {
                return Err(Error::Checkpoint(format!(
                    "unsupported dtype {dtype:?} for {name}"
                )));
            }
        };
        let bytes = &view.data()[start * shape[1] * width..end * shape[1] * width];
        Self::upload(
            bytes,
            view.dtype(),
            [end - start, shape[1]],
            precision,
            device,
        )
    }

    fn upload<B: Backend, const D: usize>(
        bytes: &[u8],
        dtype: Dtype,
        shape: [usize; D],
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let floats: Box<dyn Iterator<Item = f32> + '_> = match dtype {
            Dtype::BF16 => Box::new(
                bytes
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()),
            ),
            Dtype::F16 => Box::new(
                bytes
                    .chunks_exact(2)
                    .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()),
            ),
            Dtype::F32 => Box::new(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            ),
            _ => {
                return Err(Error::Checkpoint(format!(
                    "unsupported weight dtype: {dtype:?}"
                )));
            }
        };
        let data = match precision {
            Precision::F16 => TensorData::new(floats.map(f16::from_f32).collect::<Vec<_>>(), shape),
            Precision::F32 => TensorData::new(floats.collect::<Vec<_>>(), shape),
        };
        let dtype: burn::tensor::DType = precision.dtype().into();
        Ok(Tensor::from_data(data, (device, dtype)))
    }
}
