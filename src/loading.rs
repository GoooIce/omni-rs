//! Cooperative checkpoint loading. Each weight read/upload is an interruption point.
use crate::{Error, Precision, Result, weights::WeightSource};
use burn::tensor::{Tensor, backend::Backend};
use std::sync::atomic::{AtomicBool, Ordering};

pub fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::Interrupted)
    } else {
        Ok(())
    }
}

pub struct CancellableWeights<'a, W, F> {
    weights: &'a W,
    cancel: &'a AtomicBool,
    before: F,
}
impl<'a, W, F: Fn(&str)> CancellableWeights<'a, W, F> {
    pub fn new(weights: &'a W, cancel: &'a AtomicBool, before: F) -> Self {
        Self {
            weights,
            cancel,
            before,
        }
    }
    fn before(&self, name: &str) -> Result<()> {
        check(self.cancel)?;
        (self.before)(name);
        check(self.cancel)
    }
}
impl<B: Backend, W: WeightSource<B>, F: Fn(&str)> WeightSource<B> for CancellableWeights<'_, W, F> {
    fn contains(&self, name: &str) -> bool {
        self.weights.contains(name)
    }
    fn tensor<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        self.before(name)?;
        self.weights.tensor(name, shape, precision, device)
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
        self.before(name)?;
        self.weights
            .rows(name, shape, start, end, precision, device)
    }
}
