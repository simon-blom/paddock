//! A tagged encoder job keeps scheduling independent of tower arithmetic.
//! Gemma's image spans are bidirectional; Muse's backbone remains causal.
use super::*;
use std::time::Duration;

pub(super) enum Tower {
    Gemma(Box<vision::Vision>),
    Muse(Box<muse_vision::Vision>),
}
pub(super) enum Job {
    Gemma(vision::Job),
    Muse(muse_vision::Job),
}
impl Job {
    pub(super) fn cost(&self) -> Duration {
        match self {
            Self::Gemma(j) => j.cost,
            Self::Muse(j) => j.cost,
        }
    }
    pub(super) fn gpu_seconds(&self) -> f64 {
        match self {
            Self::Gemma(j) => j.gpu_seconds,
            Self::Muse(j) => j.gpu_seconds,
        }
    }
}
impl Tower {
    pub(super) fn load(d: &MetalDevice, path: &Path, width: usize, muse: bool) -> Result<Self> {
        if muse {
            muse_vision::Vision::load(d, path, width).map(|v| Self::Muse(Box::new(v)))
        } else {
            vision::Vision::load(d, path, width).map(|v| Self::Gemma(Box::new(v)))
        }
    }
    pub(super) fn start(&self, d: &MetalDevice, images: &[(&[u8], usize, usize)]) -> Result<Job> {
        match self {
            Self::Gemma(v) => v.start(d, images).map(Job::Gemma),
            Self::Muse(v) => v.start(d, images).map(Job::Muse),
        }
    }
    pub(super) fn step(
        &self,
        d: &MetalDevice,
        job: &mut Job,
        budget: Duration,
    ) -> Result<Option<Vec<vision::Output>>> {
        match (self, job) {
            (Self::Gemma(v), Job::Gemma(j)) => v.step(d, j, budget),
            (Self::Muse(v), Job::Muse(j)) => v.step(d, j, budget),
            _ => Err(MetalError::Model(
                "encoder job belongs to another tower".into(),
            )),
        }
    }
}
