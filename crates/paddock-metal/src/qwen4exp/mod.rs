//! Native elected Flash Next IQ3-labelled GGUF text graph and diagnostic gates.
//!
//! Explicit Metal selection only; no automatic registry qualification until
//! independent whole-generation and c=1/c=4 serving gates close. The export
//! contains no vision/MTP. Exact paired prefix reuse is still an open target.
mod affine;
mod mlx;
mod plan;
pub use mlx::FlashNextMlxPlan;
#[cfg(test)]
mod affine_tests;
#[cfg(test)]
mod mlx_stage_tests;
pub use plan::FlashNextPlan;
mod model;
pub use model::FlashNext;

// Only the whole-model owner implements Generator. Subgraphs never submit
// or publish model lengths independently inside a serving walk.
mod deltanet;
#[cfg(test)]
mod deltanet_tests;
mod moe;
mod ple;
mod qsa;
#[cfg(test)]
mod qsa_tests;
mod residual;
#[cfg(test)]
mod tests;
