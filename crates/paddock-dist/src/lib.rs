//! Distributed control plane for tensor-parallel serving.
//!
//! Phase 2 scope (TP plan): rank/world configuration and process bootstrap.
//! Deliberately NARROW - no engine, no CUDA, no NCCL here. The Communicator
//! and the GPU-side collectives arrive in later phases; this crate only
//! coordinates the two OS processes so the engine phase starts from a working
//! rank0/rank1 skeleton instead of inventing one under time pressure.
//!
//! Topology (fixed by the plan): rank 0 = coordinator, exposes the API and
//! owns scheduling; rank 1 = worker, loads its shard later and never exposes
//! a second API. Transport for control messages is length-prefixed JSON over
//! TCP - good enough for two nodes on a private RoCE fabric, and every
//! message stays far below the jumbo-frame world.

pub mod config;
pub mod protocol;
pub mod worker;
