//! Generic tensor-parallel model substrate.
//!
//! This module owns topology and sharding policy that must not be duplicated
//! by individual model families. It deliberately does not know how a model
//! schedules layers or which architecture-specific fast path it may override.

pub(crate) mod attention;
pub(crate) mod control;
pub mod cache;
pub(crate) mod ffn;
pub(crate) mod prefill;

use paddock_models::tensor_slice::{ShardKind, TensorSliceRequest};

use crate::gpu::distributed::Communicator;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TpTopologyError {
    #[error("tensor-parallel world size must be nonzero")]
    EmptyWorld,
    #[error("tensor-parallel rank {rank} is outside world size {world_size}")]
    RankOutOfRange { rank: usize, world_size: usize },
    #[error("axis {axis} cannot split evenly over {world_size} tensor-parallel ranks")]
    UnevenShard { axis: usize, world_size: usize },
}

/// Rank/world identity shared by generic TP components.
///
/// A model should depend on this rather than embedding assumptions such as
/// `rank < 2` or `local_width = width / 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpTopology {
    rank: usize,
    world_size: usize,
}

impl TpTopology {
    pub fn new(rank: usize, world_size: usize) -> Result<Self, TpTopologyError> {
        if world_size == 0 {
            return Err(TpTopologyError::EmptyWorld);
        }
        if rank >= world_size {
            return Err(TpTopologyError::RankOutOfRange { rank, world_size });
        }
        Ok(Self { rank, world_size })
    }

    pub fn from_group<C: Communicator>(group: &C) -> Result<Self, TpTopologyError> {
        Self::new(group.rank(), group.world_size())
    }

    pub fn rank(self) -> usize {
        self.rank
    }

    pub fn world_size(self) -> usize {
        self.world_size
    }

    /// Width of an evenly sharded logical axis.
    pub fn local_len(self, axis: usize) -> Result<usize, TpTopologyError> {
        if axis == 0 || !axis.is_multiple_of(self.world_size) {
            return Err(TpTopologyError::UnevenShard {
                axis,
                world_size: self.world_size,
            });
        }
        Ok(axis / self.world_size)
    }

    pub fn tensor_slice(self, kind: ShardKind) -> TensorSliceRequest {
        TensorSliceRequest {
            kind,
            rank: self.rank,
            world_size: self.world_size,
        }
    }
}

/// Standard linear-projection layouts used by transformer TP.
///
/// Model adapters may override execution for a faster architecture-specific
/// implementation, but these modes describe the default sharding contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpLinearMode {
    Replicated,
    /// Output features are split over ranks; no reduction follows.
    ColumnParallel,
    /// Input features are split over ranks; partial outputs may be summed.
    RowParallel { reduce: bool },
}

impl TpLinearMode {
    pub fn shard_kind(self) -> ShardKind {
        match self {
            Self::Replicated => ShardKind::Replicated,
            Self::ColumnParallel => ShardKind::OutputRows,
            Self::RowParallel { .. } => ShardKind::InputColumns,
        }
    }

    pub fn reduces(self) -> bool {
        matches!(self, Self::RowParallel { reduce: true })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_is_world_size_agnostic() {
        for world_size in [2usize, 3, 4] {
            for rank in 0..world_size {
                let tp = TpTopology::new(rank, world_size).unwrap();
                assert_eq!(tp.rank(), rank);
                assert_eq!(tp.world_size(), world_size);
                assert_eq!(tp.local_len(12).unwrap(), 12 / world_size);
                let req = tp.tensor_slice(ShardKind::OutputRows);
                assert_eq!(req.rank, rank);
                assert_eq!(req.world_size, world_size);
            }
        }
    }

    #[test]
    fn topology_rejects_invalid_or_uneven_layouts() {
        assert!(matches!(
            TpTopology::new(0, 0),
            Err(TpTopologyError::EmptyWorld)
        ));
        assert!(matches!(
            TpTopology::new(2, 2),
            Err(TpTopologyError::RankOutOfRange { .. })
        ));
        assert!(matches!(
            TpTopology::new(0, 3).unwrap().local_len(10),
            Err(TpTopologyError::UnevenShard { .. })
        ));
    }

    #[test]
    fn linear_modes_map_to_storage_shards() {
        assert_eq!(TpLinearMode::Replicated.shard_kind(), ShardKind::Replicated);
        assert_eq!(
            TpLinearMode::ColumnParallel.shard_kind(),
            ShardKind::OutputRows
        );
        assert_eq!(
            TpLinearMode::RowParallel { reduce: true }.shard_kind(),
            ShardKind::InputColumns
        );
        assert!(TpLinearMode::RowParallel { reduce: true }.reduces());
        assert!(!TpLinearMode::RowParallel { reduce: false }.reduces());
    }
}
