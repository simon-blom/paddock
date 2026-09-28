//! Model-supplied state snapshots for the mirrored TP checkpoint lifecycle.
//!
//! The core orders a full-layer preflight before enqueuing any rank-local
//! copies. A model owns the state layout, copy implementation, and stream.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointOp {
    Snapshot,
    Restore,
}

/// A model layer with optional rank-local checkpoint state. Stateless layers
/// implement these methods as no-ops; stateful layers check slot/index and
/// buffer geometry in `validate_checkpoint` before any layer is mutated.
pub trait TpCheckpointState<Context> {
    type Error;

    fn validate_checkpoint(
        &self,
        slot: usize,
        index: u32,
        op: CheckpointOp,
    ) -> Result<(), Self::Error>;
    fn snapshot(&mut self, context: &Context, slot: usize, index: u32) -> Result<(), Self::Error>;
    fn restore(&mut self, context: &Context, slot: usize, index: u32) -> Result<(), Self::Error>;
}

/// Fail on invalid geometry across *all* layers before the first copy. GPU
/// copy failures can still leave a partial snapshot/restore; callers must
/// treat a failed transfer as fatal rather than publish the checkpoint.
pub fn transfer_checkpoint<S, Context>(
    layers: &mut [S],
    context: &Context,
    slot: usize,
    index: u32,
    op: CheckpointOp,
) -> Result<(), S::Error>
where
    S: TpCheckpointState<Context>,
{
    for layer in layers.iter() {
        layer.validate_checkpoint(slot, index, op)?;
    }
    for layer in layers.iter_mut() {
        match op {
            CheckpointOp::Snapshot => layer.snapshot(context, slot, index)?,
            CheckpointOp::Restore => layer.restore(context, slot, index)?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Layer {
        valid: bool,
        snapshots: usize,
        restores: usize,
    }

    impl Layer {
        fn new(valid: bool) -> Self {
            Self {
                valid,
                snapshots: 0,
                restores: 0,
            }
        }
    }

    impl TpCheckpointState<()> for Layer {
        type Error = &'static str;

        fn validate_checkpoint(
            &self,
            _slot: usize,
            _index: u32,
            _op: CheckpointOp,
        ) -> Result<(), Self::Error> {
            if self.valid {
                Ok(())
            } else {
                Err("invalid layer")
            }
        }
        fn snapshot(&mut self, _: &(), _: usize, _: u32) -> Result<(), Self::Error> {
            self.snapshots += 1;
            Ok(())
        }
        fn restore(&mut self, _: &(), _: usize, _: u32) -> Result<(), Self::Error> {
            self.restores += 1;
            Ok(())
        }
    }

    #[test]
    fn invalid_later_layer_cannot_partially_copy_earlier_layers() {
        let mut layers = [Layer::new(true), Layer::new(false)];
        for op in [CheckpointOp::Snapshot, CheckpointOp::Restore] {
            assert_eq!(
                transfer_checkpoint(&mut layers, &(), 1, 3, op),
                Err("invalid layer")
            );
        }
        assert_eq!((layers[0].snapshots, layers[0].restores), (0, 0));
    }

    #[test]
    fn both_directions_visit_every_valid_layer_once() {
        let mut layers = [Layer::new(true), Layer::new(true)];
        transfer_checkpoint(&mut layers, &(), 0, 0, CheckpointOp::Snapshot).unwrap();
        transfer_checkpoint(&mut layers, &(), 0, 0, CheckpointOp::Restore).unwrap();
        for layer in &layers {
            assert_eq!((layer.snapshots, layer.restores), (1, 1));
        }
    }
}
