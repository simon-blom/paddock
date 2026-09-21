//! Layer-granular scheduling for bidirectional image prefill. Token slicing
//! cannot bound these pauses: every image token needs the full image's KV at
//! each layer. Keep a bounded GPU residual and yield only between layers.
//! This is not GPU preemption or a hard latency deadline; measured costs
//! choose the next group, with at least one layer of forward progress.
use super::*;

pub(super) struct Carry {
    pub(super) x: Buffer,
    pub(super) norm: Buffer,
    // Decode riders overwrite the drafter's staging while this prompt is
    // suspended. Preserve its partial tap set alongside the residuals.
    pub(super) taps: Option<Buffer>,
}
pub(super) struct Phase {
    rows: Vec<(usize, u32, u32)>,
    complete: Vec<(usize, usize, usize)>,
    // Slot-keyed rather than queue-indexed: more requests can arrive while
    // this cohort is suspended. Aborting a member drops the entire phase.
    grants: Vec<(usize, usize, usize)>,
    layer: usize,
    seconds_per_layer: f64,
    prefill_seconds: f64,
    decode_seconds: f64,
    ticks: usize,
    carry: Carry,
}
impl Phase {
    pub(super) fn contains(&self, slot: usize) -> bool {
        self.grants.iter().any(|g| g.0 == slot)
    }
}

impl Gemma4 {
    pub(super) fn begin_prefill_phase(
        &mut self,
        rows: &[(usize, u32, u32)],
        complete: &[(usize, usize, usize)],
        grants: &[usize],
        decodes: usize,
    ) -> Result<()> {
        assert!(
            self.prefill_phase.is_none() && !rows.is_empty() && rows.len() <= self.scratch.rows
        );
        let bytes = rows.len() * self.width * 4;
        self.prefill_phase = Some(Phase {
            rows: rows.to_vec(),
            complete: complete
                .iter()
                .map(|&(s, r, n)| (s, r - decodes, n))
                .collect(),
            grants: self
                .pending
                .iter()
                .zip(grants)
                .filter(|(_, n)| **n > 0)
                .map(|(p, &n)| (p.slot, p.offset, n))
                .collect(),
            layer: 0,
            seconds_per_layer: 0.,
            prefill_seconds: 0.,
            decode_seconds: 0.,
            ticks: 0,
            carry: Carry {
                x: self.device.alloc(bytes)?,
                norm: self.device.alloc(bytes)?,
                taps: self
                    .dflash
                    .as_ref()
                    .map(|_| self.device.alloc(bytes * 5))
                    .transpose()?,
            },
        });
        Ok(())
    }

    pub(super) fn advance_prefill_phase(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> std::result::Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        // All public decode validation happens before this call. Keeping the
        // phase installed here also rejects direct execution on its slots.
        let decode = if decodes.is_empty() {
            Vec::new()
        } else {
            self.execute(decodes, &(0..decodes.len()).collect::<Vec<_>>())?
        };
        let decode_seconds = if decodes.is_empty() {
            0.
        } else {
            self.last_gpu_seconds
        };
        if budget == 0 {
            self.last_gpu_seconds = decode_seconds;
            return Ok((decode, Vec::new()));
        }
        let mut phase = self.prefill_phase.take().expect("installed prefill phase");
        let remaining = self.layers.len() - phase.layer;
        let layers = if decodes.is_empty() {
            remaining
        } else if phase.seconds_per_layer == 0. {
            1 // Obtain a real cost for this row/attention geometry first.
        } else {
            // Muse DFlash already emits in full-block quanta. Keep image
            // service inside that latency class, instead of paying another
            // full dense weight walk every 150 ms. Other lanes stay elected
            // at 150 ms. At least one layer guarantees forward progress.
            let quantum = if self.dflash.is_some() { 0.300 } else { 0.150 };
            ((quantum - decode_seconds).max(0.) / (phase.seconds_per_layer * 1.10))
                .floor()
                .max(1.) as usize
        }
        .min(remaining);
        let outputs = phase.complete.iter().map(|r| r.1).collect::<Vec<_>>();
        let result = self.execute_slice(
            &phase.rows,
            &outputs,
            phase.layer..phase.layer + layers,
            &phase.carry,
        );
        let logits = match result {
            Ok(v) => v,
            Err(e) => {
                // Preserve ownership for the service's abort/error cleanup.
                self.prefill_phase = Some(phase);
                return Err(e.into());
            }
        };
        phase.seconds_per_layer = self.last_gpu_seconds / layers as f64;
        phase.prefill_seconds += self.last_gpu_seconds;
        phase.decode_seconds += decode_seconds;
        phase.ticks += 1;
        self.last_gpu_seconds += decode_seconds;
        phase.layer += layers;
        if phase.layer < self.layers.len() {
            self.prefill_phase = Some(phase);
            return Ok((decode, Vec::new()));
        }
        tracing::debug!(
            rows = phase.rows.len(),
            ticks = phase.ticks,
            prefill_ms = phase.prefill_seconds * 1000.,
            decode_ms = phase.decode_seconds * 1000.,
            "Metal image prefill phase completed"
        );
        for &(slot, offset, n) in &phase.grants {
            let p = self
                .pending
                .iter_mut()
                .find(|p| p.slot == slot)
                .expect("phase owns pending slot");
            assert_eq!(p.offset, offset, "suspended grant cannot advance early");
            p.offset += n;
        }
        for &(slot, _, _) in &phase.complete {
            self.publish(slot)?;
        }
        self.pending.retain(|p| p.offset < p.tokens.len());
        let done = phase
            .complete
            .iter()
            .enumerate()
            .map(|(i, &(s, _, n))| (s, logits[i * self.vocab..(i + 1) * self.vocab].to_vec(), n))
            .collect();
        Ok((decode, done))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_engine::service::MmChunk;

    #[test]
    #[ignore = "requires Gemma Q4 target and canonical BF16 mmproj"]
    fn image_layer_yields_preserve_logits_and_cancellation_ownership() {
        let model = std::env::var("PADDOCK_GEMMA4_GGUF").expect("target");
        let mm = std::env::var("PADDOCK_GEMMA4_MMPROJ").expect("vision");
        let mut m = Gemma4::load(Path::new(&model), 2048, 4, None).unwrap();
        m.attach_vision(Path::new(&mm)).unwrap();
        let setup = |m: &mut Gemma4| {
            m.reset();
            while m.evict() {}
            let requests = [0, 2].map(|slot| {
                (
                    slot,
                    vec![
                        MmChunk::Text(vec![846; 1000]),
                        MmChunk::Image {
                            rgb: vec![if slot == 0 { 255 } else { 17 }; 256 * 256 * 3],
                            w: 256,
                            h: 256,
                        },
                        MmChunk::Text(vec![106, 105, 4368, 107]),
                    ],
                )
            });
            m.admit_images(requests.into());
            while !m.encoding.is_empty() {
                m.step_images();
            }
            // Put both images across the SWA window boundary. Prefixes are
            // committed before the indivisible image-containing chunk.
            for slot in [0, 2] {
                for start in (0..1000).step_by(CHUNK) {
                    let end = (start + CHUNK).min(1000);
                    let rows = (start..end)
                        .map(|p| (slot, 846, p as u32))
                        .collect::<Vec<_>>();
                    m.execute(&rows, &[]).unwrap();
                }
                m.pending
                    .iter_mut()
                    .find(|p| p.slot == slot)
                    .unwrap()
                    .offset = 1000;
            }
            let mut rows = Vec::new();
            let mut done = Vec::new();
            let mut grants = Vec::new();
            for p in &m.pending {
                rows.extend((p.offset..p.tokens.len()).map(|i| (p.slot, p.tokens[i], i as u32)));
                done.push((p.slot, rows.len() - 1, p.tokens.len()));
                grants.push(p.tokens.len() - p.offset);
            }
            (rows, done, grants)
        };
        let (rows, complete, _) = setup(&mut m);
        let expected = m
            .execute(&rows, &complete.iter().map(|p| p.1).collect::<Vec<_>>())
            .unwrap();
        let (rows, complete, grants) = setup(&mut m);
        m.begin_prefill_phase(&rows, &complete, &grants, 0).unwrap();
        let text = vec![2, 105, 2364, 107, 846, 106, 105, 4368, 107];
        let mut next = m.forward_prefill(1, &text).unwrap();
        let mut ticks = 0;
        let completed = loop {
            let token = next
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            let pos = m.slots[1].history.len() as u32;
            let (decode, done) = m.forward_mixed(&[(1, token, pos)], CHUNK).unwrap();
            next = decode;
            ticks += 1;
            if !done.is_empty() {
                break done;
            }
            assert!(ticks < 64);
            for slot in [0, 2] {
                assert_eq!(
                    m.slots[slot].history.len(),
                    1000,
                    "partial KV published early"
                );
                assert_eq!(
                    m.pending.iter().find(|p| p.slot == slot).unwrap().offset,
                    1000
                );
            }
            let layer = m.prefill_phase.as_ref().unwrap().layer;
            let (_, no_progress) = m.forward_mixed(&[], 0).unwrap();
            assert!(no_progress.is_empty());
            assert_eq!(
                m.prefill_phase.as_ref().unwrap().layer,
                layer,
                "zero budget progressed image work"
            );
            assert!(
                m.forward_prefill(0, &text).is_err(),
                "direct prefill overwrote suspended state"
            );
        };
        assert!(
            ticks > 1,
            "test did not actually suspend a language layer group"
        );
        let actual = completed
            .iter()
            .flat_map(|r| r.1.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "yield boundaries changed full GPU logits");
        assert!(m.prefill_phase.is_none() && m.pending.is_empty());

        let (rows, complete, grants) = setup(&mut m);
        m.forward_prefill(1, &text).unwrap();
        m.begin_prefill_phase(&rows, &complete, &grants, 0).unwrap();
        m.forward_mixed(&[(1, 846, text.len() as u32)], CHUNK)
            .unwrap();
        assert!(m.prefill_phase.is_some());
        // Unrelated cancellation cannot discard the running cohort.
        m.prefill_abort(3);
        assert!(m.prefill_phase.is_some());
        // Member cancellation releases the carry and restarts the survivor
        // from its old committed offset, never from a half-built layer stack.
        let allocated = m.device.allocated_bytes();
        m.prefill_abort(0);
        assert!(m.prefill_phase.is_none());
        assert!(m.device.allocated_bytes() < allocated);
        assert_eq!(m.slots[2].history.len(), 1000);
        let (_, done) = m.forward_mixed(&[], CHUNK).unwrap();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].0, 2);
        assert_eq!(done[0].2, complete[1].2);
        assert!(done[0].1.iter().all(|x| x.is_finite()));
        m.reset();
        assert!(m.prefill_phase.is_none());
    }
}
