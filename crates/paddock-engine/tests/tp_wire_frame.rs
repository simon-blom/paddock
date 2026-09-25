//! Host-only wire-frame verification for the TP=2 KV-mirror transport
//! (upstream-readiness finding B2). No CUDA, no GPU, no engine device work:
//! builds a real `MirroredKv` at realistic serving geometry, authorizes real
//! logical operations, and measures ACTUAL `ControlMessage` frame sizes
//! through `paddock-dist`'s real `to_frame()` encoder against `MAX_FRAME`.
//!
//! Two designs are measured here:
//! - the OLD (pre-v2) shape: one full-snapshot `Event` per prompt row, which
//!   is what the review flagged as O(rows x context) on the wire;
//! - the NEW (protocol v2) shape: ordered rows plus ONE end-of-tick
//!   `kv_state` snapshot, which must stay bounded for any realistic span.
//!
//! Run with `cargo test -p paddock-engine --test tp_wire_frame`.

use paddock_dist::protocol::{ControlMessage, MAX_FRAME};
use paddock_engine::gpu_model::qwen35::tp_kv::{Event, MirroredKv, Operation, Snapshot};

/// Realistic serving geometry: 16k context, two slots. The KV pool then holds
/// ceil(16384/16)*2 = 2048 physical blocks, so one logical `Snapshot`
/// serializes two slot tables plus 2048 refcounts.
const MAX_CTX: usize = 16_384;
const SLOTS: usize = 2;
const BLOCKS: u32 = (MAX_CTX.div_ceil(16) * SLOTS) as u32;

/// Serialize a `TpSpanLaunch` the way the OLD (v1) wire shape did: one full
/// `Event` (operation + complete resulting `Snapshot`) per prompt row.
fn old_design_span_frame(row_count: usize) -> usize {
    let mut coord = MirroredKv::new(BLOCKS, SLOTS, MAX_CTX).expect("geometry");
    let mut kv_events: Vec<serde_json::Value> = Vec::with_capacity(row_count);
    for position in 0..row_count {
        let event: Event = coord
            .authorize(Operation::Ensure { slot: 0, position })
            .expect("authorize");
        kv_events.push(serde_json::to_value(&event).expect("event json"));
    }
    // The v1 shape no longer exists on the enum; serialize the events as a
    // standalone JSON array - this is byte-identical to what the v1 message
    // carried in its `kv_events` field, so the measurement stays honest.
    let json = serde_json::to_vec(&kv_events).expect("events json");
    json.len() + row_count * 8
}

/// Serialize a `TpSpanLaunch` the way the NEW (v2) wire shape does: ordered
/// rows plus ONE end-of-tick `kv_state` snapshot, through the real enum and
/// real `to_frame()` encoder.
fn new_design_span_frame(row_count: usize) -> usize {
    let mut coord = MirroredKv::new(BLOCKS, SLOTS, MAX_CTX).expect("geometry");
    let ops: Vec<Operation> = (0..row_count)
        .map(|position| Operation::Ensure { slot: 0, position })
        .collect();
    let end: Snapshot = coord.authorize_all(&ops).expect("authorize_all");
    let kv_state = serde_json::to_value(&end).expect("snapshot json");
    let msg = ControlMessage::TpSpanLaunch {
        sequence: 1,
        rows: (0..row_count)
            .map(|position| (0usize, position as u32, position))
            .collect(),
        finishers: Vec::new(),
        kv_state,
    };
    msg.to_frame().expect("frame").len()
}

#[test]
fn b2_old_design_crosses_the_frame_cap_at_realistic_spans() {
    // Small tick: 8 rows (a decode-width mixed tick). Well under the cap.
    let small = old_design_span_frame(8);
    assert!(small < MAX_FRAME as usize, "8 rows must fit: {small}");

    // 512-row prefill span at 16k context: the review's predicted overflow.
    let big = old_design_span_frame(512);
    println!("B2 old design: rows=8 frame={small}B, rows=512 frame={big}B, MAX_FRAME={MAX_FRAME}B");
    assert!(
        big > MAX_FRAME as usize,
        "expected the per-row-snapshot design to exceed the 1 MiB frame cap \
         at 512 rows x 16k context: got {big}B"
    );
}

#[test]
fn b2_old_design_cap_crossing_row_count_at_16k() {
    // Find the first row count that crosses the cap: every prefill span
    // longer than this poisons the pair under the old design.
    let mut crossing = None;
    for rows in [64usize, 128, 192, 256, 384, 512] {
        let size = old_design_span_frame(rows);
        println!("B2 old design: rows={rows} frame={size}B");
        if size > MAX_FRAME as usize {
            crossing = Some(rows);
            break;
        }
    }
    let crossing = crossing.expect("512 rows must cross the cap at 16k context");
    println!("B2 old design: frame cap crossed at {crossing} prompt rows (16k ctx, 2 slots)");
}

#[test]
fn b2_new_design_stays_bounded_far_below_the_cap() {
    // The fixed shape must keep a FULL 4k-token prefill span (and a decode
    // tick) far below MAX_FRAME, and grow only with context, not rows.
    let decode = new_design_span_frame(1);
    let span_512 = new_design_span_frame(512);
    let span_4096 = new_design_span_frame(4_096);
    println!(
        "B2 new design: decode tick={decode}B, 512-row span={span_512}B, \
         4096-row span={span_4096}B, MAX_FRAME={MAX_FRAME}B"
    );
    assert!(decode < MAX_FRAME as usize);
    assert!(span_512 < MAX_FRAME as usize);
    assert!(span_4096 < MAX_FRAME as usize);
    // Headroom invariant: even an extreme 4096-row span (far beyond the
    // coordinator's prefill chunk budget) uses under 1/8 of the frame cap,
    // and growth is ~14 B/row plus one ~4 KB snapshot - not ~4 KB/row as in
    // the old per-row-snapshot shape.
    assert!(
        span_4096 < MAX_FRAME as usize / 8,
        "4096-row span must keep >=8x headroom: {span_4096}B"
    );
}

/// Serialize a `TpBatch` (the smallest mutating command: ordered rows plus
/// the one end-of-tick snapshot) at an arbitrary context with the supported
/// maximum of two slots, through the real `to_frame()` encoder. This is the
/// per-tick floor for the mirror wire: every mutating command carries the
/// same full `Snapshot`, so its frame size bounds the whole family.
fn snapshot_frame_at_context(max_ctx: usize) -> usize {
    let blocks = (max_ctx.div_ceil(16) * SLOTS) as u32;
    let mut coord = MirroredKv::new(blocks, SLOTS, max_ctx).expect("geometry");
    // One decode row: the snapshot dominates; rows add ~40 B each.
    let ops = [Operation::Ensure {
        slot: 0,
        position: 0,
    }];
    let end: Snapshot = coord.authorize_all(&ops).expect("authorize_all");
    let kv_state = serde_json::to_value(&end).expect("snapshot json");
    let msg = ControlMessage::TpBatch {
        sequence: 1,
        rows: vec![(0usize, 7u32, 0usize)],
        kv_state,
    };
    msg.to_frame().expect("frame").len()
}

/// B2 context bound (upstream-readiness review follow-up): the v2 mirror wire
/// sends one full `Snapshot` per mutating command, so frame size grows with
/// CONTEXT. Measure the real serialized frame at the supported geometry
/// ceiling (2 slots) across the candidate context sizes. Measured growth is
/// almost exactly 0.25 B per context token (the refcount array dominates:
/// one u32 per physical block, and blocks = ctx/16 x 2 slots), so even 1M
/// context stays at ~25% of the 1 MiB frame cap and the wire does not cross
/// MAX_FRAME until ~4.19M tokens of context - two orders of magnitude above
/// any context the pinned Qwen3.8 TP lane claims. No startup gate needed;
/// this test pins the measured sizes so a future wire change that breaks
/// the bound fails here first.
#[test]
fn b2_snapshot_frame_growth_and_context_bound() {
    let mut sizes = Vec::new();
    for ctx in [16_384usize, 65_536, 131_072, 262_144, 1_048_576] {
        let size = snapshot_frame_at_context(ctx);
        sizes.push((ctx, size));
        println!(
            "B2 snapshot @ ctx={ctx} (2 slots): frame={size}B ({:.1}% of MAX_FRAME={MAX_FRAME})",
            100.0 * size as f64 / MAX_FRAME as f64
        );
    }
    let cap = MAX_FRAME as usize;
    // Headroom at every measured size: even 1M context is ~25% of the cap,
    // so no documented or plausible TP context comes near the frame limit.
    // (The 1M point is included to anchor the extrapolation below, not as a
    // supported regime.)
    for (ctx, size) in &sizes {
        assert!(
            *size <= cap,
            "snapshot frame at ctx={ctx} must stay under MAX_FRAME: {size}B"
        );
    }
    // Growth is monotone in context (the snapshot's refcount array is the
    // dominant term) and tracks the measured ~0.25 B/token closely enough
    // to extrapolate where the wire would cross the cap.
    for ((c1, s1), (c2, s2)) in sizes.iter().zip(sizes.iter().skip(1)) {
        assert!(s2 > s1, "frame must grow with context: {c1}->{c2}");
    }
    let (c1, s1) = sizes[0];
    let (c2, s2) = sizes[sizes.len() - 1];
    let slope = (s2 - s1) as f64 / (c2 - c1) as f64;
    assert!(
        (0.20..=0.30).contains(&slope),
        "snapshot growth should be ~0.25 B/context-token, got {slope}"
    );
    let extrapolated_crossing = (cap as f64 - s1 as f64) / slope + c1 as f64;
    println!(
        "B2 snapshot: frame cap crossed at ~{extrapolated_crossing:.0} tokens context (2 slots)"
    );
    assert!(
        extrapolated_crossing > 4_000_000.0,
        "the wire bound must stay far above any supported context: {extrapolated_crossing}"
    );
}

#[test]
fn b2_new_design_worker_mirror_accepts_then_fails_closed() {
    // End-to-end over the real wire: coordinator authorizes a tick, the frame
    // is encoded, decoded on the "worker" side, mirrored, and a tampered
    // snapshot is refused.
    let mut coord = MirroredKv::new(BLOCKS, SLOTS, MAX_CTX).expect("geometry");
    let rows: Vec<(usize, u32, usize)> = (0..300)
        .map(|position| (0usize, position as u32, position))
        .collect();
    let ops: Vec<Operation> = rows
        .iter()
        .map(|&(_, _, position)| Operation::Ensure { slot: 0, position })
        .collect();
    let end: Snapshot = coord.authorize_all(&ops).expect("authorize_all");
    let msg = ControlMessage::TpSpanLaunch {
        sequence: 7,
        rows: rows.clone(),
        finishers: Vec::new(),
        kv_state: serde_json::to_value(&end).expect("snapshot json"),
    };
    let frame = msg.to_frame().expect("frame under the cap");
    assert!(
        frame.len() < MAX_FRAME as usize,
        "300 rows @16k: {}",
        frame.len()
    );

    // Round-trip through a real TCP pair, the way the ranks exchange frames.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let sender = std::thread::spawn(move || {
        let mut s = std::net::TcpStream::connect(addr).expect("connect");
        msg.to_stream(&mut s).expect("send");
    });
    let (mut rx, _) = listener.accept().expect("accept");
    let decoded = ControlMessage::from_stream(&mut rx).expect("decode");
    sender.join().expect("sender");
    drop(frame);
    match decoded {
        ControlMessage::TpSpanLaunch {
            sequence,
            rows,
            kv_state,
            ..
        } => {
            assert_eq!(sequence, 7);
            assert_eq!(rows.len(), 300);
            let mut mirror = MirroredKv::new(BLOCKS, SLOTS, MAX_CTX).expect("geometry");
            let ops: Vec<Operation> = rows
                .iter()
                .map(|&(_, _, position)| Operation::Ensure { slot: 0, position })
                .collect();
            let end: Snapshot = serde_json::from_value(kv_state).expect("snapshot parse");
            mirror.mirror_tick(&ops, &end).expect("mirror tick");
            assert_eq!(mirror.snapshot(), coord.snapshot());
            // Tampered end state fails closed.
            let mut bad = end.clone();
            bad.free += 1;
            let mut poisoned = MirroredKv::new(BLOCKS, SLOTS, MAX_CTX).expect("geometry");
            assert!(poisoned.mirror_tick(&ops, &bad).is_err());
        }
        other => panic!("unexpected message: {other:?}"),
    }
}
