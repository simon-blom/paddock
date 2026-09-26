//! B-vs-C span divergence trace for the batched TP prefill span prototype.
//! Worker first, then head:
//!   qwen35_tp_abc_probe RANK MASTER_IP MODEL PACK [PORT]
//!
//! Rank 0 loads BOTH the TP=2 rank (B, `forward_prefill_span`) and the
//! existing trusted TP=1 model (C, `prefill`) on the same GPU; the arm
//! order pairs B's collectives (rank 1 drives B with it) and C never
//! enters collectives. With `PADDOCK_TP_ABC_TRACE=1` each arm records one
//! row per named stage (tp_trace.rs) and the probe prints a compact
//! per-stage B-vs-C max-abs for every layer up to (and including) the
//! first stage that exceeds the fixed reporting threshold
//! (`PADDOCK_TP_ABC_STOP`, default 1e-3) - the first material divergence
//! and nothing after it.
//!
//! Comparisons compare equivalent mathematical tensors only:
//! - whole-model stages (layer-in/post-norm/mixer-out/post-mixer/ffn-norm/
//!   ffn-out/layer-out/final-norm) compare row 0 of each engine's
//!   replicated residual/norm planes directly;
//! - sharded rank-local planes (GQA q/k/v/gate, DeltaNet mixed/z, FFN
//!   gate/up) compare rank 0's contiguous shard of the full tensor (rank r
//!   owns output rows [r*local .. (r+1)*local), so rank 0's shard is the
//!   first contiguous half on BOTH sides);
//! - the fused Q|gate planes interleave Q and gate per head (split_qg
//!   layout): compared as interleaved pairs over rank 0's Q rows;
//! - per-head f32 planes (DeltaNet gate/beta/q/k/v) slice the same way
//!   (rank r owns value heads [r*H/2 .. (r+1)*H/2)).
//!
//! Probe-only: no serving path is touched, no guard or tolerance is
//! changed, and the trace is compiled out of hardened builds. C's prefill
//! is graph-captured by default and readbacks are capture-illegal, so the
//! probe pins `PADDOCK_NO_PREFILL_GRAPH=1` itself; B's span path is always
//! eager. The arms' stages accumulate in ONE process-local trace buffer and
//! are partitioned by the b./c. stage-name prefix at drain time.
use paddock_dist::{
    config::{ParallelConfig, Resolved},
    protocol::{ControlMessage, receive_nccl_id, send_nccl_id},
    worker::{connect_worker, coordinate, shutdown_worker},
};
use paddock_engine::{
    gpu::distributed::{NcclCommunicator, create_unique_id},
    gpu::{GpuExecutor, KvDtype},
    gpu_model::qwen35::{
        GpuQwen35, Qwen35TpRank,
        tp_kv::{Event, MirroredKv, Operation},
        tp_trace,
    },
};
use paddock_models::mapped::MappedGguf;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    error::Error,
    io::{Read, Write},
    net::TcpStream,
    path::Path,
    sync::Arc,
    time::Duration,
};

const MAX_CTX: usize = 256;
const SLOTS: usize = 1;
const ROWS: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Action {
    /// One span of `rows` fresh prompt tokens starting at the running position.
    Span { rows: usize },
    /// One decode row continuing after the spans (B's post-span state check).
    Decode { token: u32, position: usize },
}

#[derive(Serialize, Deserialize)]
struct Packet {
    sequence: usize,
    action: Action,
    event: Event,
}

fn cases() -> Vec<(String, Vec<Action>)> {
    vec![(
        "16row".into(),
        vec![
            Action::Span { rows: ROWS },
            Action::Decode {
                token: 42,
                position: ROWS,
            },
        ],
    )]
}

fn operation(action: &Action) -> Operation {
    match action {
        // The coordinator's Ensure covers the span's LAST position; the pool
        // backing it covers every earlier row of the contiguous span.
        Action::Span { rows } => Operation::Ensure {
            slot: 0,
            position: rows.saturating_sub(1),
        },
        Action::Decode { position, .. } => Operation::Ensure {
            slot: 0,
            position: *position,
        },
    }
}

fn send_packet(stream: &mut TcpStream, packet: &Packet) -> Result<(), Box<dyn Error>> {
    let bytes = serde_json::to_vec(packet)?;
    if bytes.len() > 65536 {
        return Err("probe packet exceeds cap".into());
    }
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

fn recv_packet(stream: &mut TcpStream) -> Result<Packet, Box<dyn Error>> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len)?;
    let size = u32::from_le_bytes(len) as usize;
    if size == 0 || size > 65536 {
        return Err("probe packet invalid length".into());
    }
    let mut bytes = vec![0u8; size];
    stream.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn ack(stream: &mut TcpStream, rank: usize) -> Result<(), Box<dyn Error>> {
    if rank == 0 {
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte)?;
        if byte != [0x42] {
            return Err("worker acknowledgement mismatch".into());
        }
    } else {
        stream.write_all(&[0x42])?;
    }
    Ok(())
}

/// Deterministic pseudo-prompt tokens (the probe compares arms against each
/// other, not against English text).
fn prompt_token(i: usize) -> u32 {
    (i * 7919 % 32000) as u32
}

/// Run the span (and its decode continuation, if any) through the B arm.
/// Rank 1 drives the identical forwards so the collectives pair.
fn run_b(
    rank: usize,
    control: &mut TcpStream,
    tp: &mut Qwen35TpRank,
    kv: &mut MirroredKv,
    exec: &GpuExecutor,
    group: &NcclCommunicator,
    actions: &[Action],
) -> Result<(), Box<dyn Error>> {
    tp.reset()?;
    exec.synchronize()?;
    let mut position = 0usize;
    for (sequence, action) in actions.iter().enumerate() {
        let packet = if rank == 0 {
            let event = kv.authorize(operation(action))?;
            let packet = Packet {
                sequence,
                action: action.clone(),
                event,
            };
            send_packet(control, &packet)?;
            packet
        } else {
            let packet = recv_packet(control)?;
            if packet.sequence != sequence || packet.action != *action {
                return Err(format!("sequence={sequence}: command mismatch").into());
            }
            kv.mirror(&packet.event)?;
            packet
        };
        ack(control, rank)?;
        match &packet.action {
            Action::Span { rows } => {
                let tokens: Vec<u32> = (0..*rows).map(|i| prompt_token(position + i)).collect();
                tp.forward_prefill_span(group, kv, 0, &tokens, position)?;
                position += rows;
            }
            Action::Decode {
                token,
                position: pos,
            } => {
                tp.forward_token_slot(group, kv, *token, *pos, 0)?;
                position = pos + 1;
            }
        }
    }
    Ok(())
}

fn max_abs(a: &[f32], b: &[f32]) -> Result<f32, Box<dyn Error>> {
    if a.len() != b.len() {
        return Err(format!("slice length {} != {}", a.len(), b.len()).into());
    }
    let mut m = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if !d.is_finite() {
            return Err("nonfinite stage value".into());
        }
        m = m.max(d);
    }
    Ok(m)
}

/// Gather the named heads (each `head_w` lanes wide) of one full row.
/// Head codes for the mixed map: s*100 + h (s = segment 0=q, 1=k, 2=v),
/// resolved against the full row's [q: 16 heads][k: 16][v: 48] layout.
fn gather_lanes(
    row: &[f32],
    heads: &[usize],
    head_w: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    // Segment geometry of the DeltaNet mixed row (lanes).
    const Q_OFF: usize = 0; // key-q channels: heads 0..16
    const K_OFF: usize = 16 * 128; // key-k channels
    const V_OFF: usize = 2 * 16 * 128; // value channels: heads 0..48
    let seg_lane = |code: usize| -> usize {
        let (s, h) = (code / 100, code % 100);
        match s {
            0 => Q_OFF + h * head_w,
            1 => K_OFF + h * head_w,
            _ => V_OFF + h * head_w,
        }
    };
    let mut out = Vec::with_capacity(heads.len() * head_w);
    for &code in heads {
        let base = if row.len() == 5120 {
            seg_lane(code)
        } else {
            // Plain per-head plane: code IS the head index.
            code * head_w
        };
        let end = base + head_w;
        if end > row.len() {
            return Err(format!("lane gather {base}..{end} outside row {}", row.len()).into());
        }
        out.extend_from_slice(&row[base..end]);
    }
    Ok(out)
}

/// Per-row comparison class for one stage key. All compares act on row 0.
enum Cmp {
    /// Replicated plane: whole rows compare directly.
    Full,
    /// Rank-sharded plane whose rank-0 shard is the first contiguous half
    /// of the full row (GQA head groups and FFN output rows shard this way).
    Half,
    /// Rank-sharded plane whose rank-0 shard is an explicit head-indexed
    /// gather of the full row (DeltaGeometry's value-head bands), each head
    /// `head_w` lanes wide. `heads` = the rank-0 head index list.
    Lanes { heads: &'static [usize], head_w: usize },
    /// Fused Q|gate plane: `split_qg` interleaves Q and gate per head, so
    /// rank 0's Q rows are ALL of the shard's Q lanes and rank 0's heads are
    /// the first contiguous half of the full head list. `head_dim` = 256.
    Qg { head_dim: usize },
}

/// Stage keys in traversal order with their comparison class. Names match
/// the tp_trace stage tails after the `{layer}-` prefix.
const SUBSTAGES: &[(&str, Cmp)] = &[
    ("gqa-qg", Cmp::Qg { head_dim: 256 }),
    ("gqa-q", Cmp::Half),
    ("gqa-k", Cmp::Half),
    ("gqa-v", Cmp::Half),
    ("gqa-gate", Cmp::Half),
    ("gqa-qn", Cmp::Half),
    ("gqa-kn", Cmp::Half),
    ("gqa-rope-q", Cmp::Half),
    ("gqa-rope-k", Cmp::Half),
    ("gqa-attn", Cmp::Half),
    ("gqa-out", Cmp::Half),
    // DeltaNet planes compare through DeltaGeometry's exact rank maps
    // (rank 0 = value heads &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39] - bands, NOT a contiguous half). The mixed
    // shard is key-head q || key-head k || value-head v lanes of those
    // heads; per-head planes gather head-strided at 128 lanes per head.
    // dn-conv is intentionally NOT compared: B's conv runs over the
    // rank-order mixed shard while C's runs over the full mixed row, so
    // its channel lanes/windows are not elementwise mappable.
    (
        "dn-mixed",
        Cmp::Lanes {
            heads: &[
                0, 1, 2, 3, 4, 5, 6, 7, // q segment: key heads 0..8
                100, 101, 102, 103, 104, 105, 106, 107, // k segment: key heads 0..8
                200, 201, 202, 203, 204, 205, 206, 207, // v segment: value head bands
                216, 217, 218, 219, 220, 221, 222, 223, 232, 233, 234, 235,
                236, 237, 238, 239,
            ],
            head_w: 128,
        },
    ),
    (
        "dn-z",
        Cmp::Lanes {
            heads: &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39],
            head_w: 128,
        },
    ),
    (
        "dn-q",
        Cmp::Lanes {
            heads: &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39],
            head_w: 128,
        },
    ),
    (
        "dn-k",
        Cmp::Lanes {
            heads: &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39],
            head_w: 128,
        },
    ),
    (
        "dn-v",
        Cmp::Lanes {
            heads: &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39],
            head_w: 128,
        },
    ),
    (
        "dn-rec",
        Cmp::Lanes {
            heads: &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39],
            head_w: 128,
        },
    ),
    (
        "dn-core",
        Cmp::Lanes {
            heads: &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39],
            head_w: 128,
        },
    ),
    (
        "dn-gate",
        Cmp::Lanes {
            heads: &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39],
            head_w: 1,
        },
    ),
    (
        "dn-beta",
        Cmp::Lanes {
            heads: &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39],
            head_w: 1,
        },
    ),
    ("ffn-gate", Cmp::Half),
    ("ffn-up", Cmp::Half),
];

/// The whole-model spine, per layer, in traversal order (`{k}` = layer).
/// Stage keys without a C-side counterpart (post-norm/ffn-norm: C's fused
/// quantizing norm does not materialize the f32 rows; dn-in: echoed by the
/// B-side only) are intentionally absent - the probe compares stages both
/// arms actually captured.
const SPINE: &[&str] = &[
    "layer-in",
    "mixer-out",
    "post-mixer",
    "ffn-out",
    "layer-out",
];

/// One compare report line; returns the max-abs (None = skipped stage).
fn compare_stage(
    bn: &str,
    cn: &str,
    layer: usize,
    cmp: &Cmp,
    bm: &HashMap<(String, usize), Vec<f32>>,
    cm: &HashMap<(String, usize), Vec<f32>>,
) -> Result<Option<(f32, usize, usize)>, Box<dyn Error>> {
    let (Some(bd), Some(cd)) = (bm.get(&(bn.to_owned(), layer)), cm.get(&(cn.to_owned(), layer)))
    else {
        return Ok(None); // stage absent on one arm (mixer kind / arm)
    };
    let m = match cmp {
        Cmp::Full => max_abs(bd, cd)?,
        Cmp::Half => {
            if bd.len() % 2 != 0 {
                return Err(format!("{bn}: odd shard length {}", bd.len()).into());
            }
            max_abs(&bd[..bd.len() / 2], &cd[..cd.len() / 2])?
        }
        Cmp::Lanes { heads, head_w } => {
            // Both sides gather DeltaGeometry's rank-0 heads: B's shard
            // already IS that gather (identity map over its own rows), C's
            // full row gathers the same head indices. The mixed map is
            // segment-qualified (s*100+h encodes segment s, head h).
            let bg = gather_lanes(bd, heads, *head_w)?;
            let cg = gather_lanes(cd, heads, *head_w)?;
            max_abs(&bg, &cg)?
        }
        Cmp::Qg { head_dim } => {
            // split_qg layout: [head, [head_dim Q || head_dim gate]] per row.
            // Extract every head's Q block; rank 0's local heads are the
            // first contiguous half of the full Q lanes, i.e. ALL of the
            // B-side (shard) Q lanes against the same-length prefix of C's.
            let h = *head_dim;
            let bq: Vec<f32> = bd
                .chunks_exact(2 * h)
                .flat_map(|blk| blk[..h].iter().copied())
                .collect();
            let cq: Vec<f32> = cd
                .chunks_exact(2 * h)
                .flat_map(|blk| blk[..h].iter().copied())
                .collect();
            if cq.len() < bq.len() {
                return Err(format!("{bn}: full plane shorter than shard").into());
            }
            max_abs(&bq, &cq[..bq.len()])?
        }
    };
    Ok(Some((m, bd.len(), cd.len())))
}

/// Print the per-layer trace until the first material divergence. Returns
/// the number of stage pairs actually compared (both arms present) - the
/// caller treats zero pairs as an infrastructure failure even when both
/// buffers are nonempty.
fn compare_arms(
    b: Vec<(String, usize, Vec<f32>)>,
    c: Vec<(String, usize, Vec<f32>)>,
    stop: f32,
) -> Result<usize, Box<dyn Error>> {
    if b.is_empty() {
        return Err("B arm captured zero trace stages (PADDOCK_TP_ABC_TRACE \
                    did not reach the span path?)"
            .into());
    }
    if c.is_empty() {
        return Err("C arm captured zero trace stages (PADDOCK_TP_ABC_TRACE \
                    did not reach the trusted TP=1 prefill, or the single \
                    trace buffer was drained once for both arms?)"
            .into());
    }
    let index = |v: Vec<(String, usize, Vec<f32>)>| -> HashMap<(String, usize), Vec<f32>> {
        v.into_iter().map(|(s, l, d)| ((s, l), d)).collect()
    };
    let bm = index(b);
    let cm = index(c);
    let layers = bm.keys().map(|(_, l)| *l).max().unwrap_or(0) + 1;
    let mut compared = 0usize;
    for layer in 0..layers {
        // Mixer substages first (in traversal order), then the spine.
        for (key, cmp) in SUBSTAGES {
            if let Some((m, bl, cl)) =
                compare_stage(&format!("b.{layer}-{key}"), &format!("c.{layer}-{key}"), layer, cmp, &bm, &cm)?
            {
                compared += 1;
                println!(
                    "layer {layer:>2} {key:<12} len {bl}/{cl} max_abs {m:.3e}"
                );
                if m > stop {
                    println!(
                        "first material divergence: layer {layer} {key} max_abs {m:.3e} > {stop:e}"
                    );
                    return Ok(compared);
                }
            }
        }
        for key in SPINE {
            if let Some((m, bl, cl)) = compare_stage(
                &format!("b.{layer}-{key}"),
                &format!("c.{layer}-{key}"),
                layer,
                &Cmp::Full,
                &bm,
                &cm,
            )? {
                compared += 1;
                println!("layer {layer:>2} {key:<12} len {bl}/{cl} max_abs {m:.3e}");
                if m > stop {
                    println!(
                        "first material divergence: layer {layer} {key} max_abs {m:.3e} > {stop:e}"
                    );
                    return Ok(compared);
                }
            }
        }
    }
    if compared == 0 {
        return Err(
            "no comparable stage pairs between the arms (stage-name mismatch; \
             both buffers nonempty but every lookup missed)"
                .into(),
        );
    }
    // Final norm (layer key 0 on both arms).
    if let Some((m, bl, cl)) = compare_stage("b.0-final-norm", "c.0-final-norm", 0, &Cmp::Full, &bm, &cm)? {
        compared += 1;
        println!("final-norm          len {bl}/{cl} max_abs {m:.3e}");
        if m > stop {
            println!("first material divergence: final-norm max_abs {m:.3e} > {stop:e}");
            return Ok(compared);
        }
    }
    println!("no stage exceeded the reporting threshold {stop:e}");
    Ok(compared)
}

fn run(
    rank: usize,
    control: &mut TcpStream,
    resolved: &Resolved,
    model: &Path,
    pack: &Path,
) -> Result<(), Box<dyn Error>> {
    let id = if rank == 0 {
        let id = create_unique_id()?;
        send_nccl_id(control, &id)?;
        id
    } else {
        receive_nccl_id(control)?
    };
    let exec = Arc::new(GpuExecutor::new(0, pack)?);
    let group = NcclCommunicator::from_resolved(Some(resolved), exec.stream.context(), id)?
        .ok_or("TP group absent")?;
    let map = MappedGguf::open(model)?;
    let mut tp =
        Qwen35TpRank::load_slots(exec.clone(), &map, &group, MAX_CTX, KvDtype::Fp16, SLOTS)?;
    // The C arm lives on rank 0 only (single process, no collectives).
    let mut serial = if rank == 0 {
        Some(GpuQwen35::load(exec.clone(), &map, MAX_CTX)?)
    } else {
        None
    };
    let blocks = u32::try_from(MAX_CTX.div_ceil(paddock_engine::kv_pool::BLOCK_TOKENS) * SLOTS)?;
    let mut kv = MirroredKv::new(blocks, SLOTS, MAX_CTX)?;
    for (name, actions) in cases() {
        // B: the whole-model span traversal (collectives pair with rank 1).
        run_b(rank, control, &mut tp, &mut kv, &exec, &group, &actions)?;
        // C: the trusted TP=1 prefill on the same tokens, fresh state.
        if let Some(s) = &mut serial {
            s.reset();
            let tokens: Vec<u32> = (0..ROWS).map(prompt_token).collect();
            let _ = s.prefill(&tokens)?;
        }
        if rank == 0 {
            let stop: f32 = paddock_models::dev_var!("PADDOCK_TP_ABC_STOP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-3);
            // One drain for the whole case: BOTH arms' stages accumulate in
            // the single process-local trace buffer, so the partition below
            // - not two takes - is what separates the arms. (Two takes
            // back-to-back would give one arm everything and the other
            // nothing: exactly the c_stages=0 failure this fixes.)
            let all = tp_trace::take();
            let mut b = Vec::new();
            let mut c = Vec::new();
            for row in all {
                if row.0.starts_with("b.") {
                    b.push(row);
                } else if row.0.starts_with("c.") {
                    c.push(row);
                }
            }
            println!(
                "abc_probe case={name} rows={ROWS} stop={stop:e} b_stages={} c_stages={}",
                b.len(),
                c.len()
            );
            let compared = compare_arms(b, c, stop)?;
            println!("compared_stage_pairs={compared}");
        }
    }
    group.stream().synchronize()?;
    exec.synchronize()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // C's stage readbacks synchronize the stream, which is capture-illegal:
    // pin eager prefill before anything reads the env (repo pattern:
    // dnc_pf_bench). B's span path is always eager either way.
    if std::env::var_os("PADDOCK_NO_PREFILL_GRAPH").is_none() {
        // SAFETY: single-threaded startup, before any engine env read.
        unsafe { std::env::set_var("PADDOCK_NO_PREFILL_GRAPH", "1") };
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_tp_abc_probe RANK MASTER_IP MODEL PACK [PORT]".into());
    }
    let rank: usize = args[1].parse()?;
    if rank > 1 {
        return Err("rank must be 0 or 1".into());
    }
    let port: u16 = args.get(5).map_or(Ok(11567), |s| s.parse())?;
    let resolved = ParallelConfig {
        tp_size: Some(2),
        rank: Some(rank),
        master_addr: Some(args[2].clone()),
        master_port: Some(port),
    }
    .resolved(false)?
    .ok_or("expected TP=2")?;
    let (mut control, _) = if rank == 0 {
        coordinate(&resolved, false)?
    } else {
        connect_worker(&resolved)?
    };
    control.set_read_timeout(Some(Duration::from_secs(600)))?;
    control.set_write_timeout(Some(Duration::from_secs(600)))?;
    let result = run(
        rank,
        &mut control,
        &resolved,
        Path::new(&args[3]),
        Path::new(&args[4]),
    );
    if rank == 0 {
        shutdown_worker(&mut control, result.is_ok())?;
    } else {
        if let Err(ref error) = result {
            eprintln!("rank1 probe run failed before shutdown: {error}");
        }
        match ControlMessage::from_stream(&mut control)? {
            ControlMessage::Shutdown { graceful } if graceful == result.is_ok() => {}
            other => return Err(format!("unexpected shutdown {other:?}").into()),
        }
    }
    result
}
