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
//! - whole-model stages materialized by both arms (layer-in, mixer-out,
//!   ffn-out, final-norm) compare row 0 of the replicated planes directly;
//! - rank-0 GQA/FFN shards compare against C's matching full-row prefix;
//! - fused Q|gate compares both Q and gate lanes for B's local heads
//!   against the same head-aligned prefix in C;
//! - DeltaNet mixed and value-head planes compare B's local head order
//!   against explicit gathers from C's full row (value-head bands
//!   0..7, 16..23, 32..39). Unmaterialized or incompatible stages remain
//!   unmatched, not silently paired.
//! - the `gqa-*-last` stages compare the FINAL row of each pass (row 15 in
//!   the 16-row case): M-RoPE rotates every row with its own text position,
//!   so the row-0 compares above sit at position 0 where all four axes
//!   agree and a position-staging bug is invisible there. The final row is
//!   where such a bug materializes (~0.175 Q / ~0.181 K before the fix).
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
        let base = if row.len() == 10240 {
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
    /// B is already the rank-0 shard; C is the full row in head/output order.
    Prefix,
    /// B is the rank-0 DeltaNet shard in local head order; C is the full row.
    Lanes { heads: &'static [usize], head_w: usize },
}

/// Explicit stage keys and layout classes. Trace labels are literal `b.KEY`
/// and `c.KEY`; the layer number is stored separately.
const SUBSTAGES: &[(&str, Cmp)] = &[
    ("gqa-qg", Cmp::Prefix),
    ("gqa-q", Cmp::Prefix),
    ("gqa-k", Cmp::Prefix),
    ("gqa-v", Cmp::Prefix),
    ("gqa-gate", Cmp::Prefix),
    ("gqa-qn", Cmp::Prefix),
    ("gqa-qn-last", Cmp::Prefix),
    ("gqa-kn", Cmp::Prefix),
    ("gqa-kn-last", Cmp::Prefix),
    ("gqa-rope-q", Cmp::Prefix),
    // Last-row twins (final span row = highest text position, row 15 in the
    // 16-row case): M-RoPE rotates each row with its own position, so the
    // row-0 compares above sit at position 0 where all four axes agree and
    // a text-position staging bug is invisible. These are where the ~0.175
    // Q / ~0.181 K staging divergence appears (and must collapse to noise).
    ("gqa-rope-q-last", Cmp::Prefix),
    ("gqa-rope-k", Cmp::Prefix),
    ("gqa-rope-k-last", Cmp::Prefix),
    ("gqa-attn", Cmp::Prefix),
    ("gqa-attn-last", Cmp::Prefix),
    ("gqa-out", Cmp::Prefix),
    // DeltaNet planes compare through DeltaGeometry's exact rank maps
    // (rank 0 = value heads &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39] - bands, NOT a contiguous half). The mixed
    // shard is key-head q || key-head k || value-head v lanes of those
    // heads; per-head planes gather head-strided at 128 lanes per head.
    // dn-conv is intentionally NOT compared: C may use fused conv/split
    // and never materialize its conv plane.
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
    ("ffn-gate", Cmp::Prefix),
    ("ffn-up", Cmp::Prefix),
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
        Cmp::Prefix => {
            if cd.len() != bd.len() * 2 {
                return Err(format!("{bn}: expected full row twice shard length, got {}/{}", bd.len(), cd.len()).into());
            }
            max_abs(bd, &cd[..bd.len()])?
        }
        Cmp::Lanes { heads, head_w } => {
            // B's local row is already in rank-0 head order. Gather only
            // C's full row; mixed head codes qualify the q/k/v segment.
            if bd.len() != heads.len() * head_w || cd.len() != bd.len() * 2 {
                return Err(format!("{bn}: incompatible local/full lengths {}/{} for {} heads of width {head_w}", bd.len(), cd.len(), heads.len()).into());
            }
            let cg = gather_lanes(cd, heads, *head_w)?;
            max_abs(bd, &cg)?
        }
    };
    Ok(Some((m, bd.len(), cd.len())))
}

/// Canonical pairs use the trace's literal stage label plus its separate layer
/// index. Only the explicit stage tables are admitted; a shared suffix alone
/// never makes two layouts comparable.
fn compare_arms(
    b: Vec<(String, usize, Vec<f32>)>,
    c: Vec<(String, usize, Vec<f32>)>,
    stop: f32,
) -> Result<usize, Box<dyn Error>> {
    if b.is_empty() || c.is_empty() {
        return Err(format!("empty trace arm: b_stages={} c_stages={}", b.len(), c.len()).into());
    }
    let b_count = b.len();
    let c_count = c.len();
    let index = |v: Vec<(String, usize, Vec<f32>)>| -> HashMap<(String, usize), Vec<f32>> {
        v.into_iter().map(|(s, l, d)| ((s, l), d)).collect()
    };
    let bm = index(b);
    let cm = index(c);
    let layers = bm.keys().map(|(_, l)| *l).max().unwrap_or(0) + 1;
    let mut pairs = Vec::new();
    for layer in 0..layers {
        // Execution order: residual input, mixer projections, reduced output,
        // residual add, FFN projections, reduced output, residual add.
        let stages = std::iter::once(("layer-in", &Cmp::Full))
            .chain(SUBSTAGES.iter().filter(|(key, _)| !key.starts_with("ffn-")).map(|(key, cmp)| (*key, cmp)))
            .chain([("mixer-out", &Cmp::Full), ("post-mixer", &Cmp::Full)])
            .chain(SUBSTAGES.iter().filter(|(key, _)| key.starts_with("ffn-")).map(|(key, cmp)| (*key, cmp)))
            .chain([("ffn-out", &Cmp::Full), ("layer-out", &Cmp::Full)]);
        for (key, cmp) in stages {
            let bn = format!("b.{key}");
            let cn = format!("c.{key}");
            if bm.contains_key(&(bn.clone(), layer)) && cm.contains_key(&(cn.clone(), layer)) {
                pairs.push((layer, key, bn, cn, cmp));
            }
        }
    }
    if bm.contains_key(&("b.final-norm".into(), 0)) && cm.contains_key(&("c.final-norm".into(), 0)) {
        pairs.push((0, "final-norm", "b.final-norm".into(), "c.final-norm".into(), &Cmp::Full));
    }
    // Counts include duplicate capture records (e.g. b.dn-in), which have
    // no additional canonical counterpart. Stages after an early stop are
    // still matched, not spuriously reported as unmatched.
    println!("unmatched_b_stages={} unmatched_c_stages={}",
        b_count - pairs.len(), c_count - pairs.len());
    if pairs.is_empty() {
        return Err("no valid canonical stage pairs between B and C".into());
    }
    let mut compared = 0;
    let mut divergent = false;
    for (layer, key, bn, cn, cmp) in pairs {
        let (m, bl, cl) = compare_stage(&bn, &cn, layer, cmp, &bm, &cm)?
            .ok_or("canonical stage vanished from trace index")?;
        compared += 1;
        println!("layer {layer:>2} {key:<12} len {bl}/{cl} max_abs {m:.3e}");
        if m > stop {
            println!("first material divergence: layer {layer} {key} max_abs {m:.3e} > {stop:e}");
            divergent = true;
            // Keep the four row-last GQA checkpoints together: they are the
            // Task A decision surface, so a pre-RoPE divergence must still
            // report the corresponding post-RoPE Q/K values in this run.
            let required = matches!(
                key,
                "gqa-qn-last" | "gqa-kn-last" | "gqa-rope-q-last" | "gqa-rope-k-last"
            );
            if !required {
                break;
            }
        }
    }
    println!("compared_stage_pairs={compared}");
    if !divergent {
        println!("no stage exceeded the reporting threshold {stop:e}");
    }
    Ok(compared)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_stage_names_and_layer_index_pair() {
        let b = vec![("b.layer-in".into(), 3, vec![1.0, 2.0])];
        let c = vec![("c.layer-in".into(), 3, vec![1.0, 2.0])];
        assert_eq!(compare_arms(b, c, 1e-3).unwrap(), 1);
        assert!(compare_arms(
            vec![("b.dn-in".into(), 0, vec![1.0])],
            vec![("c.layer-in".into(), 0, vec![1.0])],
            1e-3,
        ).is_err());
    }

    #[test]
    fn shard_compares_whole_local_row_to_exact_full_gather() {
        let bm = HashMap::from([(("b.ffn-gate".into(), 0), vec![1.0, 2.0])]);
        let cm = HashMap::from([(("c.ffn-gate".into(), 0), vec![1.0, 2.0, 9.0, 9.0])]);
        assert_eq!(compare_stage("b.ffn-gate", "c.ffn-gate", 0, &Cmp::Prefix, &bm, &cm).unwrap().unwrap().0, 0.0);
        let bm = HashMap::from([(("b.dn-z".into(), 0), vec![1.0, 2.0])]);
        let cm = HashMap::from([(("c.dn-z".into(), 0), vec![1.0, 9.0, 2.0, 9.0])]);
        assert_eq!(compare_stage("b.dn-z", "c.dn-z", 0, &Cmp::Lanes { heads: &[0, 2], head_w: 1 }, &bm, &cm).unwrap().unwrap().0, 0.0);
    }
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
            compare_arms(b, c, stop)?;
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
