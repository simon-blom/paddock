//! tic-forestry-v1 (DINOv3 dense prediction) against its golden vectors.
//!
//! There is no llama.cpp to diff a model like this against, so the oracle is
//! the training stack's own output: twelve chips from the two held-out
//! counties, chosen to span all eight classes those counties contain (a run of
//! southern conifer would let a port pass while getting mountain birch wrong),
//! each as the raw `512x512x4` u8 block the model consumed, with the reference
//! argmax raster, the height raster, and - for the first three - full logits.
//!
//! What "matching" means here. The references were produced under bf16
//! autocast on a B200: every GEMM operand rounded to 8 mantissa bits, the
//! decoder's bilinear resize done in bf16. This engine rounds operands to f16
//! (10 bits) and keeps the residual, the norms and the resize in f32, so it is
//! the more exact of the two and they cannot be bit-equal. The gate is
//! therefore three numbers, each far tighter than any structural mistake:
//!   - argmax agreement per chip. Disagreements live on class boundaries where
//!     two logits are within noise of each other; a wrong rope, a dropped
//!     LayerScale, registers left in the grid or a mis-laid conv weight gives
//!     agreement near 1/n_classes, not 99%.
//!   - height error, against a regression whose own RMSE on real canopy is
//!     3.6 m.
//!   - logits, with an absolute tolerance - checked because an argmax agrees
//!     while logits drift, and a port that matches only the argmax is one
//!     numeric change away from disagreeing.
//!
//! The batch legs matter as much as the accuracy ones: serving is batch over
//! an area, so a chip's raster must not depend on its slot or its neighbours.
//!
//! Needs the checkpoint + fixtures under a model root as
//! `tic-forestry-v1/{model.safetensors,config.json,fixtures/}` (or
//! `DINOV3_DIR`). Heavy only in the sense of a 0.6 GB upload.

mod common;

use std::path::Path;

use paddock_engine::gpu_model::dinov3::GpuDinov3Seg;

struct Chip {
    id: String,
    input: Vec<u8>,
    classes: Vec<u8>,
    height: Vec<f32>,
    /// `[ncls][side][side]` f16, the reference layout
    logits: Option<Vec<half::f16>>,
}

fn read_chips(fix: &Path) -> Vec<Chip> {
    let man: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fix.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    let file = |c: &serde_json::Value, k: &str| -> Option<Vec<u8>> {
        let name = c["files"].get(k)?.get("name")?.as_str()?;
        Some(std::fs::read(fix.join(name)).unwrap_or_else(|e| panic!("{name}: {e}")))
    };
    man["chips"]
        .as_array()
        .expect("manifest.chips is an array")
        .iter()
        .map(|c| Chip {
            id: c["id"].as_str().expect("chip id").to_owned(),
            input: file(c, "input").expect("every chip has an input"),
            classes: file(c, "classes").expect("every chip has a class raster"),
            height: file(c, "height")
                .expect("every chip has a height raster")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            logits: file(c, "logits").map(|b| {
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|w| half::f16::from_bits(u16::from_le_bytes(*w)))
                    .collect()
            }),
        })
        .collect()
}

fn say(msg: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{msg}");
}

#[test]
fn golden_vectors_reproduce() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("DINOV3_DIR", &["tic-forestry-v1"]) else {
        return;
    };
    let fix = dir.join("fixtures");
    if !fix.join("manifest.json").exists() {
        common::missing(&format!("golden vectors not found under {}", fix.display()));
        return;
    }
    let chips = read_chips(&fix);
    assert!(chips.len() >= 4, "fixture holds {} chips", chips.len());

    let mut model = GpuDinov3Seg::load_dir(exec, &dir, 4).expect("load tic-forestry-v1");
    let ncls = model.config().n_classes;
    let side = model.config().out_size;
    let px = side * side;
    assert_eq!(chips[0].input.len(), model.chip_bytes());

    // ---- every chip, in batches of four, logits on ----
    let mut ours_cls: Vec<Vec<u8>> = Vec::new();
    let mut ours_h: Vec<Vec<f32>> = Vec::new();
    let mut ours_lg: Vec<Vec<half::f16>> = Vec::new();
    for group in chips.chunks(4) {
        let mut buf = Vec::with_capacity(group.len() * model.chip_bytes());
        for c in group {
            buf.extend_from_slice(&c.input);
        }
        let out = model.segment(&buf, group.len(), true).expect("segment");
        let lg = out.logits.expect("asked for logits");
        for i in 0..group.len() {
            ours_cls.push(out.classes[i * px..(i + 1) * px].to_vec());
            ours_h.push(out.height[i * px..(i + 1) * px].to_vec());
            ours_lg.push(lg[i * px * ncls..(i + 1) * px * ncls].to_vec());
        }
    }

    let mut worst_agree = 1.0f64;
    let mut worst_h_rmse = 0.0f64;
    let mut worst_lg = 0.0f32;
    for (i, c) in chips.iter().enumerate() {
        let same = c
            .classes
            .iter()
            .zip(&ours_cls[i])
            .filter(|(a, b)| a == b)
            .count();
        let agree = same as f64 / px as f64;
        let (mut se, mut hmax) = (0.0f64, 0.0f32);
        for (a, b) in c.height.iter().zip(&ours_h[i]) {
            let d = (a - b).abs();
            se += (d as f64) * (d as f64);
            hmax = hmax.max(d);
        }
        let rmse = (se / px as f64).sqrt();
        let mut line = format!(
            "chip {}: argmax {:.4}% ({} of {px} differ)  height rmse {:.4} m  max {:.3} m",
            c.id,
            agree * 100.0,
            px - same,
            rmse,
            hmax
        );
        if let Some(rl) = &c.logits {
            // reference is [c][y][x], ours [y][x][c]
            let (mut lmax, mut lsum) = (0.0f32, 0.0f64);
            for p in 0..px {
                for k in 0..ncls {
                    let d = (rl[k * px + p].to_f32() - ours_lg[i][p * ncls + k].to_f32()).abs();
                    lmax = lmax.max(d);
                    lsum += d as f64;
                }
            }
            line += &format!(
                "  logits max {:.3}  mean {:.4}",
                lmax,
                lsum / (px * ncls) as f64
            );
            worst_lg = worst_lg.max(lmax);
        }
        say(&line);
        worst_agree = worst_agree.min(agree);
        worst_h_rmse = worst_h_rmse.max(rmse);
    }
    say(&format!(
        "worst: argmax {:.4}%  height rmse {:.4} m  logits max {:.3}",
        worst_agree * 100.0,
        worst_h_rmse,
        worst_lg
    ));

    // A structural mistake lands near 1/ncls agreement and metres of height
    // error; rounding-class noise lands where these sit.
    assert!(worst_agree > 0.99, "argmax agreement {worst_agree}");
    assert!(worst_h_rmse < 0.25, "height rmse {worst_h_rmse} m");
    assert!(worst_lg < 1.5, "logits max abs diff {worst_lg}");

    // ---- a chip's raster is a function of the chip and the pass width ----
    // Not of its slot, and not of what else rode the pass: every op here is
    // per-row or per-chip with a fixed reduction order. That is the contract
    // the serving seam builds on (it runs every pass at one width), and it is
    // bit-exact - anything else means chips are leaking into each other.
    let (c, o1, o2) = (&chips[2].input, &chips[7].input, &chips[10].input);
    let pass = |model: &mut GpuDinov3Seg, order: [&Vec<u8>; 4]| {
        let buf: Vec<u8> = order.iter().flat_map(|x| x.iter().copied()).collect();
        model.segment(&buf, 4, false).expect("segment")
    };
    let first = pass(&mut model, [c, o1, o2, o1]);
    let third = pass(&mut model, [o2, o2, c, o1]);
    assert_eq!(
        first.classes[..px],
        third.classes[2 * px..3 * px],
        "class raster moved with the slot"
    );
    assert!(
        first.height[..px]
            .iter()
            .zip(&third.height[2 * px..3 * px])
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "height bits moved with the slot or the neighbours"
    );
    assert_eq!(
        first.classes[..px],
        ours_cls[2][..],
        "same width, different run, different classes"
    );

    // Across WIDTHS the bits may move: the tensor-core GEMM picks its tiling
    // from how full the grid is, and a narrow projection regroups its partial
    // sums when the row count changes (measured on sm_86: the three narrowest
    // tap projections, between 1, 2 and 3+ chips). It is a reorder class, not
    // an error - reported here, bounded, and kept out of serving by the seam's
    // fixed pass width.
    let alone = model.segment(c, 1, false).expect("segment alone");
    let moved = alone
        .classes
        .iter()
        .zip(&ours_cls[2])
        .filter(|(a, b)| a != b)
        .count();
    let hd = alone
        .height
        .iter()
        .zip(&ours_h[2])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    say(&format!(
        "width 1 vs width 4: {moved} of {px} class pixels, height max {hd:.4} m"
    ));
    assert!(
        moved < px / 1000,
        "{moved} class pixels moved with the pass width"
    );
    assert!(hd < 0.25, "height moved {hd} m with the pass width");

    // ---- refusals ----
    assert!(
        model.segment(&chips[0].input[1..], 1, false).is_err(),
        "short chip accepted"
    );
    let five = vec![0u8; 5 * model.chip_bytes()];
    assert!(
        model.segment(&five, 5, false).is_err(),
        "batch over max_batch accepted"
    );
}

/// Chips a second at a few pass widths - the number every performance change
/// to this tower is judged by. Ignored: it is a measurement, not a gate, and
/// the desktop shares this die. `cargo test -p paddock-engine --test
/// gpu_dinov3_golden -- --ignored --nocapture throughput`
#[test]
#[ignore = "measurement, not a gate"]
fn throughput_at_pass_widths() {
    let Some(dir) = common::model_dir("DINOV3_DIR", &["tic-forestry-v1"]) else {
        return;
    };
    let fix = dir.join("fixtures");
    if !fix.join("manifest.json").exists() {
        common::missing(&format!("golden vectors not found under {}", fix.display()));
        return;
    }
    let chips = read_chips(&fix);
    // 32 is past the serving seam's cap (MAX_PASS_WIDTH) deliberately: it is the
    // number that says whether the cap is still in the right place
    // DINOV3_WIDTHS=1,4,8 overrides the list for a finer sweep on a new card;
    // the default four are the numbers every change is judged by
    let widths: Vec<usize> = std::env::var("DINOV3_WIDTHS")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|w| w.trim().parse().expect("DINOV3_WIDTHS: a list of widths"))
                .collect()
        })
        .unwrap_or_else(|| vec![1, 8, 16, 32]);
    for width in widths {
        // one model per width: the workspace is resident at the load's width
        let Some(exec) = common::gpu_arc() else {
            return;
        };
        let mut model = GpuDinov3Seg::load_dir(exec, &dir, width).expect("load tic-forestry-v1");
        let buf: Vec<u8> = (0..width)
            .flat_map(|i| chips[i % chips.len()].input.iter().copied())
            .collect();
        for _ in 0..2 {
            model.segment(&buf, width, false).expect("segment");
        }
        // best of five timed bursts: the quiet burst is the machine's number
        let passes = (48 / width).max(3);
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..passes {
                model.segment(&buf, width, false).expect("segment");
            }
            best = best.min(t.elapsed().as_secs_f64() / passes as f64);
        }
        say(&format!(
            "width {width:2}: {:7.1} ms a pass  {:6.1} chips/s  (workspace {:.1} MB a chip)",
            best * 1e3,
            width as f64 / best,
            model.workspace_bytes() as f64 / width as f64 / 1e6
        ));
    }
}
