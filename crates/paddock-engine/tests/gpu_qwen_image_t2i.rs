//! Qwen-Image-2.1 text-to-image, end to end on the GPU: the DiT GGUF, the
//! Qwen3-VL text encoder and the official VAE, one seeded prompt at 1024^2,
//! a PNG out. The parity leg against stable-diffusion.cpp on the same files
//! and seed is the image-parity script; this gate proves the spine runs
//! and writes the image that leg compares.
//!
//! Heavy (three model uploads). Inputs, each with the env override the
//! other gates use:
//!   PADDOCK_QWEN_IMAGE_DIT  (default: Qwen-Image-2.1-GGUF/qwen-image-2.1-Q8_0.gguf)
//!   PADDOCK_QWEN_IMAGE_TE   (default: Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q8_0.gguf)
//!   PADDOCK_QWEN_IMAGE_VAE  (default: Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors)
//!   PADDOCK_QWEN_IMAGE_DIT_Q4 / PADDOCK_QWEN_IMAGE_TE_Q4 - the compact lane's
//!   pair (defaults: the Q4_K_M files beside the ones above)
//!   PADDOCK_QI_STEPS (default 4), PADDOCK_QI_SIZE (default 1024),
//!   PADDOCK_QI_SEED (default 42), PADDOCK_QI_PROMPT, PADDOCK_QI_OUT (png path)
//!
//! The device-memory lines the first gate prints are how the catalog row's
//! numbers were established: `weights_breakdown` per part, and free VRAM
//! before the load, after it, and after the render (what the render keeps).

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen_image::{
    GenerateRequest, IMAGE_CHANNELS, LATENT_CHANNELS, QwenImage, Reference, Rgba, VaeDecoder,
    VaeEncoder, t2i_prompt, ti2i_prompt,
};
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const DIT: &[&str] = &["Qwen-Image-2.1-GGUF/qwen-image-2.1-Q8_0.gguf"];
const TE: &[&str] = &["Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q8_0.gguf"];
const VAE: &[&str] = &["Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors"];
const MMPROJ: &[&str] = &["Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf"];
const DIT_Q4: &[&str] = &["Qwen-Image-2.1-GGUF/qwen-image-2.1-Q4_K_M.gguf"];
const TE_Q4: &[&str] = &["Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf"];

/// The compact lane against the full one, same seed, 4 steps at 1024^2.
/// Measured 2026-09-22 on the A6000: Q4_K_M DiT + Q4_K_M text encoder sits
/// well above this against Q8_0 + Q8_0 (the number is in the plan note); a
/// lane whose quant path is broken lands near 10 dB, where unrelated images
/// live, so the floor separates "a slightly different picture" from "not
/// this picture" with room on both sides.
const COMPACT_LANE_PSNR_FLOOR_DB: f64 = 20.0;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Tokenize the model's raw template with the text encoder's own tokenizer,
/// checking the control tokens resolve as single ids (or the system-block
/// drop count would be wrong). Returns `(ids, dropped)`.
fn tokenize(te: &Path, prompt: &str) -> (Vec<u32>, usize) {
    let te_map = MappedGguf::open(te).expect("open text encoder");
    let tok = GgufTokenizer::from_gguf(te_map.gguf()).expect("tokenizer");
    let (system, full) = t2i_prompt(prompt);
    let sys_ids = tok.encode(&system).expect("encode system");
    let ids = tok.encode(&full).expect("encode prompt");
    assert_eq!(
        &ids[..sys_ids.len()],
        &sys_ids[..],
        "system block must be a prefix of the prompt's ids"
    );
    assert_eq!(
        tok.encode("<|im_start|>")
            .expect("encode control token")
            .len(),
        1,
        "<|im_start|> must be one control token"
    );
    (ids, sys_ids.len())
}

fn free_gb(exec: &GpuExecutor) -> f64 {
    exec.device_mem_info()
        .map_or(f64::NAN, |(free, _)| free as f64 / 1e9)
}

/// Load the three parts and render one seeded image; prints the ledger and
/// the device-memory brackets the catalog row is priced from.
fn render(
    exec: Arc<GpuExecutor>,
    dit: &Path,
    te: &Path,
    vae: &Path,
    prompt: &str,
    size: usize,
    steps: usize,
    seed: u64,
) -> Rgba {
    let (ids, drop) = tokenize(te, prompt);
    eprintln!(
        "prompt: {} ids, {drop} dropped as the system block",
        ids.len()
    );
    let before = free_gb(&exec);
    let t0 = Instant::now();
    let mut model =
        QwenImage::load(exec.clone(), dit, te, vae, None, 4096).expect("load qwen-image");
    let (d, t, v) = model.weights_breakdown();
    eprintln!(
        "loaded in {:.1}s: {:.2} GB resident weights = dit {d} + text {t} + vae {v} bytes; \
         free VRAM {before:.2} -> {:.2} GB",
        t0.elapsed().as_secs_f32(),
        model.weights_bytes() as f64 / 1e9,
        free_gb(&exec)
    );

    let t1 = Instant::now();
    let img = model
        .generate(&GenerateRequest {
            prompt_ids: &ids,
            drop,
            negative: None,
            width: size,
            height: size,
            steps,
            seed,
            noise_offset: 0,
            guidance: 1.0,
            references: &[],
            image_pad_id: 0,
        })
        .expect("generate");
    let gen_s = t1.elapsed().as_secs_f32();
    eprintln!(
        "{}x{} in {gen_s:.1}s ({:.2} s/step incl. prefix+decode); free VRAM after {:.2} GB",
        img.width,
        img.height,
        gen_s / steps as f32,
        free_gb(&exec)
    );

    // an image, not a constant: the decoder produced structure
    let (mut lo, mut hi) = (255u8, 0u8);
    for &p in &img.pixels {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    assert!(
        hi - lo > 32,
        "flat output ({lo}..{hi}) - the pipeline produced no image"
    );
    img
}

fn save(img: Rgba, out: &str) {
    let rgba = image::RgbaImage::from_raw(img.width as u32, img.height as u32, img.pixels)
        .expect("rgba buffer");
    rgba.save(out).expect("write png");
    eprintln!("wrote {out}");
}

/// PSNR over the colour planes (alpha idles a few counts under 255 on every
/// lane and would only add noise to the number).
fn psnr_rgb(a: &Rgba, b: &Rgba) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut se = 0f64;
    let mut n = 0usize;
    for (pa, pb) in a
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.pixels.as_chunks::<4>().0)
    {
        for c in 0..3 {
            let d = pa[c] as f64 - pb[c] as f64;
            se += d * d;
        }
        n += 3;
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

#[test]
fn qwen_image_t2i_renders_a_seeded_image() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dit) = common::model("PADDOCK_QWEN_IMAGE_DIT", DIT) else {
        return;
    };
    let Some(te) = common::model("PADDOCK_QWEN_IMAGE_TE", TE) else {
        return;
    };
    let Some(vae) = common::model("PADDOCK_QWEN_IMAGE_VAE", VAE) else {
        return;
    };

    let steps: usize = env_or("PADDOCK_QI_STEPS", 4);
    let size: usize = env_or("PADDOCK_QI_SIZE", 1024);
    let seed: u64 = env_or("PADDOCK_QI_SEED", 42);
    let prompt = std::env::var("PADDOCK_QI_PROMPT")
        .unwrap_or_else(|_| "a lovely cat holding a sign that says 'paddock'".to_owned());
    let out = std::env::var("PADDOCK_QI_OUT").unwrap_or_else(|_| {
        format!(
            "{}/qwen-image-t2i-{size}-s{steps}-seed{seed}.png",
            env!("CARGO_MANIFEST_DIR")
        )
    });

    let img = render(exec, &dit, &te, &vae, &prompt, size, steps, seed);
    let alpha_opaque = img
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[3] == 255)
        .count();
    eprintln!(
        "alpha: {} of {} pixels opaque",
        alpha_opaque,
        img.width * img.height
    );
    save(img, &out);
}

/// The band-tiled tail of the VAE decoder against the whole-plane decode of
/// the same latent: the halo covers every convolution's reach, so the bytes
/// are identical - not close, identical - while the transient is bounded by
/// a band. A 512 x 1024 picture (32 x 64 latents, 256 rows at the band
/// resolution) at 16 band rows is sixteen bands, every one with a halo on
/// at least one side; the default band height is checked the same way.
#[test]
fn qwen_image_vae_bands_decode_exactly() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(vae) = common::model("PADDOCK_QWEN_IMAGE_VAE", VAE) else {
        return;
    };
    let vae = VaeDecoder::load(exec.clone(), &vae).expect("load vae");
    let (lw, lh) = (32usize, 64usize);
    let mut z = exec.alloc(lw * lh * LATENT_CHANNELS).expect("latent");
    exec.dit_philox_randn(&mut z, 7, 0, lw * lh, LATENT_CHANNELS)
        .expect("noise");
    let whole = vae.decode_with(&z, lw, lh, None).expect("whole decode");
    assert_eq!(whole.len(), 16 * lw * 16 * lh * 4);
    for (label, band) in [("16-row bands", Some(16)), ("the default bands", None)] {
        let banded = match band {
            Some(b) => vae.decode_with(&z, lw, lh, Some(b)),
            None => vae.decode(&z, lw, lh),
        }
        .expect("banded decode");
        assert_eq!(banded.len(), whole.len());
        let differing = whole.iter().zip(&banded).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing,
            0,
            "{label}: {differing} of {} bytes differ from the whole-plane decode - the halo \
             does not cover the tail's reach",
            whole.len()
        );
    }
    eprintln!("banded VAE decode is byte-identical to the whole-plane decode at {lw}x{lh} latents");
}

/// The VAE encoder against the decoder: a synthetic picture (gradients, a
/// few flat discs, a checker patch - the content a VAE reconstructs well)
/// encoded to latents and decoded back must come out as the same picture.
/// A broken stage lands near 10 dB, where unrelated images live; the floor
/// sits well under what the roundtrip measures so it separates "wrong
/// arithmetic" from "a VAE's own blur". The sd.cpp `-r` leg is the
/// end-to-end arbiter for editing; this is the in-engine half.
#[test]
fn qwen_image_vae_roundtrip_reconstructs_the_picture() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(vae) = common::model("PADDOCK_QWEN_IMAGE_VAE", VAE) else {
        return;
    };
    let (w, h) = (512usize, 384usize);
    let mut rgba = vec![0f32; w * h * IMAGE_CHANNELS];
    let discs = [
        (120.0, 100.0, 60.0, [0.9, 0.2, 0.1]),
        (380.0, 250.0, 80.0, [0.1, 0.4, 0.9]),
        (260.0, 300.0, 40.0, [0.2, 0.8, 0.3]),
    ];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let mut px = [fx, fy, 0.5 + 0.5 * ((fx * 6.0).sin() * (fy * 4.0).cos())];
            for (cx, cy, r, colour) in &discs {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                if dx * dx + dy * dy < r * r {
                    px = *colour;
                }
            }
            if x > 400 && y > 300 && ((x / 16) + (y / 16)) % 2 == 0 {
                px = [0.05, 0.05, 0.05];
            }
            let o = (y * w + x) * IMAGE_CHANNELS;
            for c in 0..3 {
                rgba[o + c] = px[c] * 2.0 - 1.0;
            }
            rgba[o + 3] = 1.0;
        }
    }
    let enc = VaeEncoder::load(exec.clone(), &vae).expect("load vae encoder");
    let dec = VaeDecoder::load(exec.clone(), &vae).expect("load vae decoder");
    eprintln!(
        "vae encoder {:.0} MB + decoder {:.0} MB resident",
        enc.weights_bytes as f64 / 1e6,
        dec.weights_bytes as f64 / 1e6
    );
    let d_rgba = exec.to_device(&rgba).expect("upload");
    let z = enc.encode(&d_rgba, w, h).expect("encode");
    let (lw, lh) = (w / 16, h / 16);
    let back = dec.decode(&z, lw, lh).expect("decode");
    assert_eq!(back.len(), rgba.len());
    let mut se = 0f64;
    let mut alpha_lo = 255u8;
    for (i, chunk) in back.chunks(4).enumerate() {
        for c in 0..3 {
            let want = ((rgba[i * 4 + c] * 0.5 + 0.5) * 255.0).round();
            let d = chunk[c] as f64 - want as f64;
            se += d * d;
        }
        alpha_lo = alpha_lo.min(chunk[3]);
    }
    let mse = se / (w * h * 3) as f64;
    let db = 10.0 * (255.0f64 * 255.0 / mse).log10();
    eprintln!("vae roundtrip at {w}x{h}: {db:.2} dB PSNR (rgb), alpha floor {alpha_lo}");
    if let Ok(out) = std::env::var("PADDOCK_QI_OUT") {
        save(
            Rgba {
                width: w,
                height: h,
                pixels: back,
            },
            &out,
        );
    }
    assert!(
        db >= 24.0,
        "vae roundtrip is {db:.2} dB - the encoder does not invert the decoder"
    );
    assert!(
        alpha_lo >= 200,
        "opaque input decoded with alpha {alpha_lo}"
    );
}

/// The editing lane end to end: the model with its vision tower renders a
/// reference (text-to-image), then edits it with an instruction - the
/// reference through the VAE encoder into the DiT prefix and through the
/// tower into the text encoder (DeepStack, multi-axis rope), the target
/// behind both. The edit must be a picture of the reference (well above the
/// floor where unrelated images live) and not the reference itself. Both
/// PNGs are written to `PADDOCK_QI_OUT_DIR` for the sd.cpp `-r` leg
/// (the image-parity script's `--ref` leg), which is the arbiter.
#[test]
fn qwen_image_edit_keeps_the_reference() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dit) = common::model("PADDOCK_QWEN_IMAGE_DIT", DIT) else {
        return;
    };
    let Some(te) = common::model("PADDOCK_QWEN_IMAGE_TE", TE) else {
        return;
    };
    let Some(vae) = common::model("PADDOCK_QWEN_IMAGE_VAE", VAE) else {
        return;
    };
    let Some(mmproj) = common::model("PADDOCK_QWEN_IMAGE_MMPROJ", MMPROJ) else {
        return;
    };
    let te_map = MappedGguf::open(&te).expect("open text encoder");
    let tok = GgufTokenizer::from_gguf(te_map.gguf()).expect("tokenizer");
    let pad = tok.encode("<|image_pad|>").expect("encode pad");
    assert_eq!(pad.len(), 1, "<|image_pad|> must be one control token");
    let pad_id = pad[0];
    drop(te_map);

    let (size, steps, seed) = (1024usize, env_or("PADDOCK_QI_STEPS", 4usize), 42u64);
    let ref_prompt = "a lovely cat holding a sign that says 'paddock'";
    let edit_prompt = std::env::var("PADDOCK_QI_EDIT")
        .unwrap_or_else(|_| "Make it a pencil sketch on white paper".to_owned());

    let t0 = Instant::now();
    let mut model = QwenImage::load(exec.clone(), &dit, &te, &vae, Some(&mmproj), 8192)
        .expect("load qwen-image with its tower");
    assert!(model.can_edit(), "the tower did not wire the editing lane");
    eprintln!(
        "loaded in {:.1}s: {:.2} GB resident, edit lane {:.2} GB",
        t0.elapsed().as_secs_f32(),
        model.weights_bytes() as f64 / 1e9,
        model.edit_bytes() as f64 / 1e9
    );

    // the reference: a text-to-image render at the output size
    let (system, full) = t2i_prompt(ref_prompt);
    let sys = tok.encode(&system).expect("encode");
    let ids = tok.encode(&full).expect("encode");
    let reference = model
        .generate(&GenerateRequest {
            prompt_ids: &ids,
            drop: sys.len(),
            negative: None,
            width: size,
            height: size,
            steps,
            seed,
            noise_offset: 0,
            guidance: 1.0,
            references: &[],
            image_pad_id: pad_id,
        })
        .expect("render the reference");

    // an edit: the template's one image slot expanded to the picture's
    // merged grid, the picture as RGBA in [-1, 1]
    let grid = (size / 32) * (size / 32);
    let rgba: Vec<f32> = reference
        .pixels
        .iter()
        .map(|&v| v as f32 / 255.0 * 2.0 - 1.0)
        .collect();
    let mut edit = |instruction: &str| -> Rgba {
        let (system, full) = ti2i_prompt(instruction, 1);
        let sys = tok.encode(&system).expect("encode");
        let raw = tok.encode(&full).expect("encode");
        assert_eq!(&raw[..sys.len()], &sys[..]);
        let mut ids = Vec::with_capacity(raw.len() + grid);
        for &id in &raw {
            if id == pad_id {
                ids.extend(std::iter::repeat_n(pad_id, grid));
            } else {
                ids.push(id);
            }
        }
        assert_eq!(
            ids.len(),
            raw.len() + grid - 1,
            "the template carries exactly one image slot"
        );
        let t1 = Instant::now();
        let out = model
            .generate(&GenerateRequest {
                prompt_ids: &ids,
                drop: sys.len(),
                negative: None,
                width: size,
                height: size,
                steps,
                seed,
                noise_offset: 0,
                guidance: 1.0,
                references: &[Reference {
                    rgba: &rgba,
                    width: size,
                    height: size,
                }],
                image_pad_id: pad_id,
            })
            .expect("edit");
        eprintln!(
            "'{instruction}': edited in {:.1}s ({} prompt tokens incl. {grid} of the picture)",
            t1.elapsed().as_secs_f32(),
            ids.len() - sys.len()
        );
        assert_eq!((out.width, out.height), (size, size));
        out
    };
    // Two instructions, two invariants. "Leave it as it is" must come back
    // as the reference - the model reconstructing the picture it was handed
    // is the lane working end to end (tower, encoder, prefix), and a
    // scrambled position or a wrong latent lands near 10 dB with everything
    // else. The real edit must then move it: further from the reference than
    // the identity did, but a picture still (finite, never the reference's
    // own bytes).
    let same = edit("Keep this picture exactly as it is, change nothing");
    let same_db = psnr_rgb(&reference, &same);
    let edited = edit(&edit_prompt);
    let edit_db = psnr_rgb(&reference, &edited);
    eprintln!(
        "identity edit vs reference: {same_db:.2} dB; '{edit_prompt}' vs reference: {edit_db:.2} dB (rgb PSNR)"
    );
    if let Ok(dir) = std::env::var("PADDOCK_QI_OUT_DIR") {
        save(
            reference,
            &format!("{dir}/edit-ref-{size}-s{steps}-seed{seed}.png"),
        );
        save(
            same,
            &format!("{dir}/edit-same-{size}-s{steps}-seed{seed}.png"),
        );
        save(
            edited,
            &format!("{dir}/edit-out-{size}-s{steps}-seed{seed}.png"),
        );
    }
    assert!(
        same_db >= 14.0,
        "asked to change nothing, the edit is {same_db:.2} dB from its reference - the picture \
         did not reach the model"
    );
    assert!(
        edit_db.is_finite(),
        "the edit returned the reference byte for byte - the instruction did nothing"
    );
    assert!(
        edit_db + 1.0 < same_db,
        "the instruction moved the picture no further ({edit_db:.2} dB) than leaving it alone did \
         ({same_db:.2} dB)"
    );
}

/// The compact lane (Q4_K_M DiT + Q4_K_M text encoder - the pair the catalog
/// bundles) renders the same seeded picture the full lane does. The reference
/// for BOTH lanes is stable-diffusion.cpp on the same files
/// (the image-parity script); this gate is the in-engine half, so a quant
/// route that silently broke would be caught without an sd.cpp beside it.
#[test]
fn qwen_image_compact_lane_tracks_the_full_lane() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dit) = common::model("PADDOCK_QWEN_IMAGE_DIT", DIT) else {
        return;
    };
    let Some(te) = common::model("PADDOCK_QWEN_IMAGE_TE", TE) else {
        return;
    };
    let Some(dit_q4) = common::model("PADDOCK_QWEN_IMAGE_DIT_Q4", DIT_Q4) else {
        return;
    };
    let Some(te_q4) = common::model("PADDOCK_QWEN_IMAGE_TE_Q4", TE_Q4) else {
        return;
    };
    let Some(vae) = common::model("PADDOCK_QWEN_IMAGE_VAE", VAE) else {
        return;
    };
    let prompt = "a lovely cat holding a sign that says 'paddock'";
    let (size, steps, seed) = (1024, 4, 42);
    // one lane resident at a time: the full lane's image is kept, its model
    // dropped, before the compact lane loads
    let full = render(exec.clone(), &dit, &te, &vae, prompt, size, steps, seed);
    let compact = render(exec, &dit_q4, &te_q4, &vae, prompt, size, steps, seed);
    let db = psnr_rgb(&full, &compact);
    eprintln!("compact vs full lane: {db:.2} dB PSNR (rgb) at {size}^2, {steps} steps");
    if let Ok(dir) = std::env::var("PADDOCK_QI_OUT_DIR") {
        save(
            full,
            &format!("{dir}/qwen-image-full-{size}-s{steps}-seed{seed}.png"),
        );
        save(
            compact,
            &format!("{dir}/qwen-image-compact-{size}-s{steps}-seed{seed}.png"),
        );
    }
    assert!(
        db >= COMPACT_LANE_PSNR_FLOOR_DB,
        "compact lane is {db:.2} dB from the full lane (floor {COMPACT_LANE_PSNR_FLOOR_DB} dB) - \
         a quant route is off, not a rounding difference"
    );
}
