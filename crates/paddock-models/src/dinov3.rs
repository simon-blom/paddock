//! `config.json` of a DINOv3 dense-prediction checkpoint - today that is
//! `tic-forestry-v1` (`architectures = ["TicForestryDinoV3"]`): a DINOv3 ViT
//! backbone, fine-tuned, under a four-stage multi-scale decoder with a class
//! head and a regression head.
//!
//! Same parsing stance as every other safetensors family here: every field the
//! engine consumes is validated present, and a value the graph does not
//! implement is refused by name rather than ignored. That matters more than
//! usual for this family, because its failure mode is silence - a DINOv3
//! checkpoint with the wrong band count, without LayerScale, or with the
//! registers left in the grid loads every tensor and returns noise.

use std::path::Path;

use crate::safetensors::StError;

/// The `architectures[0]` this parser answers to.
pub const TIC_FORESTRY_ARCH: &str = "TicForestryDinoV3";

/// Backbone + head + I/O contract, as the engine consumes them.
#[derive(Debug, Clone)]
pub struct Dinov3SegConfig {
    // ---- backbone (config.backbone) ----
    pub hidden: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    pub intermediate: usize,
    pub patch: usize,
    /// input bands - 4 here (RGB + near-infrared), where stock DINOv3 is 3
    pub channels: usize,
    pub rope_theta: f32,
    pub n_registers: usize,
    pub eps: f32,
    // ---- head (config.head) ----
    /// hidden-state depths the decoder taps, deepest first (24, 18, 12, 6);
    /// depth d is the output of block d, depth 0 would be the embeddings
    pub stage_depths: Vec<usize>,
    /// decoder channel width per stage, same order
    pub widths: Vec<usize>,
    pub n_classes: usize,
    pub class_names: Vec<String>,
    /// what the regression head measures (`head.outputs[1]`, e.g.
    /// "canopy_height_m") - a label for the wire, never consumed by the graph
    pub regression_name: String,
    // ---- input / output contract ----
    /// chip side in pixels (512)
    pub image_size: usize,
    /// per-band statistics in 0-1 units: divide by 255 first, then these
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub band_names: Vec<String>,
    /// ground metres per input pixel (0.5)
    pub pixel_size_m: f64,
    /// output raster side (256) and its ground metres per pixel (1.0)
    pub out_size: usize,
    pub out_pixel_size_m: f64,
    pub epsg: Option<u32>,
}

impl Dinov3SegConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
    /// patch-grid side (32 at 512 px / patch 16)
    pub fn grid(&self) -> usize {
        self.image_size / self.patch
    }
    /// tokens per chip: the patch grid plus class + register tokens (1029)
    pub fn tokens(&self) -> usize {
        self.grid() * self.grid() + 1 + self.n_registers
    }

    /// Whether `dir/config.json` names this family at all - the runner's
    /// cheap dispatch probe. A directory that is not ours answers false; one
    /// that is ours but malformed is left for [`Self::read`] to refuse loudly.
    pub fn is_ours(dir: &Path) -> bool {
        std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| {
                v.get("architectures")?
                    .as_array()?
                    .first()?
                    .as_str()
                    .map(|a| a == TIC_FORESTRY_ARCH)
            })
            .unwrap_or(false)
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| StError::Header(e.to_string()))?;
        let bad = |m: String| StError::Header(format!("dinov3 config.json: {m}"));
        let arch = v
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_str())
            .unwrap_or_default();
        if arch != TIC_FORESTRY_ARCH {
            return Err(bad(format!(
                "architectures[0] '{arch}' (want {TIC_FORESTRY_ARCH})"
            )));
        }
        let sub = |k: &str| v.get(k).ok_or_else(|| bad(format!("missing {k}")));
        let getu = |o: &serde_json::Value, scope: &str, k: &str| {
            o.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| bad(format!("missing {scope}.{k}")))
        };
        let getf = |o: &serde_json::Value, scope: &str, k: &str| {
            o.get(k)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| bad(format!("missing {scope}.{k}")))
        };
        let getb = |o: &serde_json::Value, scope: &str, k: &str| {
            o.get(k)
                .and_then(|x| x.as_bool())
                .ok_or_else(|| bad(format!("missing {scope}.{k}")))
        };
        let gets = |o: &serde_json::Value, scope: &str, k: &str| {
            o.get(k)
                .and_then(|x| x.as_str())
                .map(str::to_owned)
                .ok_or_else(|| bad(format!("missing {scope}.{k}")))
        };
        let arr = |o: &serde_json::Value, scope: &str, k: &str| {
            o.get(k)
                .and_then(|x| x.as_array())
                .cloned()
                .ok_or_else(|| bad(format!("missing {scope}.{k}")))
        };
        let usizes = |a: Vec<serde_json::Value>, what: &str| {
            a.iter()
                .map(|x| x.as_u64().map(|x| x as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| bad(format!("{what}: not all unsigned integers")))
        };
        let f32s = |a: Vec<serde_json::Value>, what: &str| {
            a.iter()
                .map(|x| x.as_f64().map(|x| x as f32))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| bad(format!("{what}: not all numbers")))
        };
        let strs = |a: Vec<serde_json::Value>, what: &str| {
            a.iter()
                .map(|x| x.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| bad(format!("{what}: not all strings")))
        };

        let b = sub("backbone")?;
        let h = sub("head")?;
        let i = sub("input")?;
        let o = sub("output")?;

        // The graph is one architecture; each of these is a switch the
        // reference implementation has and this engine does not.
        let bt = gets(b, "backbone", "model_type")?;
        if bt != "dinov3_vit" {
            return Err(bad(format!("backbone.model_type '{bt}' (want dinov3_vit)")));
        }
        if !getb(b, "backbone", "layerscale")? {
            return Err(bad(
                "backbone.layerscale false - only the LayerScale block is built".into(),
            ));
        }
        if getb(b, "backbone", "position_embeddings")? {
            return Err(bad(
                "backbone.position_embeddings true - this graph is rope-only, as DINOv3 is".into(),
            ));
        }
        let act = gets(b, "backbone", "hidden_act")?;
        if act != "gelu" {
            return Err(bad(format!(
                "backbone.hidden_act '{act}' - only the plain GELU MLP is built (not the gated one)"
            )));
        }
        let ht = gets(h, "head", "type")?;
        if ht != "multiscale_dense" {
            return Err(bad(format!("head.type '{ht}' (want multiscale_dense)")));
        }
        let dtype = gets(i, "input", "dtype")?;
        if dtype != "uint8" {
            return Err(bad(format!("input.dtype '{dtype}' (want uint8)")));
        }

        let cfg = Self {
            hidden: getu(b, "backbone", "hidden_size")?,
            n_layer: getu(b, "backbone", "num_hidden_layers")?,
            n_heads: getu(b, "backbone", "num_attention_heads")?,
            intermediate: getu(b, "backbone", "intermediate_size")?,
            patch: getu(b, "backbone", "patch_size")?,
            channels: getu(b, "backbone", "num_channels")?,
            rope_theta: getf(b, "backbone", "rope_theta")? as f32,
            n_registers: getu(b, "backbone", "num_register_tokens")?,
            eps: getf(b, "backbone", "layer_norm_eps")? as f32,
            stage_depths: usizes(arr(h, "head", "stage_depths")?, "head.stage_depths")?,
            widths: usizes(arr(h, "head", "widths")?, "head.widths")?,
            n_classes: getu(h, "head", "classes")?,
            class_names: strs(arr(h, "head", "class_names")?, "head.class_names")?,
            regression_name: h
                .get("outputs")
                .and_then(|o| o.as_array())
                .and_then(|o| o.get(1))
                .and_then(|o| o.as_str())
                .unwrap_or("regression")
                .to_owned(),
            image_size: getu(i, "input", "image_size")?,
            mean: f32s(arr(i, "input", "mean_0_1")?, "input.mean_0_1")?,
            std: f32s(arr(i, "input", "std_0_1")?, "input.std_0_1")?,
            band_names: strs(arr(i, "input", "bands")?, "input.bands")?,
            pixel_size_m: getf(i, "input", "pixel_size_m")?,
            out_size: getu(o, "output", "size")?,
            out_pixel_size_m: getf(o, "output", "pixel_size_m")?,
            epsg: v.get("epsg").and_then(|x| x.as_u64()).map(|x| x as u32),
        };
        cfg.check().map_err(bad)?;
        Ok(cfg)
    }

    /// Geometry the graph depends on. Each refusal names what it found.
    fn check(&self) -> Result<(), String> {
        let c = self;
        if c.n_heads == 0 || !c.hidden.is_multiple_of(c.n_heads) {
            return Err(format!(
                "hidden {} not a multiple of heads {}",
                c.hidden, c.n_heads
            ));
        }
        // the rope splits a head into four equal frequency blocks
        if !c.head_dim().is_multiple_of(4) {
            return Err(format!("head_dim {} not a multiple of 4", c.head_dim()));
        }
        if c.patch == 0 || !c.image_size.is_multiple_of(c.patch) {
            return Err(format!(
                "image_size {} not a multiple of patch {}",
                c.image_size, c.patch
            ));
        }
        if c.channels == 0 || c.channels > 4 {
            return Err(format!(
                "{} input bands - the patch stem takes 1 to 4",
                c.channels
            ));
        }
        if c.mean.len() != c.channels || c.std.len() != c.channels {
            return Err(format!(
                "{} bands but {} means and {} stds",
                c.channels,
                c.mean.len(),
                c.std.len()
            ));
        }
        if c.std.iter().any(|s| s.is_nan() || *s <= 0.0) {
            return Err("input.std_0_1 holds a non-positive value".into());
        }
        if c.stage_depths.is_empty() || c.stage_depths.len() != c.widths.len() {
            return Err(format!(
                "{} stage depths against {} widths",
                c.stage_depths.len(),
                c.widths.len()
            ));
        }
        if c.stage_depths.iter().any(|d| *d == 0 || *d > c.n_layer) {
            return Err(format!(
                "stage_depths {:?} outside 1..={}",
                c.stage_depths, c.n_layer
            ));
        }
        // deepest first, strictly: the decoder walks them in this order
        if c.stage_depths.windows(2).any(|w| w[0] <= w[1]) {
            return Err(format!(
                "stage_depths {:?} not strictly descending",
                c.stage_depths
            ));
        }
        if c.widths.iter().any(|w| *w == 0 || !w.is_multiple_of(8)) {
            // the f16 tensor-core GEMM stages 16-byte rows
            return Err(format!(
                "widths {:?} - every stage width must be a multiple of 8",
                c.widths
            ));
        }
        // every stage past the first doubles the grid; the last must land on
        // the output raster exactly (the reference would bilinear-resize
        // otherwise, which this graph does not build)
        let landed = c.grid() << (c.widths.len() - 1);
        if landed != c.out_size {
            return Err(format!(
                "decoder lands on {landed} px from a {}-grid through {} stages, output.size is {}",
                c.grid(),
                c.widths.len(),
                c.out_size
            ));
        }
        if c.n_classes == 0 || c.n_classes > 255 || c.class_names.len() != c.n_classes {
            return Err(format!(
                "{} classes with {} names (1..=255, one name each)",
                c.n_classes,
                c.class_names.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{
      "architectures": ["TicForestryDinoV3"],
      "backbone": {"model_type": "dinov3_vit", "hidden_size": 1024, "num_hidden_layers": 24,
        "num_attention_heads": 16, "intermediate_size": 4096, "patch_size": 16,
        "num_channels": 4, "rope_theta": 100.0, "position_embeddings": false,
        "num_register_tokens": 4, "layerscale": true, "hidden_act": "gelu",
        "layer_norm_eps": 1e-05},
      "head": {"type": "multiscale_dense", "stage_depths": [24, 18, 12, 6],
        "widths": [256, 128, 64, 32], "classes": 3, "class_names": ["a", "b", "c"]},
      "input": {"image_size": 512, "pixel_size_m": 0.5, "bands": ["r", "g", "b", "n"],
        "dtype": "uint8", "mean_0_1": [0.1, 0.2, 0.3, 0.4], "std_0_1": [0.1, 0.1, 0.1, 0.1]},
      "output": {"size": 256, "pixel_size_m": 1.0},
      "epsg": 3006
    }"#;

    fn read_str(s: &str) -> Result<Dinov3SegConfig, StError> {
        let dir = std::env::temp_dir().join(format!(
            "pd-dinov3-cfg-{}-{:x}",
            std::process::id(),
            s.len() ^ (s.as_ptr() as usize)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), s).unwrap();
        let r = Dinov3SegConfig::read(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        r
    }

    #[test]
    fn reads_the_published_shape() {
        let c = read_str(GOOD).unwrap();
        assert_eq!((c.hidden, c.n_layer, c.head_dim()), (1024, 24, 64));
        assert_eq!((c.grid(), c.tokens()), (32, 1029));
        assert_eq!(c.stage_depths, [24, 18, 12, 6]);
        assert_eq!(c.epsg, Some(3006));
    }

    #[test]
    fn refuses_what_the_graph_does_not_build() {
        // each of these loads cleanly in a careless port and serves noise
        for (from, to, needle) in [
            (
                "\"layerscale\": true",
                "\"layerscale\": false",
                "layerscale",
            ),
            (
                "\"position_embeddings\": false",
                "\"position_embeddings\": true",
                "rope-only",
            ),
            (
                "\"hidden_act\": \"gelu\"",
                "\"hidden_act\": \"silu\"",
                "hidden_act",
            ),
            ("[24, 18, 12, 6]", "[6, 12, 18, 24]", "descending"),
            ("\"size\": 256", "\"size\": 512", "lands on"),
            ("\"num_channels\": 4", "\"num_channels\": 3", "bands"),
        ] {
            let e = read_str(&GOOD.replace(from, to)).unwrap_err().to_string();
            assert!(e.contains(needle), "{from} -> {to}: {e}");
        }
    }
}
