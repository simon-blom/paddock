//! Registry/control-plane gates for the NVFP4 row, not model parity or
//! redistribution clearance.
//!
//! This row is the catalog's first COMMUNITY-produced NVFP4 export, and the
//! repo it comes from has republished DIFFERENT weights under the SAME name
//! once already (2026-09-16: a quantization-aware-distilled checkpoint, 36
//! shards where the earlier post-training quant had 34). Everything gated here
//! is a provenance fact that went wrong at least once - the row first credited
//! a mirror, then claimed Apache-2.0, then pointed at `main` while holding a
//! pinned revision. A registry row that misnames its producer or its licence
//! is a redistribution problem, so these are cheap assertions against an
//! expensive mistake.
use super::*;

const MODEL: &str = "qwen3.8-flash-next";
const ARTIFACT: &str = "nvfp4";
const CHECKPOINT: &str = "Qwen3.8-Flash-Next-NVFP4-7c4f1bc1";
const REVISION: &str = "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd";

#[test]
fn nvfp4_credits_the_producer_and_pins_the_revision_it_holds() {
    let reg = Registry::new("./models".into()).with_backend("cuda");
    let a = reg.catalog_of(MODEL).unwrap().artifact(ARTIFACT).unwrap();
    let source = a
        .source
        .as_ref()
        .expect("a community export must carry source");

    // The PRODUCING repo, never a mirror the bytes were fetched through.
    assert_eq!(source.repo, "local-inference-lab/Qwen3.8-Flash-Next-NVFP4");
    assert_eq!(source.base_model, "Qwen/Qwen3.8-Flash-Next");
    assert_ne!(
        source.repo.split('/').next(),
        source.base_model.split('/').next(),
        "a third-party export must not read as an official one"
    );

    // A bare branch name is not provenance here: `main` has already meant two
    // different checkpoints. The R2 prefix carries the revision it holds.
    assert_eq!(source.revision, REVISION);
    assert_ne!(source.revision, "main");
    assert_eq!(source.revision.len(), 40, "a full immutable sha, not a tag");
    assert!(CHECKPOINT.ends_with(&REVISION[..8]));

    // Inherited from the base model. It is NOT Apache-2.0 - an earlier cut of
    // this row said so, which would have been a false redistribution claim.
    assert_eq!(source.license, "qwen-community-1.0");
    assert!(source.license_url.contains("Qwen/Qwen3.8-Flash-Next"));

    assert_eq!(a.quant.as_deref(), Some("NVFP4"));
    assert!(a.runtime.experimental, "qualification is still open");
    assert_eq!(a.runtime.backends, ["cuda"]);
}

#[test]
fn nvfp4_bundle_carries_every_shard_and_a_licence_notice() {
    let reg = Registry::new("./models".into()).with_backend("cuda");
    let a = reg.catalog_of(MODEL).unwrap().artifact(ARTIFACT).unwrap();

    let names: std::collections::BTreeSet<_> = a
        .files
        .iter()
        .map(|f| f.dest.strip_prefix(&format!("{CHECKPOINT}/")).unwrap())
        .collect();
    assert_eq!(names.len(), a.files.len(), "no duplicated bundle members");

    // The distilled export is 36 shards. The 34-shard one it replaced is gone,
    // so a stale shard name here means the row and the origin have parted.
    for shard in 1..=36 {
        assert!(names.contains(format!("model-{shard:05}-of-00036.safetensors").as_str()));
    }
    assert!(!names.iter().any(|n| n.ends_with("-of-00034.safetensors")));
    for required in [
        // the licence notice ships WITH the weights, per the attribution rule
        "LICENSE",
        "README.md",
        "config.json",
        "hf_quant_config.json",
        "export-manifest.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ] {
        assert!(names.contains(required), "missing {required}");
    }
    // The distilled export drops ModelOpt's calibration bookkeeping; the served
    // activation scales are `input_scale` tensors inside the shards, resolved
    // through the index, so their absence is correct and not a short bundle.
    for absent in [
        "amax.safetensors",
        "amax_checkpoint.json",
        "amax_checkpoint.safetensors",
        "model-inputscales.safetensors",
        ".gitattributes", // a VCS file is not part of a model bundle
    ] {
        assert!(!names.contains(absent), "unexpected {absent}");
    }

    for f in &a.files {
        assert!(f.dest.starts_with(&format!("{CHECKPOINT}/")), "{}", f.dest);
        assert!(
            !f.dest.contains("Mia"),
            "the mirror's name is not a provenance claim: {}",
            f.dest
        );
        assert_eq!(f.sha256.len(), 64);
        assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(
            f.url,
            format!("https://models.truespar.io/models/{}", f.dest)
        );
    }
}
