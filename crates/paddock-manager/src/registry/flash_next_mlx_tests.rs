//! Registry/control-plane gates, not model parity or redistribution clearance.
use super::*;

const MODEL: &str = "qwen3.8-flash-next";
const ARTIFACT: &str = "mlx-4bit";
const CHECKPOINT: &str = "Qwen3.8-Flash-Next-MLX-4bit";
const REVISION: &str = "07b5dc6c54600a359b87f1e53e7adf6351c72a2c";

#[test]
fn upstream_q4_and_mtp_do_not_expand_metal_support() {
    let cuda = Registry::new("./models".into()).with_backend("cuda");
    let model = cuda.catalog_of(MODEL).unwrap();
    assert_eq!(model.artifact("q4").unwrap().files.len(), 4);
    let (_, mmproj, mtp) = cuda.planned_paths(MODEL, Some("q4")).unwrap();
    assert!(mmproj.is_none());
    assert!(
        mtp.unwrap()
            .ends_with("mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
    );

    let metal = cuda.with_backend("metal");
    assert!(metal.planned_paths(MODEL, Some("q4")).is_none());
    for artifact in ["iq3", ARTIFACT] {
        let (_, mmproj, mtp) = metal.planned_paths(MODEL, Some(artifact)).unwrap();
        assert!(mmproj.is_none() && mtp.is_none(), "{artifact}");
    }
}

fn tiny_registry(dir: &Path, url: &str) -> Registry {
    let reg = Registry::new(dir.into());
    let mut model = reg.catalog_of(MODEL).unwrap().clone();
    for a in &mut model.artifacts {
        for f in &mut a.files {
            f.size = 4;
            f.sha256 = hex(&Sha256::digest(b"test"));
            f.url = url.into();
        }
    }
    Registry::from_catalog(
        Catalog {
            schema: 3,
            models: vec![model],
        },
        dir.into(),
    )
    .with_backend("metal")
}

#[test]
fn pinned_bundle_includes_every_shard_template_and_notice() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    let a = reg.catalog_of(MODEL).unwrap().artifact(ARTIFACT).unwrap();
    let source = a.source.as_ref().unwrap();
    assert_eq!(source.repo, "mlx-community/Qwen3.8-Flash-Next-4bit");
    assert_eq!(source.revision, REVISION);
    assert_eq!(source.base_model, "Qwen/Qwen3.8-Flash-Next");
    assert_eq!(source.license, "qwen-community-1.0");
    assert_eq!(a.quant.as_deref(), Some("MLX-AFFINE-4-G32"));
    assert_eq!(a.files.len(), 34);
    assert_eq!(a.total_size(), 111_546_655_083);
    let names: std::collections::BTreeSet<_> = a
        .files
        .iter()
        .map(|f| f.dest.strip_prefix(&format!("{CHECKPOINT}/")).unwrap())
        .collect();
    assert_eq!(names.len(), a.files.len(), "no duplicated bundle members");
    for shard in 1..=22 {
        assert!(names.contains(format!("model-{shard:05}-of-00022.safetensors").as_str()));
    }
    for required in [
        "LICENSE",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    ] {
        assert!(names.contains(required), "missing {required}");
    }
    let license = a
        .files
        .iter()
        .find(|f| f.dest.ends_with("/LICENSE"))
        .unwrap();
    assert_eq!(
        license.url,
        format!("https://models.truespar.io/models/{CHECKPOINT}/LICENSE")
    );
    assert_eq!(license.size, 3235);
    assert_eq!(
        license.sha256,
        "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"
    );
    assert!(
        source
            .license_url
            .contains("/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/")
    );
    for f in &a.files {
        assert_eq!(f.sha256.len(), 64);
        assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(
            f.url,
            format!("https://models.truespar.io/models/{}", f.dest)
        );
    }
    assert!(
        a.runtime
            .note
            .as_ref()
            .unwrap()
            .contains("Commercial-license review")
    );
}

#[tokio::test]
async fn installed_mlx_preserves_checkpoint_identity_and_metal_requirement() {
    let dir = tempfile::tempdir().unwrap();
    let reg = tiny_registry(dir.path(), "http://127.0.0.1:1/unused");
    let model = reg.catalog_of(MODEL).unwrap();
    let a = model.artifact(ARTIFACT).unwrap();
    assert!(!a.default && !a.runtime.experimental && a.runtime.checkpoint_dir);
    assert_eq!(
        a.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(a.runtime.companions.as_deref(), Some([].as_slice()));
    assert_eq!(a.capabilities(model), &["chat", "tools", "reasoning"]);
    // Installing both formats preserves the declared default and never leaks
    // a GGUF shard into an explicitly selected MLX directory load path.
    for a in &model.artifacts {
        for f in &a.files {
            let path = dir.path().join(&f.dest);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"test").unwrap();
        }
    }
    assert!(reg.is_artifact_installed(a));
    assert_eq!(
        model.default_weights_for_backend("metal", None).unwrap().id,
        "iq3"
    );
    assert!(reg.planned_paths(MODEL, None).is_some());
    assert_eq!(model.default_bundle_for_backend("metal", None)[0].id, "iq3");
    assert_eq!(
        reg.resolve(MODEL, None, false, None)
            .await
            .unwrap()
            .unwrap()
            .weights,
        model
            .artifact("iq3")
            .unwrap()
            .entry_path(dir.path())
            .unwrap()
    );
    let r = reg
        .resolve(MODEL, Some(ARTIFACT), false, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(r.weights, dir.path().join(CHECKPOINT));
    assert!(r.mmproj.is_none() && r.mtp.is_none() && r.fp8_snapshot.is_none());
    assert!(!r.speculative && !r.drafter_declared && r.drafter_any.is_none());
    assert_eq!(
        reg.identify_weights(&r.weights),
        Some((MODEL.into(), ARTIFACT.into()))
    );
    for drafter in ["drafter", "dflash2"] {
        assert!(
            reg.resolve(MODEL, Some(ARTIFACT), false, Some(drafter))
                .await
                .is_err()
        );
    }
    let cuda = reg.with_backend("cuda");
    assert_eq!(
        cuda.catalog_of(MODEL)
            .unwrap()
            .default_weights_for_backend("cuda", None)
            .unwrap()
            .id,
        "iq3"
    );
    assert!(cuda.planned_paths(MODEL, Some(ARTIFACT)).is_none());
    assert!(cuda.start_pull(MODEL, Some(&[ARTIFACT.into()])).is_err());
    assert!(
        cuda.resolve(MODEL, Some(ARTIFACT), false, None)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn pull_requires_license_and_readme_without_fetching_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let url = super::tests::spawn_origin(b"test".to_vec()).await;
    let reg = tiny_registry(dir.path(), &url);
    let a = reg.catalog_of(MODEL).unwrap().artifact(ARTIFACT).unwrap();
    // A real registry pull must complete missing notices, even if every
    // weight/config file has already been installed at the correct size.
    for f in &a.files {
        if f.dest.ends_with("/LICENSE") || f.dest.ends_with("/README.md") {
            continue;
        }
        let path = dir.path().join(&f.dest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"test").unwrap();
    }
    assert!(!reg.is_artifact_installed(a));
    let r = reg
        .resolve(MODEL, Some(ARTIFACT), true, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(r.weights, dir.path().join(CHECKPOINT));
    assert!(reg.is_artifact_installed(a));
    assert_eq!(std::fs::read(r.weights.join("LICENSE")).unwrap(), b"test");
    assert_eq!(std::fs::read(r.weights.join("README.md")).unwrap(), b"test");
    let gguf = reg.catalog_of(MODEL).unwrap().artifact("iq3").unwrap();
    assert!(
        gguf.files
            .iter()
            .all(|f| !dir.path().join(&f.dest).exists())
    );
}

#[test]
fn mlx_memory_prices_resident_ple_native_kv_and_grouped_scratch() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    let a = reg.catalog_of(MODEL).unwrap().artifact(ARTIFACT).unwrap();
    let mem = a.runtime.memory.as_ref().unwrap();
    assert_eq!((mem.max_ctx, mem.max_batch), (4096, 4));
    assert_eq!(
        (a.runtime.default_max_ctx, a.runtime.default_max_batch),
        (Some(4096), Some(4))
    );
    assert_eq!(a.runtime.kv_cache_dtype.as_deref(), Some("auto"));
    let dtype = a
        .runtime
        .estimate_kv_dtype(paddock_estimator::KvDtype::Fp8E4m3);
    assert_eq!(dtype, paddock_estimator::KvDtype::F16);
    let mut shape = a.shape.clone().unwrap().into_model_shape(0, 0);
    mem.apply(&mut shape);
    mem.apply(&mut shape);
    assert_eq!(shape.weight_bytes, 110_626_145_280);
    assert_eq!(shape.nextn_bytes, 0);
    assert_eq!(shape.kv_reserve_sequences, 0);
    assert_eq!(shape.workspace_bytes, 1_392_445_176);
    let kv = shape.kv_per_sequence(4096, dtype) * 4;
    let state = shape.recurrent.unwrap();
    let recurrent = state.layers * 4 * (state.state_elems + state.conv_elems) * state.elem_bytes;
    // Native buffer ledger from both final 4K/c4 runs. The general manager
    // estimator still has conservative overhead; this is not OS admission.
    assert_eq!(
        shape.weight_bytes + kv + recurrent + shape.workspace_bytes,
        112_891_923_192
    );
}
