use super::*;

const MODEL: &str = "qwen3.8-27b";
const MLX: &str = "mlx-4bit";

#[test]
fn splash_package_is_pinned_complete_and_metal_only() {
    let reg = registry(Path::new("./models"), "metal");
    let model = reg.catalog_of(MODEL).unwrap();
    let artifact = model.artifact("splash-4bit").unwrap();
    assert_eq!(artifact.format, "splash-packed-q4");
    assert_eq!(artifact.runtime.backends, ["metal"]);
    assert!(!artifact.default);
    assert!(artifact.runtime.checkpoint_dir && artifact.runtime.embedded_vision);
    assert_eq!(artifact.runtime.default_envelope(), (32768, 1));
    assert_eq!(artifact.runtime.default_spec.as_deref(), Some("adaptive"));
    assert_eq!(artifact.runtime.companions, Some(vec![]));
    assert_eq!(
        artifact.entry_path(Path::new("/models")).unwrap(),
        Path::new("/models/Qwen3.8-27B-Splash")
    );
    assert_eq!(artifact.files.len(), 81);
    assert_eq!(
        artifact.files[0].sha256,
        paddock_models::splash::MANIFEST_SHA256
    );
    let source = artifact.source.as_ref().unwrap();
    assert_eq!(source.repo, "incoai/Qwen3.8-27B-Splash");
    assert_eq!(source.revision, paddock_models::splash::REVISION);
    for file in &artifact.files {
        assert!(file.dest.starts_with("Qwen3.8-27B-Splash/"));
        assert_eq!(
            file.url,
            format!("https://models.truespar.io/models/{}", file.dest)
        );
        assert!(file.size > 0 && file.sha256.len() == 64);
    }
    for (prefix, count) in [
        ("target/", 66),
        ("draft/", 6),
        ("vision/", 1),
        ("tokenizer/", 5),
    ] {
        assert_eq!(
            artifact
                .files
                .iter()
                .filter(|f| f.dest.starts_with(&format!("Qwen3.8-27B-Splash/{prefix}")))
                .count(),
            count
        );
    }
}

#[tokio::test]
async fn splash_resolves_one_package_without_gguf_companions() {
    let dir = tempfile::tempdir().unwrap();
    let reg = tiny_registry(dir.path(), "metal", "http://127.0.0.1:1/unused");
    let a = reg
        .catalog_of(MODEL)
        .unwrap()
        .artifact("splash-4bit")
        .unwrap();
    install(&reg, a);
    let resolved = reg
        .resolve(MODEL, Some("splash-4bit"), true, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(resolved.weights, dir.path().join("Qwen3.8-27B-Splash"));
    assert!(resolved.mmproj.is_none() && resolved.mtp.is_none());
}

fn registry(dir: &Path, backend: &str) -> Registry {
    Registry::new(dir.to_path_buf()).with_backend(backend)
}

/// Model bytes are irrelevant to control-plane tests. Use tiny checksum-valid
/// files, but keep the real manifest's paths, contracts and companion set.
fn tiny_registry(dir: &Path, backend: &str, url: &str) -> Registry {
    let real = registry(dir, backend);
    let mut model = real.catalog_of(MODEL).unwrap().clone();
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
        dir.to_path_buf(),
    )
    .with_backend(backend)
}

fn install(reg: &Registry, a: &CatalogArtifact) {
    for f in &a.files {
        let path = reg.models_dir.join(&f.dest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"test").unwrap();
    }
}

#[test]
fn mlx_is_a_pinned_community_export_with_mandatory_license_and_readme() {
    let reg = registry(Path::new("./models"), "metal");
    let model = reg.catalog_of(MODEL).unwrap();
    let a = model.artifact(MLX).unwrap();
    let source = a.source.as_ref().unwrap();
    assert_eq!(source.repo, "mlx-community/Qwen3.8-27B-4bit");
    assert_eq!(source.revision, "3e6447f082e89cc7f0bc6e5441afd38dfce760ff");
    assert_eq!(source.base_model, "Qwen/Qwen3.8-27B");
    assert_eq!(source.license, "apache-2.0");
    assert_eq!(a.quant.as_deref(), Some("MLX-AFFINE-4-G64"));
    assert!(
        !a.default,
        "availability must not silently replace a declared default"
    );
    assert!(!a.runtime.experimental && a.runtime.checkpoint_dir);
    assert_eq!(a.runtime.kv_cache_dtype.as_deref(), Some("auto"));
    assert_eq!(a.files.len(), 16);
    let license = a
        .files
        .iter()
        .find(|f| f.dest.ends_with("/LICENSE"))
        .unwrap();
    assert!(source.license_url.contains("/Qwen/Qwen3.8-27B/resolve/"));
    assert_eq!(
        license.sha256,
        "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
    );
    assert!(a.files.iter().any(|f| f.dest.ends_with("/README.md")));
    for f in &a.files {
        assert_eq!(
            f.url,
            format!("https://models.truespar.io/models/{}", f.dest)
        );
    }
    assert!(!a.capabilities(model).iter().any(|c| c == "vision"));
    assert!(a.capabilities(model).iter().any(|c| c == "speculative"));
    assert_eq!(a.runtime.default_spec.as_deref(), Some("adaptive"));
    assert_eq!(a.runtime.default_envelope(), (32768, 1));
    assert!(a.runtime.allows_companion("drafter2"));
    assert!(
        model.capability.iter().any(|c| c == "vision"),
        "GGUF keeps its tower"
    );
}

#[tokio::test]
async fn cuda_rejects_mlx_before_any_download_or_preview() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path(), "cuda");
    let error = reg.resolve(MODEL, Some(MLX), true, None).await.unwrap_err();
    assert!(error.to_string().contains("requires backend metal"));
    assert!(reg.planned_paths(MODEL, Some(MLX)).is_none());
    assert!(reg.start_pull(MODEL, Some(&[MLX.into()])).is_err());
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    let model = reg.catalog_of(MODEL).unwrap();
    assert_ne!(
        model.default_weights_for_backend("cuda", None).unwrap().id,
        MLX
    );
    let view = reg.catalog_annotated();
    let row = view["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == MODEL)
        .unwrap();
    let artifact = row["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == MLX)
        .unwrap();
    assert_eq!(artifact["backend_supported"], false);
}

#[tokio::test]
async fn metal_resolves_the_checkpoint_directory_with_only_its_dflash2_companion() {
    let dir = tempfile::tempdir().unwrap();
    let reg = tiny_registry(dir.path(), "metal", "http://127.0.0.1:1/unused");
    for a in &reg.catalog_of(MODEL).unwrap().artifacts {
        install(&reg, a);
    }
    let planned = reg.planned_paths(MODEL, Some(MLX)).unwrap();
    let r = reg
        .resolve(MODEL, Some(MLX), false, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(r.weights, dir.path().join("Qwen3.8-27B-MLX-4bit"));
    let draft = dir.path().join("Qwen3.8-27B-GGUF/dflash2-Q4_K_M.gguf");
    assert_eq!(planned, (r.weights.clone(), None, Some(draft.clone())));
    assert!(r.mmproj.is_none() && r.fp8_snapshot.is_none());
    assert_eq!(r.mtp, Some(draft.clone()));
    assert_eq!(r.drafter_any, Some(draft));
    assert!(r.drafter_declared && r.speculative);
    assert_eq!(
        reg.identify_weights(&r.weights),
        Some((MODEL.into(), MLX.into()))
    );
    let caps = reg.capability_of(r.weights.to_str().unwrap()).unwrap();
    assert!(caps.contains(&"chat".into()));
    assert!(!caps.contains(&"vision".into()));
    assert!(caps.contains(&"speculative".into()));
    let error = reg
        .resolve(MODEL, Some(MLX), false, Some("dflash2"))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("does not support companion"));
}

#[test]
fn a_checkpoint_without_the_license_is_not_a_complete_registry_install() {
    let dir = tempfile::tempdir().unwrap();
    let reg = tiny_registry(dir.path(), "metal", "http://127.0.0.1:1/unused");
    let a = reg.catalog_of(MODEL).unwrap().artifact(MLX).unwrap();
    let mut without_license = a.clone();
    without_license
        .files
        .retain(|f| !f.dest.ends_with("/LICENSE"));
    install(&reg, &without_license);
    assert!(!reg.is_artifact_installed(a));
    install(&reg, a);
    assert!(reg.is_artifact_installed(a));
}

#[tokio::test]
async fn pulling_mlx_fetches_the_license_and_default_dflash2_but_no_other_companions() {
    let url = super::tests::spawn_origin(b"test".to_vec()).await;
    let dir = tempfile::tempdir().unwrap();
    let reg = tiny_registry(dir.path(), "metal", &url);
    let r = reg
        .resolve(MODEL, Some(MLX), true, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(std::fs::read(r.weights.join("LICENSE")).unwrap(), b"test");
    assert_eq!(std::fs::read(r.weights.join("README.md")).unwrap(), b"test");
    for a in reg
        .catalog_of(MODEL)
        .unwrap()
        .artifacts
        .iter()
        .filter(|a| a.kind != ArtifactKind::Weights)
    {
        assert!(
            reg.is_artifact_installed(a) == (a.id == "drafter2"),
            "downloaded incompatible {}",
            a.id
        );
    }
}
