//! Publishing and platform gates, not numerical/performance qualification.
use super::*;

const MODELS: [(&str, &str); 2] = [
    ("gemma-4-31b", "Gemma-4-31B-IT-MLX-4bit"),
    ("muse-glimmer-30b", "Muse-Glimmer-30B-MLX-4bit"),
];

#[test]
fn mirrored_mlx_bundles_are_complete_and_macos_only() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    let mut mirrored_files = 0;
    for model in &reg.catalog().models {
        let Some(a) = model.artifact("mlx-4bit") else {
            continue;
        };
        mirrored_files += a.files.len();
        assert_eq!(a.runtime.backends, ["metal"]);
        assert!(a.runtime.checkpoint_dir && !a.runtime.experimental && !a.default);
        let companions: Vec<String> = if model.id == "qwen3.8-27b" {
            vec!["drafter2".into()]
        } else {
            vec![]
        };
        assert_eq!(a.runtime.companions.as_deref(), Some(companions.as_slice()));
        let source = a.source.as_ref().unwrap();
        assert!(source.repo.starts_with("mlx-community/"));
        assert_eq!(source.revision.len(), 40);
        for f in &a.files {
            assert_eq!(
                f.url,
                format!("https://models.truespar.io/models/{}", f.dest)
            );
            assert!(f.size > 0);
            assert_eq!(f.sha256.len(), 64);
            assert!(!f.dest.contains(".cache") && !f.dest.contains("benchmark"));
        }
        for name in [
            "LICENSE",
            "README.md",
            "config.json",
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ] {
            assert!(
                a.files
                    .iter()
                    .any(|f| f.dest.ends_with(&format!("/{name}")))
            );
        }
    }
    assert_eq!(mirrored_files, 78);
    for (id, folder) in MODELS {
        let model = reg.catalog_of(id).unwrap();
        let a = model.artifact("mlx-4bit").unwrap();
        assert!(a.runtime.embedded_vision);
        assert_eq!(
            a.runtime.qualification,
            Some(runtime::Qualification::Qualified)
        );
        assert!(a.capabilities(model).iter().any(|c| c == "vision"));
        assert!(!a.capabilities(model).iter().any(|c| c == "speculative"));
        assert_eq!(a.files.len(), 14);
        for n in 1..=4 {
            assert!(
                a.files
                    .iter()
                    .any(|f| f.dest == format!("{folder}/model-{n:05}-of-00004.safetensors"))
            );
        }
        assert_eq!(
            model.default_weights_for_backend("metal", None).unwrap().id,
            "q8"
        );
        assert_eq!(
            a.shape.as_ref().unwrap().source,
            paddock_estimator::ShapeSource::Probed
        );
        let memory = a.runtime.memory.as_ref().unwrap();
        assert_eq!((memory.max_ctx, memory.max_batch), (4096, 4));
        assert!(memory.workspace_bytes.unwrap() >= 2 << 30);
    }
    for backend in ["metal", "cuda"] {
        let json = Registry::new("./models".into())
            .with_backend(backend)
            .catalog_annotated();
        for model in json["models"].as_array().unwrap() {
            for a in model["artifacts"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|a| a["id"] == "mlx-4bit")
            {
                assert_eq!(a["backend_supported"], backend == "metal");
            }
        }
    }
}

#[tokio::test]
async fn multimodal_mlx_pull_includes_notices_but_never_gguf_companions() {
    let dir = tempfile::tempdir().unwrap();
    let url = super::tests::spawn_origin(b"test".to_vec()).await;
    let mut catalog = Registry::new(dir.path().into()).catalog().clone();
    catalog
        .models
        .retain(|m| MODELS.iter().any(|(id, _)| m.id == *id));
    for m in &mut catalog.models {
        for a in &mut m.artifacts {
            for f in &mut a.files {
                f.size = 4;
                f.sha256 = hex(&Sha256::digest(b"test"));
                f.url = url.clone();
            }
        }
    }
    let reg = Registry::from_catalog(catalog, dir.path().into()).with_backend("metal");
    for (id, folder) in MODELS {
        let r = reg
            .resolve(id, Some("mlx-4bit"), true, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(r.weights, dir.path().join(folder));
        assert!(r.mmproj.is_none() && r.mtp.is_none() && r.fp8_snapshot.is_none());
        assert!(!r.speculative);
        assert!(
            reg.is_artifact_installed(reg.catalog_of(id).unwrap().artifact("mlx-4bit").unwrap())
        );
        assert_eq!(std::fs::read(r.weights.join("LICENSE")).unwrap(), b"test");
        let model = reg.catalog_of(id).unwrap();
        assert!(
            model
                .artifacts
                .iter()
                .filter(|a| a.id != "mlx-4bit")
                .flat_map(|a| &a.files)
                .all(|f| !dir.path().join(&f.dest).exists())
        );
        let cuda =
            Registry::from_catalog(reg.catalog().clone(), dir.path().into()).with_backend("cuda");
        assert!(cuda.start_pull(id, Some(&["mlx-4bit".into()])).is_err());
        assert!(
            cuda.resolve(id, Some("mlx-4bit"), false, None)
                .await
                .is_err()
        );
    }
}
