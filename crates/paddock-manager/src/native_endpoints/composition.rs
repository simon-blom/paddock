//! Only registry-reviewed composition fields may replace model paths. Other
//! settings, comments, private keys and manually configured tools stay intact.
use super::*;

pub(super) async fn resolve(
    state: &AppState,
    port: u16,
    content: &str,
    choice: &Composition,
) -> Result<String, String> {
    let model = state
        .registry
        .catalog_of(&choice.model)
        .ok_or("Select a model from the downloaded catalog.")?;
    let artifact = model
        .artifact(&choice.artifact)
        .ok_or("Select a weights option from this model.")?;
    let runtime = artifact.runtime.for_backend("metal");
    if artifact.kind != crate::registry::ArtifactKind::Weights || !runtime.supports_backend("metal")
    {
        return Err("This weights option is unavailable on this Mac.".into());
    }
    let installed = state
        .registry
        .resolve(
            &choice.model,
            Some(&choice.artifact),
            false,
            choice.drafter.as_deref(),
        )
        .await
        .map_err(|_| "Download the selected weights and compatible companion files before saving.")?
        .ok_or("Select an installed catalog model.")?;
    if !installed.weights.exists() {
        return Err("The selected weights are no longer downloaded.".into());
    }
    let mut spec = state
        .supervisor
        .spec_from_config_text(content)
        .map_err(|_| "Cannot read the saved model composition.")?;
    spec.model = choice.model.clone();
    spec.artifact = Some(choice.artifact.clone());
    spec.drafter = choice.drafter.clone();
    // This control describes an optional image tower, not a built-in tower or
    // a mandatory speech encoder. Neither can be detached by a hidden toggle.
    let audio_companion = model.artifacts.iter().any(|a| {
        a.kind == crate::registry::ArtifactKind::Audio
            && a.runtime.supports_backend("metal")
            && artifact.runtime.allows_companion(&a.id)
    });
    spec.vision = if runtime.embedded_vision || audio_companion {
        None
    } else {
        Some(choice.vision)
    };
    spec.fp8_native = false;
    let rendered = state
        .supervisor
        .render_spec_config(port, spec)
        .await
        .map_err(
            |_| "This model composition cannot run. Check its vision and drafter requirements.",
        )?;
    overlay(content, &rendered, port)
}

fn overlay(content: &str, rendered: &str, port: u16) -> Result<String, String> {
    let resolved = parse(rendered, port)?;
    let mut expected = parse(content, port)?;
    let mut document: toml_edit::DocumentMut = content
        .parse()
        .map_err(|_| "Cannot prepare the model composition.")?;
    let source: toml_edit::DocumentMut = rendered
        .parse()
        .map_err(|_| "Cannot prepare the model composition.")?;
    for key in ["model", "catalog", "mmproj", "mtp", "fp8_native"] {
        if let Some(value) = resolved.get(key) {
            expected
                .as_table_mut()
                .expect("a parsed endpoint document is a table")
                .insert(key.into(), value.clone());
            document[key] = source[key].clone();
        } else {
            expected
                .as_table_mut()
                .expect("a parsed endpoint document is a table")
                .remove(key);
            document.remove(key);
        }
    }
    let text = document.to_string();
    if toml::from_str::<toml::Value>(&text).ok().as_ref() == Some(&expected) {
        Ok(text)
    } else {
        toml::to_string(&expected).map_err(|_| "Cannot serialize the model composition.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_weights_preserves_private_and_unrelated_settings() {
        let original = "# keep me\nport=12345\nmodel='old.gguf'\napi_key='private-fixture'\nmax_ctx=8192\nmmproj='old-vision.gguf'\nmtp='old-draft.gguf'\n[forensics]\nauto='images'\n[[mcp_servers]]\nserver_label='private-tools'\n[mcp_servers.headers]\nAuthorization='private-header'\n";
        let replacement = "port=12345\nmodel='new-checkpoint'\napi_key='must-not-replace'\nmax_ctx=4096\n[catalog]\nmodel='new-model'\nartifact='mlx-4bit'\n";
        let text = overlay(original, replacement, 12345).unwrap();
        let value = parse(&text, 12345).unwrap();
        assert!(text.contains("# keep me"));
        assert_eq!(value["model"].as_str(), Some("new-checkpoint"));
        assert_eq!(value["max_ctx"].as_integer(), Some(8192));
        assert_eq!(value["api_key"].as_str(), Some("private-fixture"));
        assert_eq!(
            value["mcp_servers"][0]["headers"]["Authorization"].as_str(),
            Some("private-header")
        );
        assert_eq!(value["forensics"]["auto"].as_str(), Some("images"));
        assert!(value.get("mmproj").is_none() && value.get("mtp").is_none());
    }

    #[test]
    fn changing_model_preserves_personal_concurrency_instead_of_rendered_default() {
        let original = "port=12345\nmodel='old.gguf'\nmax_ctx=4096\nmax_batch=1\n";
        let replacement = "port=12345\nmodel='new.gguf'\nmax_ctx=8192\nmax_batch=32\n";
        let text = overlay(original, replacement, 12345).unwrap();
        let value = parse(&text, 12345).unwrap();
        assert_eq!(value["max_batch"].as_integer(), Some(1));
        assert_eq!(value["max_ctx"].as_integer(), Some(4096));
    }
}
