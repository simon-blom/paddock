//! Narrow budget edits; never export or replace an operator's custom cache path.
use serde::Deserialize;
use serde_json::{Value, json};
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub enabled: bool,
    pub ram_gb: f64,
    pub nvme_gb: f64,
}
pub fn projection(doc: &toml::Value) -> Value {
    let v = doc.get("kv_offload");
    json!({
        "enabled":v.and_then(|v| v.get("enabled")).and_then(toml::Value::as_bool).unwrap_or(false),
        "ram_gb":v.and_then(|v| v.get("ram_gb")).and_then(number).unwrap_or(0.),
        "nvme_gb":v.and_then(|v| v.get("nvme_gb")).and_then(number).unwrap_or(0.)
    })
}
fn number(v: &toml::Value) -> Option<f64> {
    v.as_float().or_else(|| v.as_integer().map(|n| n as f64))
}
pub fn value(doc: &toml::Value, v: &Settings) -> Result<toml::Value, String> {
    if !v.ram_gb.is_finite()
        || !v.nvme_gb.is_finite()
        || !(0.0..=1024.0).contains(&v.ram_gb)
        || !(0.0..=8192.0).contains(&v.nvme_gb)
        || (v.enabled && v.ram_gb < 0.5)
    {
        return Err("KV offloading requires 0.5-1024 GiB RAM and 0-8192 GiB disk. The runner checks the model's complete checkpoint transfer requirement.".into());
    }
    let mut table = doc
        .get("kv_offload")
        .and_then(toml::Value::as_table)
        .cloned()
        .unwrap_or_default();
    table.insert("enabled".into(), toml::Value::Boolean(v.enabled));
    table.insert("ram_gb".into(), toml::Value::Float(v.ram_gb));
    table.insert("nvme_gb".into(), toml::Value::Float(v.nvme_gb));
    Ok(toml::Value::Table(table))
}
pub fn supported(doc: &toml::Value) -> bool {
    let Some(path) = doc
        .get("model")
        .and_then(toml::Value::as_str)
        .map(std::path::Path::new)
    else {
        return false;
    };
    if path.is_dir() {
        return paddock_models::mlx::QwenConfig::read(path).is_ok()
            || paddock_models::bonsai::BonsaiConfig::read(path).is_ok();
    }
    paddock_models::probe::probe_path(path).is_ok_and(|p| {
        matches!(
            p.architecture.as_deref(),
            Some("qwen35" | "qwen35moe" | "granite" | "gpt-oss" | "laguna")
        )
    })
}
