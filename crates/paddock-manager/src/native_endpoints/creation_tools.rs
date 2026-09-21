//! Creation carries library identities, never URLs or exported credentials.
use super::*;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreationTools {
    pub provider: String,
    pub key: String,
    pub connectors: Vec<Connector>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Connector {
    id: String,
    revision: u64,
}

pub(super) fn prepare(
    state: &AppState,
    content: &str,
    tools: &CreationTools,
) -> Result<(String, Vec<Value>), String> {
    if tools.connectors.len() > 128 {
        return Err("Select at most 128 connectors.".into());
    }
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    for choice in &tools.connectors {
        if !seen.insert(&choice.id) {
            return Err("A connector was selected twice.".into());
        }
        let row = state
            .db
            .get_connector(&choice.id)
            .map_err(|_| "Connector credentials are unavailable. Unlock them in Connectors.")?
            .ok_or("A selected connector was removed. Review system tools.")?;
        if row["revision"].as_u64() != Some(choice.revision) {
            return Err(
                "A selected connector changed. Review system tools before starting.".into(),
            );
        }
        if row["ports"]
            .as_array()
            .is_some_and(|ports| ports.len() >= 128)
            && row["system"] != true
        {
            return Err("This connector already serves 128 endpoints. Review its scope before adding another.".into());
        }
        if row["system"] != true {
            rows.push(row);
        }
    }
    let content =
        crate::integrations::patch_search(content, &tools.provider, Some(tools.key.clone()))?;
    let mut doc = parse(&content, 0)?;
    let mut entries: Vec<Value> = rows
        .iter()
        .map(|r| crate::connectors::entry_from_row(r["id"].as_str().unwrap_or_default(), r))
        .collect();
    entries.extend(crate::connectors::system_entries(&state.db, &entries));
    if !entries.is_empty() {
        doc.as_table_mut()
            .expect("a parsed endpoint document is a table")
            .insert(
                "mcp_servers".into(),
                toml::Value::try_from(entries).map_err(|_| "Cannot prepare connector settings.")?,
            );
    }
    Ok((
        toml::to_string(&doc).map_err(|_| "Cannot prepare system tools.")?,
        rows,
    ))
}

/// The new file already contains these exact entries. Add only its identity to
/// the library; do not rewrite other endpoints or change global membership.
pub(super) fn register(state: &AppState, port: u16, rows: &[Value]) -> Result<(), String> {
    for row in rows {
        let mut ports: Vec<u16> = row["ports"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| p.as_u64().and_then(|p| u16::try_from(p).ok()))
            .collect();
        ports.push(port);
        ports.sort_unstable();
        ports.dedup();
        state.db.set_connector_scope(row["id"].as_str().unwrap_or_default(), false, &ports)
            .map_err(|_| format!("Configuration saved on port {port}, but connector membership could not be saved. Nothing started. Review System tools in Edit before starting."))?;
    }
    Ok(())
}
