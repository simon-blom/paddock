//! Immutable named serving profiles. Explicit save/delete; never snapshot keys,
//! bind addresses, tools, paths, or another model's checkpoint identity.
use super::*;

impl Store {
    pub fn model_profiles(&self, model: &str, artifact: &str) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT id,name,settings,created_at FROM model_profiles WHERE model=?1 AND artifact=?2 ORDER BY name COLLATE NOCASE,id")?;
        let rows = stmt.query_map(params![model, artifact], |r| {
            let raw: String = r.get(2)?;
            Ok(json!({"id":r.get::<_,String>(0)?,"name":r.get::<_,String>(1)?,"model":model,"artifact":artifact,"settings":serde_json::from_str::<Value>(&raw).unwrap_or(Value::Null),"created_at":r.get::<_,i64>(3)?}))
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn save_model_profile(
        &self,
        name: &str,
        model: &str,
        artifact: &str,
        settings: &Value,
    ) -> Result<Value, StoreError> {
        let name = name.trim();
        if name.is_empty() || name.chars().count() > 80 || name.chars().any(char::is_control) {
            return Err(StoreError::Bad(
                "Use a profile name of 1–80 characters.".into(),
            ));
        }
        let fields = [
            "max_ctx",
            "max_batch",
            "spec",
            "no_spec",
            "kv_cache_dtype",
            "vram_budget",
            "kv_offload",
            "runtime_options",
        ];
        let mut safe = serde_json::Map::new();
        for field in fields {
            if let Some(value) = settings.get(field) {
                safe.insert(field.into(), value.clone());
            }
        }
        // The caller supplies a server-generated options projection, never raw
        // TOML. Retain numerical serving controls only, not text/path fields.
        if let Some(options) = safe
            .get_mut("runtime_options")
            .and_then(Value::as_array_mut)
        {
            options.retain(|option| {
                matches!(
                    option["kind"].as_str(),
                    Some("number" | "integer" | "boolean")
                )
            });
        }
        let conn = self.lock();
        let count: i64 = conn.query_row("SELECT count(*) FROM model_profiles", [], |r| r.get(0))?;
        if count >= 256 {
            return Err(StoreError::Bad(
                "Remove an unused profile before adding another.".into(),
            ));
        }
        let id = uuid::Uuid::new_v4().to_string();
        let now = now_ms();
        conn.execute("INSERT INTO model_profiles (id,name,model,artifact,settings,created_at) VALUES (?1,?2,?3,?4,?5,?6)", params![id,name,model,artifact,Value::Object(safe.clone()).to_string(),now])
            .map_err(|_| StoreError::Bad("A profile with that name already exists, or could not be saved.".into()))?;
        Ok(
            json!({"id":id,"name":name,"model":model,"artifact":artifact,"settings":safe,"created_at":now}),
        )
    }

    pub fn remove_model_profile(&self, id: &str) -> Result<(), StoreError> {
        self.lock()
            .execute("DELETE FROM model_profiles WHERE id=?1", params![id])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profiles_are_bounded_scoped_and_never_contain_access_settings() {
        let store = Store::open(&PathBuf::from(":memory:")).unwrap();
        let row = store.save_model_profile("Code", "qwen", "mlx", &json!({"api_key":"secret","host":"0.0.0.0","max_ctx":8192,"runtime_options":[{"id":"temp","kind":"number","value":0.3},{"id":"log_path","kind":"text","value":"private"}]})).unwrap();
        assert!(!row.to_string().contains("secret"));
        assert!(!row.to_string().contains("private"));
        assert_eq!(row["settings"]["max_ctx"], 8192);
        assert!(
            store
                .save_model_profile("Code", "qwen", "mlx", &json!({}))
                .is_err()
        );
        assert!(store.model_profiles("qwen", "gguf").unwrap().is_empty());
        store
            .remove_model_profile(row["id"].as_str().unwrap())
            .unwrap();
        assert!(store.model_profiles("qwen", "mlx").unwrap().is_empty());
    }
}
