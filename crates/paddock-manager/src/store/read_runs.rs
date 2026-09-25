//! Bounded Reads history. No full source document or credentials are accepted.
use super::*;

fn scope_ok(scope: &str) -> Result<(), StoreError> {
    if scope.is_empty()
        || scope.len() > 128
        || !scope
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return Err(StoreError::Bad("Invalid read history identity".into()));
    }
    Ok(())
}

impl Store {
    pub fn read_runs(&self, scope: &str) -> Result<Vec<Value>, StoreError> {
        scope_ok(scope)?;
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT body FROM read_runs WHERE scope=?1 ORDER BY created_at DESC,rowid DESC LIMIT 10",
        )?;
        stmt.query_map([scope], |r| r.get::<_, String>(0))?
            .map(|r| serde_json::from_str(&r?).map_err(|e| StoreError::Bad(e.to_string())))
            .collect()
    }

    pub fn save_read_run(&self, scope: &str, doc: &Value) -> Result<(), StoreError> {
        scope_ok(scope)?;
        let object = doc
            .as_object()
            .ok_or_else(|| StoreError::Bad("Invalid read result".into()))?;
        let allowed = [
            "id",
            "at",
            "fingerprint",
            "excerpt",
            "characters",
            "questions",
            "raw",
            "port",
            "elapsedMilliseconds",
        ];
        let id = doc["id"].as_str().unwrap_or("");
        let fingerprint = doc["fingerprint"].as_str().unwrap_or("");
        if object.keys().any(|k| !allowed.contains(&k.as_str()))
            || Uuid::parse_str(id).is_err()
            || fingerprint.len() != 64
            || !fingerprint.bytes().all(|c| c.is_ascii_hexdigit())
            || !doc["at"].is_number()
            || doc["excerpt"]
                .as_str()
                .is_none_or(|s| s.len() > 2048 || s.chars().count() > 512)
            || !doc["characters"].is_u64()
            || doc["port"].as_u64().is_none_or(|p| p == 0 || p > 65535)
            || doc["questions"]
                .as_array()
                .is_none_or(|q| q.is_empty() || q.len() > 64)
            || !doc["raw"]["answers"].is_object()
            || doc["raw"].as_object().is_none_or(|v| {
                v.keys()
                    .any(|k| !["model", "answers", "usage", "diagnostics"].contains(&k.as_str()))
            })
            || doc["elapsedMilliseconds"].as_f64().is_none_or(|v| v < 0.0)
        {
            return Err(StoreError::Bad(
                "Invalid read history record; store a result and short excerpt, not the document"
                    .into(),
            ));
        }
        let body = doc.to_string();
        if body.len() > 1024 * 1024 {
            return Err(StoreError::Bad("Read result exceeds 1 MiB".into()));
        }
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        // Existing IDs are immutable: retries are idempotent, not last-writer-wins.
        tx.execute(
            "INSERT OR IGNORE INTO read_runs(scope,id,created_at,body) VALUES (?1,?2,?3,?4)",
            params![scope, id, now_ms(), body],
        )?;
        tx.execute("DELETE FROM read_runs WHERE scope=?1 AND id NOT IN (SELECT id FROM read_runs WHERE scope=?1 ORDER BY created_at DESC,rowid DESC LIMIT 10)", [scope])?;
        // Global byte AND row bounds, applied atomically, including all sets.
        tx.execute("DELETE FROM read_runs WHERE rowid IN (SELECT rowid FROM (SELECT rowid, sum(length(CAST(body AS BLOB))) OVER (ORDER BY created_at DESC,rowid DESC) AS bytes, row_number() OVER (ORDER BY created_at DESC,rowid DESC) AS n FROM read_runs) WHERE bytes > 16777216 OR n > 200)", [])?;
        tx.commit()?;
        Ok(())
    }

    pub fn clear_read_runs(&self, scope: &str) -> Result<(), StoreError> {
        scope_ok(scope)?;
        self.lock()
            .execute("DELETE FROM read_runs WHERE scope=?1", [scope])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(n: usize) -> Value {
        json!({"id":Uuid::new_v4().to_string(),"at":n,"fingerprint":"a".repeat(64),"excerpt":"Short excerpt","characters":123456,"questions":[{}],"raw":{"answers":{}},"port":11543,"elapsedMilliseconds":1})
    }
    #[test]
    fn history_survives_reopen_is_bounded_and_rejects_full_documents() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("reads.db");
        let db = Store::open(&path).unwrap();
        for n in 0..14 {
            db.save_read_run("draft", &record(n)).unwrap();
        }
        let saved = db.read_runs("draft").unwrap();
        assert_eq!(saved.len(), 10);
        assert_eq!(saved[0]["at"], 13);
        db.save_read_run("draft", &saved[0]).unwrap();
        assert_eq!(db.read_runs("draft").unwrap().len(), 10);
        let mut bad = record(15);
        bad["state"] = json!("Do not retain this document");
        assert!(db.save_read_run("draft", &bad).is_err());
        assert!(db.read_runs("../escape").is_err());
        drop(db);
        let db = Store::open(&path).unwrap();
        assert_eq!(db.read_runs("draft").unwrap(), saved);
        db.clear_read_runs("draft").unwrap();
        assert!(db.read_runs("draft").unwrap().is_empty());
    }

    #[test]
    fn global_retention_bounds_bytes_and_rows_and_set_deletion_clears_history() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        for n in 0..205 {
            db.save_read_run(&format!("scope-{n}"), &record(n)).unwrap();
        }
        let count: i64 = db
            .lock()
            .query_row("SELECT count(*) FROM read_runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 200);
        for n in 0..25 {
            let mut doc = record(n);
            doc["raw"]["answers"]["payload"] = json!("x".repeat(900_000));
            db.save_read_run(&format!("large-{n}"), &doc).unwrap();
        }
        let bytes: i64 = db
            .lock()
            .query_row(
                "SELECT sum(length(CAST(body AS BLOB))) FROM read_runs",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(bytes <= 16 * 1024 * 1024);
        let saved = db
            .put_read_set(&json!({"id":"set","name":"Set","body":"{}"}))
            .unwrap();
        db.save_read_run("set", &record(0)).unwrap();
        db.delete_read_set_checked("set", saved["revision"].as_str())
            .unwrap();
        assert!(db.read_runs("set").unwrap().is_empty());
    }
}
