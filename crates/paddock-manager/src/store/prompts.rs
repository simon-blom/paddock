//! Both editors use the same SQLite records. Optional revisions keep legacy
//! API callers compatible; reviewed writes are checked inside an IMMEDIATE
//! transaction, including against another connection to this database.
use super::*;

fn row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let vars: String = r.get(3)?;
    let mut doc = json!({"id": r.get::<_, String>(0)?, "name": r.get::<_, String>(1)?,
        "body": r.get::<_, String>(2)?, "variables": serde_json::from_str::<Value>(&vars).unwrap_or_else(|_| json!([])),
        "createdAt": r.get::<_, i64>(4)?, "updatedAt": r.get::<_, i64>(5)?});
    let revision: String = Sha256::digest(doc.to_string().as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    doc["revision"] = json!(revision);
    Ok(doc)
}

fn current(conn: &Connection, id: &str) -> Result<Option<Value>, StoreError> {
    Ok(conn
        .query_row(
            "SELECT id,name,body,variables,created_at,updated_at FROM prompts WHERE id=?1",
            [id],
            row,
        )
        .optional()?)
}

fn check(existing: Option<&Value>, expected: Option<&str>) -> Result<(), StoreError> {
    if let Some(expected) = expected {
        let actual = existing.and_then(|v| v["revision"].as_str()).unwrap_or("");
        if actual != expected {
            return Err(StoreError::Conflict("This preset changed or was deleted. Reload it before saving or deleting; your draft is kept.".into()));
        }
    }
    Ok(())
}

impl Store {
    pub fn list_prompts(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT id,name,body,variables,created_at,updated_at FROM prompts ORDER BY updated_at DESC,id")?;
        Ok(stmt.query_map([], row)?.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn put_prompt(&self, doc: &Value) -> Result<Value, StoreError> {
        let id = doc["id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = doc["name"].as_str().unwrap_or("Untitled");
        let body = doc["body"].as_str().unwrap_or("");
        if id.len() > 128 || name.len() > 512 || body.len() > 128 * 1024 {
            return Err(StoreError::Bad("Preset exceeds its size limit".into()));
        }
        let expected = match doc.get("revision") {
            None => None,
            Some(Value::String(v)) => Some(v.as_str()),
            _ => return Err(StoreError::Bad("Invalid preset revision".into())),
        };
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let old = current(&tx, &id)?;
        check(old.as_ref(), expected)?;
        // Editing the text must not erase metadata the current editor doesn't expose.
        let vars = doc
            .get("variables")
            .or_else(|| old.as_ref().map(|v| &v["variables"]))
            .cloned()
            .unwrap_or_else(|| json!([]));
        let created = old
            .as_ref()
            .and_then(|v| v["createdAt"].as_i64())
            .or_else(|| doc["createdAt"].as_i64())
            .unwrap_or_else(now_ms);
        let now = now_ms().max(
            old.as_ref()
                .and_then(|v| v["updatedAt"].as_i64())
                .unwrap_or(0)
                .saturating_add(1),
        );
        tx.execute("INSERT INTO prompts (id,name,body,variables,created_at,updated_at) VALUES (?1,?2,?3,?4,?5,?6)
            ON CONFLICT(id) DO UPDATE SET name=?2,body=?3,variables=?4,updated_at=?6",
            params![id,name,body,vars.to_string(),created,now])?;
        let saved = current(&tx, &id)?.expect("inserted prompt");
        tx.commit()?;
        Ok(saved)
    }

    pub fn delete_prompt_checked(
        &self,
        id: &str,
        revision: Option<&str>,
    ) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        check(current(&tx, id)?.as_ref(), revision)?;
        tx.execute("DELETE FROM prompts WHERE id=?1", [id])?;
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn revisions_protect_update_delete_and_preserve_metadata() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        db.put_prompt(&json!({"id":"p","name":"First","body":"Original","variables":["kept"],"createdAt":1,"revision":""})).unwrap();
        let old = db.list_prompts().unwrap().remove(0);
        let mut edit = json!({"id":"p","name":"Renamed","body":"New","revision":old["revision"]});
        db.put_prompt(&edit).unwrap();
        assert!(matches!(db.put_prompt(&edit), Err(StoreError::Conflict(_))));
        assert!(
            db.delete_prompt_checked("p", old["revision"].as_str())
                .is_err()
        );
        let next = db.list_prompts().unwrap().remove(0);
        assert_eq!(next["variables"], json!(["kept"]));
        assert_eq!(next["createdAt"], 1);
        db.delete_prompt_checked("p", next["revision"].as_str())
            .unwrap();
        edit["revision"] = next["revision"].clone();
        assert!(db.put_prompt(&edit).is_err()); // A stale save cannot resurrect deletion.
        assert!(db.list_prompts().unwrap().is_empty());
    }
    #[test]
    fn invalid_or_oversize_writes_do_not_publish() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        for doc in [
            json!({"id":"p","body":"x".repeat(128*1024+1)}),
            json!({"id":"p","revision":null}),
        ] {
            assert!(db.put_prompt(&doc).is_err());
        }
        assert!(db.list_prompts().unwrap().is_empty());
    }
    #[test]
    fn competing_connections_cannot_both_replace_a_reviewed_revision() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("prompts.db");
        let first = Store::open(&path).unwrap();
        let second = Store::open(&path).unwrap();
        let original = first
            .put_prompt(&json!({"id":"p","name":"Name","body":"Before","revision":""}))
            .unwrap();
        let edit = json!({"id":"p","name":"Name","body":"After","revision":original["revision"]});
        let competing = edit.clone();
        let gate = std::sync::Arc::new(std::sync::Barrier::new(2));
        let other_gate = gate.clone();
        let handle = std::thread::spawn(move || {
            other_gate.wait();
            second.put_prompt(&competing)
        });
        gate.wait();
        let a = first.put_prompt(&edit);
        let b = handle.join().unwrap();
        assert_ne!(a.is_ok(), b.is_ok());
        let rejected = if a.is_err() { a } else { b };
        assert!(matches!(rejected, Err(StoreError::Conflict(_))));
    }
}
