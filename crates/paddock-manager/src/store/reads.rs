//! Saved read sets - the Reads page's named question sets (`/api/reads`),
//! a sibling of the prompt library with the same record shape and the same
//! reviewed-write rule: a caller that sends the revision it last saw only
//! replaces or deletes that revision, checked inside an IMMEDIATE
//! transaction so two connections cannot both win. `body` is the JSON text
//! of `{ questions, samples }` in the /v1/systemone shape; the store never
//! parses it - the Studio and the runner own that shape.
use super::*;

fn row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let mut doc = json!({"id": r.get::<_, String>(0)?, "name": r.get::<_, String>(1)?,
        "body": r.get::<_, String>(2)?, "createdAt": r.get::<_, i64>(3)?, "updatedAt": r.get::<_, i64>(4)?});
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
            "SELECT id,name,body,created_at,updated_at FROM read_sets WHERE id=?1",
            [id],
            row,
        )
        .optional()?)
}

fn check(existing: Option<&Value>, expected: Option<&str>) -> Result<(), StoreError> {
    if let Some(expected) = expected {
        let actual = existing.and_then(|v| v["revision"].as_str()).unwrap_or("");
        if actual != expected {
            return Err(StoreError::Conflict(
                "This read set changed or was deleted. Reload it before saving or deleting; your draft is kept."
                    .into(),
            ));
        }
    }
    Ok(())
}

impl Store {
    pub fn list_read_sets(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id,name,body,created_at,updated_at FROM read_sets ORDER BY updated_at DESC,id",
        )?;
        Ok(stmt.query_map([], row)?.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn put_read_set(&self, doc: &Value) -> Result<Value, StoreError> {
        let id = doc["id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = doc["name"].as_str().unwrap_or("Untitled");
        let body = doc["body"].as_str().unwrap_or("");
        if id.len() > 128 || name.len() > 512 || body.len() > 512 * 1024 {
            return Err(StoreError::Bad("Read set exceeds its size limit".into()));
        }
        let expected = match doc.get("revision") {
            None => None,
            Some(Value::String(v)) => Some(v.as_str()),
            _ => return Err(StoreError::Bad("Invalid read set revision".into())),
        };
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let old = current(&tx, &id)?;
        check(old.as_ref(), expected)?;
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
        tx.execute(
            "INSERT INTO read_sets (id,name,body,created_at,updated_at) VALUES (?1,?2,?3,?4,?5)
             ON CONFLICT(id) DO UPDATE SET name=?2,body=?3,updated_at=?5",
            params![id, name, body, created, now],
        )?;
        let saved = current(&tx, &id)?.expect("inserted read set");
        tx.commit()?;
        Ok(saved)
    }

    pub fn delete_read_set_checked(
        &self,
        id: &str,
        revision: Option<&str>,
    ) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        check(current(&tx, id)?.as_ref(), revision)?;
        tx.execute("DELETE FROM read_sets WHERE id=?1", [id])?;
        tx.execute("DELETE FROM read_runs WHERE scope=?1", [id])?;
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revisions_protect_update_and_delete() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        db.put_read_set(&json!({"id":"r","name":"Triage","body":"{}","createdAt":1,"revision":""}))
            .unwrap();
        let old = db.list_read_sets().unwrap().remove(0);
        assert_eq!(old["createdAt"], 1);
        let mut edit = json!({"id":"r","name":"Triage 2","body":"{\"questions\":{}}","revision":old["revision"]});
        db.put_read_set(&edit).unwrap();
        assert!(matches!(
            db.put_read_set(&edit),
            Err(StoreError::Conflict(_))
        ));
        assert!(
            db.delete_read_set_checked("r", old["revision"].as_str())
                .is_err()
        );
        let next = db.list_read_sets().unwrap().remove(0);
        assert_eq!(next["name"], "Triage 2");
        assert_eq!(next["createdAt"], 1);
        db.delete_read_set_checked("r", next["revision"].as_str())
            .unwrap();
        edit["revision"] = next["revision"].clone();
        assert!(db.put_read_set(&edit).is_err());
        assert!(db.list_read_sets().unwrap().is_empty());
    }

    #[test]
    fn unreviewed_writes_still_work_and_oversize_is_refused() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        let saved = db
            .put_read_set(&json!({"name":"Plain","body":"{}"}))
            .unwrap();
        assert!(saved["id"].as_str().is_some_and(|s| !s.is_empty()));
        assert!(
            db.put_read_set(&json!({"id":"x","body":"x".repeat(512*1024+1)}))
                .is_err()
        );
        assert!(db.put_read_set(&json!({"id":"x","revision":7})).is_err());
        assert_eq!(db.list_read_sets().unwrap().len(), 1);
    }
}
