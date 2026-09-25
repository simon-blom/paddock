//! Read history - the Reads page's earlier reads (`/api/read-history`), kept
//! the way conversations are: summaries for the side panel, the whole record
//! when one is opened. A read is a text, its questions and every run made of
//! them. `doc` is the Studio's JSON; the store reads only the fields the list
//! shows (title, model, how many runs, the timestamps) so the side panel
//! answers without parsing a document per row.
//!
//! The document is kept as the TEXT the Studio sent and handed back byte for
//! byte. A run's questions are a JSON object whose key order IS the order the
//! user wrote them in, and serde_json's map sorts its keys - a round trip
//! through `Value` reopened every read with its questions alphabetised.
//!
//! A summary carries `runs` as a COUNT, and a whole read carries it as the
//! array of runs. `put_read` refuses a document whose `runs` is not an array,
//! so a list row sent back by mistake cannot overwrite the read it stands
//! for - the failure the conversation list once had, where saving a stub
//! wiped the messages.
use super::*;

/// A read holds its full text once per run, so a long document re-read many
/// times is the size that matters. 16 MiB covers twenty runs of a ~800K-char
/// text; past it the Studio is asked to start a new read.
const DOC_LIMIT: usize = 16 * 1024 * 1024;

fn revision(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn check_revision(conn: &Connection, id: &str, expected: Option<&str>) -> Result<(), StoreError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let old: Option<String> = conn
        .query_row("SELECT doc FROM read_history WHERE id=?1", [id], |r| {
            r.get(0)
        })
        .optional()?;
    if old.as_deref().map(revision).unwrap_or_default() != expected {
        return Err(StoreError::Conflict("This read changed or was deleted. Reopen it before saving; your current results are kept.".into()));
    }
    Ok(())
}

fn summary(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    Ok(json!({
        "id": r.get::<_, String>(0)?,
        "title": r.get::<_, String>(1)?,
        "model": r.get::<_, String>(2)?,
        "runs": r.get::<_, i64>(3)?,
        "createdAt": r.get::<_, i64>(4)?,
        "updatedAt": r.get::<_, i64>(5)?,
    }))
}

const SUMMARY_COLS: &str = "id,title,model,runs,created_at,updated_at";

impl Store {
    /// Summaries only, the most recently run first.
    pub fn list_read_history(&self) -> Result<Vec<Value>, StoreError> {
        self.migrate_native_read_history()?;
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {SUMMARY_COLS} FROM read_history ORDER BY updated_at DESC, id"
        ))?;
        Ok(stmt
            .query_map([], summary)?
            .collect::<Result<Vec<_>, _>>()?)
    }

    /// The read as it was saved: the exact JSON text, never re-serialized.
    pub fn get_read(&self, id: &str) -> Result<Option<String>, StoreError> {
        let conn = self.lock();
        Ok(conn
            .query_row(
                "SELECT doc FROM read_history WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub fn read_snapshot(&self, id: &str) -> Result<Option<Value>, StoreError> {
        Ok(self
            .get_read(id)?
            .map(|text| json!({"revision":revision(&text), "doc":text})))
    }

    /// Create or replace a whole read from the JSON text the Studio sent;
    /// `id` is the one its address names. Returns the summary.
    pub fn put_read(&self, id: &str, text: &str) -> Result<Value, StoreError> {
        self.put_read_checked(id, text, None)
    }

    pub fn put_read_checked(
        &self,
        id: &str,
        text: &str,
        expected: Option<&str>,
    ) -> Result<Value, StoreError> {
        if text.len() > DOC_LIMIT {
            return Err(StoreError::Bad(
                "This read is too large to keep. Start a new read for further runs.".into(),
            ));
        }
        let doc: Value =
            serde_json::from_str(text).map_err(|e| StoreError::Bad(format!("Not a read: {e}")))?;
        if id.is_empty() || id.len() > 128 {
            return Err(StoreError::Bad("A read needs an id".into()));
        }
        // the address names the read; a body naming another one is refused
        // rather than rewritten, since rewriting means re-serializing
        if doc.get("id").and_then(Value::as_str) != Some(id) {
            return Err(StoreError::Bad(
                "The read's id does not match its address".into(),
            ));
        }
        let runs = doc
            .get("runs")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                StoreError::Bad(
                    "A read is saved with its runs. This looks like a list row; open the read first."
                        .into(),
                )
            })?
            .len();
        let title = doc
            .get("title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("Untitled read");
        let model = doc.get("model").and_then(Value::as_str).unwrap_or("");
        if title.len() > 512 || model.len() > 512 {
            return Err(StoreError::Bad(
                "A read's title or model is too long".into(),
            ));
        }
        let created = doc
            .get("createdAt")
            .and_then(Value::as_i64)
            .unwrap_or_else(now_ms);
        let updated = doc
            .get("updatedAt")
            .and_then(Value::as_i64)
            .unwrap_or(created);
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let identical: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM read_history WHERE id=?1 AND doc=?2)",
            params![id, text],
            |r| r.get(0),
        )?;
        // A lost acknowledgement may retry the exact same write. It cannot
        // overwrite newer content, but should not strand the client's result.
        if !identical {
            check_revision(&tx, id, expected)?;
        }
        tx.execute(
            "INSERT INTO read_history (id,title,model,runs,created_at,updated_at,doc)
             VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(id) DO UPDATE SET
               title=?2, model=?3, runs=?4, updated_at=?6, doc=?7",
            params![id, title, model, runs as i64, created, updated, text],
        )?;
        let mut saved = tx.query_row(
            &format!("SELECT {SUMMARY_COLS} FROM read_history WHERE id=?1"),
            [id],
            summary,
        )?;
        saved["revision"] = json!(revision(text));
        tx.commit()?;
        Ok(saved)
    }

    /// True when a read was removed.
    pub fn delete_read(&self, id: &str) -> Result<bool, StoreError> {
        self.delete_read_checked(id, None)
    }

    pub fn delete_read_checked(
        &self,
        id: &str,
        expected: Option<&str>,
    ) -> Result<bool, StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        check_revision(&tx, id, expected)?;
        let removed = tx.execute("DELETE FROM read_history WHERE id = ?1", params![id])? > 0;
        tx.commit()?;
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_clients_cannot_overwrite_or_delete_a_newer_revision() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("history.db");
        let web = Store::open(&path).unwrap();
        let native = Store::open(&path).unwrap();
        web.put_read_checked("a", &read("a", "Web", 1, 10), Some(""))
            .unwrap();
        let snapshot = native.read_snapshot("a").unwrap().unwrap();
        let old = snapshot["revision"].as_str().unwrap();
        native
            .put_read_checked("a", &read("a", "Native", 2, 20), Some(old))
            .unwrap();
        assert!(matches!(
            web.put_read_checked("a", &read("a", "Stale", 1, 30), Some(old)),
            Err(StoreError::Conflict(_))
        ));
        assert!(matches!(
            web.delete_read_checked("a", Some(old)),
            Err(StoreError::Conflict(_))
        ));
        drop(web);
        drop(native);
        let db = Store::open(&path).unwrap();
        let snapshot = db.read_snapshot("a").unwrap().unwrap();
        assert!(snapshot["doc"].as_str().unwrap().contains("Native"));
        db.delete_read_checked("a", snapshot["revision"].as_str())
            .unwrap();
        assert!(db.get_read("a").unwrap().is_none());
    }

    fn read(id: &str, title: &str, runs: usize, updated: i64) -> String {
        json!({
            "id": id, "title": title, "model": "DiffusionGemma 26B A4B",
            "createdAt": 1, "updatedAt": updated,
            "runs": (0..runs).map(|i| json!({"at": i, "ms": 10, "state": "Portal down again"})).collect::<Vec<_>>(),
        })
        .to_string()
    }

    #[test]
    fn the_list_is_summaries_newest_first_and_a_read_comes_back_whole() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        db.put_read("a", &read("a", "Ticket", 1, 10)).unwrap();
        db.put_read("b", &read("b", "Contract", 3, 20)).unwrap();
        let list = db.list_read_history().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0]["id"], "b");
        assert_eq!(list[0]["runs"], 3);
        assert!(list[0].get("state").is_none(), "the list carries no text");
        let whole: Value = serde_json::from_str(&db.get_read("b").unwrap().unwrap()).unwrap();
        assert_eq!(whole["runs"].as_array().unwrap().len(), 3);
        assert_eq!(whole["runs"][0]["state"], "Portal down again");
    }

    #[test]
    fn a_read_comes_back_byte_for_byte_so_question_order_survives() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        // written in the user's order, which is not alphabetical
        let text = r#"{"id":"q","title":"Order","runs":[{"questions":{"urgent":{},"about":{},"mood":{}}}]}"#;
        db.put_read("q", text).unwrap();
        assert_eq!(db.get_read("q").unwrap().unwrap(), text);
    }

    #[test]
    fn a_rerun_replaces_the_record_and_keeps_its_created_time() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        db.put_read("a", &read("a", "Ticket", 1, 10)).unwrap();
        let mut next: Value = serde_json::from_str(&read("a", "Ticket", 2, 30)).unwrap();
        next["createdAt"] = json!(999);
        let saved = db.put_read("a", &next.to_string()).unwrap();
        assert_eq!(saved["runs"], 2);
        assert_eq!(saved["createdAt"], 1);
        assert_eq!(saved["updatedAt"], 30);
    }

    #[test]
    fn a_list_row_sent_back_cannot_overwrite_the_read() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        db.put_read("a", &read("a", "Ticket", 2, 10)).unwrap();
        let stub = db.list_read_history().unwrap().remove(0);
        assert!(matches!(
            db.put_read("a", &stub.to_string()),
            Err(StoreError::Bad(_))
        ));
        let whole: Value = serde_json::from_str(&db.get_read("a").unwrap().unwrap()).unwrap();
        assert_eq!(whole["runs"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn delete_limits_and_defaults() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        let saved = db.put_read("a", &read("a", "  ", 1, 10)).unwrap();
        assert_eq!(saved["title"], "Untitled read");
        // the body must name the read its address names
        assert!(db.put_read("b", &read("a", "Ticket", 1, 10)).is_err());
        assert!(db.put_read("a", "not json").is_err());
        let big = format!(
            r#"{{"id":"big","runs":[],"pad":"{}"}}"#,
            "x".repeat(DOC_LIMIT)
        );
        assert!(db.put_read("big", &big).is_err());
        assert!(db.delete_read("a").unwrap());
        assert!(!db.delete_read("a").unwrap());
        assert!(db.get_read("a").unwrap().is_none());
        assert!(db.list_read_history().unwrap().is_empty());
    }
}
