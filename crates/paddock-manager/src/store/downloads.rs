//! Download intent lives in the product SQLite, not a second native job store.
//! Only admission and terminal transitions write here; byte progress stays in
//! memory and the range bitmap. Reopening never automatically resumes traffic.
use super::*;

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS model_downloads (
        id TEXT PRIMARY KEY, created_ms INTEGER NOT NULL, document TEXT NOT NULL
    );",
    )?;
    Ok(())
}

impl Store {
    pub fn save_download(&self, document: &Value) -> Result<(), StoreError> {
        let id = document["id"]
            .as_str()
            .ok_or_else(|| StoreError::Bad("download id missing".into()))?;
        let body = serde_json::to_string(document).map_err(|e| StoreError::Bad(e.to_string()))?;
        if body.len() > 64 * 1024 {
            return Err(StoreError::Bad("download record too large".into()));
        }
        let mut conn = self
            .conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO model_downloads(id,created_ms,document) VALUES(?1,?2,?3)
            ON CONFLICT(id) DO UPDATE SET document=excluded.document",
            params![id, document["created_ms"].as_i64().unwrap_or(0), body],
        )?;
        // Never prune interrupted work. Bound only completed history.
        tx.execute(
            "DELETE FROM model_downloads WHERE id IN (
            SELECT id FROM model_downloads WHERE json_extract(document,'$.status.state')='done'
            ORDER BY created_ms DESC LIMIT -1 OFFSET 32)",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn load_downloads(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self
            .conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut query = conn
            .prepare("SELECT document FROM model_downloads ORDER BY created_ms DESC LIMIT 128")?;
        query
            .query_map([], |row| row.get::<_, String>(0))?
            .map(|row| serde_json::from_str(&row?).map_err(|e| StoreError::Bad(e.to_string())))
            .collect()
    }
}
