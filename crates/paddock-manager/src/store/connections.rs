use super::*;
use crate::connections::{Pick, Prepared};

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    let mut stmt = conn.prepare("PRAGMA table_info(cloud_endpoints)")?;
    let columns = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    for (name, declaration) in [
        ("revision", "INTEGER NOT NULL DEFAULT 0"),
        ("allow_unauthenticated", "INTEGER NOT NULL DEFAULT 0"),
    ] {
        if !columns.iter().any(|c| c == name) {
            conn.execute(
                &format!("ALTER TABLE cloud_endpoints ADD COLUMN {name} {declaration}"),
                [],
            )?;
        }
    }
    Ok(())
}

impl Store {
    pub fn prepare_cloud_credentials(&self) -> Result<(), StoreError> {
        self.cloud_credentials.enable();
        let references = {
            let conn = self.lock();
            let mut query = conn.prepare("SELECT DISTINCT api_key FROM cloud_endpoints WHERE api_key LIKE 'paddock-keychain:v1:%'")?;
            query
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };
        for reference in references {
            if self.cloud_credentials.authorize(&reference).is_err() {
                tracing::warn!("A cloud account needs authorization in Manager before use");
            }
        }
        Ok(())
    }
    pub(crate) fn resolve_cloud_credential(&self, stored: String) -> Result<String, String> {
        self.cloud_credentials.resolve(stored)
    }
    pub fn unlock_cloud_connection(&self, id: &str, revision: u64) -> Result<(), String> {
        let (_, _, reference) = self.connection_for_revision(id, revision)?;
        self.cloud_credentials.authorize(&reference)?;
        self.connection_for_revision(id, revision)?;
        Ok(())
    }
    pub fn connection_for_revision(
        &self,
        id: &str,
        revision: u64,
    ) -> Result<(String, String, String), String> {
        self.lock()
            .query_row(
                "SELECT kind, base_url, api_key FROM cloud_endpoints WHERE id=?1 AND revision=?2",
                params![
                    id,
                    i64::try_from(revision).map_err(|_| "Invalid connection revision.")?
                ],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .map_err(|_| {
                "This connection changed or was removed. Refresh and review it before saving."
                    .into()
            })
    }

    pub fn connection_allows_unauthenticated(&self, id: &str) -> bool {
        self.lock()
            .query_row(
                "SELECT allow_unauthenticated FROM cloud_endpoints WHERE id=?1",
                [id],
                |r| r.get::<_, bool>(0),
            )
            .unwrap_or(false)
    }

    /// Keychain writes precede the SQLite transaction. A failed/stale commit
    /// removes only the new item, preserving the original key and model picks.
    pub fn save_checked_connection(
        &self,
        prepared: &Prepared,
        picks: &[Pick],
    ) -> Result<Value, String> {
        let d = &prepared.draft;
        crate::connections::validate_picks(picks, crate::connections::is_openrouter(&d.base_url))?;
        let models = serde_json::to_string(picks).map_err(|_| "Invalid model choices.")?;
        let stored = crate::credentials::protect(&prepared.key)?;
        let id = d.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        let result = (|| {
            let mut conn = self.lock();
            let tx = conn
                .transaction()
                .map_err(|_| "Could not begin saving the connection.")?;
            let old: Option<(String, i64)> = tx
                .query_row(
                    "SELECT api_key, revision FROM cloud_endpoints WHERE id=?1",
                    [&id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()
                .map_err(|_| "Could not read the saved connection.")?;
            if old.as_ref().map(|r| r.1 as u64) != d.revision {
                return Err(
                    "This connection changed or was removed. Refresh and review it before saving.",
                );
            }
            if old.is_none() {
                let count: i64 = tx
                    .query_row("SELECT count(*) FROM cloud_endpoints", [], |r| r.get(0))
                    .map_err(|_| "Could not read connections.")?;
                if count >= 64 {
                    return Err("At most 64 connections can be saved.");
                }
                tx.execute("INSERT INTO cloud_endpoints (id,name,kind,base_url,api_key,models,created_at,revision,allow_unauthenticated) VALUES (?1,?2,?3,?4,?5,?6,?7,1,?8)",
                    params![id,d.name,d.kind,d.base_url,stored,models,now_ms(),d.allow_unauthenticated]).map_err(|_| "Could not save the connection. Nothing was changed.")?;
            } else {
                tx.execute("UPDATE cloud_endpoints SET name=?2,kind=?3,base_url=?4,api_key=?5,models=?6,allow_unauthenticated=?7,revision=revision+1 WHERE id=?1",
                    params![id,d.name,d.kind,d.base_url,stored,models,d.allow_unauthenticated]).map_err(|_| "Could not save the connection. Nothing was changed.")?;
            }
            let row = tx.query_row("SELECT id,name,kind,base_url,models,api_key,created_at,revision,allow_unauthenticated FROM cloud_endpoints WHERE id=?1", [&id], Self::cloud_row).map_err(|_| "Could not confirm the saved connection.")?;
            tx.commit()
                .map_err(|_| "Could not commit the connection. Nothing was changed.")?;
            Ok((row, old.map(|r| r.0)))
        })();
        match result {
            Ok((mut row, old)) => {
                self.cloud_credentials.remember(&stored, &prepared.key);
                row["credentialReady"] = json!(self.cloud_credentials.available(&stored));
                if let Some(old) = old {
                    self.cloud_credentials.forget(&old);
                    crate::credentials::retire(&old);
                }
                Ok(row)
            }
            Err(error) => {
                crate::credentials::retire(&stored);
                Err(error.into())
            }
        }
    }

    /// A model pick edit is a single CAS, not a last-writer-wins replacement
    /// of the endpoint. It never touches the credential or address.
    pub fn set_connection_models(
        &self,
        id: &str,
        revision: u64,
        picks: &[Pick],
    ) -> Result<(), String> {
        let (_, base, _) = self.connection_for_revision(id, revision)?;
        crate::connections::validate_picks(picks, crate::connections::is_openrouter(&base))?;
        let models = serde_json::to_string(picks).map_err(|_| "Invalid model choices.")?;
        let changed = self.lock().execute("UPDATE cloud_endpoints SET models=?3,revision=revision+1 WHERE id=?1 AND revision=?2", params![id,i64::try_from(revision).map_err(|_| "Invalid connection revision.")?,models])
            .map_err(|_| "Could not save model choices. Nothing was changed.")?;
        if changed != 1 {
            return Err("This connection changed or was removed. Refresh and try again.".into());
        }
        Ok(())
    }

    pub fn remove_connection(&self, id: &str, revision: u64) -> Result<(), String> {
        let mut conn = self.lock();
        let tx = conn
            .transaction()
            .map_err(|_| "Could not begin removing the connection.")?;
        let stored: String = tx
            .query_row(
                "SELECT api_key FROM cloud_endpoints WHERE id=?1 AND revision=?2",
                params![
                    id,
                    i64::try_from(revision).map_err(|_| "Invalid connection revision.")?
                ],
                |r| r.get(0),
            )
            .map_err(|_| "This connection changed or was removed. Refresh before removing it.")?;
        tx.execute("DELETE FROM cloud_endpoints WHERE id=?1", [id])
            .map_err(|_| "Could not remove the connection.")?;
        tx.commit()
            .map_err(|_| "Could not commit removal of the connection.")?;
        drop(conn);
        self.cloud_credentials.forget(&stored);
        crate::credentials::retire(&stored);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn saved(db: &Store) -> Value {
        let draft = serde_json::from_value(json!({"name":"Session fixture", "kind":"openai-compat", "baseUrl":"https://example.invalid/v1", "apiKey":"synthetic-original"})).unwrap();
        let prepared = crate::connections::prepare(db, draft).unwrap();
        db.save_checked_connection(&prepared, &[]).unwrap()
    }
    #[test]
    fn startup_prepares_existing_keys_and_requests_never_return_to_the_vault() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        let row = saved(&db);
        let id = row["id"].as_str().unwrap();
        let (_, _, reference) = db.connection_for_revision(id, 1).unwrap();
        db.prepare_cloud_credentials().unwrap();
        assert_eq!(
            db.list_cloud_endpoints().unwrap()[0]["credentialReady"],
            true
        );
        crate::credentials::retire(&reference);
        for _ in 0..8 {
            assert_eq!(
                db.cloud_endpoint_secret(id).unwrap().unwrap().2,
                "synthetic-original"
            );
        }
        let public = serde_json::to_string(&db.list_cloud_endpoints().unwrap()).unwrap();
        assert!(!public.contains("synthetic-original"));
        assert!(!public.contains(&reference));
    }
    #[test]
    fn unavailable_startup_stays_locked_until_explicit_current_revision_unlock() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        let row = saved(&db);
        let id = row["id"].as_str().unwrap();
        let (_, _, reference) = db.connection_for_revision(id, 1).unwrap();
        crate::credentials::retire(&reference);
        db.prepare_cloud_credentials().unwrap();
        assert_eq!(
            db.list_cloud_endpoints().unwrap()[0]["credentialReady"],
            false
        );
        // Simulate OS access becoming possible; ordinary requests still must
        // not touch the vault until the user explicitly unlocks the account.
        crate::credentials::TEST_VAULT
            .lock()
            .unwrap()
            .insert(reference.clone(), "synthetic-original".into());
        for _ in 0..8 {
            assert!(db.cloud_endpoint_secret(id).is_err());
        }
        assert!(db.unlock_cloud_connection(id, 0).is_err());
        assert!(!db.cloud_credentials.available(&reference));
        db.unlock_cloud_connection(id, 1).unwrap();
        assert!(db.cloud_credentials.available(&reference));
        db.remove_connection(id, 1).unwrap();
        assert!(!db.cloud_credentials.available(&reference));
        assert!(db.unlock_cloud_connection(id, 1).is_err());
    }
    #[test]
    fn saving_and_replacing_keys_publishes_ready_cache_and_revokes_old_references() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        db.prepare_cloud_credentials().unwrap();
        let row = saved(&db);
        let id = row["id"].as_str().unwrap();
        let (_, _, old) = db.connection_for_revision(id, 1).unwrap();
        assert_eq!(row["credentialReady"], true);
        let draft = serde_json::from_value(json!({"id":id,"revision":1,"name":"Replacement", "kind":"openai-compat", "baseUrl":"https://example.invalid/v1", "apiKey":"synthetic-replacement"})).unwrap();
        let prepared = crate::connections::prepare(&db, draft).unwrap();
        db.save_checked_connection(&prepared, &[]).unwrap();
        assert!(db.resolve_cloud_credential(old).is_err());
        assert_eq!(
            db.cloud_endpoint_secret(id).unwrap().unwrap().2,
            "synthetic-replacement"
        );
        // Existing web edits must retain Keychain storage and update the cache.
        db.update_cloud_endpoint(id, &json!({"apiKey":"synthetic-web-replacement"}))
            .unwrap();
        let (_, _, current) = db.connection_for_revision(id, 3).unwrap();
        assert!(current.starts_with("paddock-keychain:v1:"));
        assert_eq!(
            db.cloud_endpoint_secret(id).unwrap().unwrap().2,
            "synthetic-web-replacement"
        );
        db.delete_cloud_endpoint(id).unwrap();
        assert!(db.resolve_cloud_credential(current).is_err());
    }
    #[test]
    fn a_failed_transaction_cannot_publish_partial_metadata_or_key_replacement() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        let row = db.create_cloud_endpoint(&json!({"name":"Original", "kind":"openai-compat", "baseUrl":"https://example.invalid/v1", "apiKey":"fixture-original", "models":[{"id":"original-model"}]})).unwrap();
        let id = row["id"].as_str().unwrap();
        let prepared = crate::connections::prepare(&db, serde_json::from_value(json!({"id":id,"revision":0,"name":"Replacement", "kind":"openai-compat","baseUrl":"https://example.invalid/v1","apiKey":"fixture-replacement"})).unwrap()).unwrap();
        db.lock().execute_batch("CREATE TRIGGER fail_connection_update BEFORE UPDATE ON cloud_endpoints BEGIN SELECT RAISE(ABORT,'synthetic failure'); END;").unwrap();
        assert!(db.save_checked_connection(&prepared, &[]).is_err());
        assert_eq!(
            db.cloud_endpoint_secret(id).unwrap().unwrap().2,
            "fixture-original"
        );
        let listed = db.list_cloud_endpoints().unwrap();
        assert_eq!(listed[0]["name"], "Original");
        assert_eq!(listed[0]["models"][0]["id"], "original-model");
        assert_eq!(listed[0]["revision"], 0);
    }
}
