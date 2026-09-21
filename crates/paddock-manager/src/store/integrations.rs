//! Native connector metadata and credential publication share the web table.
//! Resolve secrets after releasing SQLite; a locked Keychain must never hold
//! the product database hostage. Revision checks run inside the write transaction.
use super::*;

#[cfg(test)]
#[path = "connector_session_tests.rs"]
mod session_tests;

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    let mut stmt = conn.prepare("PRAGMA table_info(connectors)")?;
    let names = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !names.iter().any(|n| n == "revision") {
        conn.execute(
            "ALTER TABLE connectors ADD COLUMN revision INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }
    if !names.iter().any(|n| n == "native_credentials") {
        conn.execute(
            "ALTER TABLE connectors ADD COLUMN native_credentials INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }
    if !names.iter().any(|n| n == "oauth_revision") {
        conn.execute(
            "ALTER TABLE connectors ADD COLUMN oauth_revision INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }
    Ok(())
}

impl Store {
    /// Runs on the desktop startup blocking worker, before requests are admitted.
    /// Publication is serialized with edits/deletion, never with the SQLite lock
    /// or cache-only reads. Denial remains locked until explicit Unlock.
    pub fn prepare_connector_credentials(&self) -> Result<(), StoreError> {
        let _publication = self
            .connector_credential_updates
            .lock()
            .expect("connector publication");
        self.connector_credentials.enable();
        let references = {
            let conn = self.lock();
            let mut query = conn.prepare("SELECT headers FROM connectors WHERE headers LIKE 'paddock-keychain:v1:%' UNION SELECT oauth FROM connectors WHERE oauth LIKE 'paddock-keychain:v1:%'")?;
            query
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };
        for reference in references {
            if self.connector_credentials.authorize(&reference).is_err() {
                tracing::warn!("A connector needs authorization in Manager before use");
            }
        }
        Ok(())
    }

    pub fn unlock_connector(&self, id: &str, revision: u64) -> Result<(), StoreError> {
        let _publication = self
            .connector_credential_updates
            .lock()
            .expect("connector publication");
        let revision = i64::try_from(revision)
            .map_err(|_| StoreError::Bad("Invalid connector revision.".into()))?;
        let (headers, oauth): (String, String) = self.lock().query_row(
            "SELECT headers,oauth FROM connectors WHERE id=?1 AND revision=?2",
            params![id, revision],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        for reference in [headers, oauth] {
            self.connector_credentials
                .authorize(&reference)
                .map_err(|_| {
                    StoreError::Bad(
                        "Connector access is locked. Unlock it in Manager > Connectors before use."
                            .into(),
                    )
                })?;
        }
        let unchanged: bool = self.lock().query_row(
            "SELECT EXISTS(SELECT 1 FROM connectors WHERE id=?1 AND revision=?2)",
            params![id, revision],
            |r| r.get(0),
        )?;
        if !unchanged {
            return Err(StoreError::Bad(
                "This connector changed. Reload before unlocking.".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn resolve_connector(&self, mut row: Value) -> Result<Value, StoreError> {
        for field in ["headers", "oauth"] {
            let raw = row[field].as_str().unwrap_or_default().to_owned();
            let plain = self.connector_credentials.resolve(raw).map_err(|_| {
                StoreError::Bad(
                    "Connector access is locked. Unlock it in Manager > Connectors before use."
                        .into(),
                )
            })?;
            row[field] = if plain.is_empty() {
                Value::Null
            } else {
                serde_json::from_str(&plain)
                    .map_err(|_| StoreError::Bad("Invalid saved connector credential.".into()))?
            };
        }
        Ok(row)
    }

    /// Unlike list_connectors, this path neither fetches nor unlocks secrets.
    pub fn native_connectors(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT id,label,url,registry_key,system,ports,created_at,revision,
            headers NOT IN ('','{}'), oauth != '', (headers LIKE 'paddock-keychain:v1:%' OR oauth LIKE 'paddock-keychain:v1:%'), oauth_revision, headers, oauth
            FROM connectors ORDER BY created_at DESC")?;
        let rows = stmt.query_map([], |r| Ok(json!({
            "id":r.get::<_,String>(0)?, "label":r.get::<_,String>(1)?, "url":r.get::<_,String>(2)?,
            "registryKey":r.get::<_,String>(3)?, "system":r.get::<_,bool>(4)?,
            "ports":serde_json::from_str::<Value>(&r.get::<_,String>(5)?).unwrap_or(json!([])),
            "createdAt":r.get::<_,i64>(6)?, "revision":r.get::<_,i64>(7)?,
            "hasHeaders":r.get::<_,bool>(8)?, "connected":r.get::<_,bool>(9)?,
            "keychain":r.get::<_,bool>(10)?,"oauthRevision":r.get::<_,i64>(11)?,
            "credentialReady":self.connector_credentials.available(&r.get::<_,String>(12)?) && self.connector_credentials.available(&r.get::<_,String>(13)?)
        })))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn save_native_connector(&self, doc: &Value) -> Result<String, StoreError> {
        let _publication = self
            .connector_credential_updates
            .lock()
            .expect("connector publication");
        let (label, url, headers) = Self::connector_fields(doc)?;
        let id = doc["id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let protected = if headers == "{}" {
            headers.clone()
        } else {
            crate::credentials::protect(&headers).map_err(StoreError::Bad)?
        };
        let result = (|| {
            let mut conn = self.lock();
            let tx = conn.transaction()?;
            let old: Option<(i64, String, String, String)> = tx
                .query_row(
                    "SELECT revision,headers,oauth,url FROM connectors WHERE id=?1",
                    [&id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )
                .optional()?;
            if old.as_ref().map(|r| r.0) != doc["revision"].as_i64() {
                return Err(StoreError::Bad(
                    "This connector changed. Reload it before saving.".into(),
                ));
            }
            let duplicate: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM connectors WHERE label=?1 AND id!=?2)",
                params![label, id],
                |r| r.get(0),
            )?;
            if duplicate {
                return Err(StoreError::Bad(
                    "A connector already uses this label.".into(),
                ));
            }
            if old.is_some() {
                tx.execute("UPDATE connectors SET label=?2,oauth=CASE WHEN url!=?3 THEN '' ELSE oauth END,url=?3,headers=?4,revision=revision+1,native_credentials=1 WHERE id=?1",params![id,label,url,protected])?;
            } else {
                let count: i64 =
                    tx.query_row("SELECT count(*) FROM connectors", [], |r| r.get(0))?;
                if count >= 128 {
                    return Err(StoreError::Bad(
                        "At most 128 connectors can be saved.".into(),
                    ));
                }
                tx.execute("INSERT INTO connectors(id,label,url,headers,registry_key,created_at,revision,native_credentials) VALUES(?1,?2,?3,?4,?5,?6,1,1)",params![id,label,url,protected,doc["registryKey"].as_str().unwrap_or(""),now_ms()])?;
            }
            tx.commit()?;
            Ok(old)
        })();
        match result {
            Ok(old) => {
                self.connector_credentials.remember(&protected, &headers);
                if let Some((_, headers, oauth, old_url)) = old {
                    self.connector_credentials.forget(&headers);
                    crate::credentials::retire(&headers);
                    if old_url != url {
                        self.connector_credentials.forget(&oauth);
                        crate::credentials::retire(&oauth);
                    }
                }
                Ok(id)
            }
            Err(e) => {
                crate::credentials::retire(&protected);
                Err(e)
            }
        }
    }

    pub fn connector_uses_keychain(&self, id: &str) -> bool {
        self.lock()
            .query_row(
                "SELECT native_credentials FROM connectors WHERE id=?1",
                [id],
                |r| r.get::<_, bool>(0),
            )
            .unwrap_or(false)
    }
    pub fn invalidate_connector_revision(&self, id: &str) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE connectors SET revision=revision+1 WHERE id=?1",
            [id],
        )?;
        Ok(())
    }

    /// OAuth callback/refresh must not reanimate a deleted, disconnected or
    /// retargeted connector. Keychain publication uses the same CAS as edits.
    pub fn set_connector_oauth_checked(
        &self,
        id: &str,
        revision: u64,
        oauth: &str,
    ) -> Result<(), StoreError> {
        let _publication = self
            .connector_credential_updates
            .lock()
            .expect("connector publication");
        let revision = i64::try_from(revision)
            .map_err(|_| StoreError::Bad("Invalid connector revision.".into()))?;
        let stored = if self.connector_uses_keychain(id) {
            crate::credentials::protect(oauth).map_err(StoreError::Bad)?
        } else {
            oauth.to_owned()
        };
        let result = (|| {
            let mut conn = self.lock();
            let tx = conn.transaction()?;
            let old: String = tx.query_row(
                "SELECT oauth FROM connectors WHERE id=?1 AND revision=?2",
                params![id, revision],
                |r| r.get(0),
            )?;
            tx.execute("UPDATE connectors SET oauth=?2,revision=revision+1,oauth_revision=oauth_revision+1 WHERE id=?1",params![id,stored])?;
            tx.commit()?;
            Ok::<_, StoreError>(old)
        })();
        match result {
            Ok(old) => {
                self.connector_credentials.remember(&stored, oauth);
                self.connector_credentials.forget(&old);
                crate::credentials::retire(&old);
                Ok(())
            }
            Err(e) => {
                crate::credentials::retire(&stored);
                Err(e)
            }
        }
    }
}
