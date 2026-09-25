//! Move old native result snapshots into the same history used by both Studios.
//! The source rows are removed only in the transaction that preserves them.
use super::*;

impl Store {
    pub(super) fn migrate_native_read_history(&self) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let rows = {
            let mut stmt =
                tx.prepare("SELECT scope,id,body FROM read_runs ORDER BY created_at,rowid")?;
            stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?
        };
        for (scope, old_id, body) in rows {
            let old: Value =
                serde_json::from_str(&body).map_err(|e| StoreError::Bad(e.to_string()))?;
            let id = format!("native-{old_id}");
            let mut questions = serde_json::Map::new();
            let mut ordering = Vec::new();
            for q in old["questions"]
                .as_array()
                .ok_or_else(|| StoreError::Bad("Invalid legacy read questions".into()))?
            {
                let name = q["questionID"]
                    .as_str()
                    .ok_or_else(|| StoreError::Bad("Invalid legacy question ID".into()))?;
                let kind = q["kind"].as_str().unwrap_or("noul");
                let mut wire =
                    json!({"type":kind,"instructions":q["instructions"].as_str().unwrap_or("")});
                let mut order = vec![json!(name)];
                match kind {
                    "choice" => {
                        let mut criteria = serde_json::Map::new();
                        for option in q["options"].as_array().into_iter().flatten() {
                            let name = option["name"].as_str().unwrap_or("");
                            criteria.insert(
                                name.into(),
                                json!(option["description"].as_str().unwrap_or("")),
                            );
                            order.push(json!(name));
                        }
                        wire["criteria"] = Value::Object(criteria);
                    }
                    "score" => {
                        wire["criteria"] = Value::Array(
                            q["levels"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .map(|v| json!(v["name"].as_str().unwrap_or("")))
                                .collect(),
                        )
                    }
                    _ => {
                        wire["criteria"] = json!({"true":q["yesMeans"].as_str().unwrap_or(""),"false":q["noMeans"].as_str().unwrap_or("")})
                    }
                }
                ordering.push(Value::Array(order));
                questions.insert(name.into(), wire);
            }
            // Swift Date's default Codable epoch is 2001, not Unix milliseconds.
            let at = ((old["at"].as_f64().unwrap_or(0.0) + 978_307_200.0) * 1000.0) as i64;
            let excerpt = old["excerpt"].as_str().unwrap_or("");
            let title: String = excerpt.chars().take(60).collect();
            let title = if title.is_empty() {
                "Imported read"
            } else {
                &title
            };
            let model = old["raw"]["model"].as_str().unwrap_or("");
            let doc =
                json!({"id":id,"title":title,"model":model,"createdAt":at,"updatedAt":at,"runs":[{
                    "id":old_id,"at":at,"model":model,"port":old["port"],"excerpt":excerpt,
                    "chars":old["characters"],"state":"","stateMissing":true,"fileName":"",
                    "questions":questions,"questionOrder":ordering,"samples":"auto",
                    "response":old["raw"],"ms":old["elapsedMilliseconds"]
                }]})
                .to_string();
            tx.execute("INSERT OR IGNORE INTO read_history(id,title,model,runs,created_at,updated_at,doc) VALUES(?1,?2,?3,1,?4,?4,?5)", params![id,title,model,at,doc])?;
            tx.execute(
                "DELETE FROM read_runs WHERE scope=?1 AND id=?2",
                params![scope, old_id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_results_migrate_once_without_inventing_the_missing_input() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("reads.db");
        let db = Store::open(&path).unwrap();
        let id = Uuid::new_v4().to_string();
        let old = json!({"id":id,"at":10,"fingerprint":"a".repeat(64),"excerpt":"Earlier result","characters":1000,
            "questions":[{"questionID":"z","kind":"choice","instructions":"Pick","options":[{"name":"zebra","description":"Z"},{"name":"apple","description":"A"}]}],
            "raw":{"model":"diffusion","answers":{}},"port":11543,"elapsedMilliseconds":1.5});
        db.save_read_run("draft", &old).unwrap();
        let list = db.list_read_history().unwrap();
        assert_eq!(list.len(), 1);
        let doc: Value = serde_json::from_str(
            &db.get_read(list[0]["id"].as_str().unwrap())
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(doc["runs"][0]["stateMissing"], true);
        assert_eq!(doc["runs"][0]["state"], "");
        assert_eq!(doc["runs"][0]["at"], 978_307_210_000_i64);
        assert_eq!(
            doc["runs"][0]["questionOrder"],
            json!([["z", "zebra", "apple"]])
        );
        assert!(db.read_runs("draft").unwrap().is_empty());
        drop(db);
        let db = Store::open(&path).unwrap();
        assert_eq!(db.list_read_history().unwrap().len(), 1);
        db.delete_read(list[0]["id"].as_str().unwrap()).unwrap();
        assert!(
            db.list_read_history().unwrap().is_empty(),
            "deleted imports must not reappear"
        );
    }
}
