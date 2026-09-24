use super::*;

impl Store {
    pub fn save_native_benchmark(&self, report: &Value) -> Result<(), StoreError> {
        let id = report["id"]
            .as_str()
            .ok_or_else(|| StoreError::Bad("Missing benchmark ID.".into()))?;
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO native_benchmarks (id,created_at,report) VALUES (?1,?2,?3)",
            params![id, now_ms(), report.to_string()],
        )?;
        tx.execute("DELETE FROM native_benchmarks WHERE id NOT IN (SELECT id FROM native_benchmarks ORDER BY created_at DESC,id DESC LIMIT 50)", [])?;
        tx.commit()?;
        Ok(())
    }
    pub fn native_benchmarks(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT report FROM native_benchmarks ORDER BY created_at DESC,id DESC LIMIT 50",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.map(|raw| {
            let mut report: Value =
                serde_json::from_str(&raw?).map_err(|e| StoreError::Bad(e.to_string()))?;
            // List results without repeating thousands of event timings and
            // device snapshots. The explicit export retains the complete data.
            if let Some(object) = report.as_object_mut() {
                object.remove("hardware_before");
                object.remove("hardware_after");
            }
            if let Some(samples) = report["samples"].as_array_mut() {
                for sample in samples {
                    if let Some(object) = sample.as_object_mut() {
                        object.remove("event_gap_ms");
                    }
                }
            }
            Ok(report)
        })
        .collect()
    }

    pub fn native_benchmark(&self, id: &str) -> Result<Option<Value>, StoreError> {
        let raw: Option<String> = self
            .lock()
            .query_row(
                "SELECT report FROM native_benchmarks WHERE id=?1",
                params![id],
                |r| r.get(0),
            )
            .optional()?;
        raw.map(|raw| serde_json::from_str(&raw).map_err(|e| StoreError::Bad(e.to_string())))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_history_is_compact_but_export_preserves_full_measurements() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(&root.path().join("history.db")).unwrap();
        for index in 0..60 {
            store
                .save_native_benchmark(&json!({
                    "id":format!("report-{index:03}"),"model":"fixture",
                    "samples":[{"output_tokens":128,"event_gap_ms":[1.,2.,3.]}],
                    "hardware_before":{"name":"fixture GPU"},"hardware_after":{}
                }))
                .unwrap();
        }
        let history = store.native_benchmarks().unwrap();
        assert_eq!(history.len(), 50);
        for row in &history {
            assert!(row.get("hardware_before").is_none());
            assert!(row["samples"][0].get("event_gap_ms").is_none());
            assert_eq!(row["samples"][0]["output_tokens"], 128);
        }
        let full = store
            .native_benchmark(history[0]["id"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(full["samples"][0]["event_gap_ms"], json!([1., 2., 3.]));
        assert_eq!(full["hardware_before"]["name"], "fixture GPU");
        assert!(store.save_native_benchmark(&full).is_err());
    }
}
