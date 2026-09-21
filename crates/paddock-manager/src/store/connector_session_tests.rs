use super::*;

fn references(db: &Store, id: &str) -> (String, String) {
    db.lock()
        .query_row(
            "SELECT headers,oauth FROM connectors WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
}
fn create(db: &Store) -> String {
    db.save_native_connector(&json!({"label":"fixture", "url":"https://example.invalid/mcp", "headers":{"Authorization":"synthetic-header"}})).unwrap()
}

#[test]
fn startup_warms_headers_and_oauth_and_requests_never_read_the_vault() {
    let db = Store::open(&PathBuf::from(":memory:")).unwrap();
    let id = create(&db);
    db.set_connector_oauth_checked(
        &id,
        1,
        r#"{"access_token":"synthetic-access","refresh_token":"synthetic-refresh"}"#,
    )
    .unwrap();
    db.prepare_connector_credentials().unwrap();
    let (headers, oauth) = references(&db, &id);
    crate::credentials::retire(&headers);
    crate::credentials::retire(&oauth);
    for _ in 0..8 {
        let row = db.get_connector(&id).unwrap().unwrap();
        assert_eq!(row["headers"]["Authorization"], "synthetic-header");
        assert_eq!(row["oauth"]["refresh_token"], "synthetic-refresh");
        assert_eq!(db.list_connectors().unwrap().len(), 1);
    }
    let public = db.native_connectors().unwrap();
    assert_eq!(public[0]["credentialReady"], true);
    assert!(!public[0].to_string().contains("synthetic-"));
    assert!(!public[0].to_string().contains("paddock-keychain"));
}

#[test]
fn denied_startup_requires_explicit_unlock_not_a_passive_read() {
    let db = Store::open(&PathBuf::from(":memory:")).unwrap();
    let id = create(&db);
    let (reference, _) = references(&db, &id);
    let key = crate::credentials::TEST_VAULT
        .lock()
        .unwrap()
        .remove(&reference)
        .unwrap();
    db.prepare_connector_credentials().unwrap();
    assert_eq!(db.native_connectors().unwrap()[0]["credentialReady"], false);
    crate::credentials::TEST_VAULT
        .lock()
        .unwrap()
        .insert(reference.clone(), key);
    for _ in 0..4 {
        assert!(db.get_connector(&id).is_err());
    }
    assert!(db.unlock_connector(&id, 0).is_err());
    assert!(!db.connector_credentials.available(&reference));
    db.unlock_connector(&id, 1).unwrap();
    assert!(db.get_connector(&id).is_ok());
    db.delete_connector(&id).unwrap();
    assert!(!db.connector_credentials.available(&reference));
    assert!(db.unlock_connector(&id, 1).is_err());
}

#[test]
fn refresh_rotation_disconnect_and_retarget_evict_old_credentials() {
    let db = Store::open(&PathBuf::from(":memory:")).unwrap();
    db.prepare_connector_credentials().unwrap();
    let id = create(&db); // No read-back needed for a newly entered header.
    assert_eq!(db.native_connectors().unwrap()[0]["credentialReady"], true);
    db.set_connector_oauth_checked(&id, 1, r#"{"access_token":"first"}"#)
        .unwrap();
    let (_, first) = references(&db, &id);
    db.set_connector_oauth_checked(&id, 2, r#"{"access_token":"rotated"}"#)
        .unwrap();
    let (_, rotated) = references(&db, &id);
    assert!(!db.connector_credentials.available(&first));
    crate::credentials::retire(&rotated);
    assert_eq!(
        db.get_connector(&id).unwrap().unwrap()["oauth"]["access_token"],
        "rotated"
    );
    assert!(
        db.set_connector_oauth_checked(&id, 2, r#"{"access_token":"stale"}"#)
            .is_err()
    );
    assert!(db.connector_credentials.available(&rotated));
    db.set_connector_oauth_checked(&id, 3, "").unwrap();
    assert!(!db.connector_credentials.available(&rotated));
    assert!(db.get_connector(&id).unwrap().unwrap()["oauth"].is_null());
    db.set_connector_oauth_checked(&id, 4, r#"{"access_token":"reconnected"}"#)
        .unwrap();
    let (old_header, old_oauth) = references(&db, &id);
    db.save_native_connector(&json!({"id":id,"revision":5,"label":"fixture","url":"https://other.invalid/mcp","headers":{"Authorization":"replacement"}})).unwrap();
    assert!(!db.connector_credentials.available(&old_header));
    assert!(!db.connector_credentials.available(&old_oauth));
    let row = db.get_connector(&id).unwrap().unwrap();
    assert!(row["oauth"].is_null());
    assert_eq!(row["headers"]["Authorization"], "replacement");
    let (final_header, _) = references(&db, &id);
    db.delete_connector(&id).unwrap();
    assert!(!db.connector_credentials.available(&final_header));
}
