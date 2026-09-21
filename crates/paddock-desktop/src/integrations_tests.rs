use super::*;

#[tokio::test]
async fn typed_async_receipts_are_bounded_and_never_return_credentials() {
    let state = Arc::new(AppState::for_tests());
    let sessions = Sessions::default();
    let command=serde_json::from_value(json!({"kind":"run","operation":{"kind":"save","draft":{"label":"fixture","url":"https://example.invalid/mcp","headers":{}}}})).unwrap();
    let receipt = sessions.execute(state.clone(), command, None).unwrap();
    let id = receipt["job"]["id"].as_str().unwrap();
    for _ in 0..100 {
        let value = sessions
            .execute(state.clone(), Command::Poll { id: id.into() }, None)
            .unwrap();
        if value["job"]["status"] != "running" {
            assert_eq!(value["job"]["status"], "succeeded", "{value}");
            assert!(!value.to_string().contains("headers"));
            assert!(!sessions.saving());
            assert_eq!(state.db.native_connectors().unwrap().len(), 1);
            sessions
                .execute(state.clone(), Command::Cancel { id: id.into() }, None)
                .unwrap();
            assert!(
                sessions
                    .execute(state, Command::Poll { id: id.into() }, None)
                    .is_err()
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("native operation did not settle");
}

#[test]
fn arbitrary_routes_and_callback_origins_are_not_commands() {
    for value in [
        json!({"kind":"get","url":"https://example.com"}),
        json!({"kind":"run","operation":{"kind":"list","headers":{}}}),
        json!({"kind":"run","operation":{"kind":"sign_in","id":"x","revision":1,"origin":"https://evil.invalid"}}),
    ] {
        assert!(serde_json::from_value::<Command>(value).is_err());
    }
}
