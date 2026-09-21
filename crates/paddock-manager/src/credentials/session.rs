//! Desktop-owned credential session. Only startup and an explicit Unlock
//! operation may consult Keychain; serving and catalog refresh are cache-only.
//! Each Store owns its vault, so isolated libraries/cores cannot share secrets.
use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use zeroize::Zeroizing;

const UNAVAILABLE: &str =
    "Cloud access is not ready. Open Manager > Cloud providers and unlock this account.";

#[derive(Default)]
pub(crate) struct Session {
    enabled: AtomicBool,
    authorization: Mutex<()>,
    keys: Mutex<HashMap<String, Zeroizing<String>>>,
}

impl Session {
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }
    fn applies(&self, reference: &str) -> bool {
        self.enabled.load(Ordering::Acquire) && reference.starts_with(super::PREFIX)
    }
    pub fn available(&self, reference: &str) -> bool {
        !self.applies(reference)
            || self
                .keys
                .lock()
                .is_ok_and(|keys| keys.contains_key(reference))
    }
    pub fn resolve(&self, reference: String) -> Result<String, String> {
        if !self.applies(&reference) {
            return super::resolve(reference);
        }
        self.keys
            .lock()
            .ok()
            .and_then(|keys| keys.get(&reference).map(|key| key.to_string()))
            .ok_or_else(|| UNAVAILABLE.into())
    }
    /// Blocking, single-flight authorization. Call outside the DB mutex and
    /// outside Tokio's I/O workers. Passive reads never wait on the OS dialog.
    pub fn authorize(&self, reference: &str) -> Result<(), String> {
        if !self.applies(reference) {
            return Ok(());
        }
        let _authorization = self.authorization.lock().map_err(|_| UNAVAILABLE)?;
        if self.available(reference) {
            return Ok(());
        }
        let key = super::resolve(reference.to_owned())?;
        self.keys
            .lock()
            .map_err(|_| UNAVAILABLE)?
            .insert(reference.to_owned(), Zeroizing::new(key));
        Ok(())
    }
    pub fn remember(&self, reference: &str, key: &str) {
        if self.applies(reference) {
            let _authorization = self.authorization.lock().expect("credential authorization");
            self.keys
                .lock()
                .expect("credential session")
                .insert(reference.into(), Zeroizing::new(key.into()));
        }
    }
    pub fn forget(&self, reference: &str) {
        let _authorization = self.authorization.lock().expect("credential authorization");
        self.keys
            .lock()
            .expect("credential session")
            .remove(reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serving_never_authorizes_and_cached_access_survives_without_keychain_reads() {
        let reference = crate::credentials::protect("synthetic-secret").unwrap();
        let session = Session::default();
        session.enable();
        assert!(!session.available(&reference));
        assert!(session.resolve(reference.clone()).is_err());
        session.authorize(&reference).unwrap();
        crate::credentials::retire(&reference); // Test vault gone: repeat retrieval would fail.
        for _ in 0..8 {
            assert_eq!(
                session.resolve(reference.clone()).unwrap(),
                "synthetic-secret"
            );
        }
        session.forget(&reference);
        assert!(session.resolve(reference).is_err());
    }
    #[test]
    fn failures_do_not_trigger_requests_and_sessions_are_isolated() {
        let a = Session::default();
        let b = Session::default();
        a.enable();
        b.enable();
        let reference = "paddock-keychain:v1:missing";
        assert!(a.authorize(reference).is_err());
        for _ in 0..8 {
            assert!(a.resolve(reference.into()).is_err());
        }
        a.remember(reference, "replacement");
        assert!(a.available(reference));
        assert!(!b.available(reference));
        // A different account's OS authorization must not stall ready accounts.
        let _held = a.authorization.lock().unwrap();
        assert!(a.available(reference));
        assert_eq!(a.resolve(reference.into()).unwrap(), "replacement");
    }
}
