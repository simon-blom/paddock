//! Newly saved native credentials live in Keychain, never in a presentation
//! snapshot or SQLite. Unique accounts make replacement rollback safe: publish
//! the new reference atomically, then retire the old item. Existing web-created
//! plaintext records remain readable; this is not a silent library migration.
const PREFIX: &str = "paddock-keychain:v1:";
mod session;
pub(crate) use session::Session;
#[cfg(all(target_os = "macos", not(test)))]
const SERVICE: &str = "io.truespar.paddock.cloud";
#[cfg(test)]
pub(crate) static TEST_VAULT: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, String>>,
> = std::sync::LazyLock::new(Default::default);

#[cfg(all(target_os = "macos", not(test)))]
fn options(account: &str) -> security_framework::passwords::PasswordOptions {
    let mut options =
        security_framework::passwords::PasswordOptions::new_generic_password(SERVICE, account);
    // Provider keys are device-local; no implicit iCloud synchronization.
    options.set_access_synchronized(Some(false));
    options
}

pub(crate) fn protect(secret: &str) -> Result<String, String> {
    if secret.is_empty() {
        return Ok(String::new());
    }
    #[cfg(all(target_os = "macos", not(test)))]
    {
        let account = uuid::Uuid::new_v4().to_string();
        security_framework::passwords::set_generic_password_options(
            secret.as_bytes(),
            options(&account),
        )
        .map_err(|_| "Keychain could not save this credential. The connection was not changed.")?;
        Ok(format!("{PREFIX}{account}"))
    }
    #[cfg(test)]
    {
        let reference = format!("{PREFIX}{}", uuid::Uuid::new_v4());
        TEST_VAULT
            .lock()
            .unwrap()
            .insert(reference.clone(), secret.into());
        Ok(reference)
    }
    #[cfg(all(not(target_os = "macos"), not(test)))]
    {
        Ok(secret.into())
    }
}

pub(crate) fn resolve(stored: String) -> Result<String, String> {
    let Some(account) = stored.strip_prefix(PREFIX) else {
        return Ok(stored);
    };
    #[cfg(all(target_os = "macos", not(test)))]
    {
        let bytes = security_framework::passwords::generic_password(options(account))
            .map_err(|_| "The connection credential is unavailable in Keychain. Unlock Keychain or replace the key.")?;
        String::from_utf8(bytes).map_err(|_| "The connection credential is invalid.".into())
    }
    #[cfg(test)]
    {
        let _ = account;
        TEST_VAULT
            .lock()
            .unwrap()
            .get(&stored)
            .cloned()
            .ok_or("Synthetic Keychain item unavailable.".into())
    }
    #[cfg(all(not(target_os = "macos"), not(test)))]
    {
        let _ = account;
        Err("This credential requires the original macOS Keychain.".into())
    }
}

pub(crate) fn retire(stored: &str) {
    if let Some(account) = stored.strip_prefix(PREFIX) {
        #[cfg(all(target_os = "macos", not(test)))]
        if security_framework::passwords::delete_generic_password_options(options(account)).is_err()
        {
            // No secret/reference in logs. A failed cleanup leaves an unused
            // encrypted item, never a broken live connection.
            tracing::warn!("Unused cloud credential could not be removed from Keychain");
        }
        #[cfg(test)]
        {
            TEST_VAULT.lock().unwrap().remove(stored);
        }
        #[cfg(any(not(target_os = "macos"), test))]
        let _ = account;
    }
}
