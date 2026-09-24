//! Shared persisted runner policy and native/web administration contract.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Snapshot {
    pub phase: Phase,
    pub active_leases: usize,
    pub waiting_requests: usize,
    pub loads: u64,
    pub unloads: u64,
    pub load_failures: u64,
    pub last_load_ms: Option<u64>,
    pub last_error: Option<String>,
    pub policy: Config,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoadPolicy {
    #[default]
    AtStartup,
    OnDemand,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub load: LoadPolicy,
    pub unload_after_idle_seconds: Option<u64>,
    pub load_timeout_seconds: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            load: LoadPolicy::AtStartup,
            unload_after_idle_seconds: None,
            load_timeout_seconds: 120,
        }
    }
}
impl Config {
    pub fn enabled(&self) -> bool {
        self.load == LoadPolicy::OnDemand || self.unload_after_idle_seconds.is_some()
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=3600).contains(&self.load_timeout_seconds) {
            return Err("residency.load_timeout_seconds must be 1..3600".into());
        }
        if self.unload_after_idle_seconds.is_some_and(|s| s > 604800) {
            return Err("residency.unload_after_idle_seconds must be 0..604800".into());
        }
        Ok(())
    }
}
