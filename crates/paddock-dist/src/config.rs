//! Rank/world configuration for tensor-parallel serving.
//!
//! Phase 2 keeps this deliberately strict: TP=1 or TP=2, one GPU per node,
//! rank 0 is the coordinator and rank 1 is the worker. Broader topologies are
//! a non-goal for the first implementation (plan §Non-goals).
//!
//! Layering follows the runner convention: CLI flag > `PADDOCK_TP_*` env >
//! `paddock.toml` `[parallel]` table > defaults. Every env name read here is
//! registered in the runner's `ENV_SURFACE` (hardened builds seal
//! undocumented `PADDOCK_*` variables - see paddock-runner/src/config.rs).

use serde::{Deserialize, Serialize};

/// Errors from rank/world configuration parsing and validation.
#[derive(Debug, thiserror::Error)]
pub enum ParallelConfigError {
    #[error("{field} must be an integer, got {value:?}")]
    BadInt { field: &'static str, value: String },
    #[error("{field} must be 1..=65535, got {value}")]
    BadPort { field: &'static str, value: i64 },
    #[error("tp_size must be 1 or 2 (broader topologies are not supported), got {0}")]
    UnsupportedTpSize(usize),
    #[error("rank must be < tp_size (tp_size={tp}), got rank {rank}")]
    RankOutOfRange { tp: usize, rank: usize },
    #[error("rank 1 needs the coordinator's address: set master_addr (or PADDOCK_TP_MASTER_ADDR)")]
    EmptyMasterAddr,
    #[error(
        "rank 1 must not be started in serving mode; it is a worker (start it via the rank-0 coordinator, or set rank 0)"
    )]
    WorkerMustNotServe,
    #[error("[parallel] has no key {0:?}")]
    UnknownParallelKey(String),
}

/// Which process role this runner instance takes in the TP pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankRole {
    /// Rank 0: exposes the API, owns scheduling, coordinates the worker.
    Coordinator,
    /// Rank 1: worker. Loads its shard later; never serves requests.
    Worker,
}

/// Rank/world configuration as it arrives from toml/env/CLI - all fields
/// optional so "unset" stays distinguishable from "explicitly set".
///
/// A runner that never mentions TP keeps TP=1 semantics with zero behavior
/// change (regression gate: existing configs parse byte-identically).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelConfig {
    /// Tensor-parallel world size. 1 = the historical single-process path;
    /// 2 = one coordinator + one worker across two nodes.
    pub tp_size: Option<usize>,
    /// This process's rank: 0 = coordinator, 1 = worker. Must be < tp_size.
    pub rank: Option<usize>,
    /// Address of the rank-0 control-plane listener. Rank 0 binds it
    /// (default "0.0.0.0" - the worker is on the other node); rank 1 dials it.
    pub master_addr: Option<String>,
    /// TCP port of the rank-0 control-plane listener.
    pub master_port: Option<u16>,
}

impl ParallelConfig {
    /// The `[parallel]` toml table, if the config file has one.
    pub fn from_toml_value(v: &toml::Value) -> Result<Option<Self>, ParallelConfigError> {
        let Some(table) = v.as_table() else {
            return Ok(None);
        };
        let mut cfg = Self::default();
        for (key, val) in table {
            match key.as_str() {
                "tp_size" => cfg.tp_size = Some(as_usize("tp_size", val)?),
                "rank" => cfg.rank = Some(as_usize("rank", val)?),
                "master_addr" => match val.as_str() {
                    Some(s) => cfg.master_addr = Some(s.to_string()),
                    None => {
                        return Err(ParallelConfigError::BadInt {
                            field: "master_addr",
                            value: format!("{val}"),
                        });
                    }
                },
                "master_port" => cfg.master_port = Some(as_port("master_port", val)?),
                other => return Err(ParallelConfigError::UnknownParallelKey(other.to_string())),
            }
        }
        Ok(Some(cfg))
    }

    /// Overlay `PADDOCK_TP_*` environment variables (runner convention).
    pub fn merge_env(&mut self) -> Result<(), ParallelConfigError> {
        if let Some(v) = env_str("PADDOCK_TP_SIZE") {
            self.tp_size = Some(v.parse().map_err(|_| ParallelConfigError::BadInt {
                field: "PADDOCK_TP_SIZE",
                value: v.clone(),
            })?);
        }
        if let Some(v) = env_str("PADDOCK_TP_RANK") {
            self.rank = Some(v.parse().map_err(|_| ParallelConfigError::BadInt {
                field: "PADDOCK_TP_RANK",
                value: v.clone(),
            })?);
        }
        if let Some(v) = env_str("PADDOCK_TP_MASTER_ADDR") {
            self.master_addr = Some(v);
        }
        if let Some(v) = env_str("PADDOCK_TP_MASTER_PORT") {
            let parsed = v.parse::<u16>().map_err(|_| ParallelConfigError::BadInt {
                field: "PADDOCK_TP_MASTER_PORT",
                value: v.clone(),
            })?;
            self.master_port = Some(parsed);
        }
        Ok(())
    }

    /// Construct from environment values only (the runner wiring's parse of
    /// `PADDOCK_TP_*`); None where the variable is absent.
    pub fn from_worker_env(
        tp_size: Option<usize>,
        rank: Option<usize>,
        master_addr: Option<String>,
        master_port: Option<u16>,
    ) -> Self {
        Self {
            tp_size,
            rank,
            master_addr,
            master_port,
        }
    }

    /// Validate into a concrete role, or `None` for the historical
    /// single-process path. `serving_mode` is true when this process would
    /// bind the HTTP API - a rank-1 worker never does (the runner passes
    /// false only for its own spawned worker child, marked by
    /// [`WORKER_CHILD_ENV`]; a user-set `PADDOCK_TP_RANK=1` on an
    /// interactive run is refused, not silently demoted).
    pub fn resolved(&self, serving_mode: bool) -> Result<Option<Resolved>, ParallelConfigError> {
        match (self.tp_size, self.rank) {
            (None, None) => Ok(None),
            // A rank with no world size is a config error, not a silent TP=1.
            (None, Some(rank)) => Err(ParallelConfigError::RankOutOfRange { tp: 1, rank }),
            (Some(1), None | Some(0)) => Ok(None),
            (Some(1), Some(rank)) => Err(ParallelConfigError::RankOutOfRange { tp: 1, rank }),
            (Some(2), Some(rank)) if rank >= 2 => {
                Err(ParallelConfigError::RankOutOfRange { tp: 2, rank })
            }
            (Some(2), Some(rank)) => self.resolved_tp2(rank, serving_mode),
            // World size without a rank defaults to rank 0 (coordinator) -
            // "start the pair from this box". A rank-1 start is always
            // explicit: the spawned child's env, or a hand-set rank for a
            // manually/SSH-started worker. Both nodes defaulting to rank 0
            // is visible (both wait for a worker) and never half-serves.
            (Some(2), None) => self.resolved_tp2(0, serving_mode),
            (Some(tp), _) => Err(ParallelConfigError::UnsupportedTpSize(tp)),
        }
    }

    /// The TP=2 branch shared by the explicit-rank and default-rank arms.
    fn resolved_tp2(
        &self,
        rank: usize,
        serving_mode: bool,
    ) -> Result<Option<Resolved>, ParallelConfigError> {
        // A rank-1 worker never serves an API, no matter how it was started.
        if rank == 1 && serving_mode {
            return Err(ParallelConfigError::WorkerMustNotServe);
        }
        let master_addr = match self.master_addr.as_deref().map(str::trim) {
            Some("") | None => {
                if rank == 0 {
                    "0.0.0.0".to_string()
                } else {
                    return Err(ParallelConfigError::EmptyMasterAddr);
                }
            }
            Some(a) => a.to_string(),
        };
        let master_port = self.master_port.unwrap_or(DEFAULT_MASTER_PORT);
        Ok(Some(Resolved {
            tp_size: 2,
            rank,
            role: if rank == 0 {
                RankRole::Coordinator
            } else {
                RankRole::Worker
            },
            master_addr,
            master_port,
        }))
    }

    /// True when this process was spawned as a rank-1 worker child by a
    /// coordinator (the marker is set by [`Resolved::worker_env`], never by
    /// users).
    pub fn is_worker_child() -> bool {
        std::env::var(WORKER_CHILD_ENV).is_ok_and(|v| v == "1")
    }
}

/// Process was spawned by its rank-0 coordinator. Internal marker, part of
/// the sealed env surface so hardened builds keep it.
pub const WORKER_CHILD_ENV: &str = "PADDOCK_TP_WORKER_CHILD";

/// A validated, concrete rank configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub tp_size: usize,
    pub rank: usize,
    pub role: RankRole,
    /// Bind address for rank 0 ("0.0.0.0" default), dial target for rank 1.
    pub master_addr: String,
    pub master_port: u16,
}

impl Resolved {
    pub fn is_coordinator(&self) -> bool {
        self.role == RankRole::Coordinator
    }

    /// The exact env a coordinator sets when re-executing the rank-1 worker
    /// child. Registered in the runner's `ENV_SURFACE` - same file, edit
    /// together (see the hardened-seal note in paddock-runner/src/config.rs).
    pub fn worker_env(&self) -> Vec<(&'static str, String)> {
        let mut env = vec![
            ("PADDOCK_TP_SIZE", self.tp_size.to_string()),
            ("PADDOCK_TP_RANK", "1".to_string()),
            ("PADDOCK_TP_WORKER_CHILD", "1".to_string()),
            ("PADDOCK_TP_MASTER_PORT", self.master_port.to_string()),
        ];
        // The child dials the coordinator. "0.0.0.0"/"::" are bind
        // wildcards, not dial targets - translate them to loopback, which is
        // right for the local-spawn case where the coordinator is this very
        // machine. A two-node start sets a real master_addr explicitly.
        let dial = match self.master_addr.as_str() {
            "0.0.0.0" | "" => "127.0.0.1".to_string(),
            "::" => "::1".to_string(),
            a => a.to_string(),
        };
        env.push(("PADDOCK_TP_MASTER_ADDR", dial));
        env
    }
}

/// Default control-plane port. Distinct from the serving ports (11500/11540
/// family) so a same-box smoke test can run both roles side by side.
pub const DEFAULT_MASTER_PORT: u16 = 11560;

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn as_usize(field: &'static str, v: &toml::Value) -> Result<usize, ParallelConfigError> {
    v.as_integer()
        .map(|i| usize::try_from(i).unwrap_or(usize::MAX))
        .ok_or_else(|| ParallelConfigError::BadInt {
            field,
            value: format!("{v}"),
        })
}

fn as_port(field: &'static str, v: &toml::Value) -> Result<u16, ParallelConfigError> {
    let i = v.as_integer().ok_or_else(|| ParallelConfigError::BadInt {
        field,
        value: format!("{v}"),
    })?;
    if i <= 0 || i > u16::MAX as i64 {
        return Err(ParallelConfigError::BadPort { field, value: i });
    }
    Ok(i as u16)
}
