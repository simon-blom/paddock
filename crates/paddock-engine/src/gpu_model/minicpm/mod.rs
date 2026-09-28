//! MiniCPM5-2B GPU model namespace.
//!
//! The single-GPU MiniCPM5-2B lane rides the granite family (plain-llama
//! graph with identity multipliers); this namespace holds only the TP
//! serving shim that proves the generic `tp::serve` + `tp::conventional`
//! stack with a second model.
pub mod tp_shim;
