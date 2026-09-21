//! `prism.hadamard.*` - the rotated-basis contract PrismML's Bonsai GGUFs
//! carry in their metadata.
//!
//! Such a file stores its linear weights folded with a blockwise
//! Walsh-Hadamard rotation of the INPUT axis: the stored weight is
//! `W' = W S H`, so the runtime has to feed it `x' = H (s * x)` (per
//! `block_size`-wide strip of the activation row) instead of `x`. A table read
//! by row lookup (the token embedding) stores rotated rows and gets
//! `s * H(z)` applied to what comes out. Nothing in the tensor directory says
//! any of this - a runtime that ignores the keys loads fine and computes
//! noise, so the rule here is the same one PrismML's own llama.cpp fork
//! applies: every value we do not know how to honour is a load error that
//! names the key. That includes keys we have never seen under the prefix - a
//! newer writer's addition may change the math.
//!
//! This module only reads and validates. Which weights a model family is able
//! to route through a rotating site is that family's call (engine side).

use std::collections::HashSet;

use crate::gguf::{GgufFile, Value};

const PREFIX: &str = "prism.hadamard.";
const TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
const AXIS: &str = "input-last-dimension";
/// The one table the inverse-after-lookup form is defined for.
pub const TOKEN_EMBD: &str = "token_embd.weight";

/// Every key of contract version 1. Anything else under the prefix refuses.
const KNOWN_KEYS: [&str; 10] = [
    "version",
    "block_size",
    "transform",
    "axis",
    "sign_mode",
    "weight_names",
    "sign_widths",
    "sign_values",
    "inverse_weight_names",
    "gdn_v_grouped",
];

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum HadamardError {
    #[error("{PREFIX}{0} is missing")]
    Missing(&'static str),
    #[error("{PREFIX}{key} has the wrong value type (want {want})")]
    BadType {
        key: &'static str,
        want: &'static str,
    },
    #[error("{PREFIX}version {0} is not supported (this build reads version 1)")]
    Version(u64),
    #[error(
        "{PREFIX}{0} is not a key of contract version 1 - refusing rather than guess its meaning"
    )]
    UnknownKey(String),
    #[error("{PREFIX}block_size {0} is not a power of two")]
    BlockSize(u64),
    #[error("{PREFIX}transform \"{0}\" is not supported (want \"{TRANSFORM}\")")]
    Transform(String),
    #[error("{PREFIX}axis \"{0}\" is not supported (want \"{AXIS}\")")]
    Axis(String),
    #[error("{PREFIX}sign_mode \"{0}\" is not supported (want \"identity\" or \"explicit\")")]
    SignMode(String),
    #[error("{PREFIX}weight_names is empty")]
    NoWeights,
    #[error("{PREFIX}weight_names lists {0} twice")]
    DuplicateWeight(String),
    #[error("{PREFIX}sign_mode is explicit but sign_widths is empty")]
    NoSignWidths,
    #[error("{PREFIX}sign_widths entry {width} is invalid (block_size {block})")]
    SignWidth { width: i64, block: usize },
    #[error("{PREFIX}sign_widths lists width {0} twice")]
    DuplicateSignWidth(usize),
    #[error("{PREFIX}sign_values holds {have} values, the widths add up to {want}")]
    SignCount { have: usize, want: usize },
    #[error("{PREFIX}sign_values[{at}] is {value}, not +1 or -1")]
    SignValue { at: usize, value: i64 },
    #[error("{PREFIX}sign_widths / sign_values are set but sign_mode is identity")]
    SignsUnderIdentity,
    #[error(
        "{PREFIX}inverse_weight_names lists {0}: only {TOKEN_EMBD} has an inverse-after-lookup form"
    )]
    InverseTable(String),
    #[error("{PREFIX}: no sign vector for input width {0}")]
    NoSignVector(usize),
}

/// The validated contract of one file.
#[derive(Debug, Clone, PartialEq)]
pub struct HadamardSpec {
    /// Strip width of the transform (a power of two).
    pub block: usize,
    /// `Some` = sign_mode explicit: one +-1 vector per input width, in file
    /// order. `None` = identity (no sign flip).
    explicit_signs: Option<Vec<(usize, Vec<f32>)>>,
    /// Weights whose input must be rotated before the matmul.
    pub weights: HashSet<String>,
    /// The token embedding stores rotated rows (inverse after lookup).
    pub embd_inverse: bool,
    /// `ssm_out`'s input is permuted from tiled to grouped value-head order
    /// before the signs (qwen35 gated-delta-net layers).
    pub gdn_v_grouped: bool,
}

impl HadamardSpec {
    /// `Ok(None)` for an ordinary file. A file with ANY key under the prefix
    /// either validates completely or refuses.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Option<Self>, HadamardError> {
        let mut any = false;
        for key in gguf.metadata.keys() {
            if let Some(rest) = key.strip_prefix(PREFIX) {
                any = true;
                if !KNOWN_KEYS.contains(&rest) {
                    return Err(HadamardError::UnknownKey(rest.to_owned()));
                }
            }
        }
        if !any {
            return Ok(None);
        }
        let get = |k: &str| gguf.metadata.get(&format!("{PREFIX}{k}"));
        let uint = |k: &'static str| -> Result<u64, HadamardError> {
            get(k)
                .ok_or(HadamardError::Missing(k))?
                .as_u64()
                .ok_or(HadamardError::BadType {
                    key: k,
                    want: "an unsigned integer",
                })
        };
        let text = |k: &'static str| -> Result<&str, HadamardError> {
            get(k)
                .ok_or(HadamardError::Missing(k))?
                .as_str()
                .ok_or(HadamardError::BadType {
                    key: k,
                    want: "a string",
                })
        };
        let names = |k: &'static str| -> Result<Option<Vec<String>>, HadamardError> {
            let Some(v) = get(k) else { return Ok(None) };
            let Value::Array(items) = v else {
                return Err(HadamardError::BadType {
                    key: k,
                    want: "an array of strings",
                });
            };
            items
                .iter()
                .map(|s| {
                    s.as_str().map(str::to_owned).ok_or(HadamardError::BadType {
                        key: k,
                        want: "an array of strings",
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        };
        let ints = |k: &'static str| -> Result<Option<Vec<i64>>, HadamardError> {
            let Some(v) = get(k) else { return Ok(None) };
            let Value::Array(items) = v else {
                return Err(HadamardError::BadType {
                    key: k,
                    want: "an array of integers",
                });
            };
            items
                .iter()
                .map(|s| {
                    s.as_i64().ok_or(HadamardError::BadType {
                        key: k,
                        want: "an array of integers",
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        };

        let version = uint("version")?;
        if version != 1 {
            return Err(HadamardError::Version(version));
        }
        let block = uint("block_size")?;
        if block == 0 || !block.is_power_of_two() || block > u32::MAX as u64 {
            return Err(HadamardError::BlockSize(block));
        }
        let block = block as usize;
        let transform = text("transform")?;
        if transform != TRANSFORM {
            return Err(HadamardError::Transform(transform.to_owned()));
        }
        let axis = text("axis")?;
        if axis != AXIS {
            return Err(HadamardError::Axis(axis.to_owned()));
        }
        let sign_mode = text("sign_mode")?;

        let mut weights = HashSet::new();
        for name in names("weight_names")?.ok_or(HadamardError::Missing("weight_names"))? {
            if !weights.insert(name.clone()) {
                return Err(HadamardError::DuplicateWeight(name));
            }
        }
        if weights.is_empty() {
            return Err(HadamardError::NoWeights);
        }

        let widths = ints("sign_widths")?;
        let values = ints("sign_values")?;
        let explicit_signs = match sign_mode {
            "identity" => {
                // signs that would be silently dropped change the model function
                if widths.is_some_and(|w| !w.is_empty()) || values.is_some_and(|v| !v.is_empty()) {
                    return Err(HadamardError::SignsUnderIdentity);
                }
                None
            }
            "explicit" => {
                let widths = widths.ok_or(HadamardError::Missing("sign_widths"))?;
                let values = values.ok_or(HadamardError::Missing("sign_values"))?;
                if widths.is_empty() {
                    return Err(HadamardError::NoSignWidths);
                }
                let mut out: Vec<(usize, Vec<f32>)> = Vec::with_capacity(widths.len());
                let mut off = 0usize;
                for &width in &widths {
                    let w = usize::try_from(width).unwrap_or(0);
                    if w == 0 || !w.is_multiple_of(block) {
                        return Err(HadamardError::SignWidth { width, block });
                    }
                    if out.iter().any(|(have, _)| *have == w) {
                        return Err(HadamardError::DuplicateSignWidth(w));
                    }
                    let want = off + w;
                    let Some(run) = values.get(off..want) else {
                        return Err(HadamardError::SignCount {
                            have: values.len(),
                            want: widths.iter().map(|&x| x.max(0) as usize).sum(),
                        });
                    };
                    let mut vec = Vec::with_capacity(w);
                    for (i, &v) in run.iter().enumerate() {
                        if v != 1 && v != -1 {
                            return Err(HadamardError::SignValue {
                                at: off + i,
                                value: v,
                            });
                        }
                        vec.push(v as f32);
                    }
                    out.push((w, vec));
                    off = want;
                }
                if off != values.len() {
                    return Err(HadamardError::SignCount {
                        have: values.len(),
                        want: off,
                    });
                }
                Some(out)
            }
            other => return Err(HadamardError::SignMode(other.to_owned())),
        };

        let mut embd_inverse = false;
        for name in names("inverse_weight_names")?.unwrap_or_default() {
            // a second listing, or a table that is also a rotated matmul
            // weight, has no defined meaning
            if name != TOKEN_EMBD || embd_inverse || weights.contains(&name) {
                return Err(HadamardError::InverseTable(name));
            }
            embd_inverse = true;
        }

        let gdn_v_grouped = match get("gdn_v_grouped") {
            None => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                return Err(HadamardError::BadType {
                    key: "gdn_v_grouped",
                    want: "a bool",
                });
            }
        };

        Ok(Some(Self {
            block,
            explicit_signs,
            weights,
            embd_inverse,
            gdn_v_grouped,
        }))
    }

    /// The +-1 vector for one input width: the file's under explicit signs
    /// (a width it does not list is an error - never a silent identity), all
    /// ones under identity.
    pub fn sign_vector(&self, width: usize) -> Result<Vec<f32>, HadamardError> {
        match &self.explicit_signs {
            None => Ok(vec![1.0; width]),
            Some(sets) => sets
                .iter()
                .find(|(w, _)| *w == width)
                .map(|(_, v)| v.clone())
                .ok_or(HadamardError::NoSignVector(width)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn file(kv: Vec<(&str, Value)>) -> GgufFile {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            Value::Str("qwen35".into()),
        );
        for (k, v) in kv {
            metadata.insert(format!("{PREFIX}{k}"), v);
        }
        GgufFile {
            version: 3,
            alignment: 32,
            metadata,
            tensors: Vec::new(),
            data_offset: 0,
        }
    }

    fn strs(items: &[&str]) -> Value {
        Value::Array(items.iter().map(|s| Value::Str((*s).into())).collect())
    }

    fn i32s(items: &[i32]) -> Value {
        Value::Array(items.iter().map(|&v| Value::I32(v)).collect())
    }

    /// A well-formed version-1 contract over a 4-wide block, as key/values
    /// the refusal cases below knock one piece out of.
    fn good() -> Vec<(&'static str, Value)> {
        vec![
            ("version", Value::U32(1)),
            ("block_size", Value::U32(4)),
            ("transform", Value::Str(TRANSFORM.into())),
            ("axis", Value::Str(AXIS.into())),
            ("sign_mode", Value::Str("explicit".into())),
            (
                "weight_names",
                strs(&["output.weight", "blk.0.ffn_down.weight"]),
            ),
            ("sign_widths", i32s(&[4, 8])),
            (
                "sign_values",
                i32s(&[1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1]),
            ),
            ("inverse_weight_names", strs(&[TOKEN_EMBD])),
            ("gdn_v_grouped", Value::Bool(true)),
        ]
    }

    fn with(key: &str, value: Option<Value>) -> GgufFile {
        let mut kv = good();
        kv.retain(|(k, _)| *k != key);
        if let Some(v) = value {
            // leak is fine in a test: the key has to outlive the vec
            kv.push((Box::leak(key.to_owned().into_boxed_str()), v));
        }
        file(kv)
    }

    #[test]
    fn an_ordinary_file_has_no_contract() {
        assert_eq!(HadamardSpec::from_gguf(&file(vec![])), Ok(None));
    }

    #[test]
    fn a_well_formed_contract_reads_back() {
        let spec = HadamardSpec::from_gguf(&file(good())).unwrap().unwrap();
        assert_eq!(spec.block, 4);
        assert!(spec.embd_inverse && spec.gdn_v_grouped);
        assert!(spec.weights.contains("output.weight"));
        assert_eq!(spec.sign_vector(4).unwrap(), vec![1.0, -1.0, 1.0, 1.0]);
        assert_eq!(spec.sign_vector(8).unwrap()[..3], [-1.0, -1.0, 1.0]);
        // a width the file does not list is never a silent identity
        assert_eq!(spec.sign_vector(12), Err(HadamardError::NoSignVector(12)));
    }

    #[test]
    fn identity_signs_are_all_ones_and_take_no_vectors() {
        let mut kv = good();
        kv.retain(|(k, _)| !matches!(*k, "sign_mode" | "sign_widths" | "sign_values"));
        kv.push(("sign_mode", Value::Str("identity".into())));
        let spec = HadamardSpec::from_gguf(&file(kv.clone())).unwrap().unwrap();
        assert_eq!(spec.sign_vector(8).unwrap(), vec![1.0; 8]);
        // sign vectors under identity would be dropped on the floor
        kv.push(("sign_widths", i32s(&[4])));
        kv.push(("sign_values", i32s(&[1, 1, 1, 1])));
        assert_eq!(
            HadamardSpec::from_gguf(&file(kv)),
            Err(HadamardError::SignsUnderIdentity)
        );
    }

    #[test]
    fn every_unknown_value_refuses_and_names_its_key() {
        use HadamardError as E;
        let cases: Vec<(GgufFile, E)> = vec![
            (with("version", Some(Value::U32(2))), E::Version(2)),
            (with("version", None), E::Missing("version")),
            (
                with("block_size", Some(Value::U32(1000))),
                E::BlockSize(1000),
            ),
            (with("block_size", Some(Value::U32(0))), E::BlockSize(0)),
            (
                with("transform", Some(Value::Str("dct".into()))),
                E::Transform("dct".into()),
            ),
            (
                with("axis", Some(Value::Str("output".into()))),
                E::Axis("output".into()),
            ),
            (
                with("sign_mode", Some(Value::Str("seeded".into()))),
                E::SignMode("seeded".into()),
            ),
            (with("weight_names", Some(strs(&[]))), E::NoWeights),
            (with("weight_names", None), E::Missing("weight_names")),
            (
                with(
                    "weight_names",
                    Some(strs(&["output.weight", "output.weight"])),
                ),
                E::DuplicateWeight("output.weight".into()),
            ),
            (with("sign_widths", Some(i32s(&[]))), E::NoSignWidths),
            (
                with("sign_widths", Some(i32s(&[4, 6]))),
                E::SignWidth { width: 6, block: 4 },
            ),
            (
                with("sign_widths", Some(i32s(&[4, 4]))),
                E::DuplicateSignWidth(4),
            ),
            (
                with("sign_widths", Some(i32s(&[4]))),
                E::SignCount { have: 12, want: 4 },
            ),
            (
                with("sign_widths", Some(i32s(&[4, 16]))),
                E::SignCount { have: 12, want: 20 },
            ),
            (
                with(
                    "sign_values",
                    Some(i32s(&[1, -1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])),
                ),
                E::SignValue { at: 3, value: 0 },
            ),
            (
                with("inverse_weight_names", Some(strs(&["output.weight"]))),
                E::InverseTable("output.weight".into()),
            ),
            (
                with("gdn_v_grouped", Some(Value::U32(1))),
                E::BadType {
                    key: "gdn_v_grouped",
                    want: "a bool",
                },
            ),
            (
                with("rotation_seed", Some(Value::U32(7))),
                E::UnknownKey("rotation_seed".into()),
            ),
        ];
        for (f, want) in cases {
            assert_eq!(HadamardSpec::from_gguf(&f), Err(want));
        }
    }

    #[test]
    fn a_stray_key_without_a_version_still_refuses() {
        // half a contract is not an ordinary file
        let f = file(vec![("block_size", Value::U32(1024))]);
        assert_eq!(
            HadamardSpec::from_gguf(&f),
            Err(HadamardError::Missing("version"))
        );
    }
}
