import assert from 'node:assert/strict'
import { test } from 'node:test'
import { runnerConfigKeys } from './config-fields.mjs'

test('runtime-only fields are not editable settings', () => {
  const keys = runnerConfigKeys(`pub struct Config {
    #[serde(skip)]
    pub loaded_file: Option<toml::Value>,
    pub host: IpAddr,
    #[serde(default)]
    pub port: u16,
    #[serde(
        default, skip_deserializing
    )]
    /// Runtime metadata, not input.
    pub runtime: String,
    #[serde(skip_serializing)]
    pub input_only: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optional: Option<String>,
}
pub struct Unrelated {
    pub ignored: bool,
}`)
  assert.deepEqual([...keys], ['host', 'port', 'input_only', 'optional'])
})

test('missing or incomplete Config fails closed', () => {
  assert.equal(runnerConfigKeys('pub struct Other {}').size, 0)
  assert.equal(runnerConfigKeys('pub struct Config {\n    pub port: u16,').size, 0)
})
