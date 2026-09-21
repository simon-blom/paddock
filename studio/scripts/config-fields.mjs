/** The runner's flat Config struct, excluding fields Serde never reads.+ * Runtime bookkeeping such as loaded_file is not a TOML setting. */
export function runnerConfigKeys(source) {
  const start = source.indexOf('pub struct Config {')
  if (start < 0) return new Set()
  const end = source.indexOf('\n}', start)
  if (end < 0) return new Set()
  const body = source.slice(start, end)
  const keys = new Set()
  let attrs = ''
  for (const line of body.split('\n')) {
    const trimmed = line.trim()
    if (trimmed.startsWith('#[') || (attrs && !trimmed.startsWith('pub '))) {
      attrs += `\n${trimmed}`
    }
    const field = /^ {4}pub ([a-z0-9_]+):/.exec(line)
    if (!field) continue
    const skipped = /#\[serde\([^)]*\b(?:skip|skip_deserializing)\b[^)]*\)\]/s.test(attrs)
    if (!skipped) keys.add(field[1])
    attrs = ''
  }
  return keys
}
