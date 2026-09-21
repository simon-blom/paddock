// A finite grammar set, shared by Studio and the isolated lab. Grammars load
// on demand inside the syntax worker, not forty grammars on the UI thread.
export const MD_LANGS = [
  'text', 'bash', 'shell', 'powershell', 'json', 'yaml', 'toml', 'ini',
  'python', 'rust', 'c', 'cpp', 'go', 'javascript', 'typescript', 'tsx', 'jsx',
  'vue', 'html', 'css', 'scss', 'sql', 'markdown', 'diff', 'dockerfile', 'make',
  'xml', 'java', 'csharp', 'kotlin', 'swift', 'ruby', 'php', 'lua', 'r',
]
