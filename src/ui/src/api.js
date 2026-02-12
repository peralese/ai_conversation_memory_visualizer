const API_BASE = 'http://127.0.0.1:8000'

export async function apiGet(path) {
  const resp = await fetch(`${API_BASE}${path}`)
  if (!resp.ok) throw new Error(await resp.text())
  return resp.json()
}

export async function apiPost(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  })
  if (!resp.ok) throw new Error(await resp.text())
  return resp.json()
}

export async function importFile(file, redactPii) {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch(`${API_BASE}/import?redact_pii=${redactPii ? 'true' : 'false'}`, {
    method: 'POST',
    body: form
  })
  if (!resp.ok) throw new Error(await resp.text())
  return resp.json()
}
