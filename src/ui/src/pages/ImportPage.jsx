import { useState } from 'react'
import { apiPost, importFile } from '../api'

export default function ImportPage() {
  const [file, setFile] = useState(null)
  const [redact, setRedact] = useState(true)
  const [message, setMessage] = useState('')

  async function handleImport(e) {
    e.preventDefault()
    if (!file) return
    try {
      const result = await importFile(file, redact)
      setMessage(JSON.stringify(result, null, 2))
    } catch (err) {
      setMessage(`Import failed: ${err.message}`)
    }
  }

  async function runPipeline() {
    await apiPost('/embed?redact_pii=' + (redact ? 'true' : 'false'))
    const result = await apiPost('/cluster')
    setMessage(`Pipeline finished: ${JSON.stringify(result)}`)
  }

  return (
    <section>
      <h2>Import Screen</h2>
      <form onSubmit={handleImport} className="stack">
        <input type="file" accept=".json" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <label>
          <input type="checkbox" checked={redact} onChange={(e) => setRedact(e.target.checked)} />
          Redact PII before analysis
        </label>
        <button type="submit">Import</button>
      </form>
      <button onClick={runPipeline}>Run Embedding + Clustering</button>
      {message && <pre>{message}</pre>}
    </section>
  )
}
