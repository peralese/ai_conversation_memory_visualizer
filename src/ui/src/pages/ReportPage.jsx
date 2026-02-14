import { useEffect, useState } from 'react'
import { apiGet, apiGetText } from '../api'

export default function ReportPage() {
  const [jsonReport, setJsonReport] = useState(null)
  const [mdReport, setMdReport] = useState('')

  useEffect(() => {
    apiGet('/reports/cognitive_summary?format=json').then(setJsonReport)
    apiGetText('/reports/cognitive_summary?format=md').then(setMdReport)
  }, [])

  function download(content, filename, type) {
    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section>
      <h2>Cognitive Report</h2>
      <div className="row controls">
        <button
          type="button"
          onClick={() => download(mdReport, 'cognitive_summary.md', 'text/markdown')}
          disabled={!mdReport}
        >
          Download Markdown
        </button>
        <button
          type="button"
          onClick={() => download(JSON.stringify(jsonReport, null, 2), 'cognitive_summary.json', 'application/json')}
          disabled={!jsonReport}
        >
          Download JSON
        </button>
        <button
          type="button"
          onClick={() => navigator.clipboard?.writeText(mdReport || '')}
          disabled={!mdReport}
        >
          Copy Markdown
        </button>
      </div>
      <pre style={{ whiteSpace: 'pre-wrap', overflowX: 'auto' }}>{mdReport || 'Loading report...'}</pre>
    </section>
  )
}
