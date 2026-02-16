import { useEffect, useState } from 'react'
import { apiGet } from '../api'

export default function ConversationClustersPage() {
  const [rows, setRows] = useState([])
  const [selected, setSelected] = useState(null)
  const [detail, setDetail] = useState(null)
  const [useSemanticLabels, setUseSemanticLabels] = useState(true)
  const [showLegacyLabels, setShowLegacyLabels] = useState(false)

  useEffect(() => {
    const qs = new URLSearchParams({
      use_semantic_labels: String(useSemanticLabels),
      show_legacy_labels: String(showLegacyLabels)
    })
    apiGet(`/api/conv_clusters?${qs.toString()}`).then(res => {
      setRows(res || [])
      if (res && res.length > 0) setSelected(String(res[0].conv_cluster_id))
    })
  }, [useSemanticLabels, showLegacyLabels])

  useEffect(() => {
    if (!selected) return
    const qs = new URLSearchParams({
      use_semantic_labels: String(useSemanticLabels),
      show_legacy_labels: String(showLegacyLabels)
    })
    apiGet(`/api/conv_clusters/${encodeURIComponent(selected)}?${qs.toString()}`).then(setDetail)
  }, [selected, useSemanticLabels, showLegacyLabels])

  return (
    <section>
      <h2>Conversation Clusters</h2>
      <div className="row controls">
        <label>
          <input
            type="checkbox"
            checked={useSemanticLabels}
            onChange={e => setUseSemanticLabels(e.target.checked)}
          />
          Use semantic labels
        </label>
        <label>
          <input
            type="checkbox"
            checked={showLegacyLabels}
            onChange={e => setShowLegacyLabels(e.target.checked)}
          />
          Show legacy labels
        </label>
      </div>

      <table>
        <thead>
          <tr>
            <th>Cluster</th>
            <th>Label</th>
            <th>Conversations</th>
            <th>Messages</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.conv_cluster_id} onClick={() => setSelected(String(r.conv_cluster_id))} style={{ cursor: 'pointer' }}>
              <td>#{r.conv_cluster_id}</td>
              <td>{r.label_display}</td>
              <td>{r.conversation_count}</td>
              <td>{r.message_count}</td>
              <td>{r.semantic?.label_source || 'N/A'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {detail && (
        <div style={{ marginTop: '1rem' }}>
          <h3>#{detail.conv_cluster_id} {detail.label_display}</h3>
          <p><strong>Legacy:</strong> {detail.legacy_label}</p>
          <p><strong>Summary:</strong> {detail.semantic?.summary || 'N/A'}</p>
          <p><strong>Tags:</strong> {(detail.semantic?.tags || []).join(', ') || 'N/A'}</p>
          <h4>Representative Conversations</h4>
          <ul>
            {(detail.evidence_packet?.representative_conversations || []).map(c => (
              <li key={c.conversation_id}>
                <code>{c.conversation_id}</code> [{c.started_at} - {c.ended_at}] {c.rollup_preview}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
