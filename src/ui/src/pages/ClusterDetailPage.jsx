import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useParams } from 'react-router-dom'
import { apiGet } from '../api'

export default function ClusterDetailPage() {
  const { clusterId } = useParams()
  const [detail, setDetail] = useState(null)
  const [subclusters, setSubclusters] = useState([])
  const [showSubclusters, setShowSubclusters] = useState(false)
  const [activeSubclusterId, setActiveSubclusterId] = useState(null)
  const [excludeDomainStopwords, setExcludeDomainStopwords] = useState(true)
  const [showLabelDebug, setShowLabelDebug] = useState(false)
  const [useSemanticLabels, setUseSemanticLabels] = useState(true)
  const [showLegacyLabels, setShowLegacyLabels] = useState(false)

  useEffect(() => {
    const qs = new URLSearchParams({
      exclude_domain_stopwords: String(excludeDomainStopwords),
      use_semantic_labels: String(useSemanticLabels),
      show_legacy_labels: String(showLegacyLabels)
    })
    apiGet(`/clusters/${clusterId}?${qs.toString()}`).then(setDetail)
    apiGet(`/clusters/${clusterId}/subclusters?${qs.toString()}`).then(res => setSubclusters(res.subclusters || []))
  }, [clusterId, excludeDomainStopwords, useSemanticLabels, showLegacyLabels])

  if (!detail) return <div>Loading...</div>
  return (
    <section>
      <h2>Cluster Detail</h2>
      <label>
        <input
          type="checkbox"
          checked={excludeDomainStopwords}
          onChange={e => setExcludeDomainStopwords(e.target.checked)}
        />
        Exclude domain stopwords from labels
      </label>
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
      <p><strong>Cluster:</strong> #{detail.cluster_id}</p>
      <p>
        <strong>Label:</strong> {detail.label}{' '}
        {detail.label_low_signal && <span title={detail.label_warning || 'Label may be low-signal'}>⚠</span>}
      </p>
      <p><strong>Legacy label:</strong> {detail.legacy_label || 'N/A'}</p>
      <p><strong>Semantic subtitle:</strong> {detail.semantic?.subtitle || 'N/A'}</p>
      <p><strong>Semantic summary:</strong> {detail.semantic?.summary || 'N/A'}</p>
      <p><strong>Size:</strong> {detail.message_count} messages</p>
      <p><strong>Conversations:</strong> {detail.conversations_count}</p>
      <p><strong>% of dataset:</strong> {detail.dataset_percentage}%</p>
      <p><strong>Avg message length:</strong> {detail.average_message_length} characters</p>
      <p><strong>Top keywords:</strong> {detail.top_keywords?.join(', ') || 'N/A'}</p>
      <p>
        <button type="button" onClick={() => setShowLabelDebug(v => !v)}>
          {showLabelDebug ? 'Hide' : 'Show'} Label Debug
        </button>
      </p>
      {showLabelDebug && (
        <div>
          <p><strong>Raw top tokens:</strong> {(detail.label_debug?.raw_top_tokens || []).join(', ') || 'N/A'}</p>
          <p><strong>Final label tokens:</strong> {(detail.label_debug?.final_label_tokens || []).join(', ') || 'N/A'}</p>
          <h4>Removed by rule</h4>
          <ul>
            {Object.entries(detail.label_debug?.removed_by_rule || {}).map(([rule, tokens]) => (
              <li key={rule}>
                <strong>{rule}:</strong> {Array.isArray(tokens) ? tokens.join(', ') : 'N/A'}
              </li>
            ))}
          </ul>
        </div>
      )}
      <p><strong>First seen:</strong> {detail.first_seen}</p>
      <p><strong>Last seen:</strong> {detail.last_seen}</p>
      <p>
        <strong>Half-life:</strong>{' '}
        {detail.half_life?.half_life_weeks == null ? 'insufficient data' : `${detail.half_life.half_life_weeks} weeks`}
      </p>
      <h3>Source breakdown</h3>
      <ul>
        {Object.entries(detail.source_breakdown?.counts || {}).map(([source, count]) => (
          <li key={source}>
            {source}: {count} ({detail.source_breakdown?.percents?.[source] ?? 0}%)
          </li>
        ))}
      </ul>
      <h3>Sample messages</h3>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Role</th>
            <th>Source</th>
            <th>Conversation</th>
            <th>Snippet</th>
          </tr>
        </thead>
        <tbody>
          {(detail.sample_messages || []).map(e => (
            <tr key={e.message_id}>
              <td><code>{e.timestamp}</code></td>
              <td>{e.role}</td>
              <td>{e.source}</td>
              <td>
                <Link to={`/conversations?q=${encodeURIComponent(e.conversation_title || e.conversation_id)}`}>
                  {e.conversation_title || e.conversation_id}
                </Link>
              </td>
              <td>{e.snippet}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {Array.isArray(subclusters) && subclusters.length > 0 && (
        <>
          <h3>
            Subclusters{' '}
            <button type="button" onClick={() => setShowSubclusters(v => !v)}>
              {showSubclusters ? 'Hide' : 'Show'}
            </button>
          </h3>
          {showSubclusters && (
            <div className="row" style={{ alignItems: 'flex-start', gap: '1rem' }}>
              <ul style={{ minWidth: 320 }}>
                {subclusters.map(sc => (
                  <li key={sc.id}>
                    <button type="button" onClick={() => setActiveSubclusterId(sc.id)}>
                      {sc.label} ({sc.message_count} messages)
                    </button>
                  </li>
                ))}
              </ul>
              <div>
                {(() => {
                  const sub = subclusters.find(s => s.id === activeSubclusterId) || subclusters[0]
                  if (!sub) return null
                  return (
                    <>
                      <h4>{sub.label}</h4>
                      <p><strong>Messages:</strong> {sub.message_count}</p>
                      <p><strong>Conversations:</strong> {sub.conversations_count}</p>
                      <p><strong>% of parent cluster:</strong> {sub.dataset_percentage}%</p>
                      <p><strong>First seen:</strong> {sub.first_seen}</p>
                      <p><strong>Last seen:</strong> {sub.last_seen}</p>
                      <ul>
                        {(sub.sample_messages || []).map(m => (
                          <li key={m.message_id}>
                            <code>{m.timestamp}</code> [{m.source}/{m.role}] {m.snippet}
                          </li>
                        ))}
                      </ul>
                    </>
                  )
                })()}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  )
}
