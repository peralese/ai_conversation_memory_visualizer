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

  useEffect(() => {
    apiGet(`/clusters/${clusterId}`).then(setDetail)
    apiGet(`/clusters/${clusterId}/subclusters`).then(res => setSubclusters(res.subclusters || []))
  }, [clusterId])

  if (!detail) return <div>Loading...</div>
  return (
    <section>
      <h2>Cluster Detail</h2>
      <p><strong>Cluster:</strong> #{detail.cluster_id}</p>
      <p><strong>Label:</strong> {detail.label}</p>
      <p><strong>Size:</strong> {detail.message_count} messages</p>
      <p><strong>Conversations:</strong> {detail.conversations_count}</p>
      <p><strong>% of dataset:</strong> {detail.dataset_percentage}%</p>
      <p><strong>Avg message length:</strong> {detail.average_message_length} characters</p>
      <p><strong>Top keywords:</strong> {detail.top_keywords?.join(', ') || 'N/A'}</p>
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
