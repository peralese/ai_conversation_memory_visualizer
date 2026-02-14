import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet } from '../api'

export default function ClusterExplorerPage() {
  const [clusters, setClusters] = useState([])
  const [expanded, setExpanded] = useState({})
  const [specialization, setSpecialization] = useState({})

  useEffect(() => {
    apiGet('/clusters?include_subclusters=true&exclude_domain_stopwords=true').then(setClusters)
    apiGet('/metrics/model_specialization?level=cluster').then(res => {
      const byCluster = {}
      const baseline = res.baseline_available || {}
      ;(res.items || []).forEach(item => {
        const id = Number(item.cluster_id || item.id)
        if (!item.dominant_source || item.dominant_lift == null) {
          byCluster[id] = 'N/A'
          return
        }
        if (baseline[item.dominant_source] === false) {
          byCluster[id] = `${item.dominant_source} N/A`
          return
        }
        byCluster[id] = `${item.dominant_source} x${Number(item.dominant_lift || 0).toFixed(2)}`
      })
      setSpecialization(byCluster)
    })
  }, [])

  function toggle(clusterId) {
    setExpanded(prev => ({ ...prev, [clusterId]: !prev[clusterId] }))
  }

  return (
    <section>
      <h2>Cluster Explorer</h2>
      <ul>
        {clusters.map(c => (
          <li key={c.cluster_id} title={c.preview_snippet || ''}>
            <div className="row" style={{ alignItems: 'center' }}>
              <Link to={`/clusters/${c.cluster_id}`}>
                #{c.cluster_id} {c.label} ({c.message_count} messages) [Claude {c.source_breakdown?.counts?.CLAUDE ?? 0} | Gemini {c.source_breakdown?.counts?.GEMINI ?? 0} | ChatGPT {c.source_breakdown?.counts?.CHATGPT ?? 0}]
              </Link>
              <span className="badge">{specialization[c.cluster_id] || ''}</span>
              {Array.isArray(c.subclusters) && c.subclusters.length > 0 && (
                <button type="button" onClick={() => toggle(c.cluster_id)}>
                  {expanded[c.cluster_id] ? 'Hide subclusters' : 'Show subclusters'}
                </button>
              )}
            </div>
            {expanded[c.cluster_id] && Array.isArray(c.subclusters) && c.subclusters.length > 0 && (
              <ul>
                {c.subclusters.map(sc => (
                  <li key={sc.id} title={sc.preview_snippet || ''}>
                    {sc.id} {sc.label} ({sc.size} messages)
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}
