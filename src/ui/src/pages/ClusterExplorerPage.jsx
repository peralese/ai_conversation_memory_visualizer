import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet, apiPost } from '../api'

export default function ClusterExplorerPage() {
  const [clusters, setClusters] = useState([])
  const [expanded, setExpanded] = useState({})
  const [specialization, setSpecialization] = useState({})
  const [modes, setModes] = useState({})
  const [excludeDomainStopwords, setExcludeDomainStopwords] = useState(true)
  const [useSemanticLabels, setUseSemanticLabels] = useState(true)
  const [showLegacyLabels, setShowLegacyLabels] = useState(false)
  const [labeling, setLabeling] = useState(false)

  useEffect(() => {
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
    apiGet('/metrics/modes?level=cluster').then(res => {
      const byCluster = {}
      ;(res.per_entity_mode_weights || []).forEach(item => {
        const id = Number(item.entity_id)
        byCluster[id] = `${item.dominant_mode} ${Number(item.dominant_weight || 0).toFixed(2)}`
      })
      setModes(byCluster)
    })
  }, [])

  useEffect(() => {
    const qs = new URLSearchParams({
      include_subclusters: 'true',
      exclude_domain_stopwords: String(excludeDomainStopwords),
      use_semantic_labels: String(useSemanticLabels),
      show_legacy_labels: String(showLegacyLabels)
    })
    apiGet(`/clusters?${qs.toString()}`).then(setClusters)
  }, [excludeDomainStopwords, useSemanticLabels, showLegacyLabels])

  async function generateSemanticLabels() {
    setLabeling(true)
    try {
      await apiPost('/api/labels/clusters', { force: false })
      const qs = new URLSearchParams({
        include_subclusters: 'true',
        exclude_domain_stopwords: String(excludeDomainStopwords),
        use_semantic_labels: String(useSemanticLabels),
        show_legacy_labels: String(showLegacyLabels)
      })
      const refreshed = await apiGet(`/clusters?${qs.toString()}`)
      setClusters(refreshed)
    } finally {
      setLabeling(false)
    }
  }

  function toggle(clusterId) {
    setExpanded(prev => ({ ...prev, [clusterId]: !prev[clusterId] }))
  }

  return (
    <section>
      <h2>Cluster Explorer</h2>
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
      <button type="button" onClick={generateSemanticLabels} disabled={labeling}>
        {labeling ? 'Generating labels...' : 'Generate semantic labels'}
      </button>
      <ul>
        {clusters.map(c => (
          <li key={c.cluster_id} title={c.preview_snippet || ''}>
            <div className="row" style={{ alignItems: 'center' }}>
              <Link to={`/clusters/${c.cluster_id}`}>
                #{c.cluster_id} {c.label} ({c.message_count} messages) [Claude {c.source_breakdown?.counts?.CLAUDE ?? 0} | Gemini {c.source_breakdown?.counts?.GEMINI ?? 0} | ChatGPT {c.source_breakdown?.counts?.CHATGPT ?? 0}]
              </Link>
              {showLegacyLabels && c.legacy_label && c.legacy_label !== c.label && (
                <small style={{ opacity: 0.65 }}>legacy: {c.legacy_label}</small>
              )}
              {c.label_low_signal && <span title={c.label_warning || 'Label may be low-signal'}>⚠</span>}
              <span className="badge">{specialization[c.cluster_id] || ''}</span>
              <span className="badge">{modes[c.cluster_id] || ''}</span>
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
