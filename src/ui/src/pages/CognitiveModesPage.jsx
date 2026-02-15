import { useEffect, useMemo, useState } from 'react'
import { apiGet } from '../api'

export default function CognitiveModesPage() {
  const [data, setData] = useState(null)
  const [filterMode, setFilterMode] = useState('ALL')

  useEffect(() => {
    apiGet('/metrics/modes?level=cluster').then(setData)
  }, [])

  const rows = useMemo(() => {
    const items = data?.per_entity_mode_weights || []
    if (filterMode === 'ALL') return items
    return items.filter(r => r.dominant_mode === filterMode)
  }, [data, filterMode])

  const modeOptions = useMemo(() => {
    const taxonomy = data?.taxonomy || []
    return taxonomy.map(m => ({ id: m.id, name: m.name }))
  }, [data])

  return (
    <section>
      <h2>Cognitive Modes</h2>

      <h3>Overall Distribution</h3>
      <table>
        <thead>
          <tr><th>Mode</th><th>Weight</th></tr>
        </thead>
        <tbody>
          {Object.entries(data?.overall_mode_distribution || {}).map(([mode, weight]) => (
            <tr key={mode}>
              <td>{mode}</td>
              <td>{(Number(weight) * 100).toFixed(2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="row controls">
        <label>
          Dominant mode{' '}
          <select value={filterMode} onChange={e => setFilterMode(e.target.value)}>
            <option value="ALL">ALL</option>
            {modeOptions.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </label>
      </div>

      <h3>Clusters by Dominant Mode</h3>
      <table>
        <thead>
          <tr>
            <th>Entity</th>
            <th>Label</th>
            <th>Msgs</th>
            <th>% Dataset</th>
            <th>Dominant Mode</th>
            <th>Weight</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.entity_id}>
              <td>#{r.entity_id}</td>
              <td>{r.label}</td>
              <td>{r.message_count}</td>
              <td>{r.dataset_percentage}%</td>
              <td>{r.dominant_mode}</td>
              <td>{Number(r.dominant_weight || 0).toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
