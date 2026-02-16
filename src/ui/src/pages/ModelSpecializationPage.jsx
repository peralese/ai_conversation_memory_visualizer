import { useEffect, useMemo, useState } from 'react'
import { apiGet } from '../api'

const SOURCES = ['ALL', 'CHATGPT', 'CLAUDE', 'GEMINI']

export default function ModelSpecializationPage() {
  const [data, setData] = useState(null)
  const [dominantSource, setDominantSource] = useState('ALL')
  const [minLift, setMinLift] = useState(1.2)

  useEffect(() => {
    apiGet('/metrics/model_specialization?level=cluster').then(setData)
  }, [])

  const rows = useMemo(() => {
    const items = (data?.items || []).filter(item => item.dominant_lift >= minLift)
    const filtered = dominantSource === 'ALL' ? items : items.filter(item => item.dominant_source === dominantSource)
    return filtered.sort((a, b) => b.dominant_lift - a.dominant_lift)
  }, [data, dominantSource, minLift])

  return (
    <section>
      <h2>Model Specialization</h2>
      {Object.entries(data?.baseline_available || {})
        .filter(([, available]) => !available)
        .map(([source]) => (
          <p key={source}><strong>Note:</strong> {source} has no data and is excluded from specialization calculations.</p>
        ))}
      <div className="row controls">
        <label>
          Dominant source{' '}
          <select value={dominantSource} onChange={e => setDominantSource(e.target.value)}>
            {SOURCES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label>
          Min dominant lift: <strong>{minLift.toFixed(1)}</strong>{' '}
          <input
            type="range"
            min="1"
            max="4"
            step="0.1"
            value={minLift}
            onChange={e => setMinLift(Number(e.target.value))}
          />
        </label>
      </div>
      <table>
        <thead>
          <tr>
            <th>Cluster</th>
            <th>Label</th>
            <th>% dataset</th>
            <th>Msgs</th>
            <th>Claude %</th>
            <th>Claude lift</th>
            <th>Gemini %</th>
            <th>Gemini lift</th>
            <th>ChatGPT %</th>
            <th>ChatGPT lift</th>
            <th>Dominant</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(item => (
            <tr key={item.cluster_id || item.id}>
              <td>#{item.cluster_id || item.id}</td>
              <td>{item.label}</td>
              <td>{item.dataset_percentage}%</td>
              <td>{item.message_count}</td>
              <td>{item.source_breakdown?.percents?.CLAUDE ?? 0}%</td>
              <td>{item.lift_by_source?.CLAUDE == null ? 'N/A' : `x${Number(item.lift_by_source.CLAUDE).toFixed(2)}`}</td>
              <td>{item.source_breakdown?.percents?.GEMINI ?? 0}%</td>
              <td>{item.lift_by_source?.GEMINI == null ? 'N/A' : `x${Number(item.lift_by_source.GEMINI).toFixed(2)}`}</td>
              <td>{item.source_breakdown?.percents?.CHATGPT ?? 0}%</td>
              <td>{item.lift_by_source?.CHATGPT == null ? 'N/A' : `x${Number(item.lift_by_source.CHATGPT).toFixed(2)}`}</td>
              <td>{item.dominant_source ? `${item.dominant_source} x${Number(item.dominant_lift || 0).toFixed(2)}` : 'N/A'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
