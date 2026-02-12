import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { apiGet } from '../api'

export default function ClusterDetailPage() {
  const { clusterId } = useParams()
  const [detail, setDetail] = useState(null)

  useEffect(() => {
    apiGet(`/clusters/${clusterId}`).then(setDetail)
  }, [clusterId])

  if (!detail) return <div>Loading...</div>

  const c = detail.cluster
  return (
    <section>
      <h2>Cluster Detail</h2>
      <p><strong>Top keywords:</strong> {c.label}</p>
      <p><strong>First seen:</strong> {c.first_seen}</p>
      <p><strong>Last seen:</strong> {c.last_seen}</p>
      <p><strong>Half-life:</strong> {detail.half_life?.half_life_weeks ?? 'N/A'} weeks</p>
      <h3>Example messages</h3>
      <ul>
        {detail.examples.map(e => (
          <li key={e.id}><code>{e.timestamp}</code> {e.original_text.slice(0, 180)}</li>
        ))}
      </ul>
    </section>
  )
}
