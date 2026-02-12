import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet } from '../api'

export default function ClusterExplorerPage() {
  const [clusters, setClusters] = useState([])

  useEffect(() => {
    apiGet('/clusters').then(setClusters)
  }, [])

  return (
    <section>
      <h2>Cluster Explorer</h2>
      <ul>
        {clusters.map(c => (
          <li key={c.cluster_id}>
            <Link to={`/clusters/${c.cluster_id}`}>{c.label}</Link> ({c.message_count} messages)
          </li>
        ))}
      </ul>
    </section>
  )
}
