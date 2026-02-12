import { useEffect, useRef, useState } from 'react'
import * as echarts from 'echarts'
import { apiGet } from '../api'

export default function TimelinePage() {
  const chartRef = useRef(null)
  const [halfLife, setHalfLife] = useState([])

  useEffect(() => {
    async function run() {
      const data = await apiGet('/metrics/topic-evolution?granularity=week')
      const byCluster = {}
      data.forEach(point => {
        if (!byCluster[point.cluster_id]) byCluster[point.cluster_id] = []
        byCluster[point.cluster_id].push(point)
      })

      const chart = echarts.init(chartRef.current)
      chart.setOption({
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category' },
        yAxis: { type: 'value' },
        series: Object.entries(byCluster).map(([clusterId, points]) => ({
          name: `Cluster ${clusterId}`,
          type: 'line',
          data: points.map(p => [p.bucket, p.count])
        }))
      })

      setHalfLife(await apiGet('/metrics/idea-half-life'))
    }
    run()
  }, [])

  return (
    <section>
      <h2>Timeline View</h2>
      <div ref={chartRef} style={{ width: '100%', height: 420 }} />
      <h3>Idea Half-Life</h3>
      <ul>
        {halfLife.map(h => (
          <li key={h.cluster_id}>{h.label}: {h.half_life_weeks ?? 'N/A'} weeks</li>
        ))}
      </ul>
    </section>
  )
}
