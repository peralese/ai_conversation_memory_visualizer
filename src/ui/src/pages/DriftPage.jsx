import { useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts'
import { apiGet } from '../api'

export default function DriftPage() {
  const [rows, setRows] = useState([])
  const [selected, setSelected] = useState(null)
  const [detail, setDetail] = useState(null)
  const chartRef = useRef(null)

  useEffect(() => {
    apiGet('/metrics/drift?level=cluster').then(res => {
      const items = res.items || []
      setRows(items)
      if (items.length > 0) setSelected(String(items[0].cluster_id))
    })
  }, [])

  useEffect(() => {
    if (!selected) return
    apiGet(`/metrics/drift?level=cluster&cluster_id=${encodeURIComponent(selected)}`).then(setDetail)
  }, [selected])

  const chartModel = useMemo(() => {
    const series = detail?.series || []
    return {
      x: series.map(p => p.bucket_start_date),
      drift: series.map(p => p.week_to_week_drift ?? 0),
      counts: series.map(p => p.message_count),
    }
  }, [detail])

  useEffect(() => {
    const chart = echarts.init(chartRef.current)
    chart.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: chartModel.x },
      yAxis: [{ type: 'value', name: 'Drift' }, { type: 'value', name: 'Count' }],
      series: [
        { name: 'Week-to-week drift', type: 'line', data: chartModel.drift, yAxisIndex: 0 },
        { name: 'Message count', type: 'bar', data: chartModel.counts, yAxisIndex: 1, opacity: 0.45 }
      ]
    })
    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      chart.dispose()
    }
  }, [chartModel])

  return (
    <section>
      <h2>Drift</h2>
      <table>
        <thead>
          <tr>
            <th>Cluster</th>
            <th>Label</th>
            <th>Cumulative drift</th>
            <th>Volatility</th>
            <th>Buckets</th>
            <th>Tag</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.cluster_id} onClick={() => setSelected(String(r.cluster_id))} style={{ cursor: 'pointer' }}>
              <td>#{r.cluster_id}</td>
              <td>{r.label}</td>
              <td>{r.cumulative_drift}</td>
              <td>{r.volatility}</td>
              <td>{r.active_buckets_count}</td>
              <td>{r.stability_tag}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>{detail?.label || 'Cluster drift detail'}</h3>
      <div ref={chartRef} style={{ width: '100%', height: 420 }} />
    </section>
  )
}
