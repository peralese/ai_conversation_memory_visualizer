import { useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts'
import { apiGet } from '../api'

const SOURCES = ['ALL', 'CHATGPT', 'CLAUDE', 'GEMINI']

export default function TimelinePage() {
  const chartRef = useRef(null)
  const [halfLife, setHalfLife] = useState([])
  const [evolution, setEvolution] = useState([])
  const [source, setSource] = useState('ALL')
  const [minMessages, setMinMessages] = useState(1)
  const [topN, setTopN] = useState(15)
  const [useSubclusters, setUseSubclusters] = useState(false)
  const [excludeDomainStopwords, setExcludeDomainStopwords] = useState(true)

  useEffect(() => {
    const qs = new URLSearchParams({
      granularity: 'week',
      source,
      min_messages: String(minMessages),
      top_n: String(topN),
      use_subclusters: String(useSubclusters),
      exclude_domain_stopwords: String(excludeDomainStopwords)
    })
    apiGet(`/metrics/topic-evolution?${qs.toString()}`).then(setEvolution)
    apiGet('/metrics/idea-half-life').then(setHalfLife)
  }, [source, minMessages, topN, useSubclusters, excludeDomainStopwords])

  const heatmapModel = useMemo(() => {
    const weekSet = new Set()
    const clusterMap = {}

    evolution.forEach(point => {
      weekSet.add(point.week_start)
      if (!clusterMap[point.cluster_id]) {
        clusterMap[point.cluster_id] = {
          id: point.cluster_id,
          label: point.label,
          total: point.total_cluster_messages || 0
        }
      }
    })

    const weeks = Array.from(weekSet).sort((a, b) => new Date(a) - new Date(b))
    const clusters = Object.values(clusterMap).sort((a, b) => b.total - a.total)
    const yLabels = clusters.map(c => `#${c.id} ${c.label}`)

    const weekIndex = Object.fromEntries(weeks.map((w, i) => [w, i]))
    const clusterIndex = Object.fromEntries(clusters.map((c, i) => [c.id, i]))

    const data = evolution
      .filter(point => point.week_start in weekIndex && point.cluster_id in clusterIndex)
      .map(point => [weekIndex[point.week_start], clusterIndex[point.cluster_id], point.count])

    const xLabels = weeks.map(w => {
      const d = new Date(w)
      const week = getIsoWeek(d)
      const year = getIsoYear(d)
      return `${year}-W${String(week).padStart(2, '0')}`
    })

    return { data, xLabels, yLabels }
  }, [evolution])

  useEffect(() => {
    const chart = echarts.init(chartRef.current)
    chart.setOption({
      tooltip: {
        position: 'top',
        formatter: params => {
          const [x, y, count] = params.value
          return `${heatmapModel.yLabels[y]}<br/>${heatmapModel.xLabels[x]}: ${count} messages`
        }
      },
      grid: { height: '65%', top: 40 },
      xAxis: {
        type: 'category',
        data: heatmapModel.xLabels,
        splitArea: { show: true }
      },
      yAxis: {
        type: 'category',
        data: heatmapModel.yLabels,
        splitArea: { show: true }
      },
      visualMap: {
        min: 0,
        max: Math.max(1, ...heatmapModel.data.map(d => d[2])),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '3%'
      },
      series: [
        {
          name: 'Activity',
          type: 'heatmap',
          data: heatmapModel.data,
          label: { show: false },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.35)'
            }
          }
        }
      ]
    })

    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      chart.dispose()
    }
  }, [heatmapModel])

  return (
    <section>
      <h2>Timeline Heatmap</h2>
      <div className="row controls">
        <label>
          Source{' '}
          <select value={source} onChange={e => setSource(e.target.value)}>
            {SOURCES.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
        <label>
          Min messages: <strong>{minMessages}</strong>{' '}
          <input
            type="range"
            min="1"
            max="50"
            step="1"
            value={minMessages}
            onChange={e => setMinMessages(Number(e.target.value))}
          />
        </label>
        <label>
          Top clusters{' '}
          <input
            type="number"
            min="1"
            max="50"
            value={topN}
            onChange={e => setTopN(Math.max(1, Number(e.target.value) || 15))}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={useSubclusters}
            onChange={e => setUseSubclusters(e.target.checked)}
          />
          Use subclusters
        </label>
        <label>
          <input
            type="checkbox"
            checked={excludeDomainStopwords}
            onChange={e => setExcludeDomainStopwords(e.target.checked)}
          />
          Exclude domain stopwords from labels
        </label>
      </div>
      <div ref={chartRef} style={{ width: '100%', height: 460 }} />
      <h3>Idea Half-Life</h3>
      <ul>
        {halfLife.map(h => (
          <li key={h.cluster_id}>
            #{h.cluster_id} {h.label}: {h.half_life_weeks == null ? 'insufficient data' : `${h.half_life_weeks} weeks`}
          </li>
        ))}
      </ul>
    </section>
  )
}

function getIsoYear(date) {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7))
  return d.getUTCFullYear()
}

function getIsoWeek(date) {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7))
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1))
  return Math.ceil((((d - yearStart) / 86400000) + 1) / 7)
}
