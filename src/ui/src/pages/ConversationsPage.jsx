import { useEffect, useState } from 'react'
import { apiGet } from '../api'

export default function ConversationsPage() {
  const [items, setItems] = useState([])
  const [q, setQ] = useState('')

  async function load(query = '') {
    const data = await apiGet('/conversations' + (query ? `?q=${encodeURIComponent(query)}` : ''))
    setItems(data)
  }

  useEffect(() => { load() }, [])

  return (
    <section>
      <h2>Conversation List + Search</h2>
      <div className="row">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by title" />
        <button onClick={() => load(q)}>Search</button>
      </div>
      <table>
        <thead><tr><th>Title</th><th>Source</th><th>Updated</th></tr></thead>
        <tbody>
          {items.map(item => (
            <tr key={item.id}><td>{item.title}</td><td>{item.source}</td><td>{item.updated_at}</td></tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
