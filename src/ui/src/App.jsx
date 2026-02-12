import { NavLink, Route, Routes } from 'react-router-dom'
import ImportPage from './pages/ImportPage'
import ConversationsPage from './pages/ConversationsPage'
import ClusterExplorerPage from './pages/ClusterExplorerPage'
import TimelinePage from './pages/TimelinePage'
import ClusterDetailPage from './pages/ClusterDetailPage'

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Memory Visualizer</h1>
        <nav>
          <NavLink to="/">Import</NavLink>
          <NavLink to="/conversations">Conversations</NavLink>
          <NavLink to="/clusters">Clusters</NavLink>
          <NavLink to="/timeline">Timeline</NavLink>
        </nav>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<ImportPage />} />
          <Route path="/conversations" element={<ConversationsPage />} />
          <Route path="/clusters" element={<ClusterExplorerPage />} />
          <Route path="/timeline" element={<TimelinePage />} />
          <Route path="/clusters/:clusterId" element={<ClusterDetailPage />} />
        </Routes>
      </main>
    </div>
  )
}
