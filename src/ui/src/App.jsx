import { NavLink, Route, Routes } from 'react-router-dom'
import ImportPage from './pages/ImportPage'
import ConversationsPage from './pages/ConversationsPage'
import ClusterExplorerPage from './pages/ClusterExplorerPage'
import TimelinePage from './pages/TimelinePage'
import ClusterDetailPage from './pages/ClusterDetailPage'
import ModelSpecializationPage from './pages/ModelSpecializationPage'
import DriftPage from './pages/DriftPage'
import ReportPage from './pages/ReportPage'
import CognitiveModesPage from './pages/CognitiveModesPage'

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
          <NavLink to="/specialization">Specialization</NavLink>
          <NavLink to="/drift">Drift</NavLink>
          <NavLink to="/modes">Cognitive Modes</NavLink>
          <NavLink to="/report">Report</NavLink>
        </nav>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<ImportPage />} />
          <Route path="/conversations" element={<ConversationsPage />} />
          <Route path="/clusters" element={<ClusterExplorerPage />} />
          <Route path="/timeline" element={<TimelinePage />} />
          <Route path="/clusters/:clusterId" element={<ClusterDetailPage />} />
          <Route path="/specialization" element={<ModelSpecializationPage />} />
          <Route path="/drift" element={<DriftPage />} />
          <Route path="/modes" element={<CognitiveModesPage />} />
          <Route path="/report" element={<ReportPage />} />
        </Routes>
      </main>
    </div>
  )
}
