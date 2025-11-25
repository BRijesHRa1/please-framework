import { useState, useEffect } from 'react';
import { 
  LayoutDashboard, FolderOpen, CheckCircle2, Loader2, XCircle,
  Clock, TrendingUp, BarChart3, FileText, Trash2, Eye, RefreshCw
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../api/client';

export default function Dashboard({ onViewProject, onClose }) {
  const [dashboard, setDashboard] = useState(null);
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [dashboardRes, projectsRes] = await Promise.all([
        apiClient.getDashboard(),
        apiClient.getProjects()
      ]);
      setDashboard(dashboardRes.data);
      setProjects(projectsRes.data.projects);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch data');
      toast.error('Failed to load dashboard');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleDelete = async (projectId, projectName) => {
    if (!confirm(`Delete project "${projectName}"? This cannot be undone.`)) return;
    
    setDeletingId(projectId);
    try {
      await apiClient.deleteProject(projectId);
      toast.success('Project deleted');
      fetchData();
    } catch (err) {
      toast.error('Failed to delete project');
    } finally {
      setDeletingId(null);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case 'running':
        return <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-slate-400" />;
    }
  };

  const getStatusBadge = (status) => {
    const styles = {
      completed: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
      running: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
      failed: 'bg-red-500/20 text-red-300 border-red-500/30',
      pending: 'bg-slate-500/20 text-slate-300 border-slate-500/30'
    };
    return styles[status] || styles.pending;
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="fixed inset-0 z-50 bg-slate-900/95 backdrop-blur-sm flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-sky-400 animate-spin mx-auto mb-4" />
          <p className="text-slate-300">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/98 backdrop-blur-sm overflow-auto">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-sky-500 to-violet-500 flex items-center justify-center">
              <LayoutDashboard className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Dashboard</h1>
              <p className="text-sm text-slate-400">Overview of all projects</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={fetchData}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              Close
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300">
            {error}
          </div>
        )}

        {/* Stats Cards */}
        {dashboard && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
            <StatCard
              icon={FolderOpen}
              label="Total Projects"
              value={dashboard.total_projects}
              color="slate"
            />
            <StatCard
              icon={CheckCircle2}
              label="Completed"
              value={dashboard.completed_projects}
              color="emerald"
            />
            <StatCard
              icon={Loader2}
              label="Running"
              value={dashboard.running_projects}
              color="sky"
              animate={dashboard.running_projects > 0}
            />
            <StatCard
              icon={XCircle}
              label="Failed"
              value={dashboard.failed_projects}
              color="red"
            />
            <StatCard
              icon={TrendingUp}
              label="Avg A.G.E. Score"
              value={dashboard.avg_age_score ? `${dashboard.avg_age_score}/10` : 'N/A'}
              color="violet"
            />
           
          </div>
        )}

        {/* Projects Table */}
        <div className="glass rounded-2xl overflow-hidden">
          <div className="p-6 border-b border-slate-700/50">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <FileText className="w-5 h-5 text-sky-400" />
              All Projects
              <span className="text-sm font-normal text-slate-400">
                ({projects.length} total)
              </span>
            </h2>
          </div>

          {projects.length === 0 ? (
            <div className="p-12 text-center">
              <FolderOpen className="w-12 h-12 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400">No projects yet</p>
              <p className="text-sm text-slate-500 mt-1">
                Upload a spec sheet to start your first project
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-800/50">
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Project
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Cycles
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Updated
                    </th>
                    <th className="px-6 py-4 text-right text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {projects.map((project) => (
                    <tr 
                      key={project.id} 
                      className="hover:bg-slate-800/30 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <div>
                          <p className="font-medium text-white">{project.name}</p>
                          <p className="text-sm text-slate-400 truncate max-w-xs">
                            {project.description || 'No description'}
                          </p>
                          <p className="text-xs text-slate-500 font-mono mt-1">
                            {project.id.slice(0, 8)}...
                          </p>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`
                          inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border
                          ${getStatusBadge(project.status)}
                        `}>
                          {getStatusIcon(project.status)}
                          {project.status}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className="text-slate-300">
                          {project.current_cycle_number} / {2}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-400">
                        {formatDate(project.created_at)}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-400">
                        {formatDate(project.updated_at)}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => onViewProject(project.id)}
                            className="p-2 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 transition-colors"
                            title="View Details"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(project.id, project.name)}
                            disabled={deletingId === project.id}
                            className="p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-colors disabled:opacity-50"
                            title="Delete"
                          >
                            {deletingId === project.id ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              <Trash2 className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color, animate }) {
  const colors = {
    slate: 'from-slate-500 to-slate-600',
    emerald: 'from-emerald-500 to-emerald-600',
    sky: 'from-sky-500 to-sky-600',
    red: 'from-red-500 to-red-600',
    violet: 'from-violet-500 to-violet-600',
    amber: 'from-amber-500 to-amber-600'
  };

  const iconColors = {
    slate: 'text-slate-400',
    emerald: 'text-emerald-400',
    sky: 'text-sky-400',
    red: 'text-red-400',
    violet: 'text-violet-400',
    amber: 'text-amber-400'
  };

  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-center gap-3">
        <div className={`
          w-10 h-10 rounded-lg bg-gradient-to-br ${colors[color]} 
          flex items-center justify-center
        `}>
          <Icon className={`w-5 h-5 text-white ${animate ? 'animate-spin' : ''}`} />
        </div>
        <div>
          <p className="text-2xl font-bold text-white">{value}</p>
          <p className="text-xs text-slate-400">{label}</p>
        </div>
      </div>
    </div>
  );
}

