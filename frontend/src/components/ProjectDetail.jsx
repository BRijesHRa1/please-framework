import { useState, useEffect } from 'react';
import { 
  ArrowLeft, FileText, CheckCircle2, Loader2, XCircle, Clock,
  Brain, Cpu, BarChart3, Clipboard, TrendingUp, Target, AlertTriangle,
  Download, Share2, Printer, Copy, Check
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../api/client';
import MarkdownReport from './MarkdownReport';

export default function ProjectDetail({ projectId, onBack }) {
  const [project, setProject] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const fetchProject = async () => {
      setLoading(true);
      try {
        const response = await apiClient.getProject(projectId);
        setProject(response.data);
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to fetch project');
        toast.error('Failed to load project');
      } finally {
        setLoading(false);
      }
    };

    fetchProject();
  }, [projectId]);

  const copyReportToClipboard = () => {
    if (project?.report?.content) {
      navigator.clipboard.writeText(project.report.content);
      setCopied(true);
      toast.success('Report copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const downloadReport = () => {
    if (project?.report?.content) {
      const blob = new Blob([project.report.content], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${project.project.name.replace(/\s+/g, '_')}_report.md`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success('Report downloaded');
    }
  };

  if (loading) {
    return (
      <div className="fixed inset-0 z-50 bg-slate-900/98 flex items-center justify-center">
        <Loader2 className="w-12 h-12 text-sky-400 animate-spin" />
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="fixed inset-0 z-50 bg-slate-900/98 flex items-center justify-center">
        <div className="text-center">
          <XCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <p className="text-red-300">{error || 'Project not found'}</p>
          <button onClick={onBack} className="mt-4 text-sky-400 hover:underline">
            Go back
          </button>
        </div>
      </div>
    );
  }

  const { project: projectInfo, cycles, agent_outputs, report, gap_analysis, age_scores } = project;

  const agentIcons = {
    planner: Clipboard,
    learner: Brain,
    executor: Cpu,
    assessor: BarChart3,
    pm: FileText
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'agents', label: 'Agent Outputs', icon: Brain },
    { id: 'report', label: 'Report', icon: FileText },
  ];

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/98 overflow-auto">
      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <button
            onClick={onBack}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-white">{projectInfo.name}</h1>
            <p className="text-sm text-slate-400">{projectInfo.description}</p>
          </div>
          <StatusBadge status={projectInfo.status} />
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-slate-700 pb-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-sky-500/20 text-sky-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Stats Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="glass rounded-xl p-4">
                <p className="text-xs text-slate-400 mb-1">Cycles Completed</p>
                <p className="text-2xl font-bold text-white">
                  {projectInfo.current_cycle_number} / 2
                </p>
              </div>
              {report && (
                <>
                  <div className="glass rounded-xl p-4">
                    <p className="text-xs text-slate-400 mb-1">NIH Score</p>
                    <p className="text-2xl font-bold text-emerald-400">
                      {report.final_nih_score ?? 'N/A'}
                    </p>
                  </div>
                  <div className="glass rounded-xl p-4">
                    <p className="text-xs text-slate-400 mb-1">Satisfaction</p>
                    <p className="text-2xl font-bold text-sky-400">
                      {report.final_bimodal_score ?? 'N/A'}/5
                    </p>
                  </div>
                </>
              )}
              <div className="glass rounded-xl p-4">
                <p className="text-xs text-slate-400 mb-1">Created</p>
                <p className="text-sm font-medium text-white">
                  {new Date(projectInfo.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>

            {/* Gap Analysis */}
            {gap_analysis && gap_analysis.length > 0 && (
              <div className="glass rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-sky-400" />
                  Gap Analysis
                </h3>
                <div className="space-y-3">
                  {gap_analysis.map((gap, i) => (
                    <div key={i} className="flex items-center gap-4 p-3 bg-slate-800/50 rounded-lg">
                      <div className="flex-1">
                        <p className="font-medium text-slate-200">{gap.metric}</p>
                        <div className="flex items-center gap-4 mt-1 text-sm">
                          <span className="text-slate-400">Goal: {gap.goal}</span>
                          <span className="text-slate-400">Achieved: {gap.achieved}</span>
                          <span className={gap.gap <= 0 ? 'text-emerald-400' : 'text-amber-400'}>
                            Gap: {gap.gap}
                          </span>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        gap.status === 'met' 
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : 'bg-amber-500/20 text-amber-300'
                      }`}>
                        {gap.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* A.G.E. Scores */}
            {age_scores && Object.keys(age_scores).length > 0 && (
              <div className="glass rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-violet-400" />
                  A.G.E. Scores
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(age_scores).map(([key, score]) => (
                    <div key={key} className="p-4 bg-slate-800/50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-slate-200">{score.agent_name}</span>
                        <span className="text-lg font-bold text-violet-400">{score.score}/10</span>
                      </div>
                      <p className="text-xs text-slate-400 line-clamp-2">{score.justification}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Cycles Timeline */}
            <div className="glass rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Cycles</h3>
              <div className="space-y-3">
                {cycles.map((cycle) => (
                  <div key={cycle.id} className="flex items-center gap-4 p-3 bg-slate-800/50 rounded-lg">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      cycle.status === 'completed' ? 'bg-emerald-500/20' : 'bg-slate-700'
                    }`}>
                      <span className="font-bold text-white">{cycle.cycle_number}</span>
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-slate-200">Cycle {cycle.cycle_number}</p>
                      <p className="text-sm text-slate-400">
                        {new Date(cycle.started_at).toLocaleString()}
                      </p>
                    </div>
                    <StatusBadge status={cycle.status} small />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'agents' && (
          <div className="space-y-6">
            {Object.entries(agent_outputs).map(([cycleKey, outputs]) => (
              <div key={cycleKey} className="glass rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4 capitalize">
                  {cycleKey.replace('_', ' ')}
                </h3>
                <div className="space-y-4">
                  {outputs.map((output) => {
                    const Icon = agentIcons[output.agent_name] || FileText;
                    const data = output.output_data || {};
                    
                    return (
                      <div key={output.id} className="p-4 bg-slate-800/50 rounded-lg">
                        {/* Agent Header */}
                        <div className="flex items-center gap-3 mb-3">
                          <div className="w-8 h-8 rounded-lg bg-sky-500/20 flex items-center justify-center">
                            <Icon className="w-4 h-4 text-sky-400" />
                          </div>
                          <div className="flex-1">
                            <p className="font-medium text-slate-200 capitalize">{output.agent_name}</p>
                            <p className="text-xs text-slate-400">
                              {output.execution_time_seconds?.toFixed(2)}s
                            </p>
                          </div>
                          <StatusBadge status={output.status} small />
                        </div>

                        {/* Summary */}
                        {output.summary && (
                          <p className="text-sm text-slate-300 bg-slate-900/50 p-3 rounded-lg mb-3">
                            {output.summary}
                          </p>
                        )}

                        {/* PLANNER DETAILS */}
                        {output.agent_name === 'planner' && data && (
                          <div className="space-y-3 mt-3">
                            {/* Duration & GPU */}
                            <div className="grid grid-cols-3 gap-2">
                              <div className="p-2 bg-slate-900/50 rounded-lg text-center">
                                <p className="text-xs text-slate-500">Duration</p>
                                <p className="text-sm font-medium text-slate-300">{data.estimated_duration || 'N/A'}</p>
                              </div>
                              <div className="p-2 bg-slate-900/50 rounded-lg text-center">
                                <p className="text-xs text-slate-500">GPU Hours</p>
                                <p className="text-sm font-medium text-slate-300">{data.total_gpu_estimate || 0}h</p>
                              </div>
                              <div className="p-2 bg-slate-900/50 rounded-lg text-center">
                                <p className="text-xs text-slate-500">Tasks</p>
                                <p className="text-sm font-medium text-slate-300">{data.tasks?.length || 0}</p>
                              </div>
                            </div>

                            {/* MCP Query */}
                            {data.mcp_query && (
                              <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                                <p className="text-xs text-emerald-400 font-semibold mb-1">🔍 MCP Query (for Learner)</p>
                                <p className="text-sm text-emerald-300 italic">"{data.mcp_query}"</p>
                              </div>
                            )}

                            {/* Tasks */}
                            {data.tasks && data.tasks.length > 0 && (
                              <div className="space-y-2">
                                <p className="text-xs text-slate-400 font-semibold">📝 Tasks ({data.tasks.length})</p>
                                {data.tasks.map((task, i) => (
                                  <div key={i} className="p-2 bg-slate-900/50 rounded-lg">
                                    <div className="flex items-center gap-2">
                                      <span className="text-xs font-mono text-violet-400">[{task.task_id}]</span>
                                      <span className="text-sm font-medium text-slate-200">{task.name}</span>
                                      <span className={`ml-auto text-xs px-2 py-0.5 rounded ${
                                        task.priority === 'high' ? 'bg-red-500/20 text-red-300' :
                                        task.priority === 'medium' ? 'bg-amber-500/20 text-amber-300' :
                                        'bg-slate-500/20 text-slate-300'
                                      }`}>{task.priority}</span>
                                    </div>
                                    <p className="text-xs text-slate-400 mt-1">{task.description}</p>
                                    <div className="flex gap-3 mt-1 text-xs text-slate-500">
                                      <span>GPU: {task.gpu_hours}h</span>
                                      <span>Deps: {task.dependencies?.join(', ') || 'None'}</span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}

                            {/* Risk Factors */}
                            {data.risk_factors && data.risk_factors.length > 0 && (
                              <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                                <p className="text-xs text-amber-400 font-semibold mb-2">⚠️ Risk Factors</p>
                                <ul className="space-y-1">
                                  {data.risk_factors.map((risk, i) => (
                                    <li key={i} className="text-sm text-amber-300 flex items-start gap-2">
                                      <span className="text-amber-400">•</span>
                                      {risk}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        )}

                        {/* LEARNER DETAILS */}
                        {output.agent_name === 'learner' && data && (
                          <div className="space-y-3 mt-3">
                            {/* Key Genes */}
                            {data.key_genes && data.key_genes.length > 0 && (
                              <div className="p-3 bg-violet-500/10 border border-violet-500/20 rounded-lg">
                                <p className="text-xs text-violet-400 font-semibold mb-2">🧬 Key Genes ({data.key_genes.length})</p>
                                <div className="flex flex-wrap gap-1">
                                  {data.key_genes.slice(0, 15).map((gene, i) => (
                                    <span key={i} className="px-2 py-0.5 text-xs rounded bg-violet-500/20 text-violet-300 font-mono">{gene}</span>
                                  ))}
                                  {data.key_genes.length > 15 && (
                                    <span className="text-xs text-slate-500">+{data.key_genes.length - 15} more</span>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Model Suggestions */}
                            {data.model_suggestions && data.model_suggestions.length > 0 && (
                              <div className="p-3 bg-sky-500/10 border border-sky-500/20 rounded-lg">
                                <p className="text-xs text-sky-400 font-semibold mb-2">🤖 Model Suggestions</p>
                                <div className="flex flex-wrap gap-2">
                                  {data.model_suggestions.map((model, i) => (
                                    <span key={i} className="px-2 py-1 text-xs rounded bg-sky-500/20 text-sky-300">{model}</span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {/* EXECUTOR DETAILS */}
                        {output.agent_name === 'executor' && data && (
                          <div className="space-y-3 mt-3">
                            {data.metrics && (
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                {Object.entries(data.metrics).slice(0, 8).map(([key, value]) => (
                                  <div key={key} className="p-2 bg-slate-900/50 rounded-lg text-center">
                                    <p className="text-xs text-slate-500 truncate">{key}</p>
                                    <p className="text-sm font-medium text-slate-300">
                                      {typeof value === 'number' ? value.toFixed(4) : value}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* ASSESSOR DETAILS */}
                        {output.agent_name === 'assessor' && data && (
                          <div className="space-y-3 mt-3">
                            {/* Scores */}
                            <div className="grid grid-cols-2 gap-2">
                              {data.satisfaction_score !== undefined && (
                                <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-center">
                                  <p className="text-xs text-emerald-400">Satisfaction</p>
                                  <p className="text-xl font-bold text-emerald-300">{data.satisfaction_score}/5</p>
                                </div>
                              )}
                              {data.nih_score !== undefined && (
                                <div className="p-3 bg-sky-500/10 border border-sky-500/20 rounded-lg text-center">
                                  <p className="text-xs text-sky-400">NIH Score</p>
                                  <p className="text-xl font-bold text-sky-300">{data.nih_score}/9</p>
                                </div>
                              )}
                            </div>

                            {/* Recommendations */}
                            {data.recommendations && data.recommendations.length > 0 && (
                              <div className="p-3 bg-slate-900/50 rounded-lg">
                                <p className="text-xs text-slate-400 font-semibold mb-2">📝 Recommendations</p>
                                <ul className="space-y-1">
                                  {data.recommendations.slice(0, 3).map((rec, i) => (
                                    <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                                      <span className="text-sky-400">→</span>
                                      {rec}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'report' && (
          <div className="space-y-6">
            {report ? (
              <>
                {/* Report Header */}
                <div className="glass rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-2xl font-bold text-white mb-1">Research Report</h2>
                      <p className="text-sm text-slate-400">
                        Generated on {new Date(report.created_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={copyReportToClipboard}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors text-sm"
                      >
                        {copied ? (
                          <Check className="w-4 h-4 text-emerald-400" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                        Copy
                      </button>
                      <button
                        onClick={downloadReport}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors text-sm"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </button>
                    </div>
                  </div>

                  {/* Score Summary */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    {report.final_nih_score !== null && (
                      <ScoreDisplay 
                        label="NIH Score" 
                        value={report.final_nih_score} 
                        maxValue={9}
                        color="emerald"
                      />
                    )}
                    {report.final_bimodal_score !== null && (
                      <ScoreDisplay 
                        label="Satisfaction" 
                        value={report.final_bimodal_score}
                        maxValue={5}
                        color="sky"
                      />
                    )}
                    <div className="bg-slate-800/50 rounded-xl p-4">
                      <p className="text-xs text-slate-400 mb-1">Status</p>
                      <StatusBadge status={projectInfo.status} />
                    </div>
                    <div className="bg-slate-800/50 rounded-xl p-4">
                      <p className="text-xs text-slate-400 mb-1">Cycles Run</p>
                      <p className="text-xl font-bold text-white">{cycles.length}</p>
                    </div>
                  </div>

                  {/* Executive Summary */}
                  {report.executive_summary && (
                    <div className="bg-gradient-to-r from-sky-500/10 to-violet-500/10 border border-sky-500/20 rounded-xl p-5 mb-6">
                      <div className="flex items-start gap-3">
                        <div className="w-10 h-10 rounded-lg bg-sky-500/20 flex items-center justify-center flex-shrink-0">
                          <FileText className="w-5 h-5 text-sky-400" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-sky-300 mb-2">Executive Summary</h3>
                          <p className="text-slate-300 leading-relaxed">{report.executive_summary}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Recommendations */}
                  {report.recommendations && report.recommendations.length > 0 && (
                    <div className="bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20 rounded-xl p-5">
                      <div className="flex items-start gap-3">
                        <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                          <AlertTriangle className="w-5 h-5 text-amber-400" />
                        </div>
                        <div className="flex-1">
                          <h3 className="font-semibold text-amber-300 mb-3">Recommendations</h3>
                          <div className="space-y-2">
                            {report.recommendations.map((rec, i) => (
                              <div key={i} className="flex items-start gap-2">
                                <div className="w-6 h-6 rounded-full bg-amber-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                                  <span className="text-xs font-bold text-amber-400">{i + 1}</span>
                                </div>
                                <p className="text-slate-300">{rec}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Full Report Content */}
                <div className="glass rounded-2xl p-8">
                  <div className="flex items-center gap-2 mb-6 pb-4 border-b border-slate-700">
                    <div className="w-1 h-6 bg-gradient-to-b from-sky-400 to-violet-500 rounded-full" />
                    <h3 className="text-lg font-semibold text-white">Full Report</h3>
                  </div>
                  <MarkdownReport content={report.content} />
                </div>
              </>
            ) : (
              <div className="glass rounded-2xl p-12 text-center">
                <div className="w-20 h-20 rounded-2xl bg-slate-800 flex items-center justify-center mx-auto mb-4">
                  <FileText className="w-10 h-10 text-slate-600" />
                </div>
                <h3 className="text-xl font-semibold text-slate-300 mb-2">No Report Available</h3>
                <p className="text-slate-500">
                  The report will be generated when the project pipeline completes.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status, small }) {
  const styles = {
    completed: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    running: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
    failed: 'bg-red-500/20 text-red-300 border-red-500/30',
    pending: 'bg-slate-500/20 text-slate-300 border-slate-500/30'
  };

  const icons = {
    completed: CheckCircle2,
    running: Loader2,
    failed: XCircle,
    pending: Clock
  };

  const Icon = icons[status] || Clock;
  const isRunning = status === 'running';

  return (
    <span className={`
      inline-flex items-center gap-1.5 rounded-full border font-medium
      ${small ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'}
      ${styles[status] || styles.pending}
    `}>
      <Icon className={`${small ? 'w-3 h-3' : 'w-4 h-4'} ${isRunning ? 'animate-spin' : ''}`} />
      {status}
    </span>
  );
}

function ScoreDisplay({ label, value, maxValue, suffix = '', color = 'sky' }) {
  const percentage = maxValue ? (value / maxValue) * 100 : value;
  
  const colors = {
    sky: { text: 'text-sky-400', bg: 'from-sky-500 to-sky-400' },
    emerald: { text: 'text-emerald-400', bg: 'from-emerald-500 to-emerald-400' },
    violet: { text: 'text-violet-400', bg: 'from-violet-500 to-violet-400' },
    amber: { text: 'text-amber-400', bg: 'from-amber-500 to-amber-400' }
  };

  return (
    <div className="bg-slate-800/50 rounded-xl p-4">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <div className="flex items-baseline gap-1 mb-2">
        <span className={`text-2xl font-bold ${colors[color].text}`}>
          {value}{suffix}
        </span>
        {maxValue && (
          <span className="text-slate-500 text-sm">/ {maxValue}</span>
        )}
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div 
          className={`h-full bg-gradient-to-r ${colors[color].bg} rounded-full transition-all duration-500`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  );
}
