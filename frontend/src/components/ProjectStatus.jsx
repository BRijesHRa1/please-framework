import { useState, useEffect } from 'react';
import { 
  CheckCircle2, Loader2, XCircle, Clock, 
  Brain, Cpu, BarChart3, FileText, Clipboard,
  Wifi, WifiOff, ChevronDown, ChevronUp, ArrowRight,
  Zap, Server, Database, Code, Terminal, Target,
  AlertTriangle, TrendingUp, Package, ListChecks,
  RefreshCw, Activity, GitBranch
} from 'lucide-react';
import { apiClient } from '../api/client';

const agentIcons = {
  planner: Clipboard,
  learner: Brain,
  executor: Cpu,
  assessor: BarChart3,
  pm: FileText
};

const agentLabels = {
  planner: 'Planner Agent',
  learner: 'Learner Agent',
  executor: 'Executor Agent',
  assessor: 'Assessor Agent',
  pm: 'PM Agent'
};

const agentDescriptions = {
  planner: 'Analyzing spec sheet and creating task breakdown...',
  learner: 'Gathering research resources and methodology...',
  executor: 'Running ML pipeline and training models...',
  assessor: 'Evaluating results and analyzing gaps...',
  pm: 'Generating final report and recommendations...'
};

const agentEmojis = {
  planner: '🤖',
  learner: '🧠',
  executor: '⚙️',
  assessor: '📊',
  pm: '📋'
};

export default function ProjectStatus({ project, onComplete }) {
  const [status, setStatus] = useState(null);
  const [projectDetail, setProjectDetail] = useState(null);
  const [error, setError] = useState(null);
  const [mcpConnected, setMcpConnected] = useState(true);
  const [expandedAgents, setExpandedAgents] = useState({});
  const [connectionLog, setConnectionLog] = useState([]);
  const [previousAgent, setPreviousAgent] = useState(null);

  const addLog = (message, type = 'info') => {
    setConnectionLog(prev => {
      // Avoid duplicate consecutive messages
      if (prev.length > 0 && prev[prev.length - 1].message === message) {
        return prev;
      }
      return [...prev, { 
        message, 
        type, 
        time: new Date().toLocaleTimeString() 
      }].slice(-20);
    });
  };

  // Auto-expand running agent
  useEffect(() => {
    if (status?.current_agent && status.current_agent !== previousAgent) {
      setPreviousAgent(status.current_agent);
      // Auto-expand the current running agent
      setExpandedAgents(prev => ({
        ...prev,
        [status.current_agent]: true
      }));
    }
  }, [status?.current_agent, previousAgent]);

  useEffect(() => {
    if (!project?.project_id) return;

    addLog('🔌 Connecting to PLEASe Backend...', 'info');
    setTimeout(() => addLog('✅ MCP Connection established', 'success'), 500);
    setTimeout(() => addLog('🚀 Pipeline initialized', 'success'), 1000);

    const pollStatus = async () => {
      try {
        const [statusRes, detailRes] = await Promise.all([
          apiClient.getProjectStatus(project.project_id),
          apiClient.getProject(project.project_id).catch(() => null)
        ]);
        
        setStatus(statusRes.data);
        if (detailRes) setProjectDetail(detailRes.data);
        setMcpConnected(true);

        const currentAgent = statusRes.data.current_agent;
        const currentCycle = statusRes.data.current_cycle;

        // Add detailed agent logs
        if (currentAgent && currentAgent !== previousAgent) {
          const emoji = agentEmojis[currentAgent] || '🔄';
          const agentLabel = agentLabels[currentAgent];
          
          if (currentCycle > 1) {
            addLog(`🔄 Cycle ${currentCycle} - ${emoji} ${agentLabel} starting...`, 'agent');
          } else {
            addLog(`${emoji} ${agentLabel} starting...`, 'agent');
          }
        }

        // Check for completed agents and log their output
        if (detailRes?.data?.agent_outputs) {
          for (const [cycleKey, outputs] of Object.entries(detailRes.data.agent_outputs)) {
            for (const output of outputs) {
              const logKey = `${cycleKey}_${output.agent_name}_complete`;
              if (!connectionLog.some(l => l.key === logKey) && output.execution_time_seconds) {
                const emoji = agentEmojis[output.agent_name] || '✅';
                const agentLabel = agentLabels[output.agent_name];
                addLog(`${emoji} ${agentLabel} completed (${output.execution_time_seconds.toFixed(1)}s)`, 'success');
              }
            }
          }
        }

        if (statusRes.data.status === 'completed') {
          addLog('🎉 Pipeline completed successfully!', 'success');
          addLog('📄 Report generated and saved to database', 'success');
          onComplete?.(statusRes.data);
        } else if (statusRes.data.status === 'failed') {
          addLog('❌ Pipeline failed!', 'error');
        }
      } catch (err) {
        setError(err.message);
        setMcpConnected(false);
        addLog('⚠️ Connection error - retrying...', 'error');
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 2000);
    return () => clearInterval(interval);
  }, [project?.project_id, onComplete]);

  const toggleAgent = (agent) => {
    setExpandedAgents(prev => ({ ...prev, [agent]: !prev[agent] }));
  };

  const agents = ['planner', 'learner', 'executor', 'assessor', 'pm'];

  const getAgentStatus = (agent) => {
    if (!status?.current_agent) return 'pending';
    const currentIndex = agents.indexOf(status.current_agent);
    const agentIndex = agents.indexOf(agent);
    
    if (status.status === 'completed') return 'completed';
    if (status.status === 'failed') {
      if (agentIndex < currentIndex) return 'completed';
      if (agentIndex === currentIndex) return 'failed';
      return 'pending';
    }
    if (agentIndex < currentIndex) return 'completed';
    if (agentIndex === currentIndex) return 'running';
    return 'pending';
  };

  const getAgentOutput = (agentName) => {
    if (!projectDetail?.agent_outputs) return null;
    for (const cycleOutputs of Object.values(projectDetail.agent_outputs)) {
      const output = cycleOutputs.find(o => o.agent_name === agentName);
      if (output) return output;
    }
    return null;
  };

  // Get cycle comparison data
  const getCycleComparison = () => {
    if (!projectDetail?.agent_outputs) return null;
    
    const cycle1Executor = projectDetail.agent_outputs.cycle_1?.find(o => o.agent_name === 'executor');
    const cycle2Executor = projectDetail.agent_outputs.cycle_2?.find(o => o.agent_name === 'executor');
    
    if (!cycle1Executor || !cycle2Executor) return null;
    
    const c1 = cycle1Executor.output_data?.baseline_results || {};
    const c2 = cycle2Executor.output_data?.baseline_results || {};
    
    return { cycle1: c1, cycle2: c2 };
  };

  const cycleComparison = getCycleComparison();

  if (!status) {
    return (
      <div className="flex flex-col items-center justify-center py-12 gap-4">
        <Loader2 className="w-10 h-10 animate-spin text-sky-400" />
        <p className="text-slate-400">Initializing pipeline...</p>
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto animate-fade-in-up">
      {/* Cycle Progress Banner - Prominent Display */}
      <div className="mb-6 p-4 rounded-2xl bg-gradient-to-r from-slate-800 via-slate-800/80 to-slate-800 border border-slate-700 overflow-hidden relative">
        <div className="absolute inset-0 bg-gradient-to-r from-sky-500/10 via-violet-500/10 to-orange-500/10 opacity-50" />
        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`
              w-16 h-16 rounded-2xl flex items-center justify-center relative
              ${status.status === 'running' 
                ? 'bg-gradient-to-br from-sky-500/30 to-violet-500/30' 
                : status.status === 'completed'
                ? 'bg-gradient-to-br from-emerald-500/30 to-emerald-600/30'
                : 'bg-gradient-to-br from-red-500/30 to-red-600/30'}
            `}>
              {status.status === 'running' ? (
                <>
                  <RefreshCw className="w-7 h-7 text-sky-400 animate-spin" />
                  <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-sky-500 flex items-center justify-center">
                    <span className="text-[10px] font-bold text-white">{status.current_cycle}</span>
                  </div>
                </>
              ) : status.status === 'completed' ? (
                <>
                  <CheckCircle2 className="w-7 h-7 text-emerald-400" />
                  <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 flex items-center justify-center">
                    <span className="text-[10px] font-bold text-white">✓</span>
                  </div>
                </>
              ) : (
                <XCircle className="w-7 h-7 text-red-400" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-bold text-white">
                  Cycle {status.current_cycle}
                </h3>
                {status.status === 'running' && (
                  <span className="px-2 py-0.5 text-xs font-medium bg-sky-500/20 text-sky-300 rounded-full border border-sky-500/30 animate-pulse">
                    RUNNING
                  </span>
                )}
                {status.status === 'completed' && (
                  <span className="px-2 py-0.5 text-xs font-medium bg-emerald-500/20 text-emerald-300 rounded-full border border-emerald-500/30">
                    COMPLETED
                  </span>
                )}
              </div>
              <p className="text-sm text-slate-400 mt-0.5">
                {status.current_agent ? (
                  <>Currently: <span className="text-sky-400 font-medium">{agentLabels[status.current_agent]}</span></>
                ) : status.status === 'completed' ? (
                  'All agents finished successfully'
                ) : (
                  'Waiting to start...'
                )}
              </p>
            </div>
          </div>
          
          {/* Cycle Timeline */}
          <div className="flex items-center gap-2">
            {[1, 2].map((cycleNum) => (
              <div 
                key={cycleNum}
                className={`
                  flex items-center gap-1 px-3 py-2 rounded-lg transition-all
                  ${cycleNum < status.current_cycle 
                    ? 'bg-emerald-500/20 border border-emerald-500/30' 
                    : cycleNum === status.current_cycle
                    ? status.status === 'running' 
                      ? 'bg-sky-500/20 border border-sky-500/30'
                      : status.status === 'completed'
                      ? 'bg-emerald-500/20 border border-emerald-500/30'
                      : 'bg-slate-700/50 border border-slate-600/30'
                    : 'bg-slate-800/50 border border-slate-700/30'}
                `}
              >
                <GitBranch className={`w-4 h-4 ${
                  cycleNum < status.current_cycle 
                    ? 'text-emerald-400' 
                    : cycleNum === status.current_cycle 
                    ? status.status === 'running' ? 'text-sky-400' : 'text-emerald-400'
                    : 'text-slate-500'
                }`} />
                <span className={`text-sm font-medium ${
                  cycleNum < status.current_cycle 
                    ? 'text-emerald-300' 
                    : cycleNum === status.current_cycle 
                    ? status.status === 'running' ? 'text-sky-300' : 'text-emerald-300'
                    : 'text-slate-500'
                }`}>
                  C{cycleNum}
                </span>
                {cycleNum < status.current_cycle && (
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                )}
                {cycleNum === status.current_cycle && status.status === 'running' && (
                  <Activity className="w-3 h-3 text-sky-400 animate-pulse" />
                )}
              </div>
            ))}
          </div>
        </div>
        
        {/* Cycle Progress Bar */}
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <div className="flex justify-between text-xs mb-2">
            <span className="text-slate-400">Cycle {status.current_cycle} Progress</span>
            <span className="text-sky-400 font-mono">{Math.round(status.progress_percent)}%</span>
          </div>
          <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all duration-500 ${
                status.status === 'completed' ? 'bg-gradient-to-r from-emerald-500 to-emerald-400'
                  : status.status === 'failed' ? 'bg-gradient-to-r from-red-500 to-red-400'
                  : 'bg-gradient-to-r from-sky-500 via-violet-500 to-sky-400 bg-size-200 animate-gradient'
              }`}
              style={{ width: `${status.progress_percent}%` }}
            />
          </div>
        </div>
      </div>

      {/* MCP Connection Status */}
      <div className={`
        flex items-center justify-between p-4 rounded-xl mb-6 border
        ${mcpConnected 
          ? 'bg-emerald-500/10 border-emerald-500/30' 
          : 'bg-red-500/10 border-red-500/30'
        }
      `}>
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${mcpConnected ? 'bg-emerald-500/20' : 'bg-red-500/20'}`}>
            {mcpConnected ? <Wifi className="w-5 h-5 text-emerald-400" /> : <WifiOff className="w-5 h-5 text-red-400" />}
          </div>
          <div>
            <p className={`font-medium ${mcpConnected ? 'text-emerald-300' : 'text-red-300'}`}>MCP Connection</p>
            <p className="text-xs text-slate-400">{mcpConnected ? 'Connected to PLEASe Backend' : 'Connection lost - retrying...'}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Server className="w-4 h-4 text-slate-400" />
          <span className="text-xs text-slate-400 font-mono">localhost:8000</span>
          <div className={`w-2 h-2 rounded-full ${mcpConnected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
        </div>
      </div>

      {/* Section Header */}
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <ListChecks className="w-5 h-5 text-sky-400" />
          Agent Pipeline
        </h3>
        <p className="text-sm text-slate-400 mt-1">Click on each agent to view detailed output</p>
      </div>

      {/* Agent Pipeline */}
      <div className="space-y-3">
        {agents.map((agent) => {
          const agentStatus = getAgentStatus(agent);
          const Icon = agentIcons[agent];
          const output = getAgentOutput(agent);
          const isExpanded = expandedAgents[agent];
          const outputData = output?.output_data || {};
          
          return (
            <div
              key={agent}
              className={`rounded-xl border transition-all overflow-hidden ${
                agentStatus === 'running' ? 'border-sky-500/50 shadow-lg shadow-sky-500/10' 
                  : agentStatus === 'completed' ? 'border-emerald-500/30'
                  : agentStatus === 'failed' ? 'border-red-500/30'
                  : 'border-slate-700/50'
              }`}
            >
              {/* Agent Header */}
              <button
                onClick={() => toggleAgent(agent)}
                className={`w-full flex items-center gap-4 p-4 transition-all ${
                  agentStatus === 'running' ? 'bg-sky-500/10' 
                    : agentStatus === 'completed' ? 'bg-emerald-500/5 hover:bg-emerald-500/10'
                    : agentStatus === 'failed' ? 'bg-red-500/5'
                    : 'bg-slate-800/30 hover:bg-slate-800/50'
                }`}
              >
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                  agentStatus === 'running' ? 'bg-sky-500/20'
                    : agentStatus === 'completed' ? 'bg-emerald-500/20'
                    : agentStatus === 'failed' ? 'bg-red-500/20'
                    : 'bg-slate-700/50'
                }`}>
                  {agentStatus === 'running' ? (
                    <Loader2 className="w-6 h-6 text-sky-400 animate-spin" />
                  ) : agentStatus === 'completed' ? (
                    <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                  ) : agentStatus === 'failed' ? (
                    <XCircle className="w-6 h-6 text-red-400" />
                  ) : (
                    <Icon className="w-6 h-6 text-slate-500" />
                  )}
                </div>
                
                <div className="flex-1 text-left">
                  <p className={`font-semibold ${
                    agentStatus === 'running' ? 'text-sky-300'
                      : agentStatus === 'completed' ? 'text-emerald-300'
                      : agentStatus === 'failed' ? 'text-red-300'
                      : 'text-slate-400'
                  }`}>
                    {agentLabels[agent]}
                  </p>
                  <p className="text-xs text-slate-500">
                    {agentStatus === 'running' ? agentDescriptions[agent]
                      : agentStatus === 'completed' ? `Completed in ${output?.execution_time_seconds?.toFixed(2) || '?'}s`
                      : agentStatus === 'failed' ? 'Failed'
                      : 'Waiting...'}
                  </p>
                </div>

                {(agentStatus === 'completed' || agentStatus === 'running') && (
                  <div className="flex items-center gap-2 text-slate-400">
                    {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                  </div>
                )}

                {agentStatus === 'running' && (
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />
                    <div className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" style={{ animationDelay: '300ms' }} />
                  </div>
                )}
              </button>

              {/* Agent Details (Expanded) - Show for both running and completed */}
              {isExpanded && (agentStatus === 'completed' || agentStatus === 'running') && (
                <div className="p-4 bg-slate-900/50 border-t border-slate-700/50 space-y-4">
                  
                  {/* Running Agent Info */}
                  {agentStatus === 'running' && (
                    <div className="p-4 rounded-xl bg-sky-500/10 border border-sky-500/30">
                      <div className="flex items-center gap-3">
                        <Loader2 className="w-6 h-6 text-sky-400 animate-spin" />
                        <div>
                          <p className="text-sky-300 font-medium">{agentDescriptions[agent]}</p>
                          <p className="text-xs text-slate-400 mt-1">
                            Please wait while the agent processes the data...
                          </p>
                        </div>
                      </div>
                      <div className="mt-4 flex items-center gap-4 text-xs text-slate-400">
                        <div className="flex items-center gap-1">
                          <Activity className="w-3 h-3 animate-pulse text-sky-400" />
                          <span>Processing</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Zap className="w-3 h-3" />
                          <span>llama3.2:3b</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <RefreshCw className="w-3 h-3" />
                          <span>Cycle {status.current_cycle}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Completed Agent Details */}
              {isExpanded && agentStatus === 'completed' && outputData && Object.keys(outputData).length > 0 && (
                <div className="p-4 bg-slate-900/50 border-t border-slate-700/50 space-y-4">
                  
                  {/* PLANNER OUTPUT */}
                  {agent === 'planner' && outputData && (
                    <>
                      <OutputSection title="Summary" icon={FileText} color="sky">
                        <p className="text-slate-300 text-sm">{outputData.summary}</p>
                      </OutputSection>

                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        <MetricCard label="Duration" value={outputData.estimated_duration} />
                        <MetricCard label="GPU Hours" value={`${outputData.total_gpu_estimate}h`} />
                        <MetricCard label="Tasks" value={outputData.tasks?.length || 0} />
                      </div>

                      {outputData.tasks && (
                        <OutputSection title={`Tasks (${outputData.tasks.length})`} icon={ListChecks} color="violet">
                          <div className="space-y-2 max-h-48 overflow-y-auto">
                            {outputData.tasks.map((task, i) => (
                              <div key={i} className="p-2 bg-slate-800/50 rounded-lg">
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
                                <div className="flex gap-4 mt-1 text-xs text-slate-500">
                                  <span>GPU: {task.gpu_hours}h</span>
                                  <span>Deps: {task.dependencies?.join(', ') || 'None'}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </OutputSection>
                      )}

                      {outputData.risk_factors?.length > 0 && (
                        <OutputSection title="Risk Factors" icon={AlertTriangle} color="amber">
                          <ul className="space-y-1">
                            {outputData.risk_factors.map((risk, i) => (
                              <li key={i} className="text-sm text-amber-300 flex items-start gap-2">
                                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                                {risk}
                              </li>
                            ))}
                          </ul>
                        </OutputSection>
                      )}

                      {outputData.mcp_query && (
                        <OutputSection title="MCP Query (for Learner)" icon={Server} color="emerald">
                          <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                            <p className="text-sm text-emerald-300 italic">"{outputData.mcp_query}"</p>
                            <p className="text-xs text-slate-500 mt-2">↳ This query will be passed to the MCP server by the Learner agent</p>
                          </div>
                        </OutputSection>
                      )}
                    </>
                  )}

                  {/* LEARNER OUTPUT */}
                  {agent === 'learner' && outputData && (
                    <>
                      <OutputSection title="Summary" icon={FileText} color="sky">
                        <p className="text-slate-300 text-sm">{outputData.summary}</p>
                      </OutputSection>

                      {outputData.key_genes?.length > 0 && (
                        <OutputSection title="Key Genes" icon={Database} color="violet">
                          <div className="flex flex-wrap gap-2">
                            {outputData.key_genes.slice(0, 10).map((gene, i) => (
                              <span key={i} className="px-2 py-1 text-xs rounded bg-violet-500/20 text-violet-300 font-mono">{gene}</span>
                            ))}
                            {outputData.key_genes.length > 10 && (
                              <span className="text-xs text-slate-500">+{outputData.key_genes.length - 10} more</span>
                            )}
                          </div>
                        </OutputSection>
                      )}

                      {outputData.datasets?.length > 0 && (
                        <OutputSection title="Datasets" icon={Database} color="emerald">
                          <div className="space-y-1">
                            {outputData.datasets.map((ds, i) => (
                              <div key={i} className="text-sm text-slate-300">{ds}</div>
                            ))}
                          </div>
                        </OutputSection>
                      )}

                      {outputData.tools?.length > 0 && (
                        <OutputSection title={`Tools (${outputData.tools.length})`} icon={Code} color="amber">
                          <div className="space-y-2">
                            {outputData.tools.map((tool, i) => (
                              <div key={i} className="p-2 bg-slate-800/50 rounded-lg">
                                <span className="text-sm font-medium text-amber-300">{tool.name}</span>
                                <p className="text-xs text-slate-400">{tool.purpose}</p>
                              </div>
                            ))}
                          </div>
                        </OutputSection>
                      )}

                      {outputData.model_suggestions?.length > 0 && (
                        <OutputSection title="Model Suggestions" icon={Brain} color="sky">
                          <ul className="space-y-1">
                            {outputData.model_suggestions.map((s, i) => (
                              <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                                <ArrowRight className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                                {s}
                              </li>
                            ))}
                          </ul>
                        </OutputSection>
                      )}

                      {outputData.preprocessing_notes && (
                        <OutputSection title="Preprocessing Notes" icon={FileText} color="slate">
                          <p className="text-sm text-slate-400">{outputData.preprocessing_notes}</p>
                        </OutputSection>
                      )}

                      {outputData.references?.length > 0 && (
                        <OutputSection title="References" icon={FileText} color="sky">
                          <ul className="space-y-1">
                            {outputData.references.map((ref, i) => (
                              <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                                <span className="text-xs text-slate-500">{i + 1}.</span>
                                {ref}
                              </li>
                            ))}
                          </ul>
                        </OutputSection>
                      )}
                    </>
                  )}

                  {/* EXECUTOR OUTPUT */}
                  {agent === 'executor' && outputData && (
                    <>
                      <OutputSection title="Summary" icon={FileText} color="sky">
                        <p className="text-slate-300 text-sm">{outputData.summary}</p>
                      </OutputSection>

                      <div className="grid grid-cols-2 gap-3">
                        <MetricCard label="Tasks Completed" value={`${outputData.tasks_completed?.length || 0}/${project.specSheet?.budget_constraints?.max_iterations || '?'}`} />
                        <MetricCard label="Artifacts" value={outputData.artifacts_generated?.length || 0} />
                      </div>

                      {outputData.baseline_results && (
                        <OutputSection title="Baseline Results" icon={BarChart3} color="violet">
                          <div className="grid grid-cols-3 gap-3">
                            {Object.entries(outputData.baseline_results).map(([key, value]) => (
                              <div key={key} className="p-2 bg-slate-800/50 rounded-lg text-center">
                                <p className="text-xs text-slate-400 capitalize">{key}</p>
                                <p className="text-lg font-bold text-violet-400">
                                  {typeof value === 'number' ? value.toFixed(4) : value}
                                </p>
                              </div>
                            ))}
                          </div>
                        </OutputSection>
                      )}

                      {outputData.model_results && (
                        <OutputSection title="Model Results" icon={TrendingUp} color="emerald">
                          <div className="grid grid-cols-3 gap-3">
                            {Object.entries(outputData.model_results).map(([key, value]) => (
                              <div key={key} className="p-2 bg-slate-800/50 rounded-lg text-center">
                                <p className="text-xs text-slate-400 capitalize">{key}</p>
                                <p className="text-lg font-bold text-emerald-400">
                                  {typeof value === 'number' ? value.toFixed(4) : value}
                                </p>
                              </div>
                            ))}
                          </div>
                        </OutputSection>
                      )}

                      {outputData.artifacts_generated?.length > 0 && (
                        <OutputSection title={`Artifacts (${outputData.artifacts_generated.length})`} icon={Package} color="amber">
                          <div className="space-y-1 max-h-32 overflow-y-auto font-mono text-xs">
                            {outputData.artifacts_generated.slice(0, 10).map((a, i) => (
                              <div key={i} className="text-slate-400">{a.split('/').pop()}</div>
                            ))}
                            {outputData.artifacts_generated.length > 10 && (
                              <div className="text-slate-500">+{outputData.artifacts_generated.length - 10} more files</div>
                            )}
                          </div>
                        </OutputSection>
                      )}
                    </>
                  )}

                  {/* ASSESSOR OUTPUT */}
                  {agent === 'assessor' && outputData && (
                    <>
                      <OutputSection title="Summary" icon={FileText} color="sky">
                        <p className="text-slate-300 text-sm">{outputData.summary}</p>
                      </OutputSection>

                      <div className="grid grid-cols-3 gap-3">
                        <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                          <p className="text-xs text-slate-400">Satisfaction</p>
                          <p className="text-2xl font-bold text-sky-400">{outputData.satisfaction_score}/5</p>
                        </div>
                        <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                          <p className="text-xs text-slate-400">NIH Score</p>
                          <p className="text-2xl font-bold text-violet-400">{outputData.nih_score}/9</p>
                        </div>
                        <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                          <p className="text-xs text-slate-400">Status</p>
                          <p className={`text-lg font-bold ${outputData.overall_status === 'SUCCESS' ? 'text-emerald-400' : 'text-amber-400'}`}>
                            {outputData.overall_status}
                          </p>
                        </div>
                      </div>

                      {outputData.gap_analysis?.length > 0 && (
                        <OutputSection title="Gap Analysis" icon={Target} color="violet">
                          <div className="space-y-2">
                            {outputData.gap_analysis.map((gap, i) => (
                              <div key={i} className="flex items-center gap-3 p-2 bg-slate-800/50 rounded-lg">
                                {gap.status === 'EXCEEDED' ? (
                                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                                ) : gap.status === 'MET' ? (
                                  <CheckCircle2 className="w-5 h-5 text-amber-400" />
                                ) : (
                                  <XCircle className="w-5 h-5 text-red-400" />
                                )}
                                <div className="flex-1">
                                  <span className="text-sm font-medium text-slate-200">{gap.metric}</span>
                                  <div className="flex gap-3 text-xs text-slate-400">
                                    <span>Goal: {gap.goal?.toFixed(2)}</span>
                                    <span>Achieved: {gap.achieved?.toFixed(4)}</span>
                                    <span className={gap.gap >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                      Gap: {gap.gap >= 0 ? '+' : ''}{gap.gap?.toFixed(4)}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </OutputSection>
                      )}

                      {outputData.strengths?.length > 0 && (
                        <OutputSection title="Strengths" icon={CheckCircle2} color="emerald">
                          <ul className="space-y-1">
                            {outputData.strengths.map((s, i) => (
                              <li key={i} className="text-sm text-emerald-300 flex items-start gap-2">
                                <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                                {s}
                              </li>
                            ))}
                          </ul>
                        </OutputSection>
                      )}

                      {outputData.weaknesses?.length > 0 && (
                        <OutputSection title="Weaknesses" icon={AlertTriangle} color="amber">
                          <ul className="space-y-1">
                            {outputData.weaknesses.map((w, i) => (
                              <li key={i} className="text-sm text-amber-300 flex items-start gap-2">
                                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                                {w}
                              </li>
                            ))}
                          </ul>
                        </OutputSection>
                      )}

                      {outputData.recommendations?.length > 0 && (
                        <OutputSection title="Recommendations" icon={ArrowRight} color="sky">
                          <ul className="space-y-1">
                            {outputData.recommendations.map((r, i) => (
                              <li key={i} className="text-sm text-sky-300 flex items-start gap-2">
                                <ArrowRight className="w-4 h-4 shrink-0 mt-0.5" />
                                {r}
                              </li>
                            ))}
                          </ul>
                        </OutputSection>
                      )}

                      <div className="p-3 bg-slate-800/50 rounded-lg flex items-center gap-2">
                        <span className="text-sm text-slate-400">Should Continue:</span>
                        <span className={`font-medium ${outputData.should_continue ? 'text-amber-400' : 'text-emerald-400'}`}>
                          {outputData.should_continue ? 'No - More iterations needed' : 'No - Goals achieved'}
                        </span>
                      </div>
                    </>
                  )}

                  {/* PM OUTPUT */}
                  {agent === 'pm' && outputData && (
                    <>
                      <OutputSection title="Report Summary" icon={FileText} color="sky">
                        <p className="text-slate-300 text-sm">{outputData.report_summary}</p>
                      </OutputSection>

                      <div className="grid grid-cols-3 gap-3">
                        <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                          <p className="text-xs text-slate-400">NIH Score</p>
                          <p className="text-2xl font-bold text-emerald-400">{outputData.final_nih_score}/9</p>
                        </div>
                        <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                          <p className="text-xs text-slate-400">Satisfaction</p>
                          <p className="text-2xl font-bold text-sky-400">{outputData.final_satisfaction}/5</p>
                        </div>
                        <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                          <p className="text-xs text-slate-400">A.G.E. Score</p>
                          <p className="text-2xl font-bold text-violet-400">{outputData.overall_age_score}/10</p>
                        </div>
                      </div>

                      {outputData.age_scores && (
                        <OutputSection title="A.G.E. Scores by Agent" icon={TrendingUp} color="violet">
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                            {Object.entries(outputData.age_scores).map(([name, data]) => (
                              <div key={name} className="p-2 bg-slate-800/50 rounded-lg">
                                <div className="flex items-center justify-between">
                                  <span className="text-xs text-slate-400 capitalize">{name}</span>
                                  <span className="text-sm font-bold text-violet-400">{data.score}/10</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </OutputSection>
                      )}

                      <div className={`p-3 rounded-lg text-center ${
                        outputData.project_status === 'SUCCESS' ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-amber-500/10 border border-amber-500/30'
                      }`}>
                        <span className={`text-lg font-bold ${outputData.project_status === 'SUCCESS' ? 'text-emerald-400' : 'text-amber-400'}`}>
                          Project Status: {outputData.project_status}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Cycle Comparison (when Cycle 2 completes) */}
      {cycleComparison && (
        <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-violet-500/10 to-sky-500/10 border border-violet-500/30">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-violet-400" />
            <h3 className="text-lg font-semibold text-white">Improvement Comparison: Cycle 1 → Cycle 2</h3>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-2 px-3 text-slate-400 font-medium">Metric</th>
                  <th className="text-center py-2 px-3 text-slate-400 font-medium">Cycle 1</th>
                  <th className="text-center py-2 px-3 text-slate-400 font-medium">Cycle 2</th>
                  <th className="text-center py-2 px-3 text-slate-400 font-medium">Change</th>
                </tr>
              </thead>
              <tbody>
                {['accuracy', 'f1', 'roc_auc', 'precision', 'recall'].map((metric) => {
                  const c1Val = cycleComparison.cycle1[metric];
                  const c2Val = cycleComparison.cycle2[metric];
                  if (c1Val === undefined || c2Val === undefined) return null;
                  
                  const change = c2Val - c1Val;
                  const isImproved = change > 0;
                  
                  return (
                    <tr key={metric} className="border-b border-slate-700/50 hover:bg-slate-800/50">
                      <td className="py-2 px-3 text-slate-300 capitalize font-medium">{metric}</td>
                      <td className="py-2 px-3 text-center text-slate-400 font-mono">{c1Val.toFixed(4)}</td>
                      <td className="py-2 px-3 text-center text-sky-400 font-mono font-medium">{c2Val.toFixed(4)}</td>
                      <td className="py-2 px-3 text-center">
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono ${
                          isImproved ? 'bg-emerald-500/20 text-emerald-400' : change < 0 ? 'bg-red-500/20 text-red-400' : 'bg-slate-500/20 text-slate-400'
                        }`}>
                          {isImproved ? '📈' : change < 0 ? '📉' : '➡️'} {change >= 0 ? '+' : ''}{change.toFixed(4)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Connection Log - Enhanced */}
      <div className="mt-6 p-4 rounded-xl bg-slate-900 border border-slate-700">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-slate-400" />
            <span className="text-sm font-medium text-slate-300">Pipeline Log</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Activity className="w-3 h-3" />
            <span>{connectionLog.length} entries</span>
          </div>
        </div>
        <div className="space-y-1 max-h-48 overflow-y-auto font-mono text-xs bg-slate-950 rounded-lg p-3">
          {connectionLog.map((log, i) => (
            <div key={i} className="flex items-start gap-2 py-0.5">
              <span className="text-slate-600 shrink-0">[{log.time}]</span>
              <span className={
                log.type === 'success' ? 'text-emerald-400' :
                log.type === 'error' ? 'text-red-400' :
                log.type === 'agent' ? 'text-sky-400' :
                'text-slate-400'
              }>
                {log.message}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Model Info */}
      <div className="mt-4 p-4 rounded-xl bg-slate-800/30 border border-slate-700/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Clock className="w-5 h-5 text-slate-400" />
          <p className="text-sm text-slate-300">
            Total Cycles: <span className="font-medium text-sky-400">{status.current_cycle}</span>
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-1">
            <Zap className="w-4 h-4" />
            <span>Model: llama3.2:3b</span>
          </div>
          <div className="flex items-center gap-1">
            <Database className="w-4 h-4" />
            <span>please.db</span>
          </div>
        </div>
      </div>

      {/* View Report Button */}
      {status.status === 'completed' && (
        <button
          onClick={() => onComplete?.(status)}
          className="w-full mt-6 py-4 px-6 rounded-xl font-semibold text-white
            bg-gradient-to-r from-emerald-500 to-emerald-600 
            hover:from-emerald-400 hover:to-emerald-500
            flex items-center justify-center gap-3 transition-all shadow-lg shadow-emerald-500/25"
        >
          <FileText className="w-5 h-5" />
          View Full Report
        </button>
      )}
    </div>
  );
}

function OutputSection({ title, icon: Icon, color, children }) {
  const colors = {
    sky: 'text-sky-400 border-sky-500/30',
    violet: 'text-violet-400 border-violet-500/30',
    emerald: 'text-emerald-400 border-emerald-500/30',
    amber: 'text-amber-400 border-amber-500/30',
    slate: 'text-slate-400 border-slate-500/30'
  };

  return (
    <div className={`border-l-2 ${colors[color]} pl-3`}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${colors[color].split(' ')[0]}`} />
        <span className="text-sm font-medium text-slate-300">{title}</span>
      </div>
      {children}
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="p-3 bg-slate-800/50 rounded-lg text-center">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="text-lg font-bold text-white">{value}</p>
    </div>
  );
}
