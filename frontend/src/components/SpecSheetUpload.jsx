import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, FileJson, AlertCircle, CheckCircle2, 
  Loader2, Rocket, Sparkles, Zap, ChevronDown, ChevronUp,
  Target, Database, Settings, User, Calendar
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../api/client';

export default function SpecSheetUpload({ onProjectStarted }) {
  const [specSheet, setSpecSheet] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [expandedSections, setExpandedSections] = useState({
    metadata: true,
    problem: true,
    data: false,
    budget: false
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setError(null);
    setFileName(file.name);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);
        if (!json.project_metadata || !json.research_problem) {
          throw new Error('Invalid spec sheet structure. Missing required fields.');
        }
        setSpecSheet(json);
        toast.success('Spec sheet loaded successfully!');
      } catch (err) {
        setError(err.message || 'Failed to parse JSON file');
        setSpecSheet(null);
        toast.error('Failed to parse spec sheet');
      }
    };
    reader.readAsText(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/json': ['.json'] },
    maxFiles: 1
  });

  const handleSubmit = async () => {
    if (!specSheet) return;

    setIsSubmitting(true);
    try {
      const response = await apiClient.createProject(specSheet);
      toast.success('Project started! Agents are now working...', {
        icon: '🚀',
        duration: 4000
      });
      onProjectStarted({ ...response.data, specSheet });
    } catch (err) {
      const message = err.response?.data?.detail || 'Failed to start project';
      setError(message);
      toast.error(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const clearSpecSheet = () => {
    setSpecSheet(null);
    setFileName('');
    setError(null);
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {!specSheet ? (
        /* Upload Zone */
        <div
          {...getRootProps()}
          className={`
            relative overflow-hidden rounded-2xl border-2 border-dashed 
            transition-all duration-300 cursor-pointer group
            ${isDragActive 
              ? 'border-sky-400 bg-sky-500/10' 
              : 'border-slate-600 hover:border-sky-500/50 bg-slate-800/30'
            }
          `}
        >
          <input {...getInputProps()} />
          
          <div className="absolute inset-0 opacity-20 pointer-events-none">
            <div className="absolute top-4 right-4 w-32 h-32 bg-sky-500 rounded-full blur-3xl" />
            <div className="absolute bottom-4 left-4 w-24 h-24 bg-orange-500 rounded-full blur-3xl" />
          </div>

          <div className="relative p-16 text-center">
            <div className={`
              w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-6
              transition-all duration-300
              ${isDragActive ? 'bg-sky-500/30 scale-110' : 'bg-slate-700/50 group-hover:bg-sky-500/20'}
            `}>
              <Upload className={`
                w-10 h-10 transition-colors duration-300
                ${isDragActive ? 'text-sky-400' : 'text-slate-400 group-hover:text-sky-400'}
              `} />
            </div>
            <h3 className="text-xl font-semibold text-slate-200 mb-2">
              {isDragActive ? 'Drop your spec sheet here' : 'Upload Spec Sheet'}
            </h3>
            <p className="text-slate-400">
              Drag & drop a JSON file or click to browse
            </p>
          </div>
        </div>
      ) : (
        /* Full Spec Sheet Preview */
        <div className="space-y-4 animate-fade-in-up">
          {/* Header */}
          <div className="flex items-center justify-between p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                <CheckCircle2 className="w-6 h-6 text-emerald-400" />
              </div>
              <div>
                <h3 className="font-semibold text-emerald-300">Spec Sheet Loaded</h3>
                <p className="text-sm text-slate-400">{fileName}</p>
              </div>
            </div>
            <button
              onClick={clearSpecSheet}
              className="px-4 py-2 text-sm rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              Change File
            </button>
          </div>

          {/* Project Metadata Section */}
          <SpecSection
            title="Project Metadata"
            icon={User}
            color="sky"
            expanded={expandedSections.metadata}
            onToggle={() => toggleSection('metadata')}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <DataField 
                label="Project Name" 
                value={specSheet.project_metadata?.project_name} 
              />
              <DataField 
                label="Owner" 
                value={specSheet.project_metadata?.owner || 'Not specified'} 
              />
              <DataField 
                label="Created" 
                value={specSheet.project_metadata?.created || 'Not specified'} 
              />
            </div>
          </SpecSection>

          {/* Research Problem Section */}
          <SpecSection
            title="Research Problem"
            icon={Target}
            color="violet"
            expanded={expandedSections.problem}
            onToggle={() => toggleSection('problem')}
          >
            <div className="space-y-4">
              <DataField 
                label="Problem Statement" 
                value={specSheet.research_problem?.problem_statement}
                fullWidth
              />
              
              <div>
                <p className="text-xs text-slate-400 mb-2">Success Metrics</p>
                <div className="flex flex-wrap gap-2">
                  {specSheet.research_problem?.success_metrics?.map((metric, i) => (
                    <span 
                      key={i}
                      className="px-3 py-1.5 text-sm rounded-lg bg-violet-500/20 text-violet-300 border border-violet-500/30"
                    >
                      {metric}
                    </span>
                  ))}
                </div>
              </div>

              {specSheet.research_problem?.goal_metrics && (
                <div>
                  <p className="text-xs text-slate-400 mb-2">Goal Metrics</p>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {Object.entries(specSheet.research_problem.goal_metrics).map(([key, value]) => (
                      <div key={key} className="p-3 rounded-lg bg-slate-800/50 border border-slate-700">
                        <p className="text-xs text-slate-400 capitalize">{key}</p>
                        <p className="text-lg font-bold text-violet-400">{value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </SpecSection>

          {/* Data Sources Section */}
          <SpecSection
            title="Data Sources"
            icon={Database}
            color="emerald"
            expanded={expandedSections.data}
            onToggle={() => toggleSection('data')}
          >
            <div className="space-y-2">
              {specSheet.data_sources?.length > 0 ? (
                specSheet.data_sources.map((source, i) => (
                  <div 
                    key={i}
                    className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700"
                  >
                    <FileJson className="w-5 h-5 text-emerald-400" />
                    <span className="text-slate-300 font-mono text-sm">{source}</span>
                  </div>
                ))
              ) : (
                <p className="text-slate-500 text-sm">No data sources specified</p>
              )}
            </div>
          </SpecSection>

          {/* Budget Constraints Section */}
          <SpecSection
            title="Budget Constraints"
            icon={Settings}
            color="amber"
            expanded={expandedSections.budget}
            onToggle={() => toggleSection('budget')}
          >
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <DataField 
                label="Max Iterations" 
                value={specSheet.budget_constraints?.max_iterations || 2}
                highlight
              />
              <DataField 
                label="Max Cost" 
                value={specSheet.budget_constraints?.max_financial_cost ? `$${specSheet.budget_constraints.max_financial_cost}` : 'Unlimited'}
              />
              <DataField 
                label="Max Time" 
                value={specSheet.budget_constraints?.max_time_hours ? `${specSheet.budget_constraints.max_time_hours}h` : 'Unlimited'}
              />
              <DataField 
                label="Reporting Period" 
                value={specSheet.budget_constraints?.reporting_period_hours ? `${specSheet.budget_constraints.reporting_period_hours}h` : 'On completion'}
              />
            </div>
          </SpecSection>

          {/* Raw JSON Preview */}
          <details className="group">
            <summary className="cursor-pointer text-sm text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-2 p-2">
              <ChevronDown className="w-4 h-4 group-open:rotate-180 transition-transform" />
              View Raw JSON
            </summary>
            <pre className="mt-2 p-4 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-400 overflow-auto max-h-64 font-mono">
              {JSON.stringify(specSheet, null, 2)}
            </pre>
          </details>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={isSubmitting}
            className={`
              w-full mt-4 py-5 px-6 rounded-xl font-semibold text-white text-lg
              flex items-center justify-center gap-3
              transition-all duration-300
              ${isSubmitting 
                ? 'bg-slate-700 cursor-not-allowed' 
                : 'bg-gradient-to-r from-sky-500 to-violet-500 hover:from-sky-400 hover:to-violet-400 shadow-lg shadow-sky-500/25 hover:shadow-sky-500/40'
              }
            `}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                Connecting to Pipeline...
              </>
            ) : (
              <>
                <Rocket className="w-6 h-6" />
                Launch Research Pipeline
                <Zap className="w-5 h-5 text-yellow-300" />
              </>
            )}
          </button>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30 flex items-start gap-3 animate-fade-in-up">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-300">Error</p>
            <p className="text-sm text-red-400/80">{error}</p>
          </div>
        </div>
      )}

      {/* Features */}
      {!specSheet && (
        <div className="grid grid-cols-3 gap-4 mt-8">
          {[
            { icon: Sparkles, label: 'AI-Powered', desc: 'Multi-agent system' },
            { icon: Zap, label: 'Fast', desc: 'Automated pipeline' },
            { icon: CheckCircle2, label: 'Reliable', desc: 'Self-improving' },
          ].map((feature, i) => (
            <div 
              key={i}
              className="text-center p-4 rounded-xl bg-slate-800/30 border border-slate-700/50"
            >
              <feature.icon className="w-5 h-5 text-sky-400 mx-auto mb-2" />
              <p className="text-sm font-medium text-slate-200">{feature.label}</p>
              <p className="text-xs text-slate-500">{feature.desc}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SpecSection({ title, icon: Icon, color, expanded, onToggle, children }) {
  const colors = {
    sky: { bg: 'bg-sky-500/10', border: 'border-sky-500/30', icon: 'text-sky-400', title: 'text-sky-300' },
    violet: { bg: 'bg-violet-500/10', border: 'border-violet-500/30', icon: 'text-violet-400', title: 'text-violet-300' },
    emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', icon: 'text-emerald-400', title: 'text-emerald-300' },
    amber: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', icon: 'text-amber-400', title: 'text-amber-300' }
  };

  const c = colors[color];

  return (
    <div className={`rounded-xl border ${c.border} overflow-hidden`}>
      <button
        onClick={onToggle}
        className={`w-full flex items-center justify-between p-4 ${c.bg} hover:brightness-110 transition-all`}
      >
        <div className="flex items-center gap-3">
          <Icon className={`w-5 h-5 ${c.icon}`} />
          <span className={`font-medium ${c.title}`}>{title}</span>
        </div>
        {expanded ? (
          <ChevronUp className="w-5 h-5 text-slate-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-slate-400" />
        )}
      </button>
      {expanded && (
        <div className="p-4 bg-slate-900/30">
          {children}
        </div>
      )}
    </div>
  );
}

function DataField({ label, value, fullWidth, highlight }) {
  return (
    <div className={fullWidth ? 'col-span-full' : ''}>
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className={`text-sm ${highlight ? 'text-sky-400 font-bold text-lg' : 'text-slate-200'}`}>
        {value || 'Not specified'}
      </p>
    </div>
  );
}
