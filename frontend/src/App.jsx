import { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import { Beaker, Sparkles, Zap, Brain, BarChart3, FileText } from 'lucide-react';
import Header from './components/Header';
import SpecSheetUpload from './components/SpecSheetUpload';
import ProjectStatus from './components/ProjectStatus';
import Dashboard from './components/Dashboard';
import ProjectDetail from './components/ProjectDetail';
import Documentation from './components/Documentation';

function App() {
  const [currentProject, setCurrentProject] = useState(null);
  const [showStatus, setShowStatus] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [viewingProjectId, setViewingProjectId] = useState(null);

  const handleProjectStarted = (project) => {
    setCurrentProject(project);
    setShowStatus(true);
  };

  const handleComplete = (status) => {
    console.log('Project completed:', status);
    // Navigate to project detail to view the full report
    if (currentProject?.project_id) {
      setViewingProjectId(currentProject.project_id);
      setShowStatus(false);
    }
  };

  const resetToUpload = () => {
    setCurrentProject(null);
    setShowStatus(false);
  };

  const handleViewProject = (projectId) => {
    setViewingProjectId(projectId);
  };

  return (
    <div className="min-h-screen grid-pattern">
      <Toaster 
        position="top-right"
        toastOptions={{
          className: 'glass !bg-slate-800 !text-white',
          duration: 3000,
        }}
      />
      
      <Header 
        onDashboardClick={() => setShowDashboard(true)}
        onDocsClick={() => setShowDocs(true)}
      />

      {/* Documentation Modal */}
      {showDocs && (
        <Documentation onClose={() => setShowDocs(false)} />
      )}

      {/* Dashboard Modal */}
      {showDashboard && !viewingProjectId && !showDocs && (
        <Dashboard 
          onViewProject={handleViewProject}
          onClose={() => setShowDashboard(false)}
        />
      )}

      {/* Project Detail Modal */}
      {viewingProjectId && !showDocs && (
        <ProjectDetail 
          projectId={viewingProjectId}
          onBack={() => setViewingProjectId(null)}
        />
      )}
      
      {/* Hero Section */}
      <main className="pt-28 pb-16 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Title Section */}
          <div className="text-center mb-12 animate-fade-in-up">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-sky-500/10 border border-sky-500/30 mb-6">
              <Sparkles className="w-4 h-4 text-sky-400" />
              <span className="text-sm text-sky-300">AI-Powered Research Automation</span>
            </div>
            
            <h1 className="text-5xl md:text-6xl font-bold mb-4 tracking-tight">
              <span className="text-white">Automate Your</span>
              <br />
              <span className="gradient-text">Research Pipeline</span>
            </h1>
            
            <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
              Upload your research spec sheet and let our multi-agent AI system 
              plan, learn, execute, and assess your ML experiments automatically.
            </p>
          </div>

          {/* Main Content */}
          <div className="relative">
            {/* Decorative elements */}
            <div className="absolute -top-20 -left-20 w-72 h-72 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute -bottom-20 -right-20 w-72 h-72 bg-orange-500/10 rounded-full blur-3xl pointer-events-none" />
            
            <div className="relative glass rounded-3xl p-8 md:p-12">
              {!showStatus ? (
                <SpecSheetUpload onProjectStarted={handleProjectStarted} />
              ) : (
                <div>
                  <button
                    onClick={resetToUpload}
                    className="mb-6 text-sm text-slate-400 hover:text-sky-400 transition-colors flex items-center gap-2"
                  >
                    ← Start New Project
                  </button>
                  <ProjectStatus 
                    project={currentProject} 
                    onComplete={handleComplete}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Agent Pipeline Illustration */}
          {!showStatus && (
            <div className="mt-16 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
              <p className="text-center text-sm text-slate-500 mb-6">AGENT PIPELINE</p>
              <div className="flex items-center justify-center gap-4 flex-wrap">
                {[
                  { icon: FileText, label: 'Planner', color: 'sky' },
                  { icon: Brain, label: 'Learner', color: 'violet' },
                  { icon: Zap, label: 'Executor', color: 'amber' },
                  { icon: BarChart3, label: 'Assessor', color: 'emerald' },
                  { icon: Beaker, label: 'PM', color: 'rose' },
                ].map((agent, i) => (
                  <div key={i} className="flex items-center gap-4">
                    <div 
                      className={`
                        flex flex-col items-center p-4 rounded-xl 
                        bg-${agent.color}-500/10 border border-${agent.color}-500/20
                        hover:scale-105 transition-transform cursor-default
                      `}
                      style={{
                        background: `rgba(${agent.color === 'sky' ? '14, 165, 233' : 
                                          agent.color === 'violet' ? '139, 92, 246' :
                                          agent.color === 'amber' ? '245, 158, 11' :
                                          agent.color === 'emerald' ? '16, 185, 129' :
                                          '244, 63, 94'}, 0.1)`,
                        borderColor: `rgba(${agent.color === 'sky' ? '14, 165, 233' : 
                                            agent.color === 'violet' ? '139, 92, 246' :
                                            agent.color === 'amber' ? '245, 158, 11' :
                                            agent.color === 'emerald' ? '16, 185, 129' :
                                            '244, 63, 94'}, 0.2)`
                      }}
                    >
                      <agent.icon 
                        className="w-6 h-6 mb-2"
                        style={{
                          color: agent.color === 'sky' ? '#0ea5e9' : 
                                 agent.color === 'violet' ? '#8b5cf6' :
                                 agent.color === 'amber' ? '#f59e0b' :
                                 agent.color === 'emerald' ? '#10b981' :
                                 '#f43f5e'
                        }}
                      />
                      <span className="text-xs text-slate-300 font-medium">{agent.label}</span>
                    </div>
                    {i < 4 && (
                      <div className="w-8 h-0.5 bg-gradient-to-r from-slate-600 to-slate-700 hidden md:block" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-slate-800">
        <div className="max-w-6xl mx-auto flex items-center justify-between text-sm text-slate-500">
          <p>© 2025 PLEASe Framework. AI-Powered Research Automation.</p>
          <p>Built with multi-agent AI technology by BRIJESH</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
