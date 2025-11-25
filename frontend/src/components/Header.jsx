import { Beaker, Sparkles, LayoutDashboard, FolderOpen, BookOpen } from 'lucide-react';

export default function Header({ onDashboardClick, onDocsClick }) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-orange-500 flex items-center justify-center">
                <Beaker className="w-5 h-5 text-white" />
              </div>
              <Sparkles className="w-4 h-4 text-orange-400 absolute -top-1 -right-1 animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                <span className="gradient-text">PLEASe</span>
              </h1>
              <p className="text-xs text-slate-400 -mt-1">AI Research Framework</p>
            </div>
          </div>
          
          <nav className="flex items-center gap-2">
            <button
              onClick={onDashboardClick}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <LayoutDashboard className="w-4 h-4" />
              Dashboard
            </button>
            <button
              onClick={onDocsClick}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <BookOpen className="w-4 h-4" />
              Docs
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
}
