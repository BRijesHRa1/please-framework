import { useState, useEffect, useMemo } from 'react';
import { 
  BookOpen, X, Loader2, Search, ChevronRight, ChevronDown,
  FileText, Code, Database, Cpu, Brain, BarChart3, Beaker,
  Server, Layout, Zap, Download, Copy, Check, Menu, ArrowUp
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../api/client';
import MarkdownReport from './MarkdownReport';

export default function Documentation({ onClose }) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeSection, setActiveSection] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [copied, setCopied] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);

  // Fetch documentation on mount
  useEffect(() => {
    const fetchDocs = async () => {
      try {
        const response = await apiClient.getDocumentation();
        setContent(response.data.content);
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to load documentation');
        toast.error('Failed to load documentation');
      } finally {
        setLoading(false);
      }
    };

    fetchDocs();
  }, []);

  // Extract table of contents from markdown
  const tableOfContents = useMemo(() => {
    if (!content) return [];

    const headingRegex = /^(#{1,3})\s+(.+)$/gm;
    const toc = [];
    let match;

    while ((match = headingRegex.exec(content)) !== null) {
      const level = match[1].length;
      const title = match[2].replace(/\*\*/g, '').trim();
      const id = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
      
      toc.push({ level, title, id });
    }

    return toc;
  }, [content]);

  // Filter content based on search
  const filteredContent = useMemo(() => {
    if (!searchQuery.trim()) return content;
    
    // Highlight search terms in the content
    const regex = new RegExp(`(${searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return content.replace(regex, '**$1**');
  }, [content, searchQuery]);

  // Handle scroll to show/hide scroll-to-top button
  useEffect(() => {
    const handleScroll = (e) => {
      setShowScrollTop(e.target.scrollTop > 500);
    };

    const container = document.getElementById('docs-container');
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, []);

  const scrollToTop = () => {
    const container = document.getElementById('docs-container');
    if (container) {
      container.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const scrollToSection = (id) => {
    setActiveSection(id);
    // Find the heading element and scroll to it
    const container = document.getElementById('docs-container');
    const elements = container?.querySelectorAll('h1, h2, h3');
    
    if (elements) {
      for (const el of elements) {
        const elId = el.textContent?.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
        if (elId === id) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          break;
        }
      }
    }
  };

  const copyContent = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    toast.success('Documentation copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadDocs = () => {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'PLEASE_DOCUMENTATION.md';
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Documentation downloaded');
  };

  const getIconForSection = (title) => {
    const lower = title.toLowerCase();
    if (lower.includes('executive') || lower.includes('summary')) return FileText;
    if (lower.includes('architecture')) return Layout;
    if (lower.includes('technology') || lower.includes('stack')) return Server;
    if (lower.includes('structure') || lower.includes('project')) return Database;
    if (lower.includes('backend')) return Server;
    if (lower.includes('frontend')) return Layout;
    if (lower.includes('agent') || lower.includes('pipeline')) return Brain;
    if (lower.includes('database') || lower.includes('schema')) return Database;
    if (lower.includes('api')) return Code;
    if (lower.includes('config') || lower.includes('setup')) return Zap;
    if (lower.includes('workflow')) return BarChart3;
    if (lower.includes('troubleshoot')) return Cpu;
    if (lower.includes('planner')) return FileText;
    if (lower.includes('learner')) return Brain;
    if (lower.includes('executor')) return Cpu;
    if (lower.includes('assessor')) return BarChart3;
    if (lower.includes('pm') || lower.includes('manager')) return Beaker;
    return ChevronRight;
  };

  if (loading) {
    return (
      <div className="fixed inset-0 z-50 bg-slate-900/98 backdrop-blur-sm flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-sky-400 animate-spin mx-auto mb-4" />
          <p className="text-slate-300">Loading documentation...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fixed inset-0 z-50 bg-slate-900/98 backdrop-blur-sm flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 rounded-2xl bg-red-500/20 flex items-center justify-center mx-auto mb-4">
            <BookOpen className="w-8 h-8 text-red-400" />
          </div>
          <h2 className="text-xl font-semibold text-white mb-2">Documentation Not Found</h2>
          <p className="text-slate-400 mb-6">{error}</p>
          <button
            onClick={onClose}
            className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-slate-900">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-10 glass border-b border-slate-700">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors lg:hidden"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                <BookOpen className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">PLEASe Documentation</h1>
                <p className="text-xs text-slate-400">Complete Technical Reference</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="relative hidden md:block">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search documentation..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-64 pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm placeholder-slate-400 focus:outline-none focus:border-sky-500"
              />
            </div>

            {/* Actions */}
            <button
              onClick={copyContent}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              title="Copy to clipboard"
            >
              {copied ? <Check className="w-5 h-5 text-emerald-400" /> : <Copy className="w-5 h-5" />}
            </button>
            <button
              onClick={downloadDocs}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              title="Download"
            >
              <Download className="w-5 h-5" />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              title="Close"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="flex pt-[73px] h-screen">
        {/* Sidebar - Table of Contents */}
        <aside className={`
          fixed lg:relative inset-y-0 left-0 z-20 lg:z-0
          w-72 bg-slate-900 border-r border-slate-700
          transform transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          pt-[73px] lg:pt-0 overflow-hidden flex flex-col
        `}>
          <div className="p-4 border-b border-slate-700">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
              Table of Contents
            </h2>
          </div>
          
          <nav className="flex-1 overflow-y-auto p-4 space-y-1">
            {tableOfContents.map((item, index) => {
              const Icon = getIconForSection(item.title);
              const isActive = activeSection === item.id;
              
              return (
                <button
                  key={index}
                  onClick={() => scrollToSection(item.id)}
                  className={`
                    w-full text-left flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors
                    ${item.level === 1 ? 'font-semibold' : item.level === 2 ? 'ml-3' : 'ml-6 text-xs'}
                    ${isActive 
                      ? 'bg-sky-500/20 text-sky-300' 
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    }
                  `}
                >
                  <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-sky-400' : 'text-slate-500'}`} />
                  <span className="truncate">{item.title}</span>
                </button>
              );
            })}
          </nav>

          {/* Stats */}
          <div className="p-4 border-t border-slate-700 bg-slate-800/50">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold text-white">{tableOfContents.filter(t => t.level === 1).length}</p>
                <p className="text-xs text-slate-400">Sections</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{content.split('\n').length}</p>
                <p className="text-xs text-slate-400">Lines</p>
              </div>
            </div>
          </div>
        </aside>

        {/* Sidebar overlay for mobile */}
        {sidebarOpen && (
          <div 
            className="fixed inset-0 bg-black/50 z-10 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main 
          id="docs-container"
          className="flex-1 overflow-y-auto"
        >
          <div className="max-w-4xl mx-auto px-6 py-8">
            {/* Search on mobile */}
            <div className="mb-6 md:hidden">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search documentation..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm placeholder-slate-400 focus:outline-none focus:border-sky-500"
                />
              </div>
            </div>

            {/* Search results info */}
            {searchQuery && (
              <div className="mb-6 p-4 bg-sky-500/10 border border-sky-500/20 rounded-xl">
                <p className="text-sky-300 text-sm">
                  Searching for: <strong>"{searchQuery}"</strong>
                </p>
              </div>
            )}

            {/* Documentation Content */}
            <article className="prose prose-invert max-w-none">
              <MarkdownReport content={filteredContent} />
            </article>

            {/* Footer */}
            <footer className="mt-16 pt-8 border-t border-slate-700 text-center">
              <p className="text-slate-400 text-sm">
                PLEASe Framework Documentation v1.0.0
              </p>
              <p className="text-slate-500 text-xs mt-1">
                © 2025 PLEASe Framework - Built by BRIJESH
              </p>
            </footer>
          </div>
        </main>
      </div>

      {/* Scroll to top button */}
      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-6 right-6 p-3 bg-sky-500 hover:bg-sky-400 text-white rounded-full shadow-lg transition-all hover:scale-110 z-50"
        >
          <ArrowUp className="w-5 h-5" />
        </button>
      )}
    </div>
  );
}

