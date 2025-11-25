import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  CheckCircle2, XCircle, AlertTriangle, Info, 
  ChevronRight, ExternalLink, Copy, Check
} from 'lucide-react';
import { useState } from 'react';

export default function MarkdownReport({ content }) {
  const [copiedCode, setCopiedCode] = useState(null);

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(index);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="markdown-report">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Headings
          h1: ({ children }) => (
            <h1 className="text-3xl font-bold text-white mt-8 mb-4 pb-3 border-b border-slate-700 first:mt-0">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-2xl font-semibold text-white mt-8 mb-3 flex items-center gap-2">
              <div className="w-1 h-6 bg-gradient-to-b from-sky-400 to-violet-500 rounded-full" />
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-xl font-semibold text-slate-200 mt-6 mb-2">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-lg font-medium text-slate-300 mt-4 mb-2">
              {children}
            </h4>
          ),

          // Paragraphs
          p: ({ children }) => (
            <p className="text-slate-300 leading-relaxed mb-4">
              {children}
            </p>
          ),

          // Lists
          ul: ({ children }) => (
            <ul className="space-y-2 mb-4 ml-1">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="space-y-2 mb-4 ml-1 list-decimal list-inside">
              {children}
            </ol>
          ),
          li: ({ children, ordered }) => (
            <li className="flex items-start gap-2 text-slate-300">
              {!ordered && (
                <ChevronRight className="w-4 h-4 text-sky-400 mt-1 flex-shrink-0" />
              )}
              <span>{children}</span>
            </li>
          ),

          // Links
          a: ({ href, children }) => (
            <a 
              href={href} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sky-400 hover:text-sky-300 underline underline-offset-2 inline-flex items-center gap-1"
            >
              {children}
              <ExternalLink className="w-3 h-3" />
            </a>
          ),

          // Code blocks
          code: ({ inline, className, children, ...props }) => {
            const codeString = String(children).replace(/\n$/, '');
            const codeIndex = Math.random();

            if (inline) {
              return (
                <code className="px-1.5 py-0.5 bg-slate-800 text-sky-300 rounded text-sm font-mono">
                  {children}
                </code>
              );
            }

            return (
              <div className="relative group my-4">
                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => copyToClipboard(codeString, codeIndex)}
                    className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-300 transition-colors"
                  >
                    {copiedCode === codeIndex ? (
                      <Check className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
                <pre className="bg-slate-900 border border-slate-700 rounded-xl p-4 overflow-x-auto">
                  <code className="text-sm font-mono text-slate-300" {...props}>
                    {children}
                  </code>
                </pre>
              </div>
            );
          },

          // Blockquotes
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-sky-500 bg-sky-500/5 pl-4 py-2 my-4 italic text-slate-300">
              {children}
            </blockquote>
          ),

          // Tables
          table: ({ children }) => (
            <div className="overflow-x-auto my-6 rounded-xl border border-slate-700">
              <table className="w-full">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-slate-800/80">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-slate-700/50">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-slate-800/30 transition-colors">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="px-4 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-3 text-sm text-slate-300">
              {children}
            </td>
          ),

          // Horizontal rule
          hr: () => (
            <hr className="my-8 border-slate-700" />
          ),

          // Strong & emphasis
          strong: ({ children }) => (
            <strong className="font-semibold text-white">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-slate-200">
              {children}
            </em>
          ),

          // Images
          img: ({ src, alt }) => (
            <figure className="my-6">
              <img 
                src={src} 
                alt={alt} 
                className="rounded-xl border border-slate-700 max-w-full"
              />
              {alt && (
                <figcaption className="text-center text-sm text-slate-400 mt-2">
                  {alt}
                </figcaption>
              )}
            </figure>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

// Alert component for special callouts
export function ReportAlert({ type = 'info', title, children }) {
  const styles = {
    success: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      icon: CheckCircle2,
      iconColor: 'text-emerald-400',
      titleColor: 'text-emerald-300'
    },
    warning: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      icon: AlertTriangle,
      iconColor: 'text-amber-400',
      titleColor: 'text-amber-300'
    },
    error: {
      bg: 'bg-red-500/10',
      border: 'border-red-500/30',
      icon: XCircle,
      iconColor: 'text-red-400',
      titleColor: 'text-red-300'
    },
    info: {
      bg: 'bg-sky-500/10',
      border: 'border-sky-500/30',
      icon: Info,
      iconColor: 'text-sky-400',
      titleColor: 'text-sky-300'
    }
  };

  const style = styles[type];
  const Icon = style.icon;

  return (
    <div className={`${style.bg} border ${style.border} rounded-xl p-4 my-4`}>
      <div className="flex items-start gap-3">
        <Icon className={`w-5 h-5 ${style.iconColor} flex-shrink-0 mt-0.5`} />
        <div>
          {title && (
            <h4 className={`font-semibold ${style.titleColor} mb-1`}>{title}</h4>
          )}
          <div className="text-slate-300 text-sm">{children}</div>
        </div>
      </div>
    </div>
  );
}

// Score display component
export function ScoreCard({ label, value, maxValue, color = 'sky' }) {
  const percentage = maxValue ? (value / maxValue) * 100 : value;
  
  const colors = {
    sky: 'from-sky-500 to-sky-400',
    emerald: 'from-emerald-500 to-emerald-400',
    violet: 'from-violet-500 to-violet-400',
    amber: 'from-amber-500 to-amber-400'
  };

  return (
    <div className="bg-slate-800/50 rounded-xl p-4">
      <p className="text-xs text-slate-400 mb-2">{label}</p>
      <div className="flex items-end gap-2 mb-2">
        <span className="text-3xl font-bold text-white">{value}</span>
        {maxValue && (
          <span className="text-slate-400 text-sm mb-1">/ {maxValue}</span>
        )}
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div 
          className={`h-full bg-gradient-to-r ${colors[color]} rounded-full transition-all duration-500`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  );
}

// Metric comparison component
export function MetricComparison({ metrics }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 my-6">
      {metrics.map((metric, i) => {
        const achieved = metric.achieved >= metric.goal;
        return (
          <div 
            key={i}
            className={`rounded-xl p-4 border ${
              achieved 
                ? 'bg-emerald-500/5 border-emerald-500/20' 
                : 'bg-amber-500/5 border-amber-500/20'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-slate-200">{metric.name}</span>
              {achieved ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-amber-400" />
              )}
            </div>
            <div className="flex items-baseline gap-2">
              <span className={`text-2xl font-bold ${achieved ? 'text-emerald-400' : 'text-amber-400'}`}>
                {typeof metric.achieved === 'number' ? metric.achieved.toFixed(3) : metric.achieved}
              </span>
              <span className="text-slate-500 text-sm">
                goal: {typeof metric.goal === 'number' ? metric.goal.toFixed(3) : metric.goal}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

