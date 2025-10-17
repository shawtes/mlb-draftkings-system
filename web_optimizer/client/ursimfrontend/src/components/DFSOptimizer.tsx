import React from 'react';

interface DFSOptimizerProps {
  sport: string;
}

const DFSOptimizer = React.memo(({ sport }: DFSOptimizerProps) => {
  return (
    <div className="h-full overflow-auto p-6">
      {/* Main Card Container */}
      <div className="bg-slate-800 backdrop-blur-sm rounded-2xl border border-cyan-500/20 shadow-2xl relative overflow-hidden min-h-full">
        
        {/* Content */}
        <div className="relative z-10 flex items-center justify-center min-h-full p-12">
          <div className="text-center max-w-3xl">
          <div className="mb-8">
            <div className="w-24 h-24 mx-auto bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-cyan-500/30 mb-6">
              <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent mb-4 tracking-tight">
              DFS Optimizer
            </h1>
            <p className="text-slate-300 text-xl mb-2 font-medium">
              {sport} Lineup Optimization Tool
            </p>
          </div>
          
          <div className="bg-slate-700/40 border border-slate-600/50 rounded-2xl p-10 backdrop-blur-sm shadow-xl">
            <h2 className="text-cyan-300 text-2xl font-semibold mb-4 tracking-wide">
              Ready to Build
            </h2>
            <p className="text-slate-200 leading-relaxed text-lg">
              This component is ready for your custom DFS optimization implementation.
              Build your player selection, lineup generation, and optimization features here.
            </p>
          </div>
          
          <div className="mt-8 flex items-center justify-center gap-3">
            <div className="w-2.5 h-2.5 bg-cyan-400 rounded-full animate-pulse shadow-lg shadow-cyan-400/50" />
            <span className="text-slate-300 text-base font-medium">Clean slate ready</span>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
});

DFSOptimizer.displayName = 'DFSOptimizer';

export default DFSOptimizer;
