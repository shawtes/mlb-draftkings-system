import React from 'react';

interface GamesHubProps {
  sport: string;
}

const GamesHub = React.memo(({ sport }: GamesHubProps) => {
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
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent mb-4 tracking-tight">
                Games Hub
              </h1>
            <p className="text-slate-300 text-xl mb-2 font-medium">
              {sport.toUpperCase()} Game Research & Analysis
            </p>
        </div>

          <div className="bg-slate-700/40 border border-slate-600/50 rounded-2xl p-10 backdrop-blur-sm shadow-xl">
            <h2 className="text-cyan-300 text-2xl font-semibold mb-4 tracking-wide">
              Ready to Build
            </h2>
            <p className="text-slate-200 leading-relaxed text-lg">
              This component is ready for your custom games hub implementation.
              Build your game listings, matchup analysis, and research tools here.
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

GamesHub.displayName = 'GamesHub';

export default GamesHub;
