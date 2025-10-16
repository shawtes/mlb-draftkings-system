import React, { useState, Suspense, lazy } from 'react';
import { ThemeProvider, createTheme, Alert } from '@mui/material';
import { Toaster } from 'react-hot-toast';
import { Users, Link2, BarChart3, Target, Settings, Star, Trophy } from 'lucide-react';

// Lazy load DFS Components to prevent build crashes
const PlayersTab = lazy(() => import('./dfs/PlayersTab').catch(() => ({ default: () => <Alert severity="error">Players Tab failed to load</Alert> })));
const TeamStacksTab = lazy(() => import('./dfs/TeamStacksTab').catch(() => ({ default: () => <Alert severity="error">Team Stacks Tab failed to load</Alert> })));
const StackExposureTab = lazy(() => import('./dfs/StackExposureTab').catch(() => ({ default: () => <Alert severity="error">Stack Exposure Tab failed to load</Alert> })));
const TeamCombosTab = lazy(() => import('./dfs/TeamCombosTab').catch(() => ({ default: () => <Alert severity="error">Team Combos Tab failed to load</Alert> })));
// Use simplified versions for Control Panel and Results
const ControlPanelTab = lazy(() => import('./dfs/ControlPanelTab_Simple'));
const FavoritesTab = lazy(() => import('./dfs/FavoritesTab').catch(() => ({ default: () => <Alert severity="error">Favorites Tab failed to load</Alert> })));
const ResultsTab = lazy(() => import('./dfs/ResultsTab_Simple'));
const StatusBar = lazy(() => import('./dfs/StatusBar').catch(() => ({ default: () => <div className="h-16" /> })));

// Define Player type locally to avoid import issues
type Player = any;

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#06b6d4', // cyan-500
    },
    secondary: {
      main: '#3b82f6', // blue-500
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
});

interface DFSOptimizerProps {
  sport: string;
}

export default function DFSOptimizer({ sport }: DFSOptimizerProps) {
  const [currentTab, setCurrentTab] = useState(0);
  const [playersData, setPlayersData] = useState<Player[]>([]);
  const [teamStacks, setTeamStacks] = useState<any[]>([]);
  const [stackExposures, setStackExposures] = useState<any[]>([]);
  const [optimizationResults, setOptimizationResults] = useState<any[]>([]);

  const tabs = [
    { label: 'Players', icon: Users },
    { label: 'Team Stacks', icon: Link2 },
    { label: 'Stack Exposure', icon: BarChart3 },
    { label: 'Team Combos', icon: Target },
    { label: 'Control Panel', icon: Settings },
    { label: 'Favorites', icon: Star },
    { label: 'Results', icon: Trophy },
  ];

  return (
    <ThemeProvider theme={darkTheme}>
      <Toaster position="top-right" />
      
      <div className="h-full flex flex-col bg-black relative overflow-hidden">
        {/* Animated Grid Background */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#0a0a0a_1px,transparent_1px),linear-gradient(to_bottom,#0a0a0a_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40 pointer-events-none" />
        
        {/* Gradient Orb - Top */}
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
        
        {/* Gradient Orb - Bottom */}
        <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />

        {/* Content */}
        <div className="relative z-10 flex flex-col h-full">
          {/* Status Bar */}
          <Suspense fallback={<div className="h-16 bg-black/60 backdrop-blur-sm border-b border-cyan-500/20" />}>
            <StatusBar 
              playersCount={playersData.length}
              selectedCount={playersData.filter(p => p.selected).length}
              resultsCount={optimizationResults.length}
            />
          </Suspense>

          {/* Main Container */}
          <div className="flex-1 flex flex-col bg-black/60 backdrop-blur-sm border border-cyan-500/10 rounded-lg m-4 overflow-hidden shadow-2xl shadow-cyan-500/5">
            {/* Tabs */}
            <div className="border-b border-cyan-500/20 bg-gradient-to-r from-black via-cyan-950/10 to-black">
              <div className="flex overflow-x-auto scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent">
                {tabs.map((tab, index) => {
                  const Icon = tab.icon;
                  const isActive = currentTab === index;
                  return (
                    <button
                      key={index}
                      onClick={() => setCurrentTab(index)}
                      className={`
                        flex items-center gap-2 px-6 py-4 min-w-fit text-sm font-medium transition-all relative
                        ${isActive 
                          ? 'text-cyan-400 border-b-2 border-cyan-400 bg-cyan-500/5' 
                          : 'text-slate-400 hover:text-cyan-300 hover:bg-cyan-500/5'
                        }
                      `}
                    >
                      <Icon className={`w-5 h-5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                      <span>{tab.label}</span>
                      {isActive && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-400 to-blue-500" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-auto p-6 custom-scrollbar">
              <Suspense fallback={
                <div className="flex items-center justify-center h-full">
                  <div className="relative">
                    <div className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-8 h-8 bg-cyan-400/20 rounded-full blur-md" />
                    </div>
                  </div>
                </div>
              }>
                <div className="animate-fadeIn">
                  {currentTab === 0 && (
                    <PlayersTab 
                      players={playersData}
                      onPlayersUpdate={setPlayersData}
                    />
                  )}
                  {currentTab === 1 && (
                    <TeamStacksTab 
                      players={playersData}
                      teamStacks={teamStacks}
                      onTeamStacksUpdate={(stacks) => {
                        console.log('Team stacks updated:', stacks);
                        const stacksArray = Object.entries(stacks).map(([stackSize, teams]) => ({
                          stackSize,
                          teams
                        }));
                        setTeamStacks(stacksArray);
                      }}
                    />
                  )}
                  {currentTab === 2 && (
                    <StackExposureTab 
                      teamStacks={teamStacks}
                      stackExposures={stackExposures}
                      onStackExposuresUpdate={setStackExposures}
                    />
                  )}
                  {currentTab === 3 && (
                    <TeamCombosTab 
                      teams={[...new Set(playersData.map(p => p.team))].filter(Boolean)}
                      onTeamCombosUpdate={(combos) => {
                        console.log('Team combos updated:', combos);
                      }}
                    />
                  )}
                  {currentTab === 4 && (
                    <ControlPanelTab 
                      onStartOptimization={() => {
                        console.log('Starting optimization...');
                      }}
                    />
                  )}
                  {currentTab === 5 && (
                    <FavoritesTab 
                      players={playersData}
                      onFavoritesUpdate={(favorites) => {
                        console.log('Favorites updated:', favorites);
                      }}
                    />
                  )}
                  {currentTab === 6 && (
                    <ResultsTab 
                      results={optimizationResults}
                      onExportResults={() => {
                        console.log('Exporting results...');
                      }}
                    />
                  )}
                </div>
              </Suspense>
            </div>
          </div>
        </div>
      </div>

      {/* Add custom scrollbar styles */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(6, 182, 212, 0.2);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(6, 182, 212, 0.3);
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </ThemeProvider>
  );
}

