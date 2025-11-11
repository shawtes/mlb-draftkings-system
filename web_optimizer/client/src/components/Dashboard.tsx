import { useState, useMemo, useCallback, lazy, Suspense } from 'react';
import { Button } from './ui/button';
import { 
  TrendingUp, 
  Settings, 
  HelpCircle, 
  MessageSquare,
  LogOut,
  User,
  Trophy,
  ChevronDown
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import { DashboardLoader } from './SkeletonLoader';
import type { DashboardProps } from '../types';

// Lazy load dashboard components for better performance
const PropBettingCenter = lazy(() => import('./PropBettingCenter').catch(err => {
  console.error('Failed to load PropBettingCenter:', err);
  return { default: () => <div className="text-white p-8">Error loading Prop Betting Center. Check console.</div> };
}));
const DFSOptimizer = lazy(() => import('./DFSOptimizer')); // Original 7-tab version with all functionality
const HowToUse = lazy(() => import('./HowToUse').catch(err => {
  console.error('Failed to load HowToUse:', err);
  return { default: () => <div className="text-white p-8">Error loading How to Use. Check console.</div> };
}));
const AccountSettings = lazy(() => import('./AccountSettings'));

export default function Dashboard({ onLogout }: DashboardProps) {
  const [activeView, setActiveView] = useState('prop-betting');
  const [selectedSport, setSelectedSport] = useState<'NFL' | 'NBA' | 'MLB'>('NFL');
  const isFullScreenView = activeView === 'prop-betting' || activeView === 'dfs-optimizer';

  // Memoize navigation handlers to prevent unnecessary re-renders
  const handleNavigation = useCallback((view: string) => {
    setActiveView(view);
  }, []);

  // Memoize sport change handler
  const handleSportChange = useCallback((sport: 'NFL' | 'NBA' | 'MLB') => {
    setSelectedSport(sport);
  }, []);

  const renderMainContent = useMemo(() => {
    switch (activeView) {
      case 'prop-betting':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <PropBettingCenter sport={selectedSport} />
          </Suspense>
        );
      case 'dfs-optimizer':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <DFSOptimizer sport={selectedSport} />
          </Suspense>
        );
      case 'how-to-use':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <HowToUse />
          </Suspense>
        );
      case 'settings':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <AccountSettings />
          </Suspense>
        );
      default:
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-slate-300 mb-2">Coming Soon</h2>
              <p className="text-slate-300">This feature is under development</p>
            </div>
          </div>
        );
    }
  }, [activeView, selectedSport]);

  return (
    <div
      className={`min-h-screen flex flex-col ${isFullScreenView ? '' : 'relative overflow-hidden'}`}
      style={isFullScreenView ? undefined : { backgroundColor: '#64748b' }}
    >
      {!isFullScreenView && (
        <>
          {/* Animated Grid Background */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#64748b_1px,transparent_1px),linear-gradient(to_bottom,#64748b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-30" />

          {/* Gradient Orb - Top Left */}
          <div className="absolute top-0 -left-40 w-96 h-96 bg-cyan-500/15 rounded-full blur-[120px] pointer-events-none" />

          {/* Gradient Orb - Right */}
          <div className="absolute top-1/3 -right-40 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />
        </>
      )}

      {/* Top Navigation Header */}
      <header className="backdrop-blur-xl border-b border-slate-700/50 shadow-2xl relative z-10" style={{ backgroundColor: '#111c2e' }}>
        {/* Sport Selection & User Controls */}
        <div className="px-8 py-6 border-b border-slate-700/30" style={{ backgroundColor: '#192C57' }}>
          <div className="flex flex-wrap items-center justify-between gap-6">
            {/* Left: Sport Tabs */}
            <div className="flex items-center gap-3">
              <span className="text-slate-400 text-sm font-medium mr-2">Select Sport:</span>
              <div className="flex gap-2">
                <button
                  onClick={() => handleSportChange('NFL')}
                  className={`px-6 py-3 rounded-lg font-bold text-lg transition-all ${
                    selectedSport === 'NFL'
                      ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/30 border-2 border-cyan-400'
                      : 'bg-slate-700/40 text-slate-300 border-2 border-slate-600/30 hover:bg-slate-700 hover:border-cyan-500/50 hover:text-white'
                  }`}
                >
                  🏈 NFL
                </button>
                <button
                  onClick={() => handleSportChange('NBA')}
                  className={`px-6 py-3 rounded-lg font-bold text-lg transition-all ${
                    selectedSport === 'NBA'
                      ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/30 border-2 border-cyan-400'
                      : 'bg-slate-700/40 text-slate-300 border-2 border-slate-600/30 hover:bg-slate-700 hover:border-cyan-500/50 hover:text-white'
                  }`}
                >
                  🏀 NBA
                </button>
                <button
                  onClick={() => handleSportChange('MLB')}
                  className={`px-6 py-3 rounded-lg font-bold text-lg transition-all ${
                    selectedSport === 'MLB'
                      ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/30 border-2 border-cyan-400'
                      : 'bg-slate-700/40 text-slate-300 border-2 border-slate-600/30 hover:bg-slate-700 hover:border-cyan-500/50 hover:text-white'
                  }`}
                >
                  ⚾ MLB
                </button>
              </div>
            </div>
            
            <div className="flex items-center gap-8">
              {/* Right: Quick Stats */}
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-2xl font-bold text-blue-400">0</div>
                  <div className="text-xs text-slate-400">Active Lineups</div>
                </div>
              </div>

              {/* User Actions */}
              <div className="flex items-center gap-4">
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-slate-300 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all w-10 h-10"
                  title="Help & Support"
                  onClick={() => handleNavigation('how-to-use')}
                >
                  <HelpCircle className="w-6 h-6" />
                </Button>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-slate-300 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all relative w-10 h-10"
                  title="Notifications"
                >
                  <MessageSquare className="w-6 h-6" />
                  <div className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                </Button>

                {/* User Dropdown */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button className="flex items-center gap-3 text-slate-400 hover:text-cyan-400 px-3 py-2 rounded-lg hover:bg-cyan-500/5 transition-all border border-transparent hover:border-cyan-500/20">
                      <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center shadow-[0_0_10px_rgba(6,182,212,0.3)]">
                        <User className="w-6 h-6 text-white" />
                      </div>
                      <span className="font-medium">User</span>
                      <ChevronDown className="w-5 h-5" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-56 bg-slate-900 border-cyan-500/20">
                    <DropdownMenuItem onClick={() => handleNavigation('settings')} className="text-slate-300 hover:text-cyan-400">
                      <Settings className="w-4 h-4 mr-2" />
                      Settings
                    </DropdownMenuItem>
                    <DropdownMenuItem className="text-slate-300 hover:text-cyan-400">
                      <Trophy className="w-4 h-4 mr-2" />
                      My Lineups
                    </DropdownMenuItem>
                    <DropdownMenuSeparator className="bg-cyan-500/20" />
                    <DropdownMenuItem onClick={onLogout} className="text-red-400 focus:text-red-300">
                      <LogOut className="w-4 h-4 mr-2" />
                      Logout
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Menu Bar */}
        <nav className="px-8 py-3" style={{ backgroundColor: '#13233f' }}>
          <div className="flex items-center gap-2">
            {/* Prop Betting Section */}
            <button
              onClick={() => handleNavigation('prop-betting')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group border border-transparent text-slate-400 ${
                activeView === 'prop-betting' ? '' : 'hover:text-[#4197E8] hover:border-[#4197E8]/20'
              }`}
              style={
                activeView === 'prop-betting'
                  ? {
                      backgroundColor: 'rgba(65, 151, 232, 0.18)',
                      borderColor: 'rgba(65, 151, 232, 0.45)',
                      boxShadow: '0 0 20px rgba(65, 151, 232, 0.18)',
                      color: '#4197E8',
                    }
                  : undefined
              }
            >
              <TrendingUp className="w-5 h-5" />
              <span className="font-medium">Prop Betting</span>
              {activeView === 'prop-betting' && (
                <div
                  className="w-2 h-2 rounded-full animate-pulse"
                  style={{ backgroundColor: '#4197E8', boxShadow: '0 0 8px rgba(65, 151, 232, 0.8)' }}
                />
              )}
            </button>

            {/* Separator */}
            <div className="h-8 w-px bg-cyan-500/10" />

            {/* DFS Tools Section */}
            <button
              onClick={() => handleNavigation('dfs-optimizer')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group border border-transparent text-slate-400 ${
                activeView === 'dfs-optimizer' ? '' : 'hover:text-[#4197E8] hover:border-[#4197E8]/20'
              }`}
              style={
                activeView === 'dfs-optimizer'
                  ? {
                      backgroundColor: 'rgba(65, 151, 232, 0.18)',
                      borderColor: 'rgba(65, 151, 232, 0.45)',
                      boxShadow: '0 0 20px rgba(65, 151, 232, 0.18)',
                      color: '#4197E8',
                    }
                  : undefined
              }
            >
              <Settings className="w-5 h-5" />
              <span className="font-medium">DFS Optimizer</span>
              {activeView === 'dfs-optimizer' && (
                <div
                  className="w-2 h-2 rounded-full animate-pulse"
                  style={{ backgroundColor: '#4197E8', boxShadow: '0 0 8px rgba(65, 151, 232, 0.8)' }}
                />
              )}
            </button>

            {/* Separator */}
            <div className="h-8 w-px bg-cyan-500/10" />

            {/* How To Section */}
            <button
              onClick={() => handleNavigation('how-to-use')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group border border-transparent text-slate-400 ${
                activeView === 'how-to-use' ? '' : 'hover:text-[#4197E8] hover:border-[#4197E8]/20'
              }`}
              style={
                activeView === 'how-to-use'
                  ? {
                      backgroundColor: 'rgba(65, 151, 232, 0.18)',
                      borderColor: 'rgba(65, 151, 232, 0.45)',
                      boxShadow: '0 0 20px rgba(65, 151, 232, 0.18)',
                      color: '#4197E8',
                    }
                  : undefined
              }
            >
              <HelpCircle className="w-5 h-5" />
              <span className="font-medium">How To</span>
              {activeView === 'how-to-use' && (
                <div
                  className="w-2 h-2 rounded-full animate-pulse"
                  style={{ backgroundColor: '#4197E8', boxShadow: '0 0 8px rgba(65, 151, 232, 0.8)' }}
                />
              )}
            </button>

            {/* Separator */}
            <div className="h-8 w-px bg-cyan-500/10" />

            {/* Settings Section */}
            <button
              onClick={() => handleNavigation('settings')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group border border-transparent text-slate-400 ${
                activeView === 'settings' ? '' : 'hover:text-[#4197E8] hover:border-[#4197E8]/20'
              }`}
              style={
                activeView === 'settings'
                  ? {
                      backgroundColor: 'rgba(65, 151, 232, 0.18)',
                      borderColor: 'rgba(65, 151, 232, 0.45)',
                      boxShadow: '0 0 20px rgba(65, 151, 232, 0.18)',
                      color: '#4197E8',
                    }
                  : undefined
              }
            >
              <User className="w-5 h-5" />
              <span className="font-medium">Settings</span>
              {activeView === 'settings' && (
                <div
                  className="w-2 h-2 rounded-full animate-pulse"
                  style={{ backgroundColor: '#4197E8', boxShadow: '0 0 8px rgba(65, 151, 232, 0.8)' }}
                />
              )}
            </button>
          </div>
        </nav>
        </header>

        {/* Main Content */}
      <main
        className={`flex-1 relative z-10 overflow-auto ${isFullScreenView ? 'bg-slate-900 text-white p-0' : 'p-8'}`}
        style={isFullScreenView ? undefined : { backgroundColor: '#64748b' }}
      >
          {renderMainContent}
        </main>
    </div>
  );
}
