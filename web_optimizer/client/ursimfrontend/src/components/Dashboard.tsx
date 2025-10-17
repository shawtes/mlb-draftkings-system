import { useState, useMemo, useCallback, lazy, Suspense } from 'react';
import { Button } from './ui/button';
import { 
  TrendingUp, 
  LayoutDashboard, 
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
const GamesHub = lazy(() => import('./GamesHub').catch(err => {
  console.error('Failed to load GamesHub:', err);
  return { default: () => <div className="text-white p-8">Error loading Games Hub. Check console.</div> };
}));
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
  const [activeView, setActiveView] = useState('games');

  // Memoize navigation handlers to prevent unnecessary re-renders
  const handleNavigation = useCallback((view: string) => {
    setActiveView(view);
  }, []);

  const renderMainContent = useMemo(() => {
    switch (activeView) {
      case 'games':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <GamesHub sport="NFL" />
          </Suspense>
        );
      case 'prop-betting':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <PropBettingCenter sport="NFL" />
          </Suspense>
        );
      case 'dfs-optimizer':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <DFSOptimizer sport="NFL" />
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
  }, [activeView]);

  return (
    <div className="min-h-screen bg-slate-900 relative overflow-hidden flex flex-col">
      {/* Animated Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-30" />
      
      {/* Gradient Orb - Top Left */}
      <div className="absolute top-0 -left-40 w-96 h-96 bg-cyan-500/15 rounded-full blur-[120px] pointer-events-none" />
      
      {/* Gradient Orb - Right */}
      <div className="absolute top-1/3 -right-40 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />

      {/* Top Navigation Header */}
      <header className="bg-slate-800/95 backdrop-blur-xl border-b border-slate-700/50 shadow-2xl relative z-10">
        {/* Top Bar - Logo and User Section */}
        <div className="flex items-center justify-between px-8 py-4 border-b border-cyan-500/10">
          {/* Left: Logo */}
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/30 relative">
                <Trophy className="w-7 h-7 text-white" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-cyan-400 rounded-full border-2 border-black animate-pulse" />
              </div>
            </div>
            <div>
              <div className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">UrSim</div>
              <div className="text-xs text-cyan-400/70 font-medium">DFS Lineup Optimizer</div>
            </div>
          </div>
          
          {/* Right: User Actions */}
          <div className="flex items-center gap-3">
            <Button 
              variant="ghost" 
              size="icon" 
              className="text-slate-300 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all"
              title="Help & Support"
              onClick={() => handleNavigation('how-to-use')}
            >
              <HelpCircle className="w-5 h-5" />
            </Button>
            <Button 
              variant="ghost" 
              size="icon" 
              className="text-slate-300 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all relative"
              title="Notifications"
            >
              <MessageSquare className="w-5 h-5" />
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            </Button>

            {/* User Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="flex items-center gap-3 text-slate-400 hover:text-cyan-400 px-4 py-2 rounded-lg hover:bg-cyan-500/5 transition-all border border-transparent hover:border-cyan-500/20">
                  <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center shadow-[0_0_10px_rgba(6,182,212,0.3)]">
                    <User className="w-5 h-5 text-white" />
                  </div>
                  <span className="font-medium">User</span>
                  <ChevronDown className="w-4 h-4" />
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

        {/* Navigation Menu Bar */}
        <nav className="px-8 py-3">
          <div className="flex items-center gap-2">
            {/* Research Section */}
            <button
              onClick={() => handleNavigation('games')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group ${
                activeView === 'games' 
                  ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                  : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
              }`}
            >
              <LayoutDashboard className={`w-5 h-5 ${activeView === 'games' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
              <span className="font-medium">Games Hub</span>
              {activeView === 'games' && <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
            </button>

            {/* Separator */}
            <div className="h-8 w-px bg-cyan-500/10" />

            {/* Betting Section */}
            <button
              onClick={() => handleNavigation('prop-betting')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group ${
                activeView === 'prop-betting' 
                  ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                  : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
              }`}
            >
              <TrendingUp className={`w-5 h-5 ${activeView === 'prop-betting' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
              <span className="font-medium">Prop Betting</span>
              {activeView === 'prop-betting' && <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
            </button>

            {/* Separator */}
            <div className="h-8 w-px bg-cyan-500/10" />

            {/* DFS Tools Section */}
            <button
              onClick={() => handleNavigation('dfs-optimizer')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group ${
                activeView === 'dfs-optimizer' 
                  ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                  : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
              }`}
            >
              <Settings className={`w-5 h-5 ${activeView === 'dfs-optimizer' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
              <span className="font-medium">DFS Optimizer</span>
              {activeView === 'dfs-optimizer' && <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
            </button>

            {/* Separator */}
            <div className="h-8 w-px bg-cyan-500/10" />

            {/* How To Section */}
            <button
              onClick={() => handleNavigation('how-to-use')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group ${
                activeView === 'how-to-use' 
                  ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                  : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
              }`}
            >
              <HelpCircle className={`w-5 h-5 ${activeView === 'how-to-use' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
              <span className="font-medium">How To</span>
              {activeView === 'how-to-use' && <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
            </button>

            {/* Separator */}
            <div className="h-8 w-px bg-cyan-500/10" />

            {/* Settings Section */}
            <button
              onClick={() => handleNavigation('settings')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg transition-all group ${
                activeView === 'settings' 
                  ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                  : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
              }`}
            >
              <User className={`w-5 h-5 ${activeView === 'settings' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
              <span className="font-medium">Settings</span>
              {activeView === 'settings' && <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
            </button>
          </div>
        </nav>
        </header>

        {/* Main Content */}
      <main className="flex-1 overflow-auto p-8 relative z-10">
          {renderMainContent}
        </main>
    </div>
  );
}
