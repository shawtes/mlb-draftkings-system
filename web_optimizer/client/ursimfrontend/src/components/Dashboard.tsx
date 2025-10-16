import { useState, useMemo, lazy, Suspense } from 'react';
import { Button } from './ui/button';
import { 
  TrendingUp, 
  LayoutDashboard, 
  Target, 
  Settings, 
  HelpCircle, 
  MessageSquare,
  LogOut,
  User,
  Trophy,
  Zap,
  Menu,
  X,
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
const LineupBuilder = lazy(() => import('./LineupBuilder')); // Using original lineup builder you liked
const DFSOptimizer = lazy(() => import('./DFSOptimizer')); // Original 7-tab version with all functionality
const HowToUse = lazy(() => import('./HowToUse').catch(err => {
  console.error('Failed to load HowToUse:', err);
  return { default: () => <div className="text-white p-8">Error loading How to Use. Check console.</div> };
}));
const AccountSettings = lazy(() => import('./AccountSettings'));

export default function Dashboard({ onLogout }: DashboardProps) {
  const [activeView, setActiveView] = useState('games');
  const [sidebarOpen, setSidebarOpen] = useState(true);

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
      case 'lineup-builder':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <LineupBuilder sport="NFL" slate="main" />
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
              <p className="text-slate-500">This feature is under development</p>
            </div>
          </div>
        );
    }
  }, [activeView]);

  return (
    <div className="min-h-screen bg-black relative overflow-hidden flex">
      {/* Animated Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0a0a0a_1px,transparent_1px),linear-gradient(to_bottom,#0a0a0a_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
      
      {/* Gradient Orb - Top Left */}
      <div className="absolute top-0 -left-40 w-96 h-96 bg-cyan-500/20 rounded-full blur-[120px] pointer-events-none" />
      
      {/* Gradient Orb - Right */}
      <div className="absolute top-1/3 -right-40 w-96 h-96 bg-blue-600/15 rounded-full blur-[120px] pointer-events-none" />

      {/* Left Sidebar - Collapsible */}
      <aside className={`${sidebarOpen ? 'w-72' : 'w-0'} bg-black/80 backdrop-blur-xl border-r border-cyan-500/20 flex flex-col shadow-2xl relative z-10 transition-all duration-300 overflow-hidden`}>
        {/* Logo Header */}
        <div className="p-6 border-b border-cyan-500/20 bg-gradient-to-r from-black via-cyan-950/10 to-black">
          <div className="flex items-center gap-3 mb-2">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/30 relative">
                <Trophy className="w-7 h-7 text-white" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-cyan-400 rounded-full border-2 border-black animate-pulse" />
              </div>
              <div className="absolute inset-0 bg-cyan-400/20 blur-xl rounded-full" />
            </div>
            <div>
              <div className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">UrSim</div>
              <div className="text-xs text-cyan-400/70 font-medium">DFS Lineup Optimizer</div>
            </div>
          </div>
          
          {/* Quick Stats */}
          <div className="grid grid-cols-2 gap-2 mt-4">
            <div className="bg-cyan-500/5 border border-cyan-500/20 rounded-lg p-2.5 hover:bg-cyan-500/10 transition-colors">
              <div className="text-cyan-400/80 text-xs font-medium">Active Contests</div>
              <div className="text-white text-lg font-bold">12</div>
            </div>
            <div className="bg-cyan-500/5 border border-cyan-500/20 rounded-lg p-2.5 hover:bg-cyan-500/10 transition-colors">
              <div className="text-cyan-400/80 text-xs font-medium">Win Rate</div>
              <div className="text-white text-lg font-bold">64%</div>
            </div>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {/* RESEARCH SECTION */}
          <div className="pb-2 flex items-center gap-2">
            <Zap className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-400/80 text-xs font-bold uppercase tracking-wider">Research</span>
          </div>
          <button
            onClick={() => setActiveView('games')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all group ${
              activeView === 'games' 
                ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
            }`}
          >
            <LayoutDashboard className={`w-5 h-5 ${activeView === 'games' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
            <span className="font-medium">Games Hub</span>
            {activeView === 'games' && <div className="ml-auto w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
          </button>

          {/* BETTING SECTION */}
          <div className="pt-5 pb-2 flex items-center gap-2">
            <Target className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-400/80 text-xs font-bold uppercase tracking-wider">Betting</span>
          </div>
          <button
            onClick={() => setActiveView('prop-betting')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all group ${
              activeView === 'prop-betting' 
                ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
            }`}
          >
            <TrendingUp className={`w-5 h-5 ${activeView === 'prop-betting' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
            <span className="font-medium">Prop Betting Center</span>
            {activeView === 'prop-betting' && <div className="ml-auto w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
          </button>

          {/* DFS SECTION */}
          <div className="pt-5 pb-2 flex items-center gap-2">
            <Trophy className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-400/80 text-xs font-bold uppercase tracking-wider">DFS Tools</span>
          </div>
          <button
            onClick={() => setActiveView('lineup-builder')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all group ${
              activeView === 'lineup-builder' 
                ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
            }`}
          >
            <Trophy className={`w-5 h-5 ${activeView === 'lineup-builder' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
            <span className="font-medium">Lineup Builder</span>
            {activeView === 'lineup-builder' && <div className="ml-auto w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
          </button>
          <button
            onClick={() => setActiveView('dfs-optimizer')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all group ${
              activeView === 'dfs-optimizer' 
                ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
            }`}
          >
            <Settings className={`w-5 h-5 ${activeView === 'dfs-optimizer' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
            <span className="font-medium">DFS Optimizer</span>
            {activeView === 'dfs-optimizer' && <div className="ml-auto w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
          </button>

          {/* HELP SECTION */}
          <div className="pt-5 pb-2 flex items-center gap-2">
            <HelpCircle className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-400/80 text-xs font-bold uppercase tracking-wider">Help</span>
          </div>
          <button
            onClick={() => setActiveView('how-to-use')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all group ${
              activeView === 'how-to-use' 
                ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
            }`}
          >
            <HelpCircle className={`w-5 h-5 ${activeView === 'how-to-use' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
            <span className="font-medium">How to Use UrSim</span>
            {activeView === 'how-to-use' && <div className="ml-auto w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
          </button>

          {/* ACCOUNT SECTION */}
          <div className="pt-5 pb-2 flex items-center gap-2">
            <Settings className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-400/80 text-xs font-bold uppercase tracking-wider">Account</span>
          </div>
          <button
            onClick={() => setActiveView('settings')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all group ${
              activeView === 'settings' 
                ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]' 
                : 'text-slate-400 hover:bg-cyan-500/5 hover:text-cyan-400 border border-transparent hover:border-cyan-500/20'
            }`}
          >
            <User className={`w-5 h-5 ${activeView === 'settings' ? 'text-cyan-400' : 'group-hover:text-cyan-400'} transition-colors`} />
            <span className="font-medium">Settings</span>
            {activeView === 'settings' && <div className="ml-auto w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
          </button>

          {/* LOGOUT */}
          <div className="pt-4 border-t border-cyan-500/10 mt-4">
            <button
              onClick={onLogout}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all text-slate-400 hover:text-red-400 hover:bg-red-500/5 border border-transparent hover:border-red-500/20"
            >
              <LogOut className="w-5 h-5" />
              <span className="font-medium">Sign Out</span>
            </button>
          </div>
        </nav>

      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col relative z-10">
        {/* Top Navigation Bar - DFS Optimized */}
        <header className="h-20 bg-black/80 backdrop-blur-xl border-b border-cyan-500/20 flex items-center justify-between px-8 shadow-2xl">
          <div className="flex items-center gap-4">
            {/* Hamburger Menu */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-slate-400 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all"
            >
              {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </Button>

            {/* Live Indicator */}
            <div className="ml-4 flex items-center gap-2 px-3 py-2 bg-cyan-500/5 border border-cyan-500/20 rounded-lg">
              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
              <span className="text-cyan-400 text-sm font-medium">Live Contests</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Quick Actions */}
            <Button 
              variant="ghost" 
              size="icon" 
              className="text-slate-300 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all"
              title="Help & Support"
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
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuItem onClick={() => setActiveView('settings')}>
                  <Settings className="w-4 h-4 mr-2" />
                  Settings
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <Trophy className="w-4 h-4 mr-2" />
                  My Lineups
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={onLogout} className="text-red-400 focus:text-red-300">
                  <LogOut className="w-4 h-4 mr-2" />
                  Logout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-auto bg-black p-8">
          {renderMainContent}
        </main>
      </div>
    </div>
  );
}
