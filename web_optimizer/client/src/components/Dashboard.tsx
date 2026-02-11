import { useState, useMemo, useCallback, lazy, Suspense } from 'react';
import {
  Settings,
  HelpCircle,
  LogOut,
  User,
  ChevronDown,
  Cpu,
  Trophy,
  BarChart3,
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

// Lazy load dashboard components
const DFSOptimizer = lazy(() => import('./DFSOptimizer'));
const GamesHub = lazy(() => import('./GamesHub'));
const PropBettingCenter = lazy(() => import('./PropBettingCenter'));
const HowToUse = lazy(() => import('./HowToUse').catch(err => {
  console.error('Failed to load HowToUse:', err);
  return { default: () => <div className="text-white p-8">Error loading How to Use. Check console.</div> };
}));
const AccountSettings = lazy(() => import('./AccountSettings'));

const navItems = [
  { id: 'games-hub', label: 'Games Hub', icon: BarChart3, section: 'RESEARCH' },
  { id: 'prop-betting', label: 'Prop Betting', icon: Trophy, section: 'BETTING' },
  { id: 'dfs-optimizer', label: 'DFS Optimizer', icon: Cpu, section: 'DFS' },
  { id: 'how-to-use', label: 'How To', icon: HelpCircle, section: 'HELP' },
] as const;

export default function Dashboard({ onLogout }: DashboardProps) {
  const [activeView, setActiveView] = useState('dfs-optimizer');

  const handleNavigation = useCallback((view: string) => {
    setActiveView(view);
  }, []);

  const renderMainContent = useMemo(() => {
    switch (activeView) {
      case 'games-hub':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <GamesHub />
          </Suspense>
        );
      case 'prop-betting':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <PropBettingCenter />
          </Suspense>
        );
      case 'dfs-optimizer':
        return (
          <Suspense fallback={<DashboardLoader />}>
            <DFSOptimizer />
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
          <Suspense fallback={<DashboardLoader />}>
            <DFSOptimizer />
          </Suspense>
        );
    }
  }, [activeView]);

  return (
    <div className="h-screen flex flex-col bg-[var(--dfs-bg-primary)]">
      {/* Top bar */}
      <header
        className="flex items-center justify-between px-4 h-10 flex-shrink-0 border-b"
        style={{ backgroundColor: 'var(--dfs-bg-secondary)', borderColor: 'var(--dfs-border)' }}
      >
        {/* Left: Logo */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-white tracking-wide">UrSim</span>
          <span className="text-[10px] text-[var(--dfs-text-muted)] font-medium tracking-wider uppercase">DFS Platform</span>
        </div>

        {/* Center: Nav pills */}
        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = activeView === item.id;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigation(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-[var(--dfs-accent)]/15 text-[var(--dfs-accent)] border border-[var(--dfs-accent)]/30'
                    : 'text-[var(--dfs-text-secondary)] hover:text-[var(--dfs-text-primary)] hover:bg-[var(--dfs-bg-hover)] border border-transparent'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Right: User dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex items-center gap-1.5 text-[var(--dfs-text-secondary)] hover:text-[var(--dfs-text-primary)] px-2 py-1 rounded-md hover:bg-[var(--dfs-bg-hover)] transition-all">
              <div className="w-6 h-6 bg-gradient-to-br from-cyan-500 to-blue-600 rounded flex items-center justify-center">
                <User className="w-3.5 h-3.5 text-white" />
              </div>
              <ChevronDown className="w-3 h-3" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48 bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)]">
            <DropdownMenuItem onClick={() => handleNavigation('settings')} className="text-[var(--dfs-text-secondary)] hover:text-[var(--dfs-accent)] text-xs">
              <Settings className="w-3.5 h-3.5 mr-2" />
              Settings
            </DropdownMenuItem>
            <DropdownMenuSeparator className="bg-[var(--dfs-border)]" />
            <DropdownMenuItem onClick={onLogout} className="text-red-400 focus:text-red-300 text-xs">
              <LogOut className="w-3.5 h-3.5 mr-2" />
              Logout
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </header>

      {/* Main Content */}
      <main className="flex-1 min-h-0 overflow-hidden">
        {renderMainContent}
      </main>
    </div>
  );
}
