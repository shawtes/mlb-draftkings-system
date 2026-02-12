import { useState, useMemo, useCallback, lazy, Suspense } from 'react';
import {
  Settings,
  LogOut,
  ChevronDown,
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import { DashboardLoader } from './SkeletonLoader';
import UrSimLogo from './UrSimLogo';
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
  { id: 'dfs-optimizer', label: 'Optimizer' },
  { id: 'games-hub', label: 'Games' },
  { id: 'prop-betting', label: 'Props' },
  { id: 'how-to-use', label: 'Guide' },
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
        className="flex items-center justify-between px-5 h-11 flex-shrink-0 border-b"
        style={{ backgroundColor: 'var(--dfs-bg-secondary)', borderColor: 'var(--dfs-border)' }}
      >
        {/* Left: Emblem + wordmark */}
        <div className="flex items-center gap-2.5">
          <UrSimLogo size={26} />
          <div className="flex items-baseline gap-1.5">
            <span
              className="text-sm font-bold tracking-wide"
              style={{
                background: 'linear-gradient(135deg, #00D9FF 0%, #00FFD4 50%, #8B5CF6 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              UrSim
            </span>
            <span className="text-[10px] text-[var(--dfs-text-muted)] font-medium tracking-widest uppercase">DFS</span>
          </div>
        </div>

        {/* Center: Text-only nav */}
        <nav className="flex items-center gap-0.5">
          {navItems.map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigation(item.id)}
                className={`relative px-3.5 py-1.5 rounded text-xs font-medium transition-all ${
                  isActive
                    ? 'text-[var(--dfs-accent)]'
                    : 'text-[var(--dfs-text-secondary)] hover:text-[var(--dfs-text-primary)]'
                }`}
              >
                {item.label}
                {isActive && (
                  <span
                    className="absolute bottom-0 left-1/2 -translate-x-1/2 h-[2px] rounded-full"
                    style={{
                      width: '60%',
                      background: 'linear-gradient(90deg, #00D9FF, #00FFD4)',
                    }}
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Right: Settings + Logout text links */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => handleNavigation('settings')}
            className="text-xs text-[var(--dfs-text-muted)] hover:text-[var(--dfs-text-primary)] transition-colors"
          >
            Settings
          </button>
          <span className="text-[var(--dfs-border)]">|</span>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-1 text-xs text-[var(--dfs-text-secondary)] hover:text-[var(--dfs-text-primary)] transition-colors">
                Account
                <ChevronDown className="w-3 h-3" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-44 bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)]">
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
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 min-h-0 overflow-hidden">
        {renderMainContent}
      </main>
    </div>
  );
}
