import React from 'react';
import { Plus, X } from 'lucide-react';
import { Sport } from '../sport-config';
import { BuildState } from './types';

interface BuildBarProps {
  builds: BuildState[];
  activeBuildId: string;
  currentSport: Sport | null;
  sportLocked: boolean;
  onSwitchBuild: (buildId: string) => void;
  onAddBuild: () => void;
  onRemoveBuild: (buildId: string) => void;
  onSportChange: (sport: Sport) => void;
}

const BuildBar: React.FC<BuildBarProps> = ({
  builds,
  activeBuildId,
  currentSport,
  sportLocked,
  onSwitchBuild,
  onAddBuild,
  onRemoveBuild,
  onSportChange,
}) => {
  return (
    <div className="bg-[var(--dfs-bg-primary)] border-b border-[var(--dfs-border)] flex items-center gap-3 px-4 py-2">
      {/* Build Tabs */}
      <div className="flex items-center gap-1.5">
        {builds.map((build) => (
          <div
            key={build.id}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm cursor-pointer transition-colors ${
              build.id === activeBuildId
                ? 'bg-[var(--dfs-bg-tertiary)] text-[var(--dfs-accent)] border border-[var(--dfs-accent)]/40'
                : 'bg-[var(--dfs-bg-secondary)] text-[var(--dfs-text-secondary)] border border-transparent hover:bg-[var(--dfs-bg-hover)] hover:text-[var(--dfs-text-primary)]'
            }`}
            onClick={() => onSwitchBuild(build.id)}
          >
            <span className="font-medium">{build.name}</span>
            {build.sport && (
              <span className="text-xs text-[var(--dfs-text-muted)]">({build.sport})</span>
            )}
            {builds.length > 1 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRemoveBuild(build.id);
                }}
                className="ml-0.5 hover:text-red-400 transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        ))}
        {builds.length < 5 && (
          <button
            onClick={onAddBuild}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm bg-[var(--dfs-bg-secondary)] text-[var(--dfs-text-secondary)] hover:bg-[var(--dfs-bg-hover)] hover:text-[var(--dfs-text-primary)] transition-colors border border-transparent"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>New</span>
          </button>
        )}
      </div>

      <div className="h-6 w-px bg-[var(--dfs-border)]" />

      {/* Sport Selection */}
      <div className="flex items-center gap-2.5">
        <span className="text-xs uppercase tracking-wider text-[var(--dfs-text-muted)] font-medium">Sport</span>
        <div className="flex gap-1.5">
          {(['NFL', 'NBA', 'MLB'] as Sport[]).map((sportOption) => (
            <button
              key={sportOption}
              onClick={() => onSportChange(sportOption)}
              disabled={sportLocked}
              className={`px-3 py-1.5 rounded-md text-sm font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-95 ${
                currentSport === sportOption
                  ? 'bg-gradient-to-r from-cyan-500 to-cyan-600 text-white shadow-md shadow-cyan-500/30 border border-cyan-400/50'
                  : 'bg-[var(--dfs-bg-tertiary)] text-[var(--dfs-text-secondary)] border border-[var(--dfs-border)] hover:bg-[var(--dfs-bg-hover)] hover:border-cyan-500/50 hover:text-white'
              }`}
              title={
                sportLocked && currentSport !== sportOption
                  ? 'Sport selection is locked for this build'
                  : undefined
              }
            >
              {sportOption}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1" />

      {/* Build Status */}
      <div className="flex items-center gap-2 text-right">
        <div className="text-xs text-[var(--dfs-text-muted)]">
          <span className="font-semibold text-[var(--dfs-text-primary)]">
            {sportLocked ? `${currentSport} Active` : 'No Sport Selected'}
          </span>
          <span className="ml-2">{builds.length} {builds.length === 1 ? 'build' : 'builds'}</span>
        </div>
      </div>
    </div>
  );
};

export default BuildBar;
