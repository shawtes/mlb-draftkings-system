import React, { useState } from 'react';
import {
  Plus, X, Upload, FileText, Download, Play, Save,
  Settings, ChevronUp, Users, Trophy,
} from 'lucide-react';
import { Sport } from '../sport-config';
import { BuildState } from './types';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Checkbox } from '../ui/checkbox';

interface BuildControlBarProps {
  /* Build management */
  builds: BuildState[];
  activeBuildId: string;
  currentSport: Sport | null;
  sportLocked: boolean;
  onSwitchBuild: (buildId: string) => void;
  onAddBuild: () => void;
  onRemoveBuild: (buildId: string) => void;
  onSportChange: (sport: Sport) => void;
  /* File actions */
  onUploadCsv: () => void;
  onOpenBlending: () => void;
  onUploadOwnership: () => void;
  onLoadEntries: () => void;
  /* Lineup config */
  numLineups: number;
  onNumLineupsChange: (n: number) => void;
  minUnique: number;
  onMinUniqueChange: (n: number) => void;
  minSalary: number;
  onMinSalaryChange: (n: number) => void;
  maxSalary: number;
  sortMethod: string;
  onSortMethodChange: (s: string) => void;
  disableKelly: boolean;
  onDisableKellyChange: (v: boolean) => void;
  /* Actions */
  onOptimize: () => void;
  onExport: () => void;
  isOptimizing: boolean;
  hasPlayerData: boolean;
  hasResults: boolean;
  /* Status info */
  playerCount: number;
  selectedCount: number;
  resultsCount: number;
}

const BuildControlBar: React.FC<BuildControlBarProps> = ({
  builds,
  activeBuildId,
  currentSport,
  sportLocked,
  onSwitchBuild,
  onAddBuild,
  onRemoveBuild,
  onSportChange,
  onUploadCsv,
  onOpenBlending,
  onUploadOwnership,
  onLoadEntries,
  numLineups,
  onNumLineupsChange,
  minUnique,
  onMinUniqueChange,
  minSalary,
  onMinSalaryChange,
  maxSalary,
  sortMethod,
  onSortMethodChange,
  disableKelly,
  onDisableKellyChange,
  onOptimize,
  onExport,
  isOptimizing,
  hasPlayerData,
  hasResults,
  playerCount,
  selectedCount,
  resultsCount,
}) => {
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <div className="bg-[var(--dfs-bg-primary)] border-b border-[var(--dfs-border)] flex-shrink-0 relative">
      {/* Single merged row */}
      <div className="flex items-center gap-2 px-3 h-[40px]">
        {/* === LEFT: Build tabs + Sport pills === */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {builds.map((build) => (
            <div
              key={build.id}
              className={`flex items-center gap-1 px-2 py-1 rounded text-xs cursor-pointer transition-colors ${
                build.id === activeBuildId
                  ? 'bg-[var(--dfs-bg-tertiary)] text-[var(--dfs-accent)] border border-[var(--dfs-accent)]/40'
                  : 'bg-[var(--dfs-bg-secondary)] text-[var(--dfs-text-secondary)] border border-transparent hover:bg-[var(--dfs-bg-hover)]'
              }`}
              onClick={() => onSwitchBuild(build.id)}
            >
              <span className="font-medium">{build.name}</span>
              {build.sport && (
                <span className="text-[10px] text-[var(--dfs-text-muted)]">({build.sport})</span>
              )}
              {builds.length > 1 && (
                <button
                  onClick={(e) => { e.stopPropagation(); onRemoveBuild(build.id); }}
                  className="hover:text-red-400 transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          ))}
          {builds.length < 5 && (
            <button
              onClick={onAddBuild}
              className="flex items-center gap-0.5 px-2 py-1 rounded text-xs bg-[var(--dfs-bg-secondary)] text-[var(--dfs-text-secondary)] hover:bg-[var(--dfs-bg-hover)] transition-colors border border-transparent"
            >
              <Plus className="w-3 h-3" />
              <span>New</span>
            </button>
          )}
        </div>

        <div className="h-5 w-px bg-[var(--dfs-border)] flex-shrink-0" />

        {/* Sport pills */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {(['NFL', 'NBA', 'MLB'] as Sport[]).map((s) => (
            <button
              key={s}
              onClick={() => onSportChange(s)}
              disabled={sportLocked}
              className={`px-2 py-0.5 rounded text-xs font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-95 ${
                currentSport === s
                  ? 'bg-gradient-to-r from-cyan-500 to-cyan-600 text-white shadow-sm shadow-cyan-500/30 border border-cyan-400/50'
                  : 'bg-[var(--dfs-bg-tertiary)] text-[var(--dfs-text-secondary)] border border-[var(--dfs-border)] hover:border-cyan-500/50 hover:text-white'
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="h-5 w-px bg-[var(--dfs-border)] flex-shrink-0" />

        {/* === CENTER: Inline settings + status === */}
        {currentSport && (
          <div className="flex items-center gap-2.5 flex-1 min-w-0">
            {/* Compact inline settings */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-[var(--dfs-text-muted)] uppercase tracking-wide">LU</span>
                <Input
                  type="number"
                  min={1}
                  max={500}
                  value={numLineups}
                  onChange={(e) => onNumLineupsChange(parseInt(e.target.value, 10) || 1)}
                  className="h-6 w-12 bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-[10px] text-[var(--dfs-text-primary)] px-1.5 rounded"
                />
              </div>
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-[var(--dfs-text-muted)] uppercase tracking-wide">Uniq</span>
                <Input
                  type="number"
                  min={0}
                  max={10}
                  value={minUnique}
                  onChange={(e) => onMinUniqueChange(parseInt(e.target.value, 10) || 0)}
                  className="h-6 w-10 bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-[10px] text-[var(--dfs-text-primary)] px-1.5 rounded"
                />
              </div>
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-[var(--dfs-text-muted)] uppercase tracking-wide">MinSal</span>
                <Input
                  type="number"
                  min={0}
                  max={maxSalary}
                  step={500}
                  value={minSalary}
                  onChange={(e) => onMinSalaryChange(parseInt(e.target.value, 10))}
                  className="h-6 w-16 bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-[10px] text-[var(--dfs-text-primary)] px-1.5 rounded"
                />
                <span className="text-[10px] text-[var(--dfs-text-muted)]">/{(maxSalary / 1000).toFixed(0)}k</span>
              </div>
            </div>

            {/* Subtle status text */}
            <div className="flex items-center gap-2 text-[10px] text-[var(--dfs-text-muted)] flex-shrink-0">
              {playerCount > 0 && (
                <>
                  <span className="flex items-center gap-0.5">
                    <Users className="w-2.5 h-2.5" />
                    <span className="text-[var(--dfs-text-secondary)]">{selectedCount}</span> sel
                  </span>
                  {resultsCount > 0 && (
                    <span className="flex items-center gap-0.5">
                      <Trophy className="w-2.5 h-2.5 text-[var(--dfs-accent)]" />
                      <span className="text-[var(--dfs-accent)]">{resultsCount}</span> LU
                    </span>
                  )}
                </>
              )}
              {isOptimizing && (
                <span className="flex items-center gap-1 text-yellow-400">
                  <div className="w-2.5 h-2.5 border-[1.5px] border-yellow-400 border-t-transparent rounded-full animate-spin" />
                  Building...
                </span>
              )}
            </div>
          </div>
        )}

        {!currentSport && <div className="flex-1" />}

        {/* === RIGHT: Settings gear + CSV + BUILD LINEUPS === */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {/* Settings dropdown toggle */}
          {currentSport && (
            <button
              onClick={() => setSettingsOpen(!settingsOpen)}
              className={`flex items-center justify-center w-7 h-7 rounded transition-colors ${
                settingsOpen
                  ? 'bg-[var(--dfs-accent)]/20 text-[var(--dfs-accent)]'
                  : 'text-[var(--dfs-text-muted)] hover:text-white hover:bg-[var(--dfs-bg-hover)]'
              }`}
              title="More settings"
            >
              <Settings className="w-3.5 h-3.5" />
            </button>
          )}

          {/* Upload CSV */}
          {currentSport && (
            <Button
              variant="secondary-action"
              size="sm"
              className="h-7 px-2 text-[11px] gap-1"
              onClick={onUploadCsv}
            >
              <ChevronUp className="w-3 h-3" />
              CSV
            </Button>
          )}

          {/* Export CSV */}
          {currentSport && hasResults && (
            <Button
              variant="ghost-action"
              size="sm"
              className="h-7 px-2 text-[11px] gap-1 text-yellow-300 hover:text-yellow-200"
              onClick={onExport}
            >
              <Save className="w-3 h-3" />
            </Button>
          )}

          {/* BUILD LINEUPS — Large, prominent, accent color */}
          {currentSport && (
            <Button
              onClick={onOptimize}
              disabled={isOptimizing || !hasPlayerData}
              className="h-8 px-4 text-xs font-bold rounded-md bg-gradient-to-r from-cyan-500 to-cyan-600 text-white shadow-lg shadow-cyan-500/30 hover:from-cyan-400 hover:to-cyan-500 hover:shadow-cyan-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-1.5 border border-cyan-400/50"
            >
              <Play className="w-3.5 h-3.5" />
              {isOptimizing ? 'BUILDING...' : 'BUILD LINEUPS'}
            </Button>
          )}
        </div>
      </div>

      {/* === Settings dropdown panel (gear icon toggle) === */}
      {settingsOpen && currentSport && (
        <div
          className="absolute right-3 top-[40px] z-50 bg-[var(--dfs-bg-tertiary)] border border-[var(--dfs-border)] rounded-lg shadow-xl p-3 w-64"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-[var(--dfs-text-primary)]">Settings</span>
            <button
              onClick={() => setSettingsOpen(false)}
              className="text-[var(--dfs-text-muted)] hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="space-y-2">
            {/* Predictions / Blending */}
            <button
              onClick={() => { onOpenBlending(); setSettingsOpen(false); }}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-[var(--dfs-text-secondary)] hover:bg-[var(--dfs-bg-hover)] hover:text-white transition-colors"
            >
              <FileText className="w-3.5 h-3.5" />
              Projection Blending
            </button>

            {/* Ownership CSV */}
            <button
              onClick={() => { onUploadOwnership(); setSettingsOpen(false); }}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-orange-300 hover:bg-[var(--dfs-bg-hover)] hover:text-orange-200 transition-colors"
            >
              <Upload className="w-3.5 h-3.5" />
              Upload Own%
            </button>

            {/* Load Entries */}
            <button
              onClick={() => { onLoadEntries(); setSettingsOpen(false); }}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-emerald-300 hover:bg-[var(--dfs-bg-hover)] hover:text-emerald-200 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Load DK Entries
            </button>

            <div className="h-px bg-[var(--dfs-border)] my-1" />

            {/* Sort method */}
            <div className="flex items-center gap-2 px-2 py-1">
              <span className="text-[10px] text-[var(--dfs-text-muted)] uppercase w-10">Sort</span>
              <select
                value={sortMethod}
                onChange={(e) => onSortMethodChange(e.target.value)}
                className="flex-1 h-6 rounded bg-[var(--dfs-bg-secondary)] border border-[var(--dfs-border)] text-[var(--dfs-text-primary)] text-xs px-1.5"
              >
                <option value="points">Points</option>
                <option value="value">Value</option>
                <option value="salary">Salary</option>
              </select>
            </div>

            {/* Disable Kelly */}
            <div className="flex items-center gap-2 px-2 py-1">
              <Checkbox
                id="bcb-disable-kelly"
                checked={disableKelly}
                onCheckedChange={(checked: boolean) => onDisableKellyChange(Boolean(checked))}
                className="h-3.5 w-3.5 border-slate-600 data-[state=checked]:bg-slate-800 data-[state=checked]:border-orange-400"
              />
              <label htmlFor="bcb-disable-kelly" className="text-xs text-[var(--dfs-text-secondary)] cursor-pointer">
                Disable Kelly Criterion
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BuildControlBar;
