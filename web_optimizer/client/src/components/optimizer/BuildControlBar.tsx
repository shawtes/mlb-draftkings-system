import React, { useState } from 'react';
import {
  Plus, X, Upload, FileText, Download, Play, Save,
  Settings, ChevronUp, Users, Trophy,
} from 'lucide-react';
import { Sport } from '../sport-config';
import { BuildState, ContestPreset, CASH_FULL_PRESET, GPP_FULL_PRESET } from './types';
import { Checkbox } from '../ui/checkbox';

/* ------------------------------------------------------------------ */
/*  Props                                                              */
/* ------------------------------------------------------------------ */

export interface BuildNavBarProps {
  builds: BuildState[];
  activeBuildId: string;
  currentSport: Sport | null;
  sportLocked: boolean;
  onSwitchBuild: (id: string) => void;
  onAddBuild: () => void;
  onRemoveBuild: (id: string) => void;
  onSportChange: (sport: Sport) => void;
}

export interface FilterToolbarProps {
  currentSport: Sport | null;
  numLineups: number;
  onNumLineupsChange: (n: number) => void;
  minSalary: number;
  maxSalary: number;
  onUploadCsv: () => void;
  onOptimize: () => void;
  onExport: () => void;
  isOptimizing: boolean;
  hasPlayerData: boolean;
  hasResults: boolean;
  /* Settings panel */
  onOpenBlending: () => void;
  onUploadOwnership: () => void;
  onLoadEntries: () => void;
  minUnique: number;
  onMinUniqueChange: (n: number) => void;
  onMinSalaryChange: (n: number) => void;
  sortMethod: string;
  onSortMethodChange: (s: string) => void;
  disableKelly: boolean;
  onDisableKellyChange: (v: boolean) => void;
  /* Status */
  playerCount: number;
  selectedCount: number;
  resultsCount: number;
  /* Contest presets */
  activePreset: 'cash' | 'gpp' | 'custom';
  onApplyPreset: (preset: ContestPreset) => void;
}

interface BuildControlBarProps extends BuildNavBarProps, FilterToolbarProps {}

/* ------------------------------------------------------------------ */
/*  Shared styles                                                      */
/* ------------------------------------------------------------------ */

const dividerStyle: React.CSSProperties = {
  width: 1,
  height: 16,
  background: 'var(--dfs-border)',
  margin: '0 2px',
  flexShrink: 0,
};

const pillBtnBase: React.CSSProperties = {
  padding: '2px 8px',
  fontSize: 9,
  fontWeight: 600,
  borderRadius: 3,
  cursor: 'pointer',
  border: '1px solid var(--dfs-border)',
  lineHeight: '16px',
};

/* ------------------------------------------------------------------ */
/*  BuildNavBar  (L2 — 32px)                                           */
/* ------------------------------------------------------------------ */

export const BuildNavBar: React.FC<BuildNavBarProps> = ({
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
    <div
      style={{
        height: 32,
        display: 'flex',
        alignItems: 'center',
        padding: '0 12px',
        gap: 4,
        background: 'var(--dfs-bg-secondary)',
        borderBottom: '1px solid var(--dfs-border)',
        flexShrink: 0,
        fontSize: 9,
      }}
    >
      {/* Undo / Redo */}
      <button
        style={{
          ...pillBtnBase,
          background: 'transparent',
          color: 'var(--dfs-text-secondary)',
        }}
        title="Undo"
      >
        ↺
      </button>
      <button
        style={{
          ...pillBtnBase,
          background: 'transparent',
          color: 'var(--dfs-text-secondary)',
        }}
        title="Redo"
      >
        ↻
      </button>

      <div style={dividerStyle} />

      {/* Build tabs */}
      {builds.map((build) => (
        <button
          key={build.id}
          style={{
            ...pillBtnBase,
            background:
              build.id === activeBuildId
                ? 'var(--dfs-bg-surface3, #21262d)'
                : 'transparent',
            color:
              build.id === activeBuildId
                ? 'var(--dfs-text-primary)'
                : 'var(--dfs-text-secondary)',
          }}
          onClick={() => onSwitchBuild(build.id)}
        >
          {build.name}
          {build.sport && ` (${build.sport})`}
          {builds.length > 1 && (
            <span
              onClick={(e) => {
                e.stopPropagation();
                onRemoveBuild(build.id);
              }}
              style={{ marginLeft: 4, cursor: 'pointer' }}
            >
              ×
            </span>
          )}
        </button>
      ))}
      {builds.length < 5 && (
        <button
          style={{
            ...pillBtnBase,
            background: 'transparent',
            color: 'var(--dfs-text-secondary)',
          }}
          onClick={onAddBuild}
        >
          +
        </button>
      )}

      <div style={dividerStyle} />

      {/* Sport pills */}
      {(['NFL', 'NBA', 'MLB'] as Sport[]).map((s) => (
        <button
          key={s}
          onClick={() => onSportChange(s)}
          disabled={sportLocked}
          style={{
            ...pillBtnBase,
            background:
              currentSport === s
                ? 'linear-gradient(to right, #06b6d4, #0891b2)'
                : 'transparent',
            color: currentSport === s ? '#fff' : 'var(--dfs-text-secondary)',
            border:
              currentSport === s
                ? '1px solid rgba(34,211,238,0.5)'
                : '1px solid var(--dfs-border)',
            opacity: sportLocked ? 0.95 : 1,
            cursor: sportLocked ? 'not-allowed' : 'pointer',
          }}
        >
          {s}
        </button>
      ))}

      <div style={dividerStyle} />

      {/* Late Swap */}
      <button
        style={{
          padding: '2px 8px',
          fontSize: 9,
          fontWeight: 700,
          borderRadius: 3,
          border: 'none',
          background: '#22c55e',
          color: '#000',
          cursor: 'pointer',
          lineHeight: '16px',
        }}
      >
        Late Swap
      </button>
    </div>
  );
};

/* ------------------------------------------------------------------ */
/*  FilterToolbar  (L4 — 30px)                                         */
/* ------------------------------------------------------------------ */

export const FilterToolbar: React.FC<FilterToolbarProps> = ({
  currentSport,
  numLineups,
  onNumLineupsChange,
  minSalary,
  maxSalary,
  onUploadCsv,
  onOptimize,
  onExport,
  isOptimizing,
  hasPlayerData,
  hasResults,
  onOpenBlending,
  onUploadOwnership,
  onLoadEntries,
  minUnique,
  onMinUniqueChange,
  onMinSalaryChange,
  sortMethod,
  onSortMethodChange,
  disableKelly,
  onDisableKellyChange,
  playerCount,
  selectedCount,
  resultsCount,
  activePreset,
  onApplyPreset,
}) => {
  const [settingsOpen, setSettingsOpen] = useState(false);

  const inputStyle: React.CSSProperties = {
    width: 42,
    padding: '1px 3px',
    fontSize: 9,
    fontFamily: 'JetBrains Mono, monospace',
    background: 'var(--dfs-bg-primary)',
    border: '1px solid var(--dfs-border)',
    borderRadius: 2,
    color: 'var(--dfs-text-primary)',
    textAlign: 'center',
  };

  const labelStyle: React.CSSProperties = {
    fontSize: 8,
    color: 'var(--dfs-text-muted)',
    fontWeight: 600,
    flexShrink: 0,
  };

  const toolBtn: React.CSSProperties = {
    fontSize: 9,
    color: 'var(--dfs-text-secondary)',
    cursor: 'pointer',
    fontWeight: 500,
    background: 'transparent',
    border: 'none',
    padding: '2px 4px',
  };

  return (
    <div
      style={{
        height: 30,
        display: 'flex',
        alignItems: 'center',
        padding: '0 12px',
        gap: 6,
        background: 'var(--dfs-bg-secondary)',
        borderBottom: '1px solid var(--dfs-border)',
        flexShrink: 0,
        position: 'relative',
      }}
    >
      {/* Tool buttons */}
      {['Filters', 'Rules', 'Templates'].map((b) => (
        <button key={b} style={toolBtn}>
          {b}
        </button>
      ))}
      <button
        style={{
          ...toolBtn,
          color: settingsOpen ? 'var(--dfs-accent)' : 'var(--dfs-text-secondary)',
        }}
        onClick={() => setSettingsOpen(!settingsOpen)}
      >
        Settings
      </button>

      <div style={{ ...dividerStyle, height: 14 }} />

      {/* Cash / GPP Preset Buttons */}
      <div style={{ display: 'flex', overflow: 'hidden', borderRadius: 4, border: '1px solid var(--dfs-border)' }}>
        <button
          onClick={() => onApplyPreset(CASH_FULL_PRESET)}
          style={{
            padding: '2px 10px',
            fontSize: 9,
            fontWeight: 700,
            border: 'none',
            cursor: 'pointer',
            fontFamily: 'inherit',
            letterSpacing: 0.3,
            background: activePreset === 'cash' ? '#059669' : 'transparent',
            color: activePreset === 'cash' ? '#fff' : 'var(--dfs-text-muted)',
            transition: 'all 0.15s',
          }}
        >
          Cash
        </button>
        <div style={{ width: 1, background: 'var(--dfs-border)' }} />
        <button
          onClick={() => onApplyPreset(GPP_FULL_PRESET)}
          style={{
            padding: '2px 10px',
            fontSize: 9,
            fontWeight: 700,
            border: 'none',
            cursor: 'pointer',
            fontFamily: 'inherit',
            letterSpacing: 0.3,
            background: activePreset === 'gpp' ? '#7c3aed' : 'transparent',
            color: activePreset === 'gpp' ? '#fff' : 'var(--dfs-text-muted)',
            transition: 'all 0.15s',
          }}
        >
          GPP
        </button>
      </div>
      {activePreset === 'custom' && (
        <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontStyle: 'italic' }}>Custom</span>
      )}

      <div style={{ flex: 1 }} />

      {/* Status badges */}
      {playerCount > 0 && (
        <span style={{ fontSize: 8, color: 'var(--dfs-text-muted)', display: 'flex', alignItems: 'center', gap: 2 }}>
          <Users style={{ width: 10, height: 10 }} />
          <span style={{ color: 'var(--dfs-text-secondary)' }}>{selectedCount}</span> sel
        </span>
      )}
      {resultsCount > 0 && (
        <span style={{ fontSize: 8, color: 'var(--dfs-accent)', display: 'flex', alignItems: 'center', gap: 2 }}>
          <Trophy style={{ width: 10, height: 10 }} />
          {resultsCount} LU
        </span>
      )}
      {isOptimizing && (
        <span style={{ fontSize: 8, color: '#facc15', display: 'flex', alignItems: 'center', gap: 3 }}>
          <div
            style={{
              width: 10,
              height: 10,
              border: '1.5px solid #facc15',
              borderTopColor: 'transparent',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
            }}
          />
          Building...
        </span>
      )}

      <div style={{ ...dividerStyle, height: 14 }} />

      {/* Lineup Pool */}
      <span style={labelStyle}>Lineup Pool</span>
      <input
        type="number"
        value={numLineups}
        onChange={(e) => onNumLineupsChange(parseInt(e.target.value) || 1)}
        style={inputStyle}
      />

      {/* Min Salary */}
      <span style={labelStyle}>Min Salary</span>
      <input
        type="number"
        value={minSalary}
        onChange={(e) => onMinSalaryChange(parseInt(e.target.value))}
        style={{ ...inputStyle, width: 52 }}
      />

      {/* Max Salary */}
      <span style={labelStyle}>Max Salary</span>
      <input
        value={(maxSalary / 1000).toFixed(0) + 'k'}
        readOnly
        style={{ ...inputStyle, width: 36, cursor: 'default', opacity: 0.6 }}
      />

      {/* Upload CSV */}
      <button
        onClick={onUploadCsv}
        style={{
          padding: '2px 6px',
          fontSize: 8,
          fontWeight: 600,
          borderRadius: 3,
          border: '1px solid var(--dfs-border)',
          background: 'transparent',
          color: 'var(--dfs-text-secondary)',
          cursor: 'pointer',
        }}
      >
        CSV
      </button>

      {/* Export */}
      <button
        onClick={onExport}
        disabled={!hasResults}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          padding: '3px 10px',
          fontSize: 9,
          fontWeight: 700,
          borderRadius: 4,
          border: hasResults ? '1px solid rgba(253,224,71,0.4)' : '1px solid var(--dfs-border)',
          background: hasResults ? 'rgba(253,224,71,0.08)' : 'transparent',
          color: hasResults ? '#fde047' : 'var(--dfs-text-muted)',
          cursor: hasResults ? 'pointer' : 'default',
          opacity: hasResults ? 1 : 0.4,
          letterSpacing: '0.03em',
        }}
      >
        <Download style={{ width: 11, height: 11 }} />
        EXPORT
      </button>

      {/* BUILD LINEUPS */}
      <button
        onClick={onOptimize}
        disabled={isOptimizing || !hasPlayerData}
        style={{
          padding: '3px 10px',
          fontSize: 9,
          fontWeight: 700,
          borderRadius: 3,
          border: 'none',
          background:
            isOptimizing
              ? 'var(--dfs-bg-surface3, #21262d)'
              : 'var(--dfs-accent)',
          color: isOptimizing ? 'var(--dfs-accent)' : '#000',
          cursor:
            isOptimizing || !hasPlayerData ? 'not-allowed' : 'pointer',
          opacity: !hasPlayerData ? 0.3 : 1,
          lineHeight: '16px',
        }}
      >
        {isOptimizing ? 'Building...' : 'Rebuild Lineups'}
      </button>

      {/* Settings dropdown panel */}
      {settingsOpen && currentSport && (
        <div
          style={{
            position: 'absolute',
            right: 12,
            top: 30,
            zIndex: 50,
            background: 'var(--dfs-bg-tertiary, #161b22)',
            border: '1px solid var(--dfs-border)',
            borderRadius: 8,
            boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
            padding: 12,
            width: 256,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--dfs-text-primary)' }}>Settings</span>
            <button
              onClick={() => setSettingsOpen(false)}
              style={{ background: 'none', border: 'none', color: 'var(--dfs-text-muted)', cursor: 'pointer' }}
            >
              <X style={{ width: 14, height: 14 }} />
            </button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {/* Projection Blending */}
            <button
              onClick={() => { onOpenBlending(); setSettingsOpen(false); }}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                padding: '4px 8px', borderRadius: 4, fontSize: 12,
                color: 'var(--dfs-text-secondary)', background: 'transparent', border: 'none', cursor: 'pointer',
              }}
            >
              <FileText style={{ width: 14, height: 14 }} />
              Projection Blending
            </button>

            {/* Upload Own% */}
            <button
              onClick={() => { onUploadOwnership(); setSettingsOpen(false); }}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                padding: '4px 8px', borderRadius: 4, fontSize: 12,
                color: '#fdba74', background: 'transparent', border: 'none', cursor: 'pointer',
              }}
            >
              <Upload style={{ width: 14, height: 14 }} />
              Upload Own%
            </button>

            {/* Load DK Entries */}
            <button
              onClick={() => { onLoadEntries(); setSettingsOpen(false); }}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                padding: '4px 8px', borderRadius: 4, fontSize: 12,
                color: '#6ee7b7', background: 'transparent', border: 'none', cursor: 'pointer',
              }}
            >
              <Download style={{ width: 14, height: 14 }} />
              Load DK Entries
            </button>

            <div style={{ height: 1, background: 'var(--dfs-border)', margin: '2px 0' }} />

            {/* Sort method */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '2px 8px' }}>
              <span style={{ fontSize: 10, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', width: 36 }}>Sort</span>
              <select
                value={sortMethod}
                onChange={(e) => onSortMethodChange(e.target.value)}
                style={{
                  flex: 1, height: 24, borderRadius: 4,
                  background: 'var(--dfs-bg-secondary)', border: '1px solid var(--dfs-border)',
                  color: 'var(--dfs-text-primary)', fontSize: 12, padding: '0 6px',
                }}
              >
                <option value="points">Points</option>
                <option value="value">Value</option>
                <option value="salary">Salary</option>
              </select>
            </div>

            {/* Min Unique */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '2px 8px' }}>
              <span style={{ fontSize: 10, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', width: 36 }}>Uniq</span>
              <input
                type="number"
                min={0}
                max={10}
                value={minUnique}
                onChange={(e) => onMinUniqueChange(parseInt(e.target.value, 10) || 0)}
                style={{
                  width: 48, height: 24, borderRadius: 4,
                  background: 'var(--dfs-bg-secondary)', border: '1px solid var(--dfs-border)',
                  color: 'var(--dfs-text-primary)', fontSize: 12, padding: '0 6px', textAlign: 'center',
                }}
              />
            </div>

            {/* Disable Kelly */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '2px 8px' }}>
              <Checkbox
                id="bcb-disable-kelly"
                checked={disableKelly}
                onCheckedChange={(checked: boolean) => onDisableKellyChange(Boolean(checked))}
                className="h-3.5 w-3.5 border-slate-600 data-[state=checked]:bg-slate-800 data-[state=checked]:border-orange-400"
              />
              <label htmlFor="bcb-disable-kelly" style={{ fontSize: 12, color: 'var(--dfs-text-secondary)', cursor: 'pointer' }}>
                Disable Kelly Criterion
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/* ------------------------------------------------------------------ */
/*  Default export — backward-compatible combo                         */
/* ------------------------------------------------------------------ */

const BuildControlBar: React.FC<BuildControlBarProps> = (props) => (
  <>
    <BuildNavBar
      builds={props.builds}
      activeBuildId={props.activeBuildId}
      currentSport={props.currentSport}
      sportLocked={props.sportLocked}
      onSwitchBuild={props.onSwitchBuild}
      onAddBuild={props.onAddBuild}
      onRemoveBuild={props.onRemoveBuild}
      onSportChange={props.onSportChange}
    />
    <FilterToolbar
      currentSport={props.currentSport}
      numLineups={props.numLineups}
      onNumLineupsChange={props.onNumLineupsChange}
      minSalary={props.minSalary}
      maxSalary={props.maxSalary}
      onUploadCsv={props.onUploadCsv}
      onOptimize={props.onOptimize}
      onExport={props.onExport}
      isOptimizing={props.isOptimizing}
      hasPlayerData={props.hasPlayerData}
      hasResults={props.hasResults}
      onOpenBlending={props.onOpenBlending}
      onUploadOwnership={props.onUploadOwnership}
      onLoadEntries={props.onLoadEntries}
      minUnique={props.minUnique}
      onMinUniqueChange={props.onMinUniqueChange}
      onMinSalaryChange={props.onMinSalaryChange}
      sortMethod={props.sortMethod}
      onSortMethodChange={props.onSortMethodChange}
      disableKelly={props.disableKelly}
      onDisableKellyChange={props.onDisableKellyChange}
      playerCount={props.playerCount}
      selectedCount={props.selectedCount}
      resultsCount={props.resultsCount}
      activePreset={props.activePreset}
      onApplyPreset={props.onApplyPreset}
    />
  </>
);

export default BuildControlBar;
