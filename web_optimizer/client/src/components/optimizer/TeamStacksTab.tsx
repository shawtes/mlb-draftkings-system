import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { Checkbox } from '../ui/checkbox';
import { CheckSquare, XSquare, CheckCircle2, AlertTriangle, Link2 } from 'lucide-react';
import { Player, Team } from './types';
import { Sport } from '../sport-config';
import { TeamLogo } from '../ui/team-logo';

/* ── Debounced numeric input (mirrors PlayerTable pattern) ──── */
const DebouncedInput: React.FC<{
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  disabled?: boolean;
}> = React.memo(({ value, onChange, min = 0, max = 100, disabled }) => {
  const [local, setLocal] = useState(String(value));
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => { setLocal(String(value)); }, [value]);

  const flush = useCallback((v: string) => {
    const n = Math.max(min, Math.min(max, parseInt(v) || 0));
    if (n !== value) onChange(n);
  }, [value, onChange, min, max]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value;
    setLocal(v);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => flush(v), 500);
  }, [flush]);

  const handleBlur = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    flush(local);
  }, [flush, local]);

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  return (
    <input
      type="number" min={min} max={max}
      value={local}
      onChange={handleChange}
      onBlur={handleBlur}
      disabled={disabled}
      style={{
        width: 38, height: 20, textAlign: 'right' as const,
        fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', fontFamily: "'JetBrains Mono', monospace",
        background: 'transparent', color: disabled ? '#475569' : '#CBD5E1',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 3, outline: 'none', padding: '0 3px',
        opacity: disabled ? 0.4 : 1,
      }}
      onFocus={(e) => { e.target.style.borderColor = 'rgba(0,217,255,0.4)'; e.target.style.background = 'rgba(255,255,255,0.04)'; }}
    />
  );
});

const GRID_COLUMNS = '22px 22px 100px 60px 50px 52px 50px 38px 38px 50px';

interface TeamStacksTabProps {
  playerData: Player[];
  teamSelections: Record<number | 'all', string[]>;
  onTeamSelectionsChange: (selections: Record<number | 'all', string[]>) => void;
  onTeamExposuresChange?: (exposures: Record<string, { minExp: number; maxExp: number }>) => void;
  sport?: string;
  results?: any[];
}

const TeamStacksTab: React.FC<TeamStacksTabProps> = ({ playerData, teamSelections, onTeamSelectionsChange, onTeamExposuresChange, sport, results }) => {
  const isNBA = sport === 'NBA';
  const [activeStackSize, setActiveStackSize] = useState<'all' | number>('all');
  const [teamExposures, setTeamExposures] = useState<Record<string, { minExp: number; maxExp: number }>>({});

  const teams = useMemo(() => {
    const teamMap = new Map<string, Team>();
    playerData.forEach(player => {
      if (!teamMap.has(player.team)) {
        teamMap.set(player.team, {
          abbr: player.team, status: 'Active', gameTime: '7:00 PM', projRuns: 0, minExp: 0, maxExp: 100, playerCount: 0, batterCount: 0,
        });
      }
      const team = teamMap.get(player.team)!;
      team.playerCount++;
      if (!player.position.includes('P')) team.batterCount++;
      team.projRuns += player.projectedPoints / 10;
    });
    const result = Array.from(teamMap.values()).map(team => {
      const exp = teamExposures[team.abbr];
      if (exp) return { ...team, minExp: exp.minExp, maxExp: exp.maxExp };
      return team;
    });
    return result.sort((a, b) => a.abbr.localeCompare(b.abbr));
  }, [playerData, teamExposures]);

  const getSelectedTeams = (stackSize: 'all' | number): string[] => teamSelections[stackSize] || [];
  const getTeamCount = useCallback((t: Team) => isNBA ? t.playerCount : t.batterCount, [isNBA]);

  const actualTeamExposures = useMemo(() => {
    if (!results || results.length === 0) return new Map<string, number>();
    const totalLineups = results.length;
    const teamCounts = new Map<string, number>();
    results.forEach((lineup: any) => {
      const teamsInLineup = new Set<string>();
      (lineup.players || []).forEach((p: any) => { if (p.team) teamsInLineup.add(p.team); });
      teamsInLineup.forEach(t => { teamCounts.set(t, (teamCounts.get(t) || 0) + 1); });
    });
    const exposures = new Map<string, number>();
    teamCounts.forEach((count, team) => { exposures.set(team, parseFloat(((count / totalLineups) * 100).toFixed(1))); });
    return exposures;
  }, [results]);

  const updateTeamExposure = useCallback((abbr: string, field: 'minExp' | 'maxExp', value: number) => {
    const clamped = Math.max(0, Math.min(100, value));
    setTeamExposures(prev => {
      const current = prev[abbr] || { minExp: 0, maxExp: 100 };
      const updated = { ...current, [field]: clamped };
      if (field === 'minExp' && clamped > updated.maxExp) updated.maxExp = clamped;
      if (field === 'maxExp' && clamped < updated.minExp) updated.minExp = clamped;
      const newState = { ...prev, [abbr]: updated };
      onTeamExposuresChange?.(newState);
      return newState;
    });
  }, [onTeamExposuresChange]);

  const toggleTeam = useCallback((team: string, event?: React.MouseEvent) => {
    if (event) event.stopPropagation();
    const current = getSelectedTeams(activeStackSize);
    const updated = current.includes(team) ? current.filter(t => t !== team) : [...current, team];
    onTeamSelectionsChange({ ...teamSelections, [activeStackSize]: updated });
  }, [activeStackSize, teamSelections, onTeamSelectionsChange]);

  const handleSelectAll = () => {
    const allTeams = teams.filter(t => activeStackSize === 'all' || getTeamCount(t) >= (activeStackSize as number)).map(t => t.abbr);
    onTeamSelectionsChange({ ...teamSelections, [activeStackSize]: allTeams });
  };

  const handleDeselectAll = () => {
    onTeamSelectionsChange({ ...teamSelections, [activeStackSize]: [] });
  };

  if (teams.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: 16 }}>
            <div style={{ width: 64, height: 64, margin: '0 auto', background: 'var(--dfs-bg-secondary)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Link2 style={{ width: 32, height: 32, color: '#64748b' }} />
            </div>
          </div>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: '#E2E8F0', marginBottom: 8 }}>No Team Data</h3>
          <p style={{ fontSize: 10, color: 'var(--dfs-text-muted)' }}>Load players first to configure team stacks</p>
        </div>
      </div>
    );
  }

  const selectedCount = getSelectedTeams(activeStackSize).length;
  const eligibleTeams = activeStackSize === 'all' ? teams : teams.filter(t => getTeamCount(t) >= (activeStackSize as number));
  const allSelected = eligibleTeams.length > 0 && selectedCount === eligibleTeams.length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Stack Size Sub-Tabs */}
      <div style={{
        display: 'flex', gap: 4, padding: '6px 8px', flexShrink: 0,
        borderBottom: '1px solid var(--dfs-border)',
        background: 'var(--dfs-bg-secondary)',
        overflowX: 'auto',
      }}>
        {[
          { id: 'all' as const, label: 'All Stacks', count: getSelectedTeams('all').length },
          { id: 2, label: '2-Stack', count: getSelectedTeams(2).length },
          { id: 3, label: '3-Stack', count: getSelectedTeams(3).length },
          { id: 4, label: '4-Stack', count: getSelectedTeams(4).length },
          { id: 5, label: '5-Stack', count: getSelectedTeams(5).length },
        ].map((stack) => (
          <button
            key={stack.id}
            onClick={() => setActiveStackSize(stack.id)}
            style={{
              height: 24, fontSize: 9, padding: '0 10px',
              fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
              whiteSpace: 'nowrap', cursor: 'pointer',
              borderRadius: 3, border: '1px solid',
              borderColor: activeStackSize === stack.id ? 'rgba(0,217,255,0.4)' : 'rgba(255,255,255,0.08)',
              background: activeStackSize === stack.id ? 'rgba(0,217,255,0.12)' : 'transparent',
              color: activeStackSize === stack.id ? '#00D9FF' : '#8899AA',
              transition: 'all 0.1s',
            }}
          >
            {stack.label}
            {stack.count > 0 && (
              <span style={{
                marginLeft: 4, fontSize: 8, fontWeight: 700,
                background: '#00D9FF', color: '#0B1120',
                borderRadius: 6, padding: '1px 5px',
                display: 'inline-block', lineHeight: '14px',
              }}>{stack.count}</span>
            )}
          </button>
        ))}

        {/* Spacer + action buttons */}
        <div style={{ flex: 1 }} />
        <button
          onClick={handleSelectAll}
          style={{
            height: 24, fontSize: 8, padding: '0 8px',
            borderRadius: 3, border: '1px solid rgba(34,197,94,0.3)',
            background: 'rgba(34,197,94,0.06)', color: '#4ade80',
            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3,
            fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
          }}
        >
          <CheckSquare style={{ width: 10, height: 10 }} /> ALL
        </button>
        <button
          onClick={handleDeselectAll}
          style={{
            height: 24, fontSize: 8, padding: '0 8px',
            borderRadius: 3, border: '1px solid rgba(239,68,68,0.3)',
            background: 'rgba(239,68,68,0.06)', color: '#f87171',
            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3,
            fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
          }}
        >
          <XSquare style={{ width: 10, height: 10 }} /> NONE
        </button>
      </div>

      {/* Warning if selected teams lack enough players */}
      {activeStackSize !== 'all' && (() => {
        const selectedTeams = getSelectedTeams(activeStackSize);
        const insufficientTeams = selectedTeams.map(abbr => teams.find(t => t.abbr === abbr)).filter(t => t && getTeamCount(t) < (activeStackSize as number));
        return insufficientTeams.length > 0 ? (
          <div style={{
            padding: '4px 10px', fontSize: 8, flexShrink: 0,
            background: 'rgba(234,179,8,0.06)', borderBottom: '1px solid rgba(234,179,8,0.2)',
            color: '#facc15', fontFamily: "'JetBrains Mono', monospace",
          }}>
            {insufficientTeams.map(t => t!.abbr).join(', ')} {insufficientTeams.length === 1 ? 'has' : 'have'} fewer than {activeStackSize as number} {isNBA ? 'players' : 'batters'} — will be skipped.
          </div>
        ) : null;
      })()}

      {/* Team Exposure Summary — shown after optimization */}
      {results && results.length > 0 && actualTeamExposures.size > 0 && (
        <div style={{
          padding: '6px 8px', flexShrink: 0,
          borderBottom: '1px solid var(--dfs-border)',
          background: 'var(--dfs-bg-secondary)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 7, fontWeight: 700, color: '#00D9FF', textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: "'JetBrains Mono', monospace" }}>Team Exposure Summary</span>
            <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)' }}>{results.length} lineups</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 6 }}>
            {teams
              .filter(t => actualTeamExposures.has(t.abbr) || (teamExposures[t.abbr] && teamExposures[t.abbr].minExp > 0))
              .map(t => {
                const actual = actualTeamExposures.get(t.abbr) ?? 0;
                const exp = teamExposures[t.abbr];
                const minTarget = exp?.minExp ?? 0;
                const maxTarget = exp?.maxExp ?? 100;
                const inRange = actual >= minTarget && actual <= maxTarget;
                const barWidth = Math.min(actual, 100);
                return (
                  <div key={t.abbr} style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '4px 8px', borderRadius: 4,
                    background: inRange ? 'rgba(34,197,94,0.06)' : 'rgba(239,68,68,0.06)',
                    border: `1px solid ${inRange ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}`,
                  }}>
                    <TeamLogo team={t.abbr} sport={sport || 'MLB'} size={16} />
                    <span style={{ fontSize: 10, fontWeight: 700, color: '#E2E8F0', width: 32 }}>{t.abbr}</span>
                    <div style={{ flex: 1, position: 'relative', height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{
                        width: `${barWidth}%`, height: '100%', borderRadius: 3,
                        background: inRange ? '#22c55e' : '#ef4444',
                        transition: 'width 0.3s ease',
                      }} />
                    </div>
                    <span style={{
                      fontSize: 9, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
                      color: inRange ? '#4ade80' : '#f87171', minWidth: 36, textAlign: 'right',
                    }}>{actual.toFixed(1)}%</span>
                    {inRange
                      ? <CheckCircle2 style={{ width: 11, height: 11, color: '#22c55e', flexShrink: 0 }} />
                      : <AlertTriangle style={{ width: 11, height: 11, color: '#f87171', flexShrink: 0 }} />
                    }
                    <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', minWidth: 48, textAlign: 'right' }}>
                      {minTarget > 0 || maxTarget < 100 ? `${minTarget}-${maxTarget}%` : ''}
                    </span>
                  </div>
                );
              })}
          </div>
          {/* Summary stats */}
          {(() => {
            const exposedTeams = teams.filter(t => actualTeamExposures.has(t.abbr));
            const outOfRange = exposedTeams.filter(t => {
              const actual = actualTeamExposures.get(t.abbr) ?? 0;
              const exp = teamExposures[t.abbr];
              const minTarget = exp?.minExp ?? 0;
              const maxTarget = exp?.maxExp ?? 100;
              return actual < minTarget || actual > maxTarget;
            });
            return (
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 8, paddingTop: 4, marginTop: 4, borderTop: '1px solid var(--dfs-border)' }}>
                <span style={{ color: 'var(--dfs-text-muted)' }}>{exposedTeams.length} teams with exposure</span>
                {outOfRange.length > 0
                  ? <span style={{ color: '#f87171', fontWeight: 600 }}>{outOfRange.length} team{outOfRange.length > 1 ? 's' : ''} outside target ({outOfRange.map(t => t.abbr).join(', ')})</span>
                  : <span style={{ color: '#4ade80', fontWeight: 600 }}>All within target</span>
                }
              </div>
            );
          })()}
        </div>
      )}

      {/* Grid Header */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: GRID_COLUMNS,
          height: 26,
          alignItems: 'center',
          padding: '0 8px',
          background: 'var(--dfs-bg-secondary)',
          borderBottom: '1px solid rgba(45,212,191,0.25)',
          fontSize: 7,
          textTransform: 'uppercase' as const,
          letterSpacing: '0.3px',
          color: 'var(--dfs-text-muted, #5A6A7E)',
          fontWeight: 600,
          userSelect: 'none' as const,
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Checkbox
            checked={allSelected}
            onCheckedChange={(checked: boolean | 'indeterminate') => { if (checked) handleSelectAll(); else handleDeselectAll(); }}
            className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
            style={{ width: 12, height: 12 }}
          />
        </div>
        <div />
        <div>TEAM</div>
        <div>STATUS</div>
        <div>TIME</div>
        <div style={{ textAlign: 'right' }}>PROJ</div>
        <div style={{ textAlign: 'right' }}>{isNBA ? 'PLRS' : 'BATS'}</div>
        <div style={{ textAlign: 'right' }}>MIN</div>
        <div style={{ textAlign: 'right' }}>MAX</div>
        <div style={{ textAlign: 'right' }}>ACTUAL</div>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {teams.map((team, idx) => {
          const isSelected = getSelectedTeams(activeStackSize).includes(team.abbr);
          const canStack = activeStackSize === 'all' || getTeamCount(team) >= (activeStackSize as number);

          let rowBg = 'transparent';
          if (isSelected && canStack) rowBg = 'rgba(0,217,255,0.06)';

          return (
            <div
              key={team.abbr}
              onClick={(e) => {
                if (!canStack) return;
                const target = e.target as HTMLElement;
                if (target.tagName === 'INPUT' || target.closest('input')) return;
                toggleTeam(team.abbr, e);
              }}
              style={{
                display: 'grid',
                gridTemplateColumns: GRID_COLUMNS,
                height: 28,
                alignItems: 'center',
                padding: '0 8px',
                borderBottom: '1px solid var(--dfs-border)',
                background: rowBg,
                cursor: canStack ? 'pointer' : 'not-allowed',
                opacity: canStack ? 1 : 0.4,
                transition: 'background 0.04s',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '8.5px',
              }}
              onMouseEnter={(e) => { if (canStack) e.currentTarget.style.background = 'rgba(255,255,255,0.012)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = rowBg; }}
            >
              {/* Checkbox */}
              <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Checkbox
                  checked={isSelected}
                  onCheckedChange={(checked: boolean | 'indeterminate') => { if (!canStack) return; if (checked !== isSelected) toggleTeam(team.abbr); }}
                  disabled={!canStack}
                  className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
                  style={{ width: 12, height: 12 }}
                />
              </div>

              {/* Team logo */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <TeamLogo team={team.abbr} sport={sport || 'MLB'} size={16} />
              </div>

              {/* Team name */}
              <div style={{ fontSize: 10, fontWeight: 700, color: isSelected ? '#00D9FF' : '#E2E8F0' }}>
                {team.abbr}
              </div>

              {/* Status */}
              <div>
                <span style={{
                  fontSize: 7, fontWeight: 600, padding: '2px 5px', borderRadius: 3,
                  background: team.status === 'Active' ? 'rgba(34,197,94,0.15)' : team.status === 'Postponed' ? 'rgba(239,68,68,0.15)' : 'rgba(100,116,139,0.15)',
                  color: team.status === 'Active' ? '#4ade80' : team.status === 'Postponed' ? '#f87171' : '#94a3b8',
                }}>{team.status}</span>
              </div>

              {/* Game time */}
              <div style={{ color: '#8899AA' }}>{team.gameTime}</div>

              {/* Proj */}
              <div style={{
                textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontWeight: 500,
                color: team.projRuns > 5.0 ? '#E2E8F0' : team.projRuns > 4.0 ? '#facc15' : '#64748b',
              }}>
                {team.projRuns.toFixed(1)}
              </div>

              {/* Player/Batter count */}
              <div style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: '#00D9FF', fontWeight: 500 }}>
                {isNBA ? team.playerCount : team.batterCount}
              </div>

              {/* Min Exposure */}
              <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <DebouncedInput value={team.minExp} onChange={(v) => updateTeamExposure(team.abbr, 'minExp', v)} disabled={!canStack} />
              </div>

              {/* Max Exposure */}
              <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <DebouncedInput value={team.maxExp} onChange={(v) => updateTeamExposure(team.abbr, 'maxExp', v)} disabled={!canStack} />
              </div>

              {/* Actual Exposure */}
              <div style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                {(() => {
                  const actual = actualTeamExposures.get(team.abbr);
                  if (actual === undefined) return <span style={{ color: '#475569' }}>{'\u2014'}</span>;
                  const exp = teamExposures[team.abbr];
                  const minTarget = exp?.minExp ?? 0;
                  const maxTarget = exp?.maxExp ?? 100;
                  const inRange = actual >= minTarget && actual <= maxTarget;
                  return <span style={{ color: inRange ? '#4ade80' : '#f87171', fontWeight: 600 }}>{actual.toFixed(1)}%</span>;
                })()}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer bar */}
      <div style={{
        height: 22,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 10px',
        fontSize: 8,
        color: 'var(--dfs-text-muted, #5A6A7E)',
        borderTop: '1px solid var(--dfs-border)',
        flexShrink: 0,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        <span>
          <span style={{ color: '#00D9FF', fontWeight: 600 }}>{activeStackSize === 'all' ? 'All Stacks' : `${activeStackSize}-Stack`}</span>
          {' | '}
          {selectedCount > 0
            ? <><span style={{ color: '#00D9FF', fontWeight: 600 }}>{selectedCount}</span> teams: {getSelectedTeams(activeStackSize).join(', ')}</>
            : <span style={{ color: '#475569' }}>No teams selected</span>
          }
        </span>
        {activeStackSize !== 'all' && (
          <span style={{ color: 'var(--dfs-text-muted)' }}>
            Eligible: {eligibleTeams.length} teams with {activeStackSize}+ {isNBA ? 'players' : 'batters'}
          </span>
        )}
      </div>
    </div>
  );
};

export default TeamStacksTab;
