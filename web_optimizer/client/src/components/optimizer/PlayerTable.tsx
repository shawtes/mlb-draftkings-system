import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { Input } from '../ui/input';
import { Checkbox } from '../ui/checkbox';
import { HoverCard, HoverCardTrigger, HoverCardContent } from '../ui/hover-card';
import {
  Users, ArrowUpDown, ArrowUp, ArrowDown, Lock, Ban,
} from 'lucide-react';
import { Sport, filterPlayersByPosition } from '../sport-config';
import { Player } from './types';
import { dfsApi } from '../../services/dfs-api';
import { TeamLogo } from '../ui/team-logo';

function standaloneCDF(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1.0 / (1.0 + p * absX);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX / 2);
  return 0.5 * (1.0 + sign * y);
}

// Mirror quant-engine.js _getPlayerStdDev priority chain
const getEffectiveStdDev = (player: Player): number => {
  const proj = player.projectedPoints || 0;
  if (player.garchConditionalVolatility && proj > 0) return player.garchConditionalVolatility * proj;
  if (player.garchVolatility && proj > 0) return player.garchVolatility * proj;
  if (player.stdDev) return player.stdDev;
  const ceiling = player.ceiling || proj * 1.3;
  const floor = player.floor || proj * 0.6;
  return (ceiling - floor) / 4;
};

const hasQuantProfile = (player: Player): boolean =>
  player.garchVolatility !== undefined || player.volatilityRegime !== undefined || player.rollingSharpe !== undefined;

const computeBoomPct = (player: Player): number => {
  // Use training pipeline's calibrated probability if available
  if (player.probOver15 !== undefined) return player.probOver15 * 100;
  const proj = player.projectedPoints || 0;
  if (proj <= 0) return 0;
  const stdDev = getEffectiveStdDev(player) || proj * 0.25;
  return (1 - standaloneCDF(0.5 * proj / stdDev)) * 100;
};

const computeLeverage = (player: Player): number => {
  const ceiling = player.ceiling || (player.projectedPoints || 0) * 1.4;
  const ownership = player.ownership || 0;
  const salary = player.salary || 1;
  return (ceiling * (1 - ownership / 100)) / salary * 1000;
};

// Composite quant score mirroring quant-engine.js scorePlayersQuant
const computeQuantScore = (player: Player, contestMode: 'gpp' | 'cash'): number => {
  const proj = player.projectedPoints || 0;
  const sal = player.salary || 1;
  const stdDev = getEffectiveStdDev(player) || proj * 0.25;
  const sharpe = stdDev > 0 ? (proj / sal * 1000) / stdDev : 0;
  const leverage = computeLeverage(player);
  const boomPct = computeBoomPct(player) / 100;
  const floor = player.floor || proj * 0.6;
  if (contestMode === 'cash') {
    return 0.4 * sharpe + 0.3 * (floor / sal * 1000) + 0.2 * (proj / sal * 1000) - 0.1 * stdDev;
  }
  return 0.35 * (leverage / 10) + 0.25 * sharpe + 0.2 * (boomPct * 100) + 0.2 * (proj / sal * 1000);
};

// Debounced numeric input for exposure fields — local state + flush on blur/idle
const DebouncedInput: React.FC<{
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
}> = React.memo(({ value, onChange, min = 0, max = 100 }) => {
  const [local, setLocal] = useState(String(value));
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync when parent value changes externally
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
      style={{
        width: 38, height: 20, textAlign: 'right',
        fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', fontFamily: "'JetBrains Mono', monospace",
        background: 'transparent', color: '#CBD5E1',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 3, outline: 'none', padding: '0 3px',
      }}
      onFocus={(e) => { e.target.style.borderColor = 'rgba(0,217,255,0.4)'; e.target.style.background = 'rgba(255,255,255,0.04)'; }}
    />
  );
});

const GRID_COLUMNS = '22px 22px 22px minmax(110px,1.1fr) 52px 34px 50px 46px 38px 38px 42px 42px 46px 42px 42px 44px 44px 38px 38px 38px 44px 38px 36px 50px 50px 50px';

interface PlayerTableProps {
  playerData: Player[];
  selectedPlayers: string[];
  sport: Sport;
  onPlayersChange: (ids: string[]) => void;
  onPlayerDataChange: (data: Player[]) => void;
  contestMode?: 'gpp' | 'cash';
  onContestModeChange?: (mode: 'gpp' | 'cash') => void;
  // Parent controls filters
  positionFilter?: string;
  searchQuery?: string;
  leveragePlaysFilter?: boolean;
  showAdvancedCols?: boolean;
}

type SortKey = 'name' | 'salary' | 'points' | 'value' | 'ownership' | 'ceiling' | 'floor' | 'stddev' | 'leverage' | 'boom' | 'bust' | 'garch' | 'sharpe' | 'hurst' | 'entropy' | 'corr' | 'evtceil' | 'exceed' | 'qscore';

const PlayerTable: React.FC<PlayerTableProps> = ({
  playerData, selectedPlayers, sport, onPlayersChange, onPlayerDataChange,
  contestMode = 'gpp', onContestModeChange,
  positionFilter: positionFilterProp,
  searchQuery: searchQueryProp,
  leveragePlaysFilter: leveragePlaysFilterProp,
  showAdvancedCols: showAdvancedColsProp,
}) => {
  // Use props with fallback defaults
  const positionFilter = positionFilterProp ?? (sport === 'MLB' ? 'all-batters' : sport === 'NFL' ? 'all-offense' : 'all');
  const searchQuery = searchQueryProp ?? '';
  const showAdvancedCols = showAdvancedColsProp ?? false;
  const leveragePlaysFilter = leveragePlaysFilterProp ?? false;

  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [sortBy, setSortBy] = useState<SortKey>('salary');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [editingProjection, setEditingProjection] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState('');
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(searchQuery), 200);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const handleColumnSort = (key: string) => {
    const sortKey = key as SortKey;
    if (sortBy === sortKey) setSortDirection(prev => prev === 'desc' ? 'asc' : 'desc');
    else { setSortBy(sortKey); setSortDirection('desc'); }
  };

  const debouncedSyncWithBackend = useCallback(async (playerId: string, selected: boolean) => {
    try { await dfsApi.updatePlayer(playerId, { selected }); } catch (e) { console.error('Sync error', e); }
  }, []);

  const filteredPlayers = useMemo(() => {
    let filtered = filterPlayersByPosition(playerData, positionFilter, sport);
    if (debouncedSearch) {
      const q = debouncedSearch.toLowerCase();
      filtered = filtered.filter((p: Player) =>
        p.name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q) || p.position.toLowerCase().includes(q)
      );
    }
    filtered.sort((a: Player, b: Player) => {
      let aVal: number | string = 0, bVal: number | string = 0;
      switch (sortBy) {
        case 'name': aVal = a.name.toLowerCase(); bVal = b.name.toLowerCase(); break;
        case 'salary': aVal = a.salary; bVal = b.salary; break;
        case 'points': aVal = a.projectedPoints; bVal = b.projectedPoints; break;
        case 'value': aVal = a.salary > 0 ? a.projectedPoints / a.salary * 1000 : 0; bVal = b.salary > 0 ? b.projectedPoints / b.salary * 1000 : 0; break;
        case 'ownership': aVal = a.ownership; bVal = b.ownership; break;
        case 'ceiling': aVal = a.ceiling || a.projectedPoints * 1.3; bVal = b.ceiling || b.projectedPoints * 1.3; break;
        case 'floor': aVal = a.floor || a.projectedPoints * 0.6; bVal = b.floor || b.projectedPoints * 0.6; break;
        case 'stddev': aVal = getEffectiveStdDev(a); bVal = getEffectiveStdDev(b); break;
        case 'leverage': aVal = computeLeverage(a); bVal = computeLeverage(b); break;
        case 'boom': aVal = computeBoomPct(a); bVal = computeBoomPct(b); break;
        case 'bust': { const sA = getEffectiveStdDev(a), sB = getEffectiveStdDev(b); const fA = (a.floor || a.projectedPoints * 0.6) * 1.2, fB = (b.floor || b.projectedPoints * 0.6) * 1.2; aVal = sA > 0 ? standaloneCDF((fA - a.projectedPoints) / sA) * 100 : 0; bVal = sB > 0 ? standaloneCDF((fB - b.projectedPoints) / sB) * 100 : 0; break; }
        case 'garch': aVal = a.garchVolatility || 0; bVal = b.garchVolatility || 0; break;
        case 'sharpe': aVal = a.rollingSharpe || 0; bVal = b.rollingSharpe || 0; break;
        case 'hurst': aVal = a.hurstExponent || 0; bVal = b.hurstExponent || 0; break;
        case 'entropy': aVal = a.entropy || 0; bVal = b.entropy || 0; break;
        case 'corr': aVal = a.avgPlayerCorrelation || 0; bVal = b.avgPlayerCorrelation || 0; break;
        case 'evtceil': aVal = a.evtReturnLevel || 0; bVal = b.evtReturnLevel || 0; break;
        case 'exceed': aVal = a.exceedanceProb || 0; bVal = b.exceedanceProb || 0; break;
        case 'qscore': aVal = computeQuantScore(a, contestMode); bVal = computeQuantScore(b, contestMode); break;
      }
      if (typeof aVal === 'string' && typeof bVal === 'string') return sortDirection === 'desc' ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
      return sortDirection === 'desc' ? (bVal as number) - (aVal as number) : (aVal as number) - (bVal as number);
    });
    // Compute leverage scores
    const salaries = filtered.map((p: Player) => p.salary > 0 ? p.projectedPoints / p.salary * 1000 : 0);
    salaries.sort((a, b) => a - b);
    const withLeverage = filtered.map((p: Player) => {
      const pVal = p.salary > 0 ? p.projectedPoints / p.salary * 1000 : 0;
      const rank = salaries.filter(v => v <= pVal).length;
      const valuePercentile = salaries.length > 0 ? (rank / salaries.length) * 100 : 0;
      const leverageScore = parseFloat((valuePercentile - p.ownership).toFixed(1));
      return { ...p, leverageScore };
    });
    if (leveragePlaysFilter) {
      const leverages = withLeverage.map((p: Player) => computeLeverage(p));
      const sortedLev = [...leverages].sort((a, b) => a - b);
      const medianLev = sortedLev.length > 0 ? sortedLev[Math.floor(sortedLev.length / 2)] : 0;
      return withLeverage.filter((p: Player) => computeLeverage(p) > medianLev && p.ownership < 15);
    }
    return withLeverage;
  }, [playerData, positionFilter, sortBy, sortDirection, debouncedSearch, sport, leveragePlaysFilter]);

  // Compute value percentiles for color-coding
  const valuePercentiles = useMemo(() => {
    const values = filteredPlayers.map((p: Player) => p.salary > 0 ? p.projectedPoints / p.salary * 1000 : 0);
    if (values.length === 0) return { p25: 0, p75: 0 };
    const sorted = [...values].sort((a, b) => a - b);
    const p25 = sorted[Math.floor(sorted.length * 0.25)] ?? 0;
    const p75 = sorted[Math.floor(sorted.length * 0.75)] ?? 0;
    return { p25, p75 };
  }, [filteredPlayers]);

  // Bar width scaling ranges for visual heatmap bars
  const barRanges = useMemo(() => ({
    maxSalary: Math.max(...filteredPlayers.map(p => p.salary), 1),
    maxProj: Math.max(...filteredPlayers.map(p => p.projectedPoints), 1),
    maxValue: Math.max(...filteredPlayers.map(p => p.salary > 0 ? p.projectedPoints / p.salary * 1000 : 0), 1),
    maxBoom: Math.max(...filteredPlayers.map(p => computeBoomPct(p)), 1),
    maxLeverage: Math.max(...filteredPlayers.map(p => computeLeverage(p)), 1),
  }), [filteredPlayers]);

  const getValueColor = useCallback((val: number) => {
    if (val >= valuePercentiles.p75) return 'text-green-400';
    if (val <= valuePercentiles.p25) return 'text-red-400';
    return 'text-cyan-400';
  }, [valuePercentiles]);

  const getValueBarColor = useCallback((val: number): string => {
    if (val >= valuePercentiles.p75) return '#22c55e';
    if (val <= valuePercentiles.p25) return '#ef4444';
    return '#00D9FF';
  }, [valuePercentiles]);

  const handleSelectAll = () => {
    const currentIds = filteredPlayers.map((p: Player) => p.id);
    const newSelected = Array.from(new Set([...selectedPlayers, ...currentIds]));
    onPlayersChange(newSelected);
  };

  const handleDeselectAll = () => {
    const currentIds = new Set(filteredPlayers.map((p: Player) => p.id));
    const newSelected = selectedPlayers.filter(id => !currentIds.has(id));
    onPlayersChange(newSelected);
  };

  const togglePlayer = useCallback((playerId: string, event?: React.MouseEvent) => {
    if (event) event.stopPropagation();
    const isCurrentlySelected = selectedPlayers.includes(playerId);
    const newSelected = isCurrentlySelected
      ? selectedPlayers.filter(id => id !== playerId)
      : [...selectedPlayers, playerId];
    onPlayersChange(newSelected);
    debouncedSyncWithBackend(playerId, !isCurrentlySelected).catch(console.error);
  }, [selectedPlayers, onPlayersChange, debouncedSyncWithBackend]);

  useEffect(() => {
    return () => { if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current); };
  }, []);

  const updateExposure = (playerId: string, field: 'minExp' | 'maxExp', value: number) => {
    const updated = playerData.map(p => {
      if (p.id === playerId) {
        const newValue = Math.max(0, Math.min(100, value));
        if (field === 'minExp' && newValue > p.maxExp) return { ...p, minExp: newValue, maxExp: newValue };
        if (field === 'maxExp' && newValue < p.minExp) return { ...p, maxExp: newValue, minExp: newValue };
        return { ...p, [field]: newValue };
      }
      return p;
    });
    onPlayerDataChange(updated);
  };

  const toggleLock = useCallback((playerId: string) => {
    const updated = playerData.map(p => {
      if (p.id === playerId) {
        const newLocked = !p.locked;
        return { ...p, locked: newLocked, excluded: newLocked ? false : p.excluded, selected: newLocked ? true : p.selected };
      }
      return p;
    });
    onPlayerDataChange(updated);
    const player = updated.find(p => p.id === playerId);
    if (player) {
      dfsApi.updatePlayer(playerId, { locked: player.locked, excluded: player.excluded, selected: player.selected }).catch(console.error);
      if (player.locked && !selectedPlayers.includes(playerId)) onPlayersChange([...selectedPlayers, playerId]);
    }
  }, [playerData, selectedPlayers, onPlayerDataChange, onPlayersChange]);

  const toggleExclude = useCallback((playerId: string) => {
    const updated = playerData.map(p => {
      if (p.id === playerId) {
        const newExcluded = !p.excluded;
        return { ...p, excluded: newExcluded, locked: newExcluded ? false : p.locked, selected: newExcluded ? false : p.selected };
      }
      return p;
    });
    onPlayerDataChange(updated);
    const player = updated.find(p => p.id === playerId);
    if (player) {
      dfsApi.updatePlayer(playerId, { locked: player.locked, excluded: player.excluded, selected: player.selected }).catch(console.error);
      if (player.excluded && selectedPlayers.includes(playerId)) onPlayersChange(selectedPlayers.filter(id => id !== playerId));
    }
  }, [playerData, selectedPlayers, onPlayerDataChange, onPlayersChange]);

  const commitProjectionEdit = useCallback((playerId: string, newValue: string) => {
    const numVal = parseFloat(newValue);
    if (isNaN(numVal) || numVal < 0) { setEditingProjection(null); return; }
    const updated = playerData.map(p => {
      if (p.id === playerId) {
        return { ...p, originalProjection: p.originalProjection ?? p.projectedPoints, projectedPoints: numVal, projectionEdited: true };
      }
      return p;
    });
    onPlayerDataChange(updated);
    dfsApi.updatePlayer(playerId, { projection: numVal }).catch(console.error);
    setEditingProjection(null);
  }, [playerData, onPlayerDataChange]);

  const resetProjection = useCallback((playerId: string) => {
    const updated = playerData.map(p => {
      if (p.id === playerId && p.originalProjection !== undefined) {
        return { ...p, projectedPoints: p.originalProjection, projectionEdited: false, originalProjection: undefined };
      }
      return p;
    });
    onPlayerDataChange(updated);
    const player = updated.find(p => p.id === playerId);
    if (player) dfsApi.updatePlayer(playerId, { projection: player.projectedPoints }).catch(console.error);
  }, [playerData, onPlayerDataChange]);

  const normalCDF = (x: number, mean: number, sd: number): number => {
    if (sd <= 0) return x >= mean ? 1 : 0;
    const z = (x - mean) / sd;
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const d = 0.3989422804014327 * Math.exp(-z * z / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744))));
    return z > 0 ? 1 - p : p;
  };

  if (playerData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="mb-4">
            <div className="w-16 h-16 mx-auto bg-slate-700 rounded-full flex items-center justify-center">
              <Users className="w-8 h-8 text-slate-400" />
            </div>
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">No Player Data</h3>
          <p className="text-[var(--dfs-text-secondary)] mb-4">Load a CSV file to view and select players</p>
        </div>
      </div>
    );
  }

  const selectedInFiltered = filteredPlayers.filter((p: Player) => selectedPlayers.includes(p.id)).length;
  const allFilteredSelected = filteredPlayers.length > 0 && selectedInFiltered === filteredPlayers.length;

  const lockedSalary = playerData.filter(p => p.locked).reduce((sum, p) => sum + p.salary, 0);

  // Column header definitions
  const columns: Array<{ key: string; label: string; noSort?: boolean }> = [
    { key: '_check', label: '', noSort: true },
    { key: '_lock', label: '', noSort: true },
    { key: '_exclude', label: '', noSort: true },
    { key: 'name', label: 'PLAYER' },
    { key: 'salary', label: 'SALARY' },
    { key: 'pos', label: 'POS', noSort: true },
    { key: 'team', label: 'TM', noSort: true },
    { key: 'points', label: 'PROJ' },
    { key: 'value', label: 'VAL' },
    { key: 'ownership', label: 'OWN%' },
    { key: 'boom', label: 'BOOM%' },
    { key: 'bust', label: 'BUST%' },
    { key: 'leverage', label: 'LEV' },
    { key: 'ceiling', label: 'CEIL' },
    { key: 'floor', label: 'FLR' },
    { key: 'stddev', label: 'STDEV' },
    { key: 'garch', label: 'GARCH' },
    { key: 'sharpe', label: 'SHRP' },
    { key: 'hurst', label: 'HURST' },
    { key: 'entropy', label: 'ENTR' },
    { key: 'corr', label: 'CORR' },
    { key: 'evtceil', label: 'EVT' },
    { key: '_minExp', label: 'MIN', noSort: true },
    { key: '_maxExp', label: 'MAX', noSort: true },
    { key: '_actExp', label: 'ACT', noSort: true },
  ];

  const sortKeyMap: Record<string, string> = { team: 'name' };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Scrollable container — header is sticky inside so it locks during both vertical & horizontal scroll */}
      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {/* Sticky header row */}
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
            textTransform: 'uppercase',
            letterSpacing: '0.3px',
            color: 'var(--dfs-text-muted, #5A6A7E)',
            fontWeight: 600,
            userSelect: 'none',
            position: 'sticky',
            top: 0,
            zIndex: 10,
          }}
        >
          {columns.map((col) => {
            const resolvedKey = sortKeyMap[col.key] || col.key;
            const isSorted = !col.noSort && sortBy === resolvedKey;
            const canSort = !col.noSort && !col.key.startsWith('_');
            return (
              <div
                key={col.key}
                onClick={() => canSort && handleColumnSort(resolvedKey)}
                style={{
                  cursor: canSort ? 'pointer' : 'default',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  justifyContent: col.key === 'name' ? 'flex-start' : 'flex-end',
                }}
              >
                {col.key === '_check' ? (
                  <Checkbox
                    checked={allFilteredSelected}
                    onCheckedChange={(checked: boolean | 'indeterminate') => {
                      if (checked) handleSelectAll(); else handleDeselectAll();
                    }}
                    className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
                    style={{ width: 12, height: 12 }}
                  />
                ) : (
                  <>
                    {col.label}
                    {isSorted ? (
                      sortDirection === 'desc' ? <ArrowDown style={{ width: 8, height: 8, color: '#00D9FF' }} /> : <ArrowUp style={{ width: 8, height: 8, color: '#00D9FF' }} />
                    ) : canSort ? (
                      <ArrowUpDown style={{ width: 8, height: 8, opacity: 0.2 }} />
                    ) : null}
                  </>
                )}
              </div>
            );
          })}
        </div>

        {/* Data rows */}
        {filteredPlayers.map((player, idx) => {
          const numericValue = player.salary > 0 ? player.projectedPoints / player.salary * 1000 : 0;
          const value = numericValue.toFixed(2);
          const isSelected = selectedPlayers.includes(player.id);
          const pCeiling = player.ceiling || player.projectedPoints * 1.3;
          const pFloor = player.floor || player.projectedPoints * 0.6;
          const pStdDev = getEffectiveStdDev(player);
          const hasQuant = hasQuantProfile(player);
          const boomThreshold = pCeiling * 0.85;
          const bustThreshold = pFloor * 1.2;
          const playerBoom = computeBoomPct(player);
          const playerLeverage = computeLeverage(player);
          const boomPct = player.probOver15 !== undefined
            ? player.probOver15 * 100
            : (pStdDev > 0 ? ((1 - normalCDF(boomThreshold, player.projectedPoints, pStdDev)) * 100) : 0);
          const bustPct = pStdDev > 0 ? (normalCDF(bustThreshold, player.projectedPoints, pStdDev) * 100) : 0;
          const isLocked = player.locked;
          const isExcluded = player.excluded;
          const ownColor = player.ownership > 25 ? 'text-orange-400 font-bold'
            : player.ownership > 15 ? 'text-yellow-400'
            : player.ownership > 5 ? 'text-white'
            : player.ownership > 0 ? 'text-green-400' : 'text-slate-500';
          const valueColor = getValueColor(numericValue);
          const valueBarColor = getValueBarColor(numericValue);
          const boomColor = playerBoom > 20 ? '#4ade80' : playerBoom >= 10 ? '#facc15' : '#64748b';
          const levColor = playerLeverage > 8 ? '#4ade80' : playerLeverage >= 5 ? '#facc15' : '#64748b';

          let rowBg = 'transparent';
          if (isLocked) rowBg = 'rgba(34,197,94,0.12)';
          else if (isExcluded) rowBg = 'rgba(239,68,68,0.08)';
          else if (isSelected) rowBg = 'rgba(0,217,255,0.04)';

          return (
            <div
              key={player.id}
              onClick={(e) => {
                const target = e.target as HTMLElement;
                if (target.tagName === 'INPUT' || target.closest('input') || target.closest('button')) return;
                togglePlayer(player.id, e);
              }}
              style={{
                display: 'grid',
                gridTemplateColumns: GRID_COLUMNS,
                height: 30,
                alignItems: 'center',
                padding: '0 8px',
                borderBottom: '1px solid var(--dfs-border)',
                background: rowBg,
                transition: 'background 0.04s',
                cursor: 'pointer',
                position: 'relative',
                opacity: isExcluded ? 0.18 : 1,
                animation: 'fadeIn 0.04s ease both',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.012)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = rowBg; }}
            >
              {/* Locked left bar — green accent */}
              {isLocked && (
                <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: '#22c55e', boxShadow: '0 0 6px rgba(34,197,94,0.6)' }} />
              )}
              {/* Excluded left bar — red accent */}
              {isExcluded && (
                <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: '#ef4444', boxShadow: '0 0 6px rgba(239,68,68,0.5)' }} />
              )}

              {/* Checkbox */}
              <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Checkbox
                  checked={isSelected}
                  onCheckedChange={(checked: boolean | 'indeterminate') => { if (checked !== isSelected) togglePlayer(player.id); }}
                  className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
                  style={{ width: 12, height: 12 }}
                />
              </div>

              {/* Lock button */}
              <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <button
                  onClick={() => toggleLock(player.id)}
                  style={{
                    padding: 2, borderRadius: 3,
                    border: isLocked ? '1px solid rgba(34,197,94,0.6)' : 'none',
                    background: isLocked ? 'rgba(34,197,94,0.25)' : 'transparent',
                    color: isLocked ? '#4ade80' : '#475569', cursor: 'pointer',
                    boxShadow: isLocked ? '0 0 8px rgba(34,197,94,0.6), inset 0 0 4px rgba(34,197,94,0.15)' : 'none',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    transition: 'all 0.15s ease',
                  }}
                  title={isLocked ? 'Unlock player' : 'Lock player into lineups'}
                >
                  <Lock style={{ width: 11, height: 11, strokeWidth: isLocked ? 2.5 : 2 }} />
                </button>
              </div>

              {/* Exclude button */}
              <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <button
                  onClick={() => toggleExclude(player.id)}
                  style={{
                    padding: 2, borderRadius: 3,
                    border: isExcluded ? '1px solid rgba(239,68,68,0.6)' : 'none',
                    background: isExcluded ? 'rgba(239,68,68,0.25)' : 'transparent',
                    color: isExcluded ? '#f87171' : '#475569', cursor: 'pointer',
                    boxShadow: isExcluded ? '0 0 8px rgba(239,68,68,0.6), inset 0 0 4px rgba(239,68,68,0.15)' : 'none',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    transition: 'all 0.15s ease',
                  }}
                  title={isExcluded ? 'Include player' : 'Exclude player from lineups'}
                >
                  <Ban style={{ width: 11, height: 11, strokeWidth: isExcluded ? 2.5 : 2 }} />
                </button>
              </div>

              {/* Name */}
              <div style={{
                fontSize: 10, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                color: isLocked ? '#4ade80' : isExcluded ? '#64748b' : '#E2E8F0',
                textDecoration: isExcluded ? 'line-through' : 'none',
                textDecorationColor: isExcluded ? 'rgba(239,68,68,0.5)' : undefined,
              }}>
                <HoverCard openDelay={300}>
                  <HoverCardTrigger asChild>
                    <span className="cursor-help hover:text-cyan-300 transition-colors" style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                      {player.name}
                      {isLocked && <span style={{ fontSize: 7, fontWeight: 700, color: '#22c55e', letterSpacing: '0.05em' }}>LOCK</span>}
                      {isExcluded && <span style={{ fontSize: 7, fontWeight: 700, color: '#ef4444', letterSpacing: '0.05em' }}>OUT</span>}
                      {hasQuant && !isLocked && !isExcluded && <span style={{ display: 'inline-block', width: 5, height: 5, borderRadius: '50%', background: '#a78bfa' }} title="Quant profile available" />}
                    </span>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-72 bg-slate-800 border-slate-600 text-white p-4" side="right">
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <TeamLogo team={player.team} sport={sport} size={20} />
                        <div>
                          <h4 className="font-semibold text-cyan-400 text-sm">{player.name}</h4>
                          <p className="text-xs text-slate-400">{player.position} - {player.team}{player.opponent ? ` vs ${player.opponent}` : ''}</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div><span className="text-slate-400">Salary:</span> <span className="text-white font-medium">${player.salary.toLocaleString()}</span></div>
                        <div><span className="text-slate-400">Proj:</span> <span className={`font-medium ${player.projectionEdited ? 'text-yellow-400' : 'text-white'}`}>{player.projectedPoints.toFixed(2)}</span></div>
                        <div><span className="text-slate-400">Own%:</span> <span className={`font-medium ${ownColor}`}>{player.ownership > 0 ? `${player.ownership.toFixed(1)}%` : '\u2014'}</span></div>
                        <div><span className="text-slate-400">Value:</span> <span className="text-cyan-400 font-medium">{value}</span></div>
                      </div>
                      <div className="border-t border-slate-700 pt-2">
                        <div className="text-xs text-slate-400 mb-1.5">Floor-to-Ceiling Range</div>
                        <div className="relative h-3 bg-slate-700 rounded-full overflow-hidden mb-2">
                          <div className="absolute h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full" style={{ left: `${Math.max(0, (pFloor / pCeiling) * 100)}%`, width: `${Math.min(100, 100 - (pFloor / pCeiling) * 100)}%` }} />
                          <div className="absolute h-full w-0.5 bg-white" style={{ left: `${Math.min(100, (player.projectedPoints / pCeiling) * 100)}%` }} />
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div><span className="text-slate-400">Floor:</span> <span className="text-red-400">{pFloor.toFixed(1)}</span></div>
                          <div className="text-center"><span className="text-slate-400">StdDev:</span> <span className="text-slate-300">{pStdDev.toFixed(1)}</span></div>
                          <div className="text-right"><span className="text-slate-400">Ceil:</span> <span className="text-green-400">{pCeiling.toFixed(1)}</span></div>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-xs mt-1">
                          <div><span className="text-slate-400">Bust%:</span> <span className="text-red-400">{bustPct.toFixed(1)}%</span></div>
                          <div className="text-center"><span className="text-slate-400">Lev:</span> <span className={`font-medium ${(player.leverageScore || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>{(player.leverageScore || 0) > 0 ? '+' : ''}{(player.leverageScore || 0).toFixed(1)}</span></div>
                          <div className="text-right"><span className="text-slate-400">Boom%:</span> <span className="text-green-400">{boomPct.toFixed(1)}%</span></div>
                        </div>
                      </div>
                      {hasQuant && (
                        <div className="border-t border-slate-700 pt-2">
                          <div className="text-xs text-purple-400 mb-1.5 font-semibold">Quant Profile</div>
                          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                            {player.garchVolatility !== undefined && (
                              <div><span className="text-slate-400">GARCH Vol:</span> <span className={`font-medium ${player.garchVolatility > 0.3 ? 'text-red-400' : player.garchVolatility > 0.15 ? 'text-yellow-400' : 'text-green-400'}`}>{(player.garchVolatility * 100).toFixed(1)}%</span></div>
                            )}
                            {player.rollingSharpe !== undefined && (
                              <div><span className="text-slate-400">Sharpe:</span> <span className={`font-medium ${player.rollingSharpe > 1.0 ? 'text-green-400' : player.rollingSharpe > 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>{player.rollingSharpe.toFixed(2)}</span></div>
                            )}
                            {player.hurstExponent !== undefined && (
                              <div><span className="text-slate-400">Hurst:</span> <span className={`font-medium ${player.hurstExponent > 0.5 ? 'text-green-400' : 'text-slate-300'}`}>{player.hurstExponent.toFixed(2)}</span></div>
                            )}
                            {player.entropy !== undefined && (
                              <div><span className="text-slate-400">Entropy:</span> <span className="text-slate-300 font-medium">{player.entropy.toFixed(2)}</span></div>
                            )}
                            {player.avgPlayerCorrelation !== undefined && (
                              <div><span className="text-slate-400">Corr:</span> <span className="text-slate-300 font-medium">{player.avgPlayerCorrelation.toFixed(2)}</span></div>
                            )}
                            {player.evtReturnLevel !== undefined && (
                              <div><span className="text-slate-400">EVT Ceil:</span> <span className="text-purple-400 font-medium">{player.evtReturnLevel.toFixed(1)}</span></div>
                            )}
                          </div>
                          {/* Regime badges */}
                          <div className="flex flex-wrap gap-1 mt-1.5">
                            {player.bullRegime === 1 && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-green-500/20 text-green-400 border border-green-500/30">BULL</span>}
                            {player.volatilityRegime === 1 && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-orange-500/20 text-orange-400 border border-orange-500/30">HIGH VOL</span>}
                            {player.consistencyRegime === 1 && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30">CONSISTENT</span>}
                            {player.momentumRegime === 1 && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">MOMENTUM</span>}
                          </div>
                        </div>
                      )}
                      {player.projectionEdited && (
                        <div className="border-t border-slate-700 pt-2 flex items-center justify-between text-xs">
                          <span className="text-yellow-400">Edited (was {player.originalProjection?.toFixed(2)})</span>
                          <button onClick={() => resetProjection(player.id)} className="text-cyan-400 hover:text-cyan-300">Reset</button>
                        </div>
                      )}
                    </div>
                  </HoverCardContent>
                </HoverCard>
              </div>

              {/* Salary with bar */}
              <div style={{ position: 'relative', overflow: 'hidden', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', color: '#E2E8F0' }}>
                <div style={{
                  position: 'absolute', inset: '2px 0', right: 0,
                  width: `${(player.salary / barRanges.maxSalary) * 100}%`,
                  marginLeft: 'auto',
                  background: 'linear-gradient(90deg, transparent 0%, rgba(0,217,255,0.18) 100%)',
                  borderRadius: 2, pointerEvents: 'none',
                }} />
                <span style={{ position: 'relative', zIndex: 1, fontVariantNumeric: 'tabular-nums' }}>
                  ${player.salary.toLocaleString()}
                </span>
              </div>

              {/* Position */}
              <div style={{ fontSize: '8.5px', color: '#8899AA', textAlign: 'right' }}>{player.position}</div>

              {/* Team */}
              <div style={{ fontSize: '8.5px', color: '#5A6A7E', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 2 }}>
                <TeamLogo team={player.team} sport={sport} size={12} />
                {player.team}
              </div>

              {/* Projection with bar + inline edit */}
              <div
                style={{ position: 'relative', overflow: 'hidden', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px' }}
                onDoubleClick={(e) => { e.stopPropagation(); setEditingProjection(player.id); setEditingValue(player.projectedPoints.toFixed(2)); }}
                onClick={(e) => e.stopPropagation()}
              >
                <div style={{
                  position: 'absolute', inset: '2px 0', right: 0,
                  width: `${(player.projectedPoints / barRanges.maxProj) * 100}%`,
                  marginLeft: 'auto',
                  background: 'linear-gradient(90deg, transparent 0%, rgba(34,197,94,0.20) 100%)',
                  borderRadius: 2, pointerEvents: 'none',
                }} />
                {editingProjection === player.id ? (
                  <Input type="number" step="0.01" value={editingValue} onChange={(e) => setEditingValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') commitProjectionEdit(player.id, editingValue); if (e.key === 'Escape') setEditingProjection(null); }} onBlur={() => commitProjectionEdit(player.id, editingValue)} autoFocus className="bg-slate-700 border-cyan-500 text-white text-xs h-6 w-14 text-right relative z-[1]" />
                ) : (
                  <span style={{ position: 'relative', zIndex: 1, fontVariantNumeric: 'tabular-nums', color: player.projectionEdited ? '#facc15' : '#E2E8F0', fontWeight: 500 }} title="Double-click to edit">
                    {player.projectedPoints.toFixed(2)}
                  </span>
                )}
              </div>

              {/* Value with bar */}
              <div className={valueColor} style={{ position: 'relative', overflow: 'hidden', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontWeight: 500 }}>
                <div style={{
                  position: 'absolute', inset: '2px 0', right: 0,
                  width: `${(numericValue / barRanges.maxValue) * 100}%`,
                  marginLeft: 'auto',
                  background: `linear-gradient(90deg, transparent 0%, ${valueBarColor}33 100%)`,
                  borderRadius: 2, pointerEvents: 'none',
                }} />
                <span style={{ position: 'relative', zIndex: 1, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
              </div>

              {/* Ownership */}
              <div className={ownColor} style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontVariantNumeric: 'tabular-nums' }}>
                {player.ownership > 0 ? `${player.ownership.toFixed(1)}%` : '\u2014'}
              </div>

              {/* Boom% with bar */}
              <div style={{ position: 'relative', overflow: 'hidden', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontWeight: 500, color: boomColor }}>
                <div style={{
                  position: 'absolute', inset: '2px 0', right: 0,
                  width: `${(playerBoom / barRanges.maxBoom) * 100}%`,
                  marginLeft: 'auto',
                  background: `linear-gradient(90deg, transparent 0%, ${boomColor}30 100%)`,
                  borderRadius: 2, pointerEvents: 'none',
                }} />
                <span style={{ position: 'relative', zIndex: 1, fontVariantNumeric: 'tabular-nums' }}>{playerBoom.toFixed(1)}%</span>
              </div>

              {/* Bust% */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontWeight: 500, color: bustPct > 30 ? '#f87171' : bustPct > 15 ? '#facc15' : '#64748b', fontVariantNumeric: 'tabular-nums' }}>
                {bustPct.toFixed(1)}%
              </div>

              {/* Leverage with bar */}
              <div style={{ position: 'relative', overflow: 'hidden', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontWeight: 500, color: levColor }}>
                <div style={{
                  position: 'absolute', inset: '2px 0', right: 0,
                  width: `${(playerLeverage / barRanges.maxLeverage) * 100}%`,
                  marginLeft: 'auto',
                  background: `linear-gradient(90deg, transparent 0%, ${levColor}30 100%)`,
                  borderRadius: 2, pointerEvents: 'none',
                }} />
                <span style={{ position: 'relative', zIndex: 1, fontVariantNumeric: 'tabular-nums' }}>{playerLeverage.toFixed(1)}</span>
              </div>

              {/* Ceiling */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', color: '#4ade80', fontVariantNumeric: 'tabular-nums' }}>
                {pCeiling.toFixed(1)}
              </div>

              {/* Floor */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', color: '#f87171', fontVariantNumeric: 'tabular-nums' }}>
                {pFloor.toFixed(1)}
              </div>

              {/* StdDev */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', color: pStdDev > 8 ? '#facc15' : '#8b949e', fontVariantNumeric: 'tabular-nums' }}>
                {pStdDev.toFixed(1)}
              </div>

              {/* GARCH Volatility */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', color: player.garchVolatility != null ? (player.garchVolatility > 0.3 ? '#f87171' : player.garchVolatility > 0.15 ? '#facc15' : '#4ade80') : '#484f58' }}>
                {player.garchVolatility != null ? `${(player.garchVolatility * 100).toFixed(1)}%` : '\u2014'}
              </div>

              {/* Rolling Sharpe */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', color: player.rollingSharpe != null ? (player.rollingSharpe > 1.0 ? '#4ade80' : player.rollingSharpe > 0.5 ? '#facc15' : '#f87171') : '#484f58' }}>
                {player.rollingSharpe != null ? player.rollingSharpe.toFixed(2) : '\u2014'}
              </div>

              {/* Hurst Exponent */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', color: player.hurstExponent != null ? (player.hurstExponent > 0.5 ? '#4ade80' : '#8b949e') : '#484f58' }}>
                {player.hurstExponent != null ? player.hurstExponent.toFixed(2) : '\u2014'}
              </div>

              {/* Entropy */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', color: player.entropy != null ? '#a78bfa' : '#484f58' }}>
                {player.entropy != null ? player.entropy.toFixed(2) : '\u2014'}
              </div>

              {/* Avg Correlation */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', color: player.avgPlayerCorrelation != null ? (player.avgPlayerCorrelation > 0.3 ? '#3b82f6' : '#8b949e') : '#484f58' }}>
                {player.avgPlayerCorrelation != null ? player.avgPlayerCorrelation.toFixed(2) : '\u2014'}
              </div>

              {/* EVT Return Level (Ceiling) */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', color: player.evtReturnLevel != null ? '#c084fc' : '#484f58' }}>
                {player.evtReturnLevel != null ? player.evtReturnLevel.toFixed(1) : '\u2014'}
              </div>

              {/* Min Exposure */}
              <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <DebouncedInput value={player.minExp} onChange={(v) => updateExposure(player.id, 'minExp', v)} />
              </div>

              {/* Max Exposure */}
              <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <DebouncedInput value={player.maxExp} onChange={(v) => updateExposure(player.id, 'maxExp', v)} />
              </div>

              {/* Actual Exposure */}
              <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: '8.5px', color: '#64748b', fontVariantNumeric: 'tabular-nums' }}>
                {player.actualExp !== undefined ? `${player.actualExp.toFixed(1)}%` : '\u2014'}
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
      }}>
        <span>
          <span style={{ color: '#00D9FF', fontWeight: 600 }}>{filteredPlayers.length}</span> players
          {' | '}
          <span style={{ color: '#00D9FF', fontWeight: 600 }}>{selectedPlayers.length}</span> selected
          {(() => { const lockedCount = playerData.filter(p => p.locked).length; return lockedCount > 0 ? <>{' | '}<span style={{ color: '#4ade80', fontWeight: 600 }}>{lockedCount}</span> locked (${lockedSalary.toLocaleString()})</> : null; })()}
          {(() => { const excludedCount = playerData.filter(p => p.excluded).length; return excludedCount > 0 ? <>{' | '}<span style={{ color: '#f87171', fontWeight: 600 }}>{excludedCount}</span> excluded</> : null; })()}
        </span>
        {selectedPlayers.length < 30 && (
          <span style={{ color: '#facc15' }}>Select 30+ for diverse lineups</span>
        )}
      </div>
    </div>
  );
};

export default PlayerTable;
