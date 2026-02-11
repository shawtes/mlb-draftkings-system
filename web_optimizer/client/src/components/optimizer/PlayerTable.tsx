import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Checkbox } from '../ui/checkbox';
import { HoverCard, HoverCardTrigger, HoverCardContent } from '../ui/hover-card';
import {
  Users, Search, CheckSquare, XSquare, Eye, EyeOff,
  ArrowUpDown, ArrowUp, ArrowDown, Lock, Ban,
} from 'lucide-react';
import { Sport, getPositionFilters, getPositionCount, filterPlayersByPosition } from '../sport-config';
import { Player } from './types';
import { dfsApi } from '../../services/dfs-api';

interface PlayerTableProps {
  playerData: Player[];
  selectedPlayers: string[];
  sport: Sport;
  onPlayersChange: (ids: string[]) => void;
  onPlayerDataChange: (data: Player[]) => void;
}

type SortKey = 'name' | 'salary' | 'points' | 'value' | 'ownership' | 'ceiling' | 'leverage';

const PlayerTable: React.FC<PlayerTableProps> = ({
  playerData, selectedPlayers, sport, onPlayersChange, onPlayerDataChange,
}) => {
  const [positionFilter, setPositionFilter] = useState<string>(() => {
    if (sport === 'MLB') return 'all-batters';
    if (sport === 'NFL') return 'all-offense';
    return 'all';
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [sortBy, setSortBy] = useState<SortKey>('salary');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [showAdvancedCols, setShowAdvancedCols] = useState(false);
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
        case 'leverage': aVal = a.leverageScore || 0; bVal = b.leverageScore || 0; break;
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
    return withLeverage;
  }, [playerData, positionFilter, sortBy, sortDirection, debouncedSearch, sport]);

  // Compute value percentiles for color-coding (Change #4)
  const valuePercentiles = useMemo(() => {
    const values = filteredPlayers.map((p: Player) => p.salary > 0 ? p.projectedPoints / p.salary * 1000 : 0);
    if (values.length === 0) return { p25: 0, p75: 0 };
    const sorted = [...values].sort((a, b) => a - b);
    const p25 = sorted[Math.floor(sorted.length * 0.25)] ?? 0;
    const p75 = sorted[Math.floor(sorted.length * 0.75)] ?? 0;
    return { p25, p75 };
  }, [filteredPlayers]);

  const getValueColor = useCallback((val: number) => {
    if (val >= valuePercentiles.p75) return 'text-green-400';
    if (val <= valuePercentiles.p25) return 'text-red-400';
    return 'text-cyan-400';
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

  return (
    <div className="flex flex-col h-full space-y-2">
      {/* Change #5 & #6: Merged position pills + search + actions into one row */}
      <div className="flex flex-wrap items-center gap-1.5 pb-1">
        {getPositionFilters(sport).map((pos) => {
          const count = getPositionCount(playerData, pos.id, sport);
          return (
            <button
              key={pos.id}
              onClick={() => setPositionFilter(pos.id)}
              className={`px-2.5 py-1 rounded whitespace-nowrap transition-all text-xs ${
                positionFilter === pos.id
                  ? 'bg-cyan-500/20 text-[var(--dfs-accent)] border border-cyan-500/40'
                  : 'bg-[var(--dfs-bg-tertiary)] text-[var(--dfs-text-secondary)] border border-[var(--dfs-border)] hover:bg-[var(--dfs-bg-hover)] hover:text-white'
              }`}
            >
              {pos.label} <span className="text-[10px] opacity-70">({count})</span>
            </button>
          );
        })}
        {/* Separator */}
        <div className="w-px h-5 bg-[var(--dfs-border)] mx-1" />
        {/* Search inline */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-slate-400" />
          <Input
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-7 h-7 w-40 bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white text-xs"
          />
        </div>
        {/* Action buttons inline */}
        <Button variant="secondary-action" size="sm" onClick={handleSelectAll} className="border-green-500/30 bg-green-500/5 h-7 px-2 text-xs">
          <CheckSquare className="w-3.5 h-3.5 mr-1" /> All
        </Button>
        <Button variant="secondary-action" size="sm" onClick={handleDeselectAll} className="border-red-500/30 bg-red-500/5 h-7 px-2 text-xs">
          <XSquare className="w-3.5 h-3.5 mr-1" /> None
        </Button>
        <Button
          variant="secondary-action"
          size="sm"
          onClick={() => setShowAdvancedCols(prev => !prev)}
          className={`h-7 px-2 text-xs ${showAdvancedCols ? 'border-purple-500/40 bg-purple-500/10 text-purple-300' : ''}`}
        >
          {showAdvancedCols ? <EyeOff className="w-3.5 h-3.5 mr-1" /> : <Eye className="w-3.5 h-3.5 mr-1" />}
          Boom/Bust
        </Button>
      </div>

      {/* Player Table */}
      <div className="flex-1 overflow-auto max-h-96 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
        <table className="w-full text-sm">
          <thead className="bg-[var(--dfs-bg-tertiary)] sticky top-0 z-10">
            <tr className="border-b border-[var(--dfs-border)]">
              {/* Change #7: text-[11px] for header font */}
              <th className="px-2 py-2 text-left text-[11px] font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-24">
                <Checkbox
                  checked={allFilteredSelected}
                  onCheckedChange={(checked: boolean | 'indeterminate') => {
                    if (checked) handleSelectAll(); else handleDeselectAll();
                  }}
                  className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
                />
              </th>
              {[
                { key: 'name', label: 'Name', align: 'left', width: 'min-w-[140px]' },
                { key: 'team', label: 'Team', align: 'left', width: 'w-14' },
                { key: 'opp', label: 'Opp', align: 'left', width: 'w-14', noSort: true },
                { key: 'pos', label: 'Pos', align: 'left', width: 'w-16', noSort: true },
                { key: 'salary', label: 'Salary', align: 'right', width: 'w-20' },
                { key: 'points', label: 'Proj', align: 'right', width: 'w-16' },
                { key: 'value', label: 'Value', align: 'right', width: 'w-16' },
                { key: 'ownership', label: 'Own%', align: 'right', width: 'w-14' },
              ].map(col => (
                <th
                  key={col.key}
                  className={`px-2 py-2 text-${col.align} text-[11px] font-semibold text-[var(--dfs-accent)] uppercase tracking-wider ${col.width} ${col.noSort ? '' : 'cursor-pointer hover:text-cyan-300 select-none'}`}
                  onClick={() => !col.noSort && handleColumnSort(col.key === 'team' ? 'name' : col.key)}
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    {!col.noSort && sortBy === (col.key === 'team' ? 'name' : col.key) ? (
                      sortDirection === 'desc' ? <ArrowDown className="w-3 h-3" /> : <ArrowUp className="w-3 h-3" />
                    ) : !col.noSort ? (
                      <ArrowUpDown className="w-3 h-3 opacity-30" />
                    ) : null}
                  </span>
                </th>
              ))}
              {showAdvancedCols && (
                <>
                  <th className="px-2 py-2 text-right text-[11px] font-semibold text-purple-400 uppercase tracking-wider w-14 cursor-pointer hover:text-purple-300 select-none" onClick={() => handleColumnSort('ceiling')}>
                    <span className="inline-flex items-center gap-1">Ceil {sortBy === 'ceiling' ? (sortDirection === 'desc' ? <ArrowDown className="w-3 h-3" /> : <ArrowUp className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 opacity-30" />}</span>
                  </th>
                  <th className="px-2 py-2 text-right text-[11px] font-semibold text-purple-400 uppercase tracking-wider w-14">Floor</th>
                  <th className="px-2 py-2 text-right text-[11px] font-semibold text-purple-400 uppercase tracking-wider w-14">StdDev</th>
                  <th className="px-2 py-2 text-right text-[11px] font-semibold text-purple-400 uppercase tracking-wider w-16 cursor-pointer hover:text-purple-300 select-none" onClick={() => handleColumnSort('leverage')}>
                    <span className="inline-flex items-center gap-1">Lev {sortBy === 'leverage' ? (sortDirection === 'desc' ? <ArrowDown className="w-3 h-3" /> : <ArrowUp className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 opacity-30" />}</span>
                  </th>
                </>
              )}
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-20">Min Exp</th>
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-20">Max Exp</th>
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-20">Actual</th>
            </tr>
          </thead>
          <tbody>
            {filteredPlayers.map((player, idx) => {
              const numericValue = player.salary > 0 ? player.projectedPoints / player.salary * 1000 : 0;
              const value = numericValue.toFixed(2);
              const isSelected = selectedPlayers.includes(player.id);
              const pCeiling = player.ceiling || player.projectedPoints * 1.3;
              const pFloor = player.floor || player.projectedPoints * 0.6;
              const pStdDev = player.stdDev || (pCeiling - pFloor) / 4;
              const boomThreshold = pCeiling * 0.85;
              const bustThreshold = pFloor * 1.2;
              const boomPct = pStdDev > 0 ? ((1 - normalCDF(boomThreshold, player.projectedPoints, pStdDev)) * 100) : 0;
              const bustPct = pStdDev > 0 ? (normalCDF(bustThreshold, player.projectedPoints, pStdDev) * 100) : 0;
              const rowClass = player.locked ? 'bg-green-500/10 border-l-2 border-l-green-400'
                : player.excluded ? 'bg-red-500/10 opacity-60 border-l-2 border-l-red-400'
                : isSelected ? 'bg-cyan-500/10 hover:bg-cyan-500/15' : '';
              // Change #3: More aggressive ownership color coding
              const ownColor = player.ownership > 25 ? 'text-orange-400 font-bold'
                : player.ownership > 15 ? 'text-yellow-400'
                : player.ownership > 5 ? 'text-white'
                : player.ownership > 0 ? 'text-green-400' : 'text-slate-500';
              // Change #4: Value color from percentile
              const valueColor = getValueColor(numericValue);

              return (
                <tr
                  key={player.id}
                  onClick={(e) => {
                    const target = e.target as HTMLElement;
                    if (target.tagName === 'INPUT' || target.closest('input') || target.closest('button')) return;
                    togglePlayer(player.id, e);
                  }}
                  className={`border-b border-slate-700/50 hover:bg-[var(--dfs-bg-hover)] transition-colors cursor-pointer ${idx % 2 === 0 ? 'bg-slate-800/20' : ''} ${rowClass}`}
                >
                  {/* Change #1: Reduced cell padding px-2 py-1 for ~28px rows */}
                  <td className="px-2 py-1" onClick={(e) => e.stopPropagation()}>
                    <div className="flex items-center gap-0.5">
                      <Checkbox checked={isSelected} onCheckedChange={(checked: boolean | 'indeterminate') => { if (checked !== isSelected) togglePlayer(player.id); }} className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer" />
                      <button onClick={() => toggleLock(player.id)} className={`p-0.5 rounded transition-colors ${player.locked ? 'text-green-400 bg-green-500/20' : 'text-slate-500 hover:text-green-400'}`} title={player.locked ? 'Unlock player' : 'Lock player into lineups'}>
                        <Lock className="w-3 h-3" />
                      </button>
                      <button onClick={() => toggleExclude(player.id)} className={`p-0.5 rounded transition-colors ${player.excluded ? 'text-red-400 bg-red-500/20' : 'text-slate-500 hover:text-red-400'}`} title={player.excluded ? 'Include player' : 'Exclude player from lineups'}>
                        <Ban className="w-3 h-3" />
                      </button>
                    </div>
                  </td>
                  <td className="px-2 py-1 text-white font-medium text-xs">
                    <HoverCard openDelay={300}>
                      <HoverCardTrigger asChild>
                        <span className="cursor-help hover:text-cyan-300 transition-colors">{player.name}</span>
                      </HoverCardTrigger>
                      <HoverCardContent className="w-72 bg-slate-800 border-slate-600 text-white p-4" side="right">
                        <div className="space-y-3">
                          <div>
                            <h4 className="font-semibold text-cyan-400 text-sm">{player.name}</h4>
                            <p className="text-xs text-slate-400">{player.position} - {player.team}{player.opponent ? ` vs ${player.opponent}` : ''}</p>
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
                          {player.projectionEdited && (
                            <div className="border-t border-slate-700 pt-2 flex items-center justify-between text-xs">
                              <span className="text-yellow-400">Edited (was {player.originalProjection?.toFixed(2)})</span>
                              <button onClick={() => resetProjection(player.id)} className="text-cyan-400 hover:text-cyan-300">Reset</button>
                            </div>
                          )}
                        </div>
                      </HoverCardContent>
                    </HoverCard>
                  </td>
                  <td className="px-2 py-1 text-white text-xs">{player.team}</td>
                  {/* Change #2: Opp column */}
                  <td className="px-2 py-1 text-xs text-[var(--dfs-text-muted)]">{player.opponent ? `vs ${player.opponent}` : '\u2014'}</td>
                  <td className="px-2 py-1 text-white text-xs">{player.position}</td>
                  <td className="px-2 py-1 text-right text-white font-mono text-xs">${player.salary.toLocaleString()}</td>
                  <td className="px-2 py-1 text-right font-medium text-xs" onDoubleClick={(e) => { e.stopPropagation(); setEditingProjection(player.id); setEditingValue(player.projectedPoints.toFixed(2)); }} onClick={(e) => e.stopPropagation()}>
                    {editingProjection === player.id ? (
                      <Input type="number" step="0.01" value={editingValue} onChange={(e) => setEditingValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') commitProjectionEdit(player.id, editingValue); if (e.key === 'Escape') setEditingProjection(null); }} onBlur={() => commitProjectionEdit(player.id, editingValue)} autoFocus className="bg-slate-700 border-cyan-500 text-white text-xs h-6 w-16 text-right" />
                    ) : (
                      <span className={player.projectionEdited ? 'text-yellow-400' : 'text-white'} title="Double-click to edit">{player.projectedPoints.toFixed(2)}</span>
                    )}
                  </td>
                  {/* Change #4: Value color from percentile */}
                  <td className={`px-2 py-1 text-right font-medium font-mono text-xs ${valueColor}`}>{value}</td>
                  {/* Change #3: Ownership color coding with font-bold on chalk */}
                  <td className={`px-2 py-1 text-right text-xs font-medium ${ownColor}`}>{player.ownership > 0 ? `${player.ownership.toFixed(1)}%` : '\u2014'}</td>
                  {showAdvancedCols && (
                    <>
                      <td className="px-2 py-1 text-right text-xs text-green-400">{pCeiling.toFixed(1)}</td>
                      <td className="px-2 py-1 text-right text-xs text-red-400">{pFloor.toFixed(1)}</td>
                      <td className="px-2 py-1 text-right text-xs text-slate-300">{pStdDev.toFixed(1)}</td>
                      <td className={`px-2 py-1 text-right text-xs font-medium ${(player.leverageScore || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>{(player.leverageScore || 0) > 0 ? '+' : ''}{(player.leverageScore || 0).toFixed(1)}</td>
                    </>
                  )}
                  <td className="px-2 py-1" onClick={(e) => e.stopPropagation()}>
                    <Input type="number" min="0" max="100" value={player.minExp} onChange={(e) => updateExposure(player.id, 'minExp', parseInt(e.target.value) || 0)} className="bg-slate-700 border-slate-600 text-white text-xs h-6 w-16 text-right" />
                  </td>
                  <td className="px-2 py-1" onClick={(e) => e.stopPropagation()}>
                    <Input type="number" min="0" max="100" value={player.maxExp} onChange={(e) => updateExposure(player.id, 'maxExp', parseInt(e.target.value) || 0)} className="bg-slate-700 border-slate-600 text-white text-xs h-6 w-16 text-right" />
                  </td>
                  <td className="px-2 py-1 text-right text-slate-400 text-xs">{player.actualExp !== undefined ? `${player.actualExp.toFixed(1)}%` : '\u2014'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Status Bar */}
      <div className="bg-[var(--dfs-bg-tertiary)] border border-[var(--dfs-border)] rounded-lg p-2">
        <div className="flex items-center justify-between text-xs">
          <div className="text-white">
            <span className="font-semibold text-[var(--dfs-accent)]">{selectedPlayers.length}</span> / {playerData.length} players selected
            <span className="text-[var(--dfs-text-muted)] ml-2">({playerData.length > 0 ? ((selectedPlayers.length / playerData.length) * 100).toFixed(1) : '0.0'}%)</span>
          </div>
          {selectedPlayers.length < 30 && (
            <div className="text-yellow-400 text-[11px]">Select at least 30 players for diverse lineups</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PlayerTable;
