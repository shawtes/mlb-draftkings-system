import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Checkbox } from './ui/checkbox';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Users, Link2, BarChart3, Target, Cpu, Star, Upload, Play, Save, FileText, Download, Plus, CheckSquare, XSquare, Trophy, RefreshCcw, Check, X } from 'lucide-react';
import { Sport, SPORT_CONFIGS, getPositionFilters, filterPlayersByPosition, getPositionCount, getStackDescription } from './sport-config';
import LineupsTab from './LineupsTab';
import { dfsApi } from '../services/dfs-api';
import { toast } from 'react-hot-toast';

// Verbose logging
const DEBUG_LOG = true;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const vlog = (...args: any[]) => {
  if (DEBUG_LOG) {
    // eslint-disable-next-line no-console
    console.log('[DFSOptimizer]', ...args);
  }
};

// Player data interface
interface Player {
  id: string;
  name: string;
  team: string;
  position: string;
  salary: number;
  projectedPoints: number;
  minExp: number;
  maxExp: number;
  actualExp?: number;
  selected: boolean;
}

// Players Tab Component
interface PlayersTabProps {
  playerData: Player[];
  selectedPlayers: string[];
  sport: Sport;
  onPlayersChange: (players: string[]) => void;
  onPlayerDataChange: (players: Player[]) => void;
}

const PlayersTab: React.FC<PlayersTabProps> = ({ playerData, selectedPlayers, sport, onPlayersChange, onPlayerDataChange }) => {
  const sportConfig = SPORT_CONFIGS[sport];
  const [positionFilter, setPositionFilter] = useState(sport === 'MLB' ? 'all-batters' : 'all-offense');
  const [sortBy, setSortBy] = useState('points');
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isSyncingRef = useRef(false);
  const selectedPlayersRef = useRef(selectedPlayers);

  // Keep ref in sync with selectedPlayers
  useEffect(() => {
    selectedPlayersRef.current = selectedPlayers;
  }, [selectedPlayers]);

  // Update filter when sport changes
  useEffect(() => {
    setPositionFilter(sport === 'MLB' ? 'all-batters' : 'all-offense');
  }, [sport]);

  // Debounced function to sync player selection with backend
  const debouncedSyncWithBackend = useCallback(async (playerId: string, isSelected: boolean) => {
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Set new timer
    debounceTimerRef.current = setTimeout(async () => {
      if (isSyncingRef.current) return;
      
      try {
        isSyncingRef.current = true;
        await dfsApi.updatePlayer(playerId, { selected: isSelected });
        // Success feedback is optional to avoid too many toasts
      } catch (error) {
        console.error('Failed to sync player selection with backend:', error);
        toast.error(`Failed to ${isSelected ? 'select' : 'deselect'} player`);
        // Revert the selection on error using current ref value
        const currentSelected = selectedPlayersRef.current;
        const revertedSelected = isSelected
          ? currentSelected.filter(id => id !== playerId)
          : [...currentSelected, playerId];
        onPlayersChange(revertedSelected);
      } finally {
        isSyncingRef.current = false;
      }
    }, 300); // 300ms debounce delay
  }, [onPlayersChange]);

  // Position counts
  const positionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    (playerData || []).forEach(p => {
      if (sport === 'MLB') {
        const positions = p.position.split('/');
        positions.forEach(pos => {
          counts[pos] = (counts[pos] || 0) + 1;
        });
      } else if (sport === 'NBA') {
        // NBA positions can be multi-position (e.g., "PG/SG")
        const positions = p.position.split('/');
        positions.forEach(pos => {
          counts[pos] = (counts[pos] || 0) + 1;
        });
      } else {
        // NFL positions are single
        counts[p.position] = (counts[p.position] || 0) + 1;
      }
    });
    
    // Add flex position counts for NBA
    if (sport === 'NBA') {
      counts['G'] = (counts['PG'] || 0) + (counts['SG'] || 0);
      counts['F'] = (counts['SF'] || 0) + (counts['PF'] || 0);
      counts['UTIL'] = (playerData || []).length;
    }
    
    return counts;
  }, [playerData, sport]);

  // Filter players by position using sport config
  const filteredPlayers = useMemo(() => {
    const filtered = filterPlayersByPosition(playerData || [], positionFilter, sport);

    // Sort
    const sorted = [...filtered].sort((a, b) => {
      vlog('sorting players', { sortBy });
      switch (sortBy) {
        case 'points':
          return b.projectedPoints - a.projectedPoints;
        case 'value':
          return (b.projectedPoints / b.salary * 1000) - (a.projectedPoints / a.salary * 1000);
        case 'salary':
          return b.salary - a.salary;
        case 'name':
          return a.name.localeCompare(b.name);
        default:
          return 0;
      }
    });

    vlog('filteredPlayers computed', {
      sport,
      positionFilter,
      sortBy,
      totalPlayers: playerData?.length || 0,
      filteredCount: sorted.length,
      sample: sorted.slice(0, 3).map(p => ({ id: p.id, name: p.name, pos: p.position, team: p.team }))
    });

    return sorted;
  }, [playerData, positionFilter, sortBy]);

  // Handle select all
  const handleSelectAll = () => {
    const currentIds = filteredPlayers.map(p => p.id);
    const newSelected = Array.from(new Set([...selectedPlayers, ...currentIds]));
    vlog('handleSelectAll', {
      filteredCount: filteredPlayers.length,
      selectedBefore: selectedPlayers.length,
      selectedAfter: newSelected.length
    });
    onPlayersChange(newSelected);
  };

  // Handle deselect all
  const handleDeselectAll = () => {
    const currentIds = new Set(filteredPlayers.map(p => p.id));
    const newSelected = selectedPlayers.filter(id => !currentIds.has(id));
    vlog('handleDeselectAll', {
      filteredCount: filteredPlayers.length,
      selectedBefore: selectedPlayers.length,
      selectedAfter: newSelected.length
    });
    onPlayersChange(newSelected);
  };

  // Toggle player selection with backend sync
  const togglePlayer = useCallback((playerId: string, event?: React.MouseEvent) => {
    // Prevent event propagation if called from row click to avoid double-toggling
    if (event) {
      event.stopPropagation();
    }
    
    const isCurrentlySelected = selectedPlayers.includes(playerId);
    const newSelected = isCurrentlySelected
      ? selectedPlayers.filter(id => id !== playerId)
      : [...selectedPlayers, playerId];
    
    // Update local state immediately for responsive UI
    vlog('togglePlayer', {
      playerId,
      wasSelected: selectedPlayers.includes(playerId),
      selectedAfter: newSelected.includes(playerId),
      totalSelectedAfter: newSelected.length
    });
    onPlayersChange(newSelected);
    
    // Sync with backend (debounced) - fire and forget
    debouncedSyncWithBackend(playerId, !isCurrentlySelected).catch(error => {
      console.error('Backend sync error:', error);
    });
  }, [selectedPlayers, onPlayersChange, debouncedSyncWithBackend]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  // Update exposure
  const updateExposure = (playerId: string, field: 'minExp' | 'maxExp', value: number) => {
    const updated = playerData.map(p => {
      if (p.id === playerId) {
        const newValue = Math.max(0, Math.min(100, value));
        if (field === 'minExp' && newValue > p.maxExp) {
          return { ...p, minExp: newValue, maxExp: newValue };
        } else if (field === 'maxExp' && newValue < p.minExp) {
          return { ...p, maxExp: newValue, minExp: newValue };
        }
        return { ...p, [field]: newValue };
      }
      return p;
    });
    onPlayerDataChange(updated);
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
          <p className="text-white mb-4">Load a CSV file to view and select players</p>
          <button
            type="button"
            className="inline-flex items-center justify-center gap-2 rounded-md border border-cyan-500/40 bg-slate-700/60 px-4 py-2 text-sm font-semibold text-white shadow-none transition-none"
          >
            <Upload className="w-4 h-4" />
            Load CSV
          </button>
        </div>
      </div>
    );
  }

  // Header checkbox state (Players table)
  const selectedInFiltered = filteredPlayers.filter(p => selectedPlayers.includes(p.id)).length;
  const allFilteredSelected = filteredPlayers.length > 0 && selectedInFiltered === filteredPlayers.length;
  const someFilteredSelected = selectedInFiltered > 0 && selectedInFiltered < filteredPlayers.length;
  vlog('playersHeaderCheckbox', {
    filteredCount: filteredPlayers.length,
    selectedInFiltered,
    allFilteredSelected,
    someFilteredSelected
  });

                  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Position Sub-Tabs - Dynamic based on sport */}
      <div className="flex gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent pb-2">
        {getPositionFilters(sport).map((pos) => {
          const count = getPositionCount(playerData, pos.id, sport);
          return (
            <button
              key={pos.id}
              onClick={() => setPositionFilter(pos.id)}
              className={`px-4 py-2 rounded-lg whitespace-nowrap transition-all ${
                positionFilter === pos.id
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                  : 'bg-slate-700/40 text-white border border-slate-600/30 hover:bg-slate-700 hover:text-white'
              }`}
            >
              {pos.label} <span className="text-xs opacity-70">({count})</span>
            </button>
          );
        })}
      </div>

      {/* Action Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-700/40 border border-slate-600/50 rounded-lg p-3">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleSelectAll}
            className="border-green-500/30 bg-green-500/5 text-white transition-none"
          >
            <CheckSquare className="w-4 h-4 mr-2" />
            Select All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleDeselectAll}
            className="border-red-500/30 bg-red-500/5 text-white transition-none"
          >
            <XSquare className="w-4 h-4 mr-2" />
            Deselect All
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-sm text-white">Sort by:</Label>
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-40 bg-slate-700 border-slate-600 text-white text-sm h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-cyan-500/20">
              <SelectItem value="points" className="text-white">Points ↓</SelectItem>
              <SelectItem value="value" className="text-white">Value ↓</SelectItem>
              <SelectItem value="salary" className="text-white">Salary ↓</SelectItem>
              <SelectItem value="name" className="text-white">Name A-Z</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Player Table */}
      <div className="flex-1 overflow-auto max-h-96 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
        <table className="w-full text-sm">
          <thead className="bg-slate-700 sticky top-0 z-10">
            <tr className="border-b border-slate-600">
              <th className="px-3 py-3 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider w-12">
                <Checkbox
                  checked={allFilteredSelected}
                  onCheckedChange={(checked: boolean | 'indeterminate') => {
                    vlog('playersHeaderCheckbox.onCheckedChange', { checked });
                    if (checked) {
                      handleSelectAll();
                    } else {
                      handleDeselectAll();
                    }
                  }}
                  className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
                />
              </th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider min-w-[150px]">Name</th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider w-16">Team</th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider w-20">Pos</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Salary</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-20">Proj</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-20">Value</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Min Exp</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Max Exp</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Actual</th>
            </tr>
          </thead>
          <tbody>
            {filteredPlayers.map((player, idx) => {
              const value = (player.projectedPoints / player.salary * 1000).toFixed(2);
              const isSelected = selectedPlayers.includes(player.id);
              
              return (
                <tr
                  key={player.id}
                  onClick={(e) => {
                    // Don't toggle if clicking on input fields
                    const target = e.target as HTMLElement;
                    if (target.tagName === 'INPUT' || target.closest('input')) {
                      return;
                    }
                    togglePlayer(player.id, e);
                  }}
                  className={`border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors cursor-pointer ${
                    idx % 2 === 0 ? 'bg-slate-800/20' : ''
                  } ${isSelected ? 'bg-cyan-500/10 hover:bg-cyan-500/15' : ''}`}
                >
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={(checked) => {
                        // Only toggle if the checked state doesn't match current state
                        if (checked !== isSelected) {
                          togglePlayer(player.id);
                        }
                      }}
                      className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer transition-all duration-200"
                      style={{ 
                        accentColor: '#1f2937'
                      }}
                    />
                  </td>
                  <td className="px-3 py-2 text-white font-medium">{player.name}</td>
                  <td className="px-3 py-2 text-white">{player.team}</td>
                  <td className="px-3 py-2 text-white">{player.position}</td>
                  <td className="px-3 py-2 text-right text-white">${player.salary.toLocaleString()}</td>
                  <td className="px-3 py-2 text-right text-white font-medium">{player.projectedPoints.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right text-cyan-400 font-medium">{value}</td>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <Input
                      type="number"
                      min="0"
                      max="100"
                      value={player.minExp}
                      onChange={(e) => updateExposure(player.id, 'minExp', parseInt(e.target.value) || 0)}
                      className="bg-slate-700 border-slate-600 text-white text-xs h-8 w-20 text-right"
                    />
                  </td>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <Input
                      type="number"
                      min="0"
                      max="100"
                      value={player.maxExp}
                      onChange={(e) => updateExposure(player.id, 'maxExp', parseInt(e.target.value) || 0)}
                      className="bg-slate-700 border-slate-600 text-white text-xs h-8 w-20 text-right"
                    />
                  </td>
                  <td className="px-3 py-2 text-right text-slate-400 text-xs">
                    {player.actualExp !== undefined ? `${player.actualExp.toFixed(1)}%` : '—'}
                  </td>
                </tr>
                  );
                })}
          </tbody>
        </table>
              </div>

      {/* Status Bar */}
      <div className="bg-slate-700/40 border border-slate-600/50 rounded-lg p-3">
        <div className="flex items-center justify-between text-sm">
          <div className="text-white">
            <span className="font-semibold text-cyan-400">{selectedPlayers.length}</span> / {playerData.length} players selected
            <span className="text-slate-500 ml-2">({((selectedPlayers.length / playerData.length) * 100).toFixed(1)}%)</span>
            </div>
          {selectedPlayers.length < 30 && (
            <div className="text-yellow-400 text-xs">
              ⚠ Select at least 30 players for diverse lineups
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Team data interface
interface Team {
  abbr: string;
  status: 'Active' | 'Postponed' | 'Final';
  gameTime: string;
  projRuns: number;
  minExp: number;
  maxExp: number;
  actualExp?: number;
  playerCount: number;
  batterCount: number;
}

// Team Stacks Tab Component
interface TeamStacksTabProps {
  playerData: Player[];
  teamSelections: Record<number | 'all', string[]>;
  onTeamSelectionsChange: (selections: Record<number | 'all', string[]>) => void;
}

const TeamStacksTab: React.FC<TeamStacksTabProps> = ({ playerData, teamSelections, onTeamSelectionsChange }) => {
  const [activeStackSize, setActiveStackSize] = useState<'all' | number>('all');

  // Generate teams from player data
  const teams = useMemo(() => {
    const teamMap = new Map<string, Team>();
    
    playerData.forEach(player => {
      if (!teamMap.has(player.team)) {
        teamMap.set(player.team, {
          abbr: player.team,
          status: 'Active',
          gameTime: '7:00 PM',
          projRuns: 0,
          minExp: 0,
          maxExp: 100,
          playerCount: 0,
          batterCount: 0,
        });
      }
      
      const team = teamMap.get(player.team)!;
      team.playerCount++;
      if (!player.position.includes('P')) {
        team.batterCount++;
      }
      team.projRuns += player.projectedPoints / 10; // Rough estimation
    });

    return Array.from(teamMap.values()).sort((a, b) => a.abbr.localeCompare(b.abbr));
  }, [playerData]);

  // Get selected teams for current stack size
  const getSelectedTeams = (stackSize: 'all' | number): string[] => {
    return teamSelections[stackSize] || [];
  };

  // Toggle team selection
  const toggleTeam = useCallback((team: string, event?: React.MouseEvent) => {
    // Prevent event propagation if called from row click to avoid double-toggling
    if (event) {
      event.stopPropagation();
    }
    
    const current = getSelectedTeams(activeStackSize);
    const isCurrentlySelected = current.includes(team);
    const updated = isCurrentlySelected
      ? current.filter(t => t !== team)
      : [...current, team];
    
    onTeamSelectionsChange({
      ...teamSelections,
      [activeStackSize]: updated,
    });
  }, [activeStackSize, teamSelections, onTeamSelectionsChange]);

  // Select all teams in current stack size
  const handleSelectAll = () => {
    const allTeams = teams.filter(t => {
      if (activeStackSize === 'all') return true;
      return t.batterCount >= (activeStackSize as number);
    }).map(t => t.abbr);
    
    onTeamSelectionsChange({
      ...teamSelections,
      [activeStackSize]: allTeams,
    });
  };

  // Deselect all teams in current stack size
  const handleDeselectAll = () => {
    onTeamSelectionsChange({
      ...teamSelections,
      [activeStackSize]: [],
    });
  };

  // Test detection - log selections
  const handleTestDetection = () => {
    console.log('===== TEAM SELECTION DEBUG =====');
    console.log('✓ Found team selections:');
    Object.entries(teamSelections).forEach(([size, teams]) => {
      if (teams.length > 0) {
        console.log(`  ${size === 'all' ? 'All Stacks' : `${size}-Stack`}: [${teams.join(', ')}]`);
      }
    });
    console.log('================================');
    alert('Team selections logged to console. Press F12 to view.');
  };

  if (teams.length === 0) {
    return (
                <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="mb-4">
            <div className="w-16 h-16 mx-auto bg-slate-700 rounded-full flex items-center justify-center">
              <Link2 className="w-8 h-8 text-slate-400" />
                    </div>
                  </div>
          <h3 className="text-xl font-semibold text-white mb-2">No Team Data</h3>
          <p className="text-white mb-4">Load players first to configure team stacks</p>
                </div>
      </div>
    );
  }

  const selectedCount = getSelectedTeams(activeStackSize).length;

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Stack Size Sub-Tabs */}
      <div className="flex gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent pb-2">
        {[
          { id: 'all', label: 'All Stacks', count: getSelectedTeams('all').length },
          { id: 2, label: '2 Stack', count: getSelectedTeams(2).length },
          { id: 3, label: '3 Stack', count: getSelectedTeams(3).length },
          { id: 4, label: '4 Stack', count: getSelectedTeams(4).length },
          { id: 5, label: '5 Stack', count: getSelectedTeams(5).length },
        ].map((stack) => (
          <button
            key={stack.id}
            onClick={() => setActiveStackSize(stack.id as 'all' | number)}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition-all ${
              activeStackSize === stack.id
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                : 'bg-slate-700/40 text-white border border-slate-600/30 hover:bg-slate-700 hover:text-white'
            }`}
          >
            {stack.label} <span className="text-xs opacity-70">({stack.count})</span>
          </button>
        ))}
      </div>

      {/* Action Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-700/40 border border-slate-600/50 rounded-lg p-3">
        <div className="flex gap-2 flex-wrap">
          <Button
            variant="outline"
            size="sm"
            onClick={handleSelectAll}
            className="border-green-500/30 bg-green-500/5 text-white transition-none"
          >
            <CheckSquare className="w-4 h-4 mr-2" />
            Select All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleDeselectAll}
            className="border-red-500/30 bg-red-500/5 text-white transition-none"
          >
            <XSquare className="w-4 h-4 mr-2" />
            Deselect All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleTestDetection}
            className="border-cyan-500/30 bg-cyan-500/5 text-white transition-none"
          >
            <Target className="w-4 h-4 mr-2" />
            Test Detection
          </Button>
        </div>

        {activeStackSize !== 'all' && (
          <div className="text-xs text-slate-400">
            Teams with {activeStackSize}+ batters: {teams.filter(t => t.batterCount >= (activeStackSize as number)).length}
          </div>
        )}
      </div>

      {/* Team Stack Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-700 sticky top-0 z-10">
            <tr className="border-b border-slate-600">
              <th className="px-3 py-3 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider w-12">
                <Checkbox
                  checked={
                    teams.length > 0 &&
                    (getSelectedTeams(activeStackSize).length === (
                      activeStackSize === 'all'
                        ? teams.length
                        : teams.filter(t => t.batterCount >= (activeStackSize as number)).length
                    ))
                  }
                  onCheckedChange={(checked: boolean | 'indeterminate') => {
                    if (checked) {
                      handleSelectAll();
                    } else {
                      handleDeselectAll();
                    }
                  }}
                  className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
                />
              </th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider w-20">Team</th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Status</th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Time</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Proj Runs</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Batters</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Min Exp</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Max Exp</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-cyan-400 uppercase tracking-wider w-24">Actual</th>
            </tr>
          </thead>
          <tbody>
            {teams.map((team, idx) => {
              const isSelected = getSelectedTeams(activeStackSize).includes(team.abbr);
              const canStack = activeStackSize === 'all' || team.batterCount >= (activeStackSize as number);
              
              return (
                <tr
                  key={team.abbr}
                  onClick={(e) => {
                    // Don't toggle if team can't stack or if clicking on disabled inputs
                    if (!canStack) return;
                    const target = e.target as HTMLElement;
                    if (target.tagName === 'INPUT' || target.closest('input')) {
                      return;
                    }
                    toggleTeam(team.abbr, e);
                  }}
                  className={`border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors ${
                    idx % 2 === 0 ? 'bg-slate-800/20' : ''
                  } ${!canStack ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'} ${
                    isSelected && canStack ? 'bg-cyan-500/10 hover:bg-cyan-500/15' : ''
                  }`}
                >
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={(checked) => {
                        // Only toggle if the checked state doesn't match current state and team can stack
                        if (!canStack) return;
                        if (checked !== isSelected) {
                          toggleTeam(team.abbr);
                        }
                      }}
                      disabled={!canStack}
                      className="cursor-pointer transition-all duration-200"
                    />
                  </td>
                  <td className="px-3 py-2 text-white font-bold">{team.abbr}</td>
                  <td className="px-3 py-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      team.status === 'Active' ? 'bg-green-500/20 text-white' :
                      team.status === 'Postponed' ? 'bg-red-500/20 text-red-400' :
                      'bg-slate-500/20 text-slate-400'
                    }`}>
                      {team.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-white">{team.gameTime}</td>
                  <td className="px-3 py-2 text-right">
                    <span className={`font-medium ${
                      team.projRuns > 5.0 ? 'text-white' :
                      team.projRuns > 4.0 ? 'text-yellow-400' :
                      'text-slate-400'
                    }`}>
                      {team.projRuns.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right text-cyan-400 font-medium">{team.batterCount}</td>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <Input
                      type="number"
                      min="0"
                      max="100"
                      value={team.minExp}
                      className="bg-slate-700 border-slate-600 text-white text-xs h-8 w-20 text-right"
                      disabled
                    />
                  </td>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <Input
                      type="number"
                      min="0"
                      max="100"
                      value={team.maxExp}
                      className="bg-slate-700 border-slate-600 text-white text-xs h-8 w-20 text-right"
                      disabled
                    />
                  </td>
                  <td className="px-3 py-2 text-right text-slate-400 text-xs">
                    {team.actualExp !== undefined ? `${team.actualExp.toFixed(1)}%` : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Status Bar */}
      <div className="bg-slate-700/40 border border-slate-600/50 rounded-lg p-3">
        <div className="flex items-center justify-between text-sm flex-wrap gap-2">
          <div className="text-white">
            <span className="font-semibold text-cyan-400">
              {activeStackSize === 'all' ? 'All Stacks' : `${activeStackSize}-Stack`}:
            </span>
            {' '}
            {selectedCount > 0 ? (
              <span className="text-white">
                {getSelectedTeams(activeStackSize).join(', ')}
              </span>
            ) : (
              <span className="text-slate-500">No teams selected</span>
            )}
            <span className="text-slate-500 ml-2">({selectedCount}/{teams.length})</span>
          </div>
          
          {activeStackSize !== 'all' && selectedCount > 0 && (
            <div className="text-xs text-cyan-400">
              ✓ {selectedCount} team{selectedCount !== 1 ? 's' : ''} configured for {activeStackSize}-stacks
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Stack Type interface
interface StackType {
  id: string;
  label: string;
  minExp: number;
  maxExp: number;
  lineupExp?: number;
  poolExp?: number;
  entryExp?: number;
  enabled: boolean;
}

// Stack Exposure Tab Component
interface StackExposureTabProps {
  stackSettings: StackType[];
  sport: Sport;
  onStackSettingsChange: (settings: StackType[]) => void;
}

const StackExposureTab: React.FC<StackExposureTabProps> = ({ stackSettings, sport, onStackSettingsChange }) => {
  const sportConfig = SPORT_CONFIGS[sport];
  // Toggle stack type enabled
  const toggleStackType = useCallback((id: string, event?: React.MouseEvent) => {
    // Prevent event propagation if called from row click to avoid double-toggling
    if (event) {
      event.stopPropagation();
    }
    
    const updated = stackSettings.map(s => 
      s.id === id ? { ...s, enabled: !s.enabled } : s
    );
    onStackSettingsChange(updated);
  }, [stackSettings, onStackSettingsChange]);

  // Update exposure values
  const updateExposure = (id: string, field: 'minExp' | 'maxExp', value: number) => {
    const updated = stackSettings.map(s => {
      if (s.id === id) {
        const newValue = Math.max(0, Math.min(100, value));
        if (field === 'minExp' && newValue > s.maxExp) {
          return { ...s, minExp: newValue, maxExp: newValue };
        } else if (field === 'maxExp' && newValue < s.minExp) {
          return { ...s, maxExp: newValue, minExp: newValue };
        }
        return { ...s, [field]: newValue };
      }
      return s;
    });
    onStackSettingsChange(updated);
  };

  // Calculate totals
  const enabledStacks = stackSettings.filter(s => s.enabled);
  const totalMinExp = enabledStacks.reduce((sum, s) => sum + s.minExp, 0);
  const totalMaxExp = enabledStacks.reduce((sum, s) => sum + s.maxExp, 0);

  // Validation
  const hasConflict = totalMinExp > 100;
  const hasNoSelection = enabledStacks.length === 0;

  const minRemaining = Math.max(0, 100 - totalMinExp);
  const maxHeadroom = Math.max(0, 100 - totalMaxExp);

  const handleEnableAll = () => {
    if (stackSettings.every(s => s.enabled)) return;
    const updated = stackSettings.map(s => ({ ...s, enabled: true }));
    onStackSettingsChange(updated);
  };

  const handleDisableAll = () => {
    if (enabledStacks.length === 0) return;
    const updated = stackSettings.map(s => ({ ...s, enabled: false }));
    onStackSettingsChange(updated);
  };

  const handleResetRanges = () => {
    const updated = stackSettings.map(s => ({
      ...s,
      minExp: 0,
      maxExp: 100,
    }));
    onStackSettingsChange(updated);
  };

  const getExposureBadges = (stack: StackType) => {
    const badges: Array<{ label: string; value: number }> = [];
    if (typeof stack.lineupExp === 'number') {
      badges.push({ label: 'Lineup', value: stack.lineupExp });
    }
    if (typeof stack.poolExp === 'number') {
      badges.push({ label: 'Pool', value: stack.poolExp });
    }
    if (typeof stack.entryExp === 'number') {
      badges.push({ label: 'Entries', value: stack.entryExp });
    }
    return badges;
  };

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Stack Exposure</h2>
          <p className="text-sm text-slate-300">
            Toggle the stacks you want and set simple min/max exposure targets.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            onClick={handleEnableAll}
            disabled={stackSettings.length === 0 || stackSettings.every(s => s.enabled)}
            className="gap-2 border-cyan-500/40 bg-slate-900/60 text-white hover:bg-cyan-500/20 disabled:opacity-60"
          >
            <CheckSquare className="h-4 w-4" />
            Enable All
          </Button>
          <Button
            variant="outline"
            onClick={handleDisableAll}
            disabled={enabledStacks.length === 0}
            className="gap-2 border-slate-700/70 bg-slate-900/60 text-white hover:bg-slate-800 disabled:opacity-60"
          >
            <XSquare className="h-4 w-4" />
            Disable All
          </Button>
          <Button
            variant="outline"
            onClick={handleResetRanges}
            disabled={stackSettings.length === 0}
            className="gap-2 border-amber-500/40 bg-slate-900/60 text-white hover:bg-amber-500/20 disabled:opacity-60"
          >
            <RefreshCcw className="h-4 w-4" />
            Reset Exposure Ranges
          </Button>
        </div>
      </div>

      {(hasNoSelection || hasConflict) && (
        <div className="space-y-2">
          {hasNoSelection && (
            <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
              Select at least one stack to include in generated lineups.
            </div>
          )}
          {hasConflict && (
            <div className="rounded-md border border-yellow-400/40 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-100">
              Total minimum exposure exceeds 100%. Lower a few minimums to proceed.
            </div>
          )}
        </div>
      )}

      <div className="flex-1">
        {stackSettings.length === 0 ? (
          <div className="flex h-full items-center justify-center px-6 py-10 text-sm text-slate-400">
            No stack types available for {sport}. Configure sport settings to populate options.
          </div>
        ) : (
          <div className="mx-auto h-full w-full max-w-4xl overflow-hidden rounded-lg border border-slate-700">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-slate-800/90 backdrop-blur">
                <tr className="text-left text-slate-300">
                  <th className="w-14 px-2 py-2 font-semibold">Use</th>
                  <th className="px-2 py-2 font-semibold">Stack Type</th>
                  <th className="w-28 px-2 py-2 text-right font-semibold">Min %</th>
                  <th className="w-28 px-2 py-2 text-right font-semibold">Max %</th>
                  <th className="px-2 py-2 text-right font-semibold">Live Exposure</th>
                </tr>
              </thead>
              <tbody>
                {stackSettings.map((stack, index) => {
                  const exposureBadges = getExposureBadges(stack);
                  const exposureSummary = exposureBadges.length
                    ? exposureBadges.map((badge) => `${badge.label}: ${badge.value.toFixed(1)}%`).join(' • ')
                    : '—';

                  return (
                    <tr
                      key={stack.id}
                      onClick={(e) => {
                        // Don't toggle if clicking on input fields
                        const target = e.target as HTMLElement;
                        if (target.tagName === 'INPUT' || target.closest('input')) {
                          return;
                        }
                        toggleStackType(stack.id, e);
                      }}
                      className={`border-t border-slate-800 cursor-pointer transition-colors hover:bg-slate-800/50 ${
                        index % 2 === 0 ? 'bg-slate-900/70' : 'bg-slate-900/50'
                      } ${stack.enabled ? 'bg-cyan-500/10 hover:bg-cyan-500/15' : ''}`}
                    >
                      <td className="px-2 py-2" onClick={(e) => e.stopPropagation()}>
                        <Checkbox
                          checked={stack.enabled}
                          onCheckedChange={(checked) => {
                            // Only toggle if the checked state doesn't match current state
                            if (checked !== stack.enabled) {
                              toggleStackType(stack.id);
                            }
                          }}
                          className="cursor-pointer transition-all duration-200"
                        />
                      </td>
                      <td className="px-2 py-2 align-middle">
                        <div className={`font-medium ${stack.enabled ? 'text-white' : 'text-slate-400'}`}>
                          {stack.label}
                        </div>
                        <div className="text-xs text-slate-500">
                          {getStackDescription(stack.label, sport)}
                        </div>
                      </td>
                      <td className="px-2 py-2 text-right" onClick={(e) => e.stopPropagation()}>
                        <Input
                          type="number"
                          min="0"
                          max="100"
                          value={stack.minExp}
                          onChange={(e) => updateExposure(stack.id, 'minExp', parseInt(e.target.value) || 0)}
                          disabled={!stack.enabled}
                          className={`h-9 w-full text-right ${
                            stack.enabled ? 'bg-slate-800 text-white' : 'bg-slate-900 text-slate-500'
                          }`}
                        />
                      </td>
                      <td className="px-2 py-2 text-right" onClick={(e) => e.stopPropagation()}>
                        <Input
                          type="number"
                          min="0"
                          max="100"
                          value={stack.maxExp}
                          onChange={(e) => updateExposure(stack.id, 'maxExp', parseInt(e.target.value) || 0)}
                          disabled={!stack.enabled}
                          className={`h-9 w-full text-right ${
                            stack.enabled ? 'bg-slate-800 text-white' : 'bg-slate-900 text-slate-500'
                          }`}
                        />
                      </td>
                      <td className="px-2 py-2 text-right text-slate-300">
                        {stack.enabled ? exposureSummary : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="mx-auto mt-auto flex w-full max-w-4xl flex-wrap items-center justify-between gap-3 rounded-md border border-slate-700 bg-slate-900/60 px-4 py-3 text-sm text-slate-300">
        <span>
          Active stacks:{' '}
          <span className="font-semibold text-white">
            {enabledStacks.length}
          </span>{' '}
          / {stackSettings.length}
        </span>
        <span>
          Total min:{' '}
          <span className={`font-semibold ${hasConflict ? 'text-red-300' : 'text-white'}`}>
            {totalMinExp}%
          </span>
          <span className="ml-3">
            Total max: <span className="font-semibold text-white">{totalMaxExp}%</span>
          </span>
        </span>
        <span className="text-xs text-slate-500">
          Min remaining: {minRemaining}% • Max headroom: {maxHeadroom}%
        </span>
      </div>
    </div>
  );
};

// Team Combination interface
interface TeamCombination {
  id: string;
  teams: string[];
  stackSizes: number[];
  display: string;
  lineupsPerCombo: number;
  status: 'ready' | 'generating' | 'complete' | 'error';
  enabled: boolean;
}

// Team Combinations Tab Component
interface TeamCombinationsTabProps {
  playerData: Player[];
}

const TeamCombinationsTab: React.FC<TeamCombinationsTabProps> = ({ playerData }) => {
  console.log('TeamCombinationsTab rendered with playerData:', playerData.length, 'players');
  
  const [selectedTeams, setSelectedTeams] = useState<string[]>([]);
  const [stackPattern, setStackPattern] = useState('4');
  const [defaultLineupsPerCombo, setDefaultLineupsPerCombo] = useState(5);
  const [combinations, setCombinations] = useState<TeamCombination[]>([]);

  // Extract teams from player data
  const teams = useMemo(() => {
    const teamSet = new Set((playerData || []).map(p => p.team));
    return Array.from(teamSet).sort();
  }, [playerData]);

  // Auto-select all teams when playerData is loaded (e.g., after CSV upload)
  const prevTeamsRef = useRef<string[]>([]);
  useEffect(() => {
    if (teams.length > 0) {
      // Check if teams have changed (new CSV uploaded)
      const teamsChanged = teams.length !== prevTeamsRef.current.length || 
        teams.some(team => !prevTeamsRef.current.includes(team));
      
      if (teamsChanged) {
        setSelectedTeams([...teams]);
        prevTeamsRef.current = [...teams];
      }
    } else {
      prevTeamsRef.current = [];
    }
  }, [teams]);

  // Toggle team selection
  const toggleTeam = (team: string) => {
    setSelectedTeams(prev =>
      prev.includes(team) ? prev.filter(t => t !== team) : [...prev, team]
    );
  };

  // Select/deselect all teams
  const selectAllTeams = () => setSelectedTeams([...teams]);
  const deselectAllTeams = () => setSelectedTeams([]);

  // Generate all combinations
  const generateCombinations = () => {
    const stackSizes = stackPattern.split('|').map(s => parseInt(s));
    const teamsNeeded = stackSizes.length;

    if (selectedTeams.length < teamsNeeded) {
      alert(`Pattern "${stackPattern}" requires ${teamsNeeded} teams. Only ${selectedTeams.length} selected.`);
      return;
    }

    const combos: TeamCombination[] = [];
    
    // Generate all combinations of teams
    const teamCombos = getCombinations(selectedTeams, teamsNeeded);
    
    // For each combination, generate all permutations
    teamCombos.forEach(teamCombo => {
      const perms = getPermutations(teamCombo);
      perms.forEach(perm => {
        const display = perm.map((t, i) => `${t}(${stackSizes[i]})`).join(' + ');
        combos.push({
          id: `combo-${combos.length}`,
          teams: perm,
          stackSizes: stackSizes,
          display: display,
          lineupsPerCombo: defaultLineupsPerCombo,
          status: 'ready',
          enabled: true,
        });
      });
    });

    // Warn if too many combinations
    if (combos.length > 50) {
      if (!confirm(`This will create ${combos.length} combinations (${combos.length * defaultLineupsPerCombo} total lineups). Continue?`)) {
        return;
      }
    }

    setCombinations(combos);
  };

  // Helper: Get combinations (n choose k)
  const getCombinations = (arr: string[], k: number): string[][] => {
    if (k === 0) return [[]];
    if (arr.length === 0) return [];
    
    const [first, ...rest] = arr;
    const withoutFirst = getCombinations(rest, k);
    const withFirst = getCombinations(rest, k - 1).map(c => [first, ...c]);
    
    return [...withFirst, ...withoutFirst];
  };

  // Helper: Get permutations
  const getPermutations = (arr: string[]): string[][] => {
    if (arr.length === 0) return [[]];
    if (arr.length === 1) return [arr];
    
    const result: string[][] = [];
    for (let i = 0; i < arr.length; i++) {
      const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
      const perms = getPermutations(rest);
      perms.forEach(p => result.push([arr[i], ...p]));
    }
    return result;
  };

  // Toggle combination enabled
  const toggleCombination = (id: string) => {
    setCombinations(prev => 
      prev.map(c => c.id === id ? { ...c, enabled: !c.enabled } : c)
    );
  };

  // Update lineups per combo
  const updateLineupsPerCombo = (id: string, value: number) => {
    setCombinations(prev =>
      prev.map(c => c.id === id ? { ...c, lineupsPerCombo: Math.max(1, Math.min(100, value)) } : c)
    );
  };

  // Calculate totals
  const enabledCombos = combinations.filter(c => c.enabled);
  const totalLineups = enabledCombos.reduce((sum, c) => sum + c.lineupsPerCombo, 0);

  if (teams.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="mb-3">
            <Target className="w-10 h-10 mx-auto text-slate-500" />
          </div>
          <h3 className="text-sm font-medium text-slate-300 mb-1">No Team Data</h3>
          <p className="text-xs text-slate-400">Load players first to generate team combinations</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-8 p-8">
      {/* Header - Minimalistic */}
      <div className="pb-2">
        <h2 className="text-xl font-medium text-white mb-1">
          Team Combinations
        </h2>
        <p className="text-sm text-slate-400">
          {playerData.length} players · {teams.length} teams
        </p>
      </div>

      {/* Controls Section - Modern Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left: Team Selection */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-slate-300">
              Select Teams
            </h3>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={selectAllTeams}
                className="h-8 px-3 text-xs text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <Check className="w-3 h-3 mr-1.5" />
                All
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={deselectAllTeams}
                className="h-8 px-3 text-xs text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <X className="w-3 h-3 mr-1.5" />
                None
              </Button>
            </div>
          </div>

          {/* Team Checkboxes - Clean Grid */}
          <div className="max-h-80 overflow-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
            <div className="grid grid-cols-3 gap-2">
              {teams.map(team => (
                <div 
                  key={team} 
                  className="flex items-center gap-2 p-2.5 rounded-md hover:bg-slate-800/50 transition-colors cursor-pointer"
                  onClick={() => toggleTeam(team)}
                >
                  <Checkbox
                    checked={selectedTeams.includes(team)}
                    onCheckedChange={() => toggleTeam(team)}
                    className="h-4 w-4 border-slate-600 data-[state=checked]:bg-slate-700 data-[state=checked]:border-slate-500"
                  />
                  <Label className="text-sm text-slate-300 cursor-pointer font-normal">
                    {team}
                  </Label>
                </div>
              ))}
            </div>
          </div>

          {/* Selection Counter */}
          <div className="text-xs text-slate-400 pt-2">
            <span className="text-slate-300">{selectedTeams.length}</span> of <span className="text-slate-300">{teams.length}</span> selected
          </div>
        </div>

        {/* Right: Stack Settings - Minimalistic */}
        <div className="space-y-5">
          <h3 className="text-sm font-medium text-slate-300">
            Stack Settings
          </h3>

          <div className="space-y-4">
            <div>
              <Label className="text-xs text-slate-400 block mb-2">Stack Pattern</Label>
              <Select value={stackPattern} onValueChange={setStackPattern}>
                <SelectTrigger className="w-full bg-slate-800/50 border border-slate-700 text-white text-sm h-10">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-700">
                  <SelectItem value="5" className="text-sm">5</SelectItem>
                  <SelectItem value="4" className="text-sm">4</SelectItem>
                  <SelectItem value="3" className="text-sm">3</SelectItem>
                  <SelectItem value="No Stacks" className="text-sm">No Stacks</SelectItem>
                  <SelectItem value="5|2" className="text-sm">5|2</SelectItem>
                  <SelectItem value="4|2" className="text-sm">4|2</SelectItem>
                  <SelectItem value="4|2|2" className="text-sm">4|2|2</SelectItem>
                  <SelectItem value="3|3|2" className="text-sm">3|3|2</SelectItem>
                  <SelectItem value="3|2|2" className="text-sm">3|2|2</SelectItem>
                  <SelectItem value="2|2|2" className="text-sm">2|2|2</SelectItem>
                  <SelectItem value="5|3" className="text-sm">5|3</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-xs text-slate-400 block mb-2">Lineups per Combination</Label>
              <Input
                type="number"
                min="1"
                max="50"
                value={defaultLineupsPerCombo}
                onChange={(e) => setDefaultLineupsPerCombo(parseInt(e.target.value) || 5)}
                className="w-full bg-slate-800/50 border border-slate-700 text-white text-sm h-10"
                placeholder="5"
              />
            </div>

            <Button
              onClick={generateCombinations}
              className="w-full bg-slate-700 hover:bg-slate-600 text-white text-sm h-10 font-medium"
            >
              Generate Combinations
            </Button>
          </div>
        </div>
      </div>

      {/* Combinations List - Card-based Modern Layout */}
      {combinations.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-slate-300">
              Generated Combinations
            </h3>
            <div className="text-xs text-slate-400">
              Total: <span className="text-slate-300 font-medium">{totalLineups}</span> lineups
            </div>
          </div>

          <div className="overflow-auto max-h-96 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent space-y-2">
            {combinations.map(combo => (
              <div 
                key={combo.id} 
                className="flex items-center gap-4 p-4 bg-slate-800/30 border border-slate-700/50 rounded-lg hover:bg-slate-800/50 transition-colors"
              >
                <Checkbox
                  checked={combo.enabled}
                  onCheckedChange={() => toggleCombination(combo.id)}
                  className="h-4 w-4 border-slate-600 data-[state=checked]:bg-slate-700 data-[state=checked]:border-slate-500"
                />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-white font-medium truncate">
                    {combo.display}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <Label className="text-xs text-slate-400">Lineups:</Label>
                    <Input
                      type="number"
                      min="1"
                      max="100"
                      value={combo.lineupsPerCombo}
                      onChange={(e) => updateLineupsPerCombo(combo.id, parseInt(e.target.value) || 5)}
                      className="w-16 h-8 bg-slate-800/50 border border-slate-700 text-white text-xs text-center"
                      onClick={(e) => e.stopPropagation()}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="flex justify-end pt-2">
            <Button
              className="bg-slate-700 hover:bg-slate-600 text-white text-sm h-10 px-6 font-medium"
              disabled={totalLineups === 0}
            >
              Generate Lineups ({totalLineups})
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

// Advanced Quant Settings interface
interface AdvancedQuantSettings {
  enabled: boolean;
  strategy: string;
  riskTolerance: number;
  varConfidence: number;
  targetVolatility: number;
  monteCarloSims: number;
  timeHorizon: number;
  garchP: number;
  garchQ: number;
  lookbackPeriod: number;
  copulaFamily: string;
  dependencyThreshold: number;
  maxKellyFraction: number;
  expectedWinRate: number;
}

// Default Advanced Quant Settings
const DEFAULT_ADVANCED_QUANT_SETTINGS: AdvancedQuantSettings = {
  enabled: false,
  strategy: 'combined',
  riskTolerance: 1.0,
  varConfidence: 0.95,
  targetVolatility: 0.20,
  monteCarloSims: 10000,
  timeHorizon: 1,
  garchP: 1,
  garchQ: 1,
  lookbackPeriod: 100,
  copulaFamily: 'gaussian',
  dependencyThreshold: 0.3,
  maxKellyFraction: 0.25,
  expectedWinRate: 0.20,
};

// Advanced Quant Tab Component
interface AdvancedQuantTabProps {
  settings: AdvancedQuantSettings;
  onSettingsChange: (settings: AdvancedQuantSettings) => void;
}

const AdvancedQuantTab: React.FC<AdvancedQuantTabProps> = ({ settings, onSettingsChange }) => {
  // Merge with defaults to ensure all properties exist
  const safeSettings: AdvancedQuantSettings = {
    ...DEFAULT_ADVANCED_QUANT_SETTINGS,
    ...settings,
  };

  const updateSetting = <K extends keyof AdvancedQuantSettings>(key: K, value: AdvancedQuantSettings[K]) => {
    onSettingsChange({ ...safeSettings, [key]: value });
  };

  return (
    <div className="flex flex-col h-full space-y-4 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Cpu className="w-6 h-6 text-cyan-400" />
            Advanced Quant
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Financial-grade quantitative optimization settings
          </p>
        </div>
      </div>

      {/* Master Enable Toggle */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Checkbox
              checked={safeSettings.enabled}
              onCheckedChange={(checked: boolean) => updateSetting('enabled', checked as boolean)}
              className="border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
              style={{ 
                accentColor: '#1f2937'
              }}
            />
            <div>
              <Label className="text-white font-semibold text-base cursor-pointer" onClick={() => updateSetting('enabled', !safeSettings.enabled)}>
                Enable Advanced Quantitative Optimization
              </Label>
              <p className="text-xs text-slate-400 mt-1">
                Master switch for financial-grade risk modeling
              </p>
            </div>
          </div>
          {safeSettings.enabled && (
            <div className="flex items-center gap-2 text-white">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-xs font-medium">ENABLED</span>
            </div>
          )}
        </div>
      </Card>

      {/* Optimization Strategy */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider mb-3">Optimization Strategy</h3>
        <div>
          <Label className="text-white block mb-2 text-sm">Strategy</Label>
          <Select 
            value={safeSettings.strategy} 
            onValueChange={(v: string) => updateSetting('strategy', v)}
            disabled={!safeSettings.enabled}
          >
            <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-cyan-500/20">
              <SelectItem value="combined" className="text-white">Combined (Recommended)</SelectItem>
              <SelectItem value="kelly_criterion" className="text-white">Kelly Criterion</SelectItem>
              <SelectItem value="risk_parity" className="text-white">Risk Parity</SelectItem>
              <SelectItem value="mean_variance" className="text-white">Mean-Variance</SelectItem>
              <SelectItem value="equal_weight" className="text-white">Equal Weight</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-slate-400 mt-1">
            {safeSettings.strategy === 'combined' && 'Combines multiple optimization techniques for balanced approach'}
            {safeSettings.strategy === 'kelly_criterion' && 'Pure Kelly optimal betting strategy - maximizes long-term growth'}
            {safeSettings.strategy === 'risk_parity' && 'Equal risk contribution - balances volatility across lineup'}
            {safeSettings.strategy === 'mean_variance' && 'Classic Markowitz optimization - maximizes return for given risk'}
            {safeSettings.strategy === 'equal_weight' && 'Simple equal allocation - baseline strategy'}
          </p>
        </div>
      </Card>

      {/* Risk Parameters */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider mb-3">Risk Parameters</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">Risk Tolerance</Label>
              <span className="text-cyan-400 font-medium text-sm">{(safeSettings.riskTolerance ?? 1.0).toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={safeSettings.riskTolerance}
              onChange={(e) => updateSetting('riskTolerance', parseFloat(e.target.value))}
              disabled={!safeSettings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">Range: 0.1 - 2.0 (1.0 = neutral, &lt;1.0 = conservative, &gt;1.0 = aggressive)</p>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">VaR Confidence Level</Label>
              <span className="text-cyan-400 font-medium text-sm">{((safeSettings.varConfidence ?? 0.95) * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.90"
              max="0.99"
              step="0.01"
              value={safeSettings.varConfidence}
              onChange={(e) => updateSetting('varConfidence', parseFloat(e.target.value))}
              disabled={!safeSettings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">Range: 90% - 99% (probability level for Value-at-Risk)</p>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">Target Volatility</Label>
              <span className="text-cyan-400 font-medium text-sm">{((safeSettings.targetVolatility ?? 0.20) * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.50"
              step="0.01"
              value={safeSettings.targetVolatility}
              onChange={(e) => updateSetting('targetVolatility', parseFloat(e.target.value))}
              disabled={!safeSettings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">Range: 5% - 50% (target standard deviation of returns)</p>
          </div>
        </div>
      </Card>

      {/* Monte Carlo Simulation */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider mb-3">Monte Carlo Simulation</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label className="text-white block mb-2 text-sm">Simulations</Label>
            <Input
              type="number"
              min="1000"
              max="50000"
              step="1000"
              value={safeSettings.monteCarloSims}
              onChange={(e) => updateSetting('monteCarloSims', parseInt(e.target.value) || 10000)}
              disabled={!safeSettings.enabled}
              className="bg-slate-700 border-slate-600 text-white"
            />
            <p className="text-xs text-slate-400 mt-1">1K - 50K (10K recommended)</p>
          </div>
          <div>
            <Label className="text-white block mb-2 text-sm">Time Horizon (days)</Label>
            <Input
              type="number"
              min="1"
              max="30"
              value={safeSettings.timeHorizon}
              onChange={(e) => updateSetting('timeHorizon', parseInt(e.target.value) || 1)}
              disabled={!safeSettings.enabled}
              className="bg-slate-700 border-slate-600 text-white"
            />
            <p className="text-xs text-slate-400 mt-1">1 - 30 days (1 = single slate)</p>
          </div>
        </div>
      </Card>

      {/* GARCH Volatility Modeling */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider mb-3">GARCH Volatility Modeling</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <Label className="text-white block mb-2 text-sm">GARCH p</Label>
            <Input
              type="number"
              min="1"
              max="5"
              value={safeSettings.garchP}
              onChange={(e) => updateSetting('garchP', parseInt(e.target.value) || 1)}
              disabled={!safeSettings.enabled}
              className="bg-slate-700 border-slate-600 text-white"
            />
            <p className="text-xs text-slate-400 mt-1">1 - 5 (ARCH terms)</p>
          </div>
          <div>
            <Label className="text-white block mb-2 text-sm">GARCH q</Label>
            <Input
              type="number"
              min="1"
              max="5"
              value={safeSettings.garchQ}
              onChange={(e) => updateSetting('garchQ', parseInt(e.target.value) || 1)}
              disabled={!safeSettings.enabled}
              className="bg-slate-700 border-slate-600 text-white"
            />
            <p className="text-xs text-slate-400 mt-1">1 - 5 (GARCH terms)</p>
          </div>
          <div>
            <Label className="text-white block mb-2 text-sm">Lookback Period</Label>
            <Input
              type="number"
              min="30"
              max="365"
              step="10"
              value={safeSettings.lookbackPeriod}
              onChange={(e) => updateSetting('lookbackPeriod', parseInt(e.target.value) || 100)}
              disabled={!safeSettings.enabled}
              className="bg-slate-700 border-slate-600 text-white"
            />
            <p className="text-xs text-slate-400 mt-1">30 - 365 days</p>
          </div>
        </div>
        <div className="mt-2 text-xs text-slate-400">
          GARCH(1,1) is most common. Longer lookback = more stable, shorter = more responsive.
        </div>
      </Card>

      {/* Copula Dependency Modeling */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider mb-3">Copula Dependency Modeling</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label className="text-white block mb-2 text-sm">Copula Family</Label>
            <Select 
              value={safeSettings.copulaFamily} 
              onValueChange={(v: string) => updateSetting('copulaFamily', v)}
              disabled={!safeSettings.enabled}
            >
              <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-900 border-cyan-500/20">
                <SelectItem value="gaussian" className="text-white">Gaussian</SelectItem>
                <SelectItem value="t" className="text-white">t-Copula</SelectItem>
                <SelectItem value="clayton" className="text-white">Clayton</SelectItem>
                <SelectItem value="frank" className="text-white">Frank</SelectItem>
                <SelectItem value="gumbel" className="text-white">Gumbel</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-slate-400 mt-1">
              {safeSettings.copulaFamily === 'gaussian' && 'Normal distribution - symmetric, general use'}
              {safeSettings.copulaFamily === 't' && 'Student\'s t - heavy tails, extreme events'}
              {safeSettings.copulaFamily === 'clayton' && 'Lower tail dependence - fail together'}
              {safeSettings.copulaFamily === 'frank' && 'Weak tail dependence - more independent'}
              {safeSettings.copulaFamily === 'gumbel' && 'Upper tail dependence - succeed together'}
            </p>
          </div>
          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">Dependency Threshold</Label>
              <span className="text-cyan-400 font-medium text-sm">{((safeSettings.dependencyThreshold ?? 0.3) * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={safeSettings.dependencyThreshold}
              onChange={(e) => updateSetting('dependencyThreshold', parseFloat(e.target.value))}
              disabled={!safeSettings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">Min correlation to model (10% - 90%)</p>
          </div>
        </div>
      </Card>

      {/* Kelly Criterion Position Sizing */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider mb-3">Kelly Criterion Position Sizing</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">Max Kelly Fraction</Label>
              <span className="text-cyan-400 font-medium text-sm">{((safeSettings.maxKellyFraction ?? 0.25) * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={safeSettings.maxKellyFraction}
              onChange={(e) => updateSetting('maxKellyFraction', parseFloat(e.target.value))}
              disabled={!safeSettings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">10% - 100% of bankroll (25% = quarter Kelly, recommended)</p>
          </div>
          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">Expected Win Rate</Label>
              <span className="text-cyan-400 font-medium text-sm">{((safeSettings.expectedWinRate ?? 0.20) * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={safeSettings.expectedWinRate}
              onChange={(e) => updateSetting('expectedWinRate', parseFloat(e.target.value))}
              disabled={!safeSettings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">10% - 90% (50/50: 50%, GPP top 20%: 20%)</p>
          </div>
        </div>
      </Card>

      {/* Status & Information */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider mb-3">Status & Information</h3>
        <div className="space-y-3">
          <div className={`flex items-center gap-2 ${safeSettings.enabled ? 'text-white' : 'text-slate-500'}`}>
            {safeSettings.enabled ? '✓' : '○'} 
            <span className="font-medium">
              Advanced quantitative optimization {safeSettings.enabled ? 'ENABLED' : 'DISABLED'}
            </span>
          </div>

          {safeSettings.enabled && (
            <>
              <div className="border-t border-slate-600 pt-3">
                <div className="text-sm text-white mb-2">Library Status:</div>
                <div className="space-y-1 text-xs">
                  <div className="flex items-center gap-2 text-white">
                    ✓ <span>ARCH (GARCH): Available</span>
                  </div>
                  <div className="flex items-center gap-2 text-yellow-400">
                    ⚠ <span>Copulas: Optional - limited dependency modeling</span>
                  </div>
                  <div className="flex items-center gap-2 text-white">
                    ✓ <span>SciPy: Available</span>
                  </div>
                  <div className="flex items-center gap-2 text-white">
                    ✓ <span>Scikit-learn: Available</span>
                  </div>
                </div>
              </div>

              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3 mt-3">
                <div className="flex items-start gap-2">
                  <div className="text-blue-400 text-sm">💡</div>
                  <p className="text-xs text-white">
                    <span className="font-semibold text-blue-400">Performance Note:</span> Advanced quant adds 30-60 seconds to optimization time. 
                    High Monte Carlo simulations (50K) may use up to 1.5 GB memory.
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </Card>

      {/* Tip */}
      <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <div className="text-blue-400">💡</div>
          <p className="text-sm text-white">
            <span className="font-semibold text-blue-400">Tip:</span> These settings enable financial-grade risk modeling 
            for professional DFS portfolio management. Start with default "Combined" strategy and adjust based on contest type.
          </p>
        </div>
      </div>
    </div>
  );
};

// Favorite Lineup interface
interface FavoriteLineup {
  id: string;
  players: Player[];
  totalPoints: number;
  totalSalary: number;
  runNumber: number;
  dateAdded: string;
  selected: boolean;
}

// My Entries Tab Component
interface MyEntriesTabProps {
  results: any[];
  sport: Sport;
}

const MyEntriesTab: React.FC<MyEntriesTabProps> = ({ results, sport }) => {
  const [favorites, setFavorites] = useState<FavoriteLineup[]>([]);
  const [currentRun, setCurrentRun] = useState(1);
  const [sortBy, setSortBy] = useState('points-desc');
  const [filterRun, setFilterRun] = useState<number | 'all'>('all');

  // Add current pool to favorites
  const handleAddCurrentPool = () => {
    if (results.length === 0) {
      alert('No lineups available. Run optimization first.');
      return;
    }

    const count = parseInt(prompt(`Add how many lineups from current pool?\n\nAvailable: ${results.length} lineups\nCurrent favorites: ${favorites.length}`, 
      Math.min(30, results.length).toString()) || '0');

    if (count <= 0 || count > results.length) {
      return;
    }

    // Add top N lineups
    const newFavorites = results.slice(0, count).map((result, idx) => ({
      id: `fav-${Date.now()}-${idx}`,
      players: result.players || [],
      totalPoints: result.points || 0,
      totalSalary: result.salary || 0,
      runNumber: currentRun,
      dateAdded: new Date().toLocaleString(),
      selected: true,
    }));

    setFavorites([...favorites, ...newFavorites]);
    setCurrentRun(currentRun + 1);
    alert(`Added ${count} lineups to favorites as Run #${currentRun}`);
  };

  // Clear all favorites
  const handleClearAll = () => {
    if (favorites.length === 0) return;
    
    if (confirm(`Delete all ${favorites.length} favorite lineups?\n\nThis action cannot be undone.`)) {
      setFavorites([]);
      setCurrentRun(1);
    }
  };

  // Export favorites
  const handleExport = () => {
    if (favorites.length === 0) {
      alert('No favorites to export.');
      return;
    }

    const count = parseInt(prompt(`Export how many lineups?\n\nAvailable: ${favorites.length} favorites`, 
      favorites.length.toString()) || '0');

    if (count <= 0) return;

    // TODO: Implement CSV export
    alert(`Would export ${Math.min(count, favorites.length)} lineups to DraftKings CSV format.\n\nExport functionality will be connected to backend.`);
  };

  // Toggle lineup selection
  const toggleLineup = (id: string) => {
    setFavorites(prev => 
      prev.map(f => f.id === id ? { ...f, selected: !f.selected } : f)
    );
  };

  // Delete lineup
  const deleteLineup = (id: string) => {
    if (confirm('Delete this lineup from favorites?')) {
      setFavorites(prev => prev.filter(f => f.id !== id));
    }
  };

  // Get run numbers
  const runNumbers = Array.from(new Set(favorites.map(f => f.runNumber))).sort();

  // Filter and sort favorites
  const displayedFavorites = useMemo(() => {
    let filtered = favorites;
    
    if (filterRun !== 'all') {
      filtered = favorites.filter(f => f.runNumber === filterRun);
    }

    const sorted = [...filtered].sort((a, b) => {
      switch (sortBy) {
        case 'points-desc':
          return b.totalPoints - a.totalPoints;
        case 'points-asc':
          return a.totalPoints - b.totalPoints;
        case 'salary-desc':
          return b.totalSalary - a.totalSalary;
        case 'salary-asc':
          return a.totalSalary - b.totalSalary;
        case 'run':
          return a.runNumber - b.runNumber;
        case 'date':
          return new Date(a.dateAdded).getTime() - new Date(b.dateAdded).getTime();
        default:
          return 0;
      }
    });

    return sorted;
  }, [favorites, sortBy, filterRun]);

  // Calculate statistics
  const totalFavorites = favorites.length;
  const selectedCount = favorites.filter(f => f.selected).length;
  const pointsRange = favorites.length > 0 ? {
    min: Math.min(...favorites.map(f => f.totalPoints)),
    max: Math.max(...favorites.map(f => f.totalPoints))
  } : null;

  if (favorites.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <div className="mb-4">
            <div className="w-20 h-20 mx-auto bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-2xl flex items-center justify-center border border-cyan-500/30">
              <Star className="w-10 h-10 text-cyan-400" />
            </div>
          </div>
          <h3 className="text-2xl font-semibold text-white mb-3">No Favorites Yet</h3>
          <p className="text-white mb-6 leading-relaxed">
            Run optimizations and add your best lineups to favorites. 
            Build a portfolio of lineups from multiple runs, then export when ready.
          </p>
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
            <div className="flex items-start gap-2">
              <div className="text-blue-400">💡</div>
              <p className="text-sm text-white text-left">
                <span className="font-semibold text-blue-400">Tip:</span> Generate lineups using different strategies, 
                save the best from each run, then export your final portfolio for the contest.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Action Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-700/40 border border-slate-600/50 rounded-lg p-3">
        <div className="flex gap-2 flex-wrap">
          <Button
            variant="outline"
            size="sm"
            onClick={handleAddCurrentPool}
            className="border-green-500/30 hover:bg-green-500/10 text-white"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Current Pool
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleClearAll}
            disabled={favorites.length === 0}
            className="border-red-500/30 hover:bg-red-500/10 text-white"
          >
            <XSquare className="w-4 h-4 mr-2" />
            Clear All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={favorites.length === 0}
            className="border-cyan-500/30 hover:bg-cyan-500/10 text-white"
          >
            <Download className="w-4 h-4 mr-2" />
            Export Favorites
          </Button>
        </div>

        <div className="text-sm text-white">
          <span className="font-semibold text-cyan-400">{selectedCount}</span> / {totalFavorites} selected
        </div>
      </div>

      {/* Statistics Display */}
      <Card className="bg-slate-700/40 border-slate-600/50 p-4">
        <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          Portfolio Statistics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div className="text-xs text-slate-400 mb-1">Total Lineups</div>
            <div className="text-2xl font-bold text-cyan-400">{totalFavorites}</div>
          </div>
          <div>
            <div className="text-xs text-slate-400 mb-1">Runs</div>
            <div className="text-2xl font-bold text-white">{runNumbers.length}</div>
          </div>
          {pointsRange && (
            <>
              <div>
                <div className="text-xs text-slate-400 mb-1">Point Range</div>
                <div className="text-sm font-medium text-white">
                  {pointsRange.min.toFixed(1)} - {pointsRange.max.toFixed(1)}
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-400 mb-1">Avg Points</div>
                <div className="text-lg font-bold text-white">
                  {(favorites.reduce((sum, f) => sum + f.totalPoints, 0) / favorites.length).toFixed(1)}
                </div>
              </div>
            </>
          )}
        </div>

        {runNumbers.length > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-600">
            <div className="text-xs text-slate-400 mb-2">By Run:</div>
            <div className="flex flex-wrap gap-2">
              {runNumbers.map(run => {
                const count = favorites.filter(f => f.runNumber === run).length;
                return (
                  <div key={run} className="bg-slate-800/40 rounded px-3 py-1 text-xs">
                    <span className="text-cyan-400 font-semibold">Run {run}:</span>
                    <span className="text-white ml-1">{count} lineups</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </Card>

      {/* Sort and Filter */}
      <div className="flex flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <Label className="text-sm text-white">Sort by:</Label>
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-40 bg-slate-700 border-slate-600 text-white text-sm h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-cyan-500/20">
              <SelectItem value="points-desc" className="text-white">Points (High)</SelectItem>
              <SelectItem value="points-asc" className="text-white">Points (Low)</SelectItem>
              <SelectItem value="salary-desc" className="text-white">Salary (High)</SelectItem>
              <SelectItem value="salary-asc" className="text-white">Salary (Low)</SelectItem>
              <SelectItem value="run" className="text-white">Run Number</SelectItem>
              <SelectItem value="date" className="text-white">Date Added</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-sm text-white">Filter:</Label>
          <Select value={filterRun.toString()} onValueChange={(v: string) => setFilterRun(v === 'all' ? 'all' : parseInt(v))}>
            <SelectTrigger className="w-32 bg-slate-700 border-slate-600 text-white text-sm h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-cyan-500/20">
              <SelectItem value="all" className="text-white">All Runs</SelectItem>
              {runNumbers.map(run => (
                <SelectItem key={run} value={run.toString()} className="text-white">Run {run}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Favorites List */}
      <div className="flex-1 overflow-auto max-h-96 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
        <div className="space-y-3">
          {displayedFavorites.map((favorite) => (
            <Card key={favorite.id} className="bg-slate-700/40 border-slate-600/50 p-4 hover:border-cyan-500/40 transition-colors">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-start gap-3">
                  <Checkbox
                    checked={favorite.selected}
                    onCheckedChange={() => toggleLineup(favorite.id)}
                    className="border-slate-500 mt-1"
                  />
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        favorite.runNumber === 1 ? 'bg-blue-500/20 text-blue-400' :
                        favorite.runNumber === 2 ? 'bg-green-500/20 text-white' :
                        favorite.runNumber === 3 ? 'bg-yellow-500/20 text-yellow-400' :
                        favorite.runNumber === 4 ? 'bg-orange-500/20 text-orange-400' :
                        'bg-purple-500/20 text-purple-400'
                      }`}>
                        Run #{favorite.runNumber}
                      </span>
                      <span className="text-xs text-slate-400">{favorite.dateAdded}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <div>
                        <span className="text-slate-400">Total Points:</span>
                        <span className="ml-2 font-bold text-white">{favorite.totalPoints.toFixed(1)}</span>
                      </div>
                      <div>
                        <span className="text-slate-400">Total Salary:</span>
                        <span className="ml-2 font-bold text-white">${favorite.totalSalary.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => deleteLineup(favorite.id)}
                  className="border-red-500/30 hover:bg-red-500/10 text-red-400"
                >
                  <XSquare className="w-4 h-4" />
                </Button>
              </div>

              {/* Player List */}
              {favorite.players.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                  {favorite.players.slice(0, 10).map((player, pidx) => (
                    <div key={pidx} className="bg-slate-800/40 rounded px-2 py-1.5">
                      <div className="text-white font-medium truncate">{player.name}</div>
                      <div className="text-slate-400 text-xs">
                        {player.position} • {player.team} • ${(player.salary / 1000).toFixed(1)}k
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          ))}
        </div>
      </div>

      {/* Summary Bar */}
      <div className="bg-slate-700/40 border border-slate-600/50 rounded-lg p-3">
        <div className="flex items-center justify-between text-sm flex-wrap gap-2">
          <div className="text-white">
            Showing <span className="font-semibold text-cyan-400">{displayedFavorites.length}</span> of {totalFavorites} favorites
            {filterRun !== 'all' && <span className="text-slate-500 ml-2">(Run #{filterRun} only)</span>}
          </div>
          {selectedCount < totalFavorites && (
            <div className="text-xs text-yellow-400">
              {selectedCount} lineups selected for export
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

type BuildSport = Sport | null;

// Build state interface for multi-build support
interface BuildState {
  id: string;
  name: string;
  sport: BuildSport;
  activeTab: string;
  playerData: Player[];
  selectedPlayers: string[];
  teamSelections: Record<number | 'all', string[]>;
  stackSettings: StackType[];
  advancedQuantSettings: any;
  results: any[];
  minSalary: number | null;
}

const DFSOptimizer = React.memo(() => {
  const createEmptyTeamSelections = (): Record<number | 'all', string[]> => ({
    all: [],
    2: [],
    3: [],
    4: [],
    5: [],
  });

  function initializeStackSettings(sport: Sport): StackType[] {
    const config = SPORT_CONFIGS[sport];
    return config.stackTypes.map((stackType, index) => ({
      id: `stack-${index}`,
      label: stackType,
      minExp: 0,
      maxExp: 100,
      enabled: false
    }));
  }

  const createBuildState = (id: string, name: string, sport: BuildSport): BuildState => ({
    id,
    name,
    sport,
    activeTab: 'team-combos',
    playerData: [],
    selectedPlayers: [],
    teamSelections: createEmptyTeamSelections(),
    stackSettings: sport ? initializeStackSettings(sport) : [],
    advancedQuantSettings: { ...DEFAULT_ADVANCED_QUANT_SETTINGS },
    results: [],
    minSalary: sport ? SPORT_CONFIGS[sport].defaultMinSalary : null,
  });

  // Build management state
  const [builds, setBuilds] = useState<BuildState[]>([
    createBuildState('build-1', 'Build 1', null)
  ]);
  const [activeBuildId, setActiveBuildId] = useState<string>('build-1');

  // Get current build
  const currentBuild = builds.find(build => build.id === activeBuildId) || builds[0];
  
  const currentSport = currentBuild?.sport ?? null;
  const sportConfig = currentSport ? SPORT_CONFIGS[currentSport] : undefined;
  const sportLocked = Boolean(currentSport);
  const sportStatusLabel = sportLocked
    ? `Activate Optimizer: ${currentSport}`
    : 'Activate Optimizer: No Sport Selected';
  
  const getHighestBuildNumber = (buildList: BuildState[]) =>
    buildList.reduce((max, build) => {
      const match = build.id.match(/build-(\d+)/);
      if (!match) return max;
      const parsed = parseInt(match[1], 10);
      return Number.isNaN(parsed) ? max : Math.max(max, parsed);
    }, 0);

  // Build management functions
  const addNewBuild = () => {
    if (builds.length >= 5) return; // Max 5 builds
    
    const newBuildNumber = getHighestBuildNumber(builds) + 1;
    const newBuild = createBuildState(`build-${newBuildNumber}`, `Build ${newBuildNumber}`, null);
    
    setBuilds(prev => [...prev, newBuild]);
    setActiveBuildId(newBuild.id);
  };

  const removeBuild = (buildId: string) => {
    if (builds.length <= 1) return; // Keep at least one build
    
    setBuilds(prev => {
      const newBuilds = prev.filter(build => build.id !== buildId);
      // If we're removing the active build, switch to the first remaining build
      if (buildId === activeBuildId) {
        setActiveBuildId(newBuilds[0].id);
      }
      return newBuilds;
    });
  };

  const switchBuild = (buildId: string) => {
    setActiveBuildId(buildId);
  };

  const updateCurrentBuild = (updates: Partial<BuildState>) => {
    setBuilds(prev => prev.map(build => 
      build.id === activeBuildId ? { ...build, ...updates } : build
    ));
  };

  // Get current build data
  const activeTab = currentBuild.activeTab;
  const playerData = currentBuild.playerData;
  const selectedPlayers = currentBuild.selectedPlayers;
  const teamSelections = currentBuild.teamSelections;
  const stackSettings = currentBuild.stackSettings;
  const advancedQuantSettings = currentBuild.advancedQuantSettings;
  const results = currentBuild.results;

  // Initialize stack settings for current build if not set
  useEffect(() => {
    if (!currentSport) return;
    if (stackSettings.length === 0) {
      updateCurrentBuild({ stackSettings: initializeStackSettings(currentSport) });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSport, stackSettings.length]);

  // Build state setters
  const setActiveTab = (tab: string) => updateCurrentBuild({ activeTab: tab });
  const setPlayerData = (data: Player[]) => updateCurrentBuild({ playerData: data });
  const setSelectedPlayers = (players: string[]) => updateCurrentBuild({ selectedPlayers: players });
  const setTeamSelections = (selections: Record<number | 'all', string[]>) => updateCurrentBuild({ teamSelections: selections });
  const setStackSettings = (settings: StackType[]) => updateCurrentBuild({ stackSettings: settings });
  const setAdvancedQuantSettings = (settings: any) => updateCurrentBuild({ advancedQuantSettings: settings });
  const setResults = (newResults: any[]) => updateCurrentBuild({ results: newResults });

  // Define syncSelectionsWithBackend before handleSelectedPlayersChange
  const syncSelectionsWithBackend = useCallback(async (selectedIds: string[]) => {
    try {
      await dfsApi.bulkUpdatePlayers({ action: 'deselect', filters: {} });
      if (selectedIds.length === 0) {
        return;
      }
      await Promise.all(
        selectedIds.map((id) =>
          dfsApi.updatePlayer(id, { selected: true })
        )
      );
    } catch (error) {
      console.error('Failed to sync selected players with backend', error);
      throw error;
    }
  }, []);

  const handleSelectedPlayersChange = useCallback(async (playerIds: string[]) => {
    setSelectedPlayers(playerIds);
    const updatedPlayers = (currentBuild.playerData || []).map((player) => ({
      ...player,
      selected: playerIds.includes(player.id),
    }));
    setPlayerData(updatedPlayers);
    
    // Sync with backend
    try {
      await syncSelectionsWithBackend(playerIds);
    } catch (error) {
      console.error('Failed to sync player selections with backend:', error);
      // Don't show toast here as individual toggles will handle their own errors
    }
  }, [currentBuild.playerData, syncSelectionsWithBackend]);

  // Optimization Settings
  const [numLineups, setNumLineups] = useState(100);
  const [minUnique, setMinUnique] = useState(3);
  const minSalary = currentBuild.minSalary ?? (sportConfig?.defaultMinSalary ?? 0);
  const setMinSalary = (value: number) => {
    if (!sportConfig) return;
    const { maxSalary, defaultMinSalary } = sportConfig;
    const parsed = Number.isFinite(value) ? value : defaultMinSalary;
    const clamped = Math.max(0, Math.min(parsed, maxSalary));
    updateCurrentBuild({ minSalary: clamped });
  };
  const [disableKelly, setDisableKelly] = useState(false);
  
  // Sorting
  const [sortMethod, setSortMethod] = useState('points');

  // Right sidebar tab state
  const [rightSidebarTab, setRightSidebarTab] = useState<'lineups' | 'favorites' | 'results'>('lineups');
  const [selectedLineups, setSelectedLineups] = useState<Set<number>>(new Set());
  
  // Generated Teams - Now connected to backend
  const [generatedTeams, setGeneratedTeams] = useState<any[]>([]);
  
  // Optimization state
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isRunningCombinations, setIsRunningCombinations] = useState(false);
  const [dkEntriesLoaded, setDkEntriesLoaded] = useState(false);
  const workspaceCsvInputRef = useRef<HTMLInputElement | null>(null);
  const projCsvInputRef = useRef<HTMLInputElement | null>(null);
  const [favorites, setFavorites] = useState<FavoriteLineup[]>([]);
  const [favoriteRunCounter, setFavoriteRunCounter] = useState(1);

  // Keep backend sport mode in sync with current selection
  useEffect(() => {
    if (!currentSport) return;
    let isMounted = true;
    dfsApi.setSport(currentSport).catch((error) => {
      if (!isMounted) return;
      console.error('Failed to set sport on backend', error);
    });
    return () => {
      isMounted = false;
    };
  }, [currentSport]);
  
  // Lineups state management
  const [lineups, setLineups] = useState<Array<{
    id: string;
    players: Array<{
      id: string;
      name: string;
      team: string;
      position: string;
      salary: number;
      projection: number;
      value: number;
    }>;
    totalSalary: number;
    totalProjection: number;
    value: number;
    strategy: string;
    stacks: Array<{
      team: string;
      players: number;
      positions: string;
      type: string;
    }>;
    timestamp: string;
  }>>([]);
  const [isLoadingLineups, setIsLoadingLineups] = useState(false);
  
  const handleBuildSportChange = (sport: Sport) => {
    if (!currentBuild || currentBuild.sport) {
      return;
    }

    updateCurrentBuild({
      sport,
      activeTab: 'team-combos',
      playerData: [],
      selectedPlayers: [],
      teamSelections: createEmptyTeamSelections(),
      stackSettings: initializeStackSettings(sport),
      advancedQuantSettings: { ...DEFAULT_ADVANCED_QUANT_SETTINGS },
      results: [],
      minSalary: SPORT_CONFIGS[sport].defaultMinSalary,
    });

    setGeneratedTeams([]);
    setLineups([]);
    setFavorites([]);
    setFavoriteRunCounter(1);
    setDkEntriesLoaded(false);
  };
  
  // Function to fetch lineups from backend
  const fetchLineups = async () => {
    if (!currentSport) {
      setLineups([]);
      setIsLoadingLineups(false);
      return;
    }
    setIsLoadingLineups(true);
    try {
      // First try to get lineups from local results state
      if (results.length > 0) {
        // Transform results to lineups format
        const transformedLineups = results.map(result => ({
          id: result.id,
          players: result.players.map((p: any) => ({
            id: `${p.name}_${p.position}`,
            name: p.name,
            team: p.team,
            position: p.position,
            salary: p.salary,
            projection: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
            projectedPoints: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
            value: (p.projection || p.Predicted_DK_Points || p.projectedPoints || 0) ? 
                   ((p.projection || p.Predicted_DK_Points || p.projectedPoints || 0) / p.salary * 1000) : 0
          })),
          totalSalary: result.salary,
          totalProjection: result.points,
          value: result.points / result.salary * 1000,
          strategy: 'Optimized',
          stacks: [],
          timestamp: new Date().toISOString()
        }));
        setLineups(transformedLineups);
        setIsLoadingLineups(false);
        return;
      }

      // If no local results, try to fetch from backend
      const response = await fetch(`/api/lineups/${currentSport}`);
      if (response.ok) {
        const data = await response.json();
        setLineups(data.lineups || []);
      } else {
        console.error('Failed to fetch lineups:', response.statusText);
        setLineups([]);
      }
    } catch (error) {
      console.error('Error fetching lineups:', error);
      setLineups([]);
    } finally {
      setIsLoadingLineups(false);
    }
  };

  // Fetch lineups when lineups tab is activated or when results change
  useEffect(() => {
    if (activeTab === 'lineups') {
      fetchLineups();
    }
  }, [activeTab, currentSport, results]);
  
  // Control panel state
  const tabs = [
    { id: 'players', label: 'Players', icon: Users },
    { id: 'team-stacks', label: 'Team Stacks', icon: Link2 },
    { id: 'stack-exposure', label: 'Stack Exposure', icon: BarChart3 },
    { id: 'team-combos', label: 'Team Combinations', icon: Target },
    { id: 'advanced-quant', label: 'Advanced Quant', icon: Cpu },
    { id: 'lineups', label: 'Generated Lineups', icon: Trophy },
    { id: 'my-entries', label: 'My Entries', icon: Star },
  ];

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!currentSport || !sportConfig) {
      alert('Select a sport before uploading player data.');
      return;
    }

    try {
      const uploadResult = await dfsApi.uploadPlayers(file);

      if (uploadResult?.success) {
        console.log(`✅ Uploaded ${uploadResult.playersCount} players to backend`);

        const playersResponse = await dfsApi.getPlayers();
        const backendPlayers = (playersResponse?.players ?? []) as any[];

        const transformedPlayers: Player[] = backendPlayers.map((p: any) => ({
          id: p.id,
          name: p.name,
          team: p.team,
          position: p.position,
          salary: p.salary,
          projectedPoints: p.projection || p.projectedPoints || 0,
          minExp: p.minExposure ?? 0,
          maxExp: p.maxExposure ?? 100,
          selected: Boolean(p.selected),
        }));

        setPlayerData(transformedPlayers);
        // Select all players by default
        setSelectedPlayers(
          transformedPlayers.map((player) => player.id)
        );
        // Extract all unique teams and populate all stack sizes
        const uniqueTeams = [...new Set(transformedPlayers.map(p => p.team))].filter(Boolean);
        setTeamSelections({
          all: uniqueTeams,
          2: uniqueTeams,
          3: uniqueTeams,
          4: uniqueTeams,
          5: uniqueTeams,
        });
        // Initialize stack settings and enable all stacks by default
        const initialStackSettings = initializeStackSettings(currentSport);
        setStackSettings(initialStackSettings.map(s => ({ ...s, enabled: true })));
        setActiveTab('players');
        alert(`✅ Loaded ${transformedPlayers.length} players successfully!`);
      } else {
        alert(`❌ Upload failed: ${uploadResult?.error || 'Unknown error'}`);
      }
    } catch (error) {
      const apiError = dfsApi.handleApiError(error);
      alert(`❌ Upload failed: ${apiError.message}`);
    }
  };

  const handleRunOptimization = async () => {
    // Validate inputs
    if (!currentSport || !sportConfig) {
      alert('Select a sport before running the optimizer.');
      return;
    }

    if (playerData.length === 0) {
      alert('❌ Please load player data first');
      return;
    }

    const selectedCount = selectedPlayers.length;
    const minRequired = sportConfig.lineupSize;
    
    if (selectedCount < minRequired) {
      alert(`❌ Please select at least ${minRequired} players for ${currentSport}`);
      return;
    }

    setIsOptimizing(true);
    
    try {
      await dfsApi.setSport(currentSport);
      await syncSelectionsWithBackend(selectedPlayers);

      // Prepare stack settings
      const enabledStacks = stackSettings.filter(s => s.enabled);

      const stackSizeEntries = Object.entries(teamSelections)
        .filter(([sizeKey, teams]) => sizeKey !== 'all' && Array.isArray(teams) && teams.length > 0);

      const stackTeamsSet = new Set<string>();
      stackSizeEntries.forEach(([, teams]) => {
        teams.forEach(team => stackTeamsSet.add(team));
      });

      const fallbackTeams = playerData
        .filter(player => selectedPlayers.includes(player.id))
        .map(player => player.team)
        .filter(Boolean);

      if (stackTeamsSet.size === 0) {
        fallbackTeams.forEach(team => stackTeamsSet.add(team));
      }

      const stackTeams = Array.from(stackTeamsSet);

      const stackSizeValues = stackSizeEntries
        .map(([sizeKey]) => Number(sizeKey))
        .filter(size => !Number.isNaN(size) && size > 0);

      const minPlayersPerTeam = stackSizeValues.length > 0 ? Math.min(...stackSizeValues) : 2;
      const maxPlayersPerTeam = stackSizeValues.length > 0 ? Math.max(...stackSizeValues) : 4;

      // Run optimization
      console.log('🚀 Starting optimization...');
      const sortingMethodMap: Record<string, string> = {
        points: 'Points',
        value: 'Value',
        salary: 'Salary',
      };

      const exposureSettingsPayload = enabledStacks.reduce<Record<string, { min: number; max: number }>>(
        (acc, stack) => {
          acc[stack.label] = { min: stack.minExp, max: stack.maxExp };
          return acc;
        },
        {}
      );

      const optimizationResponse = await dfsApi.optimizeLineups({
        sport: currentSport,
        numLineups,
        minSalary,
        maxSalary: sportConfig.maxSalary,
        stackSettings: {
          enabled: enabledStacks.length > 0 && stackTeams.length > 0,
          teams: stackTeams,
          minPlayersPerTeam,
          maxPlayersPerTeam,
        },
        uniquePlayers: minUnique,
        maxExposure: 40,
        sortingMethod: sortingMethodMap[sortMethod] ?? 'Points',
        minUniquePlayersBetweenLineups: minUnique,
        disableKellySizing: disableKelly,
        exposureSettings: exposureSettingsPayload,
        contestMode: 'gpp',
        monteCarloIterations: 100,
        advancedQuantSettings: {
          ...DEFAULT_ADVANCED_QUANT_SETTINGS,
          ...advancedQuantSettings,
        },
      });

      if (optimizationResponse.success) {
        console.log('✅ Optimization complete:', optimizationResponse.summary);
        
        // Transform lineups for display
        const transformedResults = optimizationResponse.lineups.map((lineup: any) => ({
          id: lineup.id,
          players: lineup.players.map((p: any) => ({
            name: p.name,
            position: p.position,
            team: p.team,
            salary: p.salary,
            projection: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
            projectedPoints: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
          })),
          points: lineup.totalProjection,
          salary: lineup.totalSalary,
        }));

        setResults(transformedResults);
        // Also update lineups state for the lineups tab (ensure projections are included)
        const transformedLineups = (optimizationResponse.lineups || []).map((lineup: any) => ({
          ...lineup,
          players: lineup.players.map((p: any) => ({
            ...p,
            projection: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
            projectedPoints: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
          }))
        }));
        setLineups(transformedLineups);
        
        // Switch to Generated Lineups tab to show results
        setActiveTab('lineups');
        
        alert(`✅ Generated ${transformedResults.length} optimal lineups!\nAvg Projection: ${optimizationResponse.summary.avgProjection.toFixed(1)} pts`);
      } else {
        alert(`❌ Optimization failed: ${optimizationResponse.error || 'Unknown error'}`);
      }
    } catch (error) {
      const apiError = dfsApi.handleApiError(error);
      alert(`❌ Optimization failed: ${apiError.message}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  const handleRunCombinations = async () => {
    if (!currentSport || !sportConfig) {
      alert('Select a sport before running combinations.');
      return;
    }

    if (generatedTeams.length === 0) {
      alert('No teams available to run combinations. Generate teams first.');
      return;
    }
    
    setIsRunningCombinations(true);
    console.log('Running combinations for teams:', generatedTeams);
    
    try {
      await dfsApi.setSport(currentSport);
      await syncSelectionsWithBackend(selectedPlayers);

      const comboTeams = generatedTeams
        .map((team) => team.team || (team.name ? team.name.split(' ')[0] : ''))
        .filter((team: string) => Boolean(team));

      const sortingMethodMap: Record<string, string> = {
        points: 'Points',
        value: 'Value',
        salary: 'Salary',
      };

      const comboResponse = await dfsApi.optimizeLineups({
        advancedQuantSettings: {
          ...DEFAULT_ADVANCED_QUANT_SETTINGS,
          ...advancedQuantSettings,
        },
        sport: currentSport,
        numLineups: 5,
        minSalary,
        maxSalary: sportConfig.maxSalary,
        stackSettings: {
          enabled: comboTeams.length > 0,
          teams: comboTeams,
          minPlayersPerTeam: 2,
          maxPlayersPerTeam: 5,
        },
        uniquePlayers: minUnique,
        maxExposure: 40,
        sortingMethod: sortingMethodMap[sortMethod] ?? 'Points',
        minUniquePlayersBetweenLineups: minUnique,
        disableKellySizing: disableKelly,
        contestMode: 'gpp',
        monteCarloIterations: 100,
      });

      console.log('Combinations complete:', comboResponse);
      
      if (comboResponse.lineups && comboResponse.lineups.length > 0) {
        const transformedCombo = comboResponse.lineups.map((lineup: any) => ({
          id: lineup.id,
          players: lineup.players.map((p: any) => ({
            name: p.name,
            position: p.position,
            team: p.team,
            salary: p.salary,
          })),
          points: lineup.totalProjection,
          salary: lineup.totalSalary,
        }));

        setResults(transformedCombo);
        setLineups(comboResponse.lineups);
        setActiveTab('lineups');
        console.log(`✅ Generated ${comboResponse.lineups.length} lineups from backend`);
      }
      
    } catch (error) {
      console.error('Error running combinations:', error);
      const apiError = dfsApi.handleApiError(error);
      alert(`Failed to run combinations: ${apiError.message}`);
    } finally {
      setIsRunningCombinations(false);
    }
  };

  // Function to generate teams from backend
  const handleGenerateTeams = async () => {
    if (playerData.length === 0) {
      alert('Please load player data first.');
      return;
    }
    
    try {
      // Call backend to generate team combinations
      const response = await fetch('/api/generate-teams', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sport: currentSport,
          players: playerData.filter(p => p.selected),
          numTeams: 5
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Generated teams:', data);
      
      // Update generated teams with backend data
      if (data.teams && data.teams.length > 0) {
        setGeneratedTeams(data.teams);
        console.log(`✅ Generated ${data.teams.length} teams from backend`);
      }
      
    } catch (error) {
      console.error('Error generating teams:', error);
      // Fallback: Create teams from selected players
      const selectedPlayers = playerData.filter(p => p.selected);
      const teams = [...new Set(selectedPlayers.map(p => p.team))].slice(0, 5);
      const generatedTeams = teams.map((team, index) => ({
        id: `team_${index + 1}`,
        name: `${team} Stack`,
        team: team,
        players: selectedPlayers.filter(p => p.team === team).map(p => `${p.position}: ${p.name}`),
        type: 'stack'
      }));
      setGeneratedTeams(generatedTeams);
      console.log(`✅ Generated ${generatedTeams.length} fallback teams`);
    }
  };

  // Function to update generated teams (called by backend)
  const updateGeneratedTeams = (teams: any[]) => {
    setGeneratedTeams(teams);
  };

  // Function to run combinations (called by backend)
  const runCombinations = () => {
    handleRunCombinations();
  };

  const handleLoadEntries = useCallback(() => {
    setDkEntriesLoaded(true);
    alert('✅ DraftKings entries file loaded (frontend only preview).');
  }, []);

  const handleExportDraftKings = useCallback(async () => {
    if (results.length === 0) {
      alert('❌ No optimized lineups available to export yet.');
      return;
    }

    try {
      const response = await fetch(`/api/export/draftkings?sport=${currentSport}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentSport.toLowerCase()}_lineups_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      alert(`✅ Exported ${results.length} lineups`);
    } catch (error) {
      console.error('Export failed:', error);
      alert('❌ Export failed');
    }
  }, [currentSport, results]);

  const handleFillEntriesWithOptimized = useCallback(() => {
    if (!dkEntriesLoaded) {
      alert('❌ Load your DraftKings entries file first.');
      return;
    }
    if (results.length === 0) {
      alert('❌ Run an optimization before filling entries.');
      return;
    }
    alert('✅ Filled entries with optimized lineups (frontend preview only).');
  }, [dkEntriesLoaded, results]);

  const handleAddFavoritesFromResults = useCallback(() => {
    if (results.length === 0) {
      alert('No lineups available. Run optimization first.');
      return;
    }

    const defaultCount = Math.min(30, results.length);
    const countInput = prompt(
      `Add how many lineups from current pool?\n\nAvailable: ${results.length} lineups\nCurrent favorites: ${favorites.length}`,
      defaultCount.toString()
    );

    if (countInput === null) {
      return;
    }

    const count = parseInt(countInput, 10);

    if (Number.isNaN(count) || count <= 0 || count > results.length) {
      return;
    }

    const runNumber = favoriteRunCounter;
    const timestamp = Date.now();
    const newFavorites = results.slice(0, count).map((result, idx) => ({
      id: `fav-${timestamp}-${idx}`,
      players: result.players || [],
      totalPoints: result.points || 0,
      totalSalary: result.salary || 0,
      runNumber,
      dateAdded: new Date().toLocaleString(),
      selected: true,
    }));

    setFavorites((prev) => [...prev, ...newFavorites]);
    setFavoriteRunCounter((prev) => prev + 1);
    alert(`Added ${count} lineups to favorites as Run #${runNumber}`);
  }, [results, favorites.length, favoriteRunCounter]);

  const handleClearFavorites = useCallback(() => {
    setFavorites((prev) => {
      if (prev.length === 0) {
        return prev;
      }

      if (confirm(`Delete all ${prev.length} favorite lineups?\n\nThis action cannot be undone.`)) {
        setFavoriteRunCounter(1);
        return [];
      }

      return prev;
    });
  }, []);

  const handleExportFavorites = useCallback(() => {
    if (favorites.length === 0) {
      alert('No favorites to export.');
      return;
    }

    const countInput = prompt(
      `Export how many lineups?\n\nAvailable: ${favorites.length} favorites`,
      favorites.length.toString()
    );

    if (countInput === null) {
      return;
    }

    const count = parseInt(countInput, 10);

    if (Number.isNaN(count) || count <= 0) {
      return;
    }

    alert(
      `Would export ${Math.min(count, favorites.length)} lineups to DraftKings CSV format.\n\nExport functionality will be connected to backend.`
    );
  }, [favorites]);

  // Expose functions globally for backend access
  React.useEffect(() => {
    (window as any).updateGeneratedTeams = updateGeneratedTeams;
    (window as any).runCombinations = runCombinations;
    return () => {
      delete (window as any).updateGeneratedTeams;
      delete (window as any).runCombinations;
    };
  }, []);

  return (
    <div className="flex min-h-full w-full flex-col bg-slate-900 text-white overflow-hidden">

      <div className="flex-1 min-h-0 overflow-hidden px-2 pb-3 pt-3">
        <div className="flex h-full flex-col overflow-hidden border border-slate-700 bg-slate-900 w-full max-w-[1920px] mx-auto">
          <div className="bg-slate-800 border-b border-slate-700 flex items-center px-2 py-1">
            <div className="flex items-center gap-1">
              {builds.map((build) => (
                <div
                  key={build.id}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-t-lg text-sm cursor-pointer transition-colors ${
                    build.id === activeBuildId
                      ? 'bg-slate-900 text-cyan-400 border-b-2 border-cyan-400'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                  onClick={() => switchBuild(build.id)}
                >
                  <span className="font-medium">{build.name}</span>
                  {builds.length > 1 && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeBuild(build.id);
                      }}
                      className="ml-1 hover:text-red-400 transition-colors"
                    >
                      <XSquare className="w-3 h-3" />
                    </button>
                  )}
                </div>
              ))}
              {builds.length < 5 && (
                <button
                  onClick={addNewBuild}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-t-lg text-sm bg-slate-700 text-slate-300 hover:bg-slate-600 transition-colors"
                >
                  <Plus className="w-3 h-3" />
                  <span>New Build</span>
                </button>
              )}
            </div>
          </div>

          <div className="bg-slate-800 border-b border-slate-700 px-4 py-4 flex flex-wrap items-center justify-between gap-6">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-3 bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-2">
                <span className="uppercase tracking-wide text-[11px] text-slate-400">Build / Select Sport</span>
                <div className="flex gap-2">
                  {(['NFL', 'NBA', 'MLB'] as Sport[]).map((sportOption) => (
                    <button
                      key={sportOption}
                      onClick={() => handleBuildSportChange(sportOption)}
                      disabled={sportLocked}
                      className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all disabled:cursor-not-allowed disabled:opacity-95 ${
                        currentSport === sportOption
                          ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/30 border-2 border-cyan-400'
                          : 'bg-slate-700/40 text-slate-300 border-2 border-slate-600/30 hover:bg-slate-700 hover:border-cyan-500/50 hover:text-white'
                      }`}
                      title={
                        sportLocked && currentSport !== sportOption
                          ? 'Sport selection is locked for this build'
                          : undefined
                      }
                    >
                      {sportOption === 'NFL' && '🏈 NFL'}
                      {sportOption === 'NBA' && '🏀 NBA'}
                      {sportOption === 'MLB' && '⚾ MLB'}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-3 bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-2">
                <span className="uppercase tracking-wide text-[11px] text-slate-400">Active Build</span>
                <span className="text-sm font-semibold text-white">{currentBuild?.name ?? 'Build'}</span>
              </div>
            </div>
            <div className="flex flex-col items-end gap-1 text-right">
              <span className="uppercase tracking-wide text-[11px] text-slate-400">Build Status</span>
              <span className="text-sm font-semibold text-white">{sportStatusLabel}</span>
              <span className="text-[11px] text-slate-400">
                {sportLocked ? 'Sport selection locked for this build' : 'Select a sport to unlock optimizer tools'}
              </span>
              <span className="text-[11px] text-slate-500">
                {builds.length} {builds.length === 1 ? 'build' : 'builds'} available
              </span>
            </div>
          </div>

          {currentSport && sportConfig ? (
            <div className="flex-1 flex h-full min-h-0">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col h-full min-h-0">
            <div className="bg-slate-900/70 border-b border-slate-800 px-3 py-2">
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300">
                <div className="flex items-center gap-1.5 bg-slate-900/60 border border-slate-700 rounded px-2 py-1">
                  <span className="uppercase tracking-wide text-slate-400">Files</span>
                  <div className="flex items-center gap-1.5">
                    <Button
                      size="sm"
                      className="h-7 px-2 text-[11px] font-medium bg-slate-800 hover:bg-slate-700 text-slate-100 border border-cyan-500/40"
                      onClick={() => workspaceCsvInputRef.current?.click()}
                    >
                      <Upload className="w-3.5 h-3.5 mr-1" />
                      Upload CSV
                    </Button>
                    <input
                      ref={workspaceCsvInputRef}
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={handleFileUpload}
                    />
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2 text-[11px] font-medium text-slate-300 hover:text-white hover:bg-slate-800"
                      onClick={() => projCsvInputRef.current?.click()}
                    >
                      <FileText className="w-3.5 h-3.5 mr-1" />
                      Load Draftkings Predictions
                    </Button>
                    <input
                      ref={projCsvInputRef}
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={handleFileUpload}
                    />
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2 text-[11px] font-medium text-emerald-300 hover:text-emerald-200 hover:bg-emerald-500/10"
                      onClick={handleLoadEntries}
                    >
                      <Download className="w-3.5 h-3.5 mr-1" />
                      Load Entries CSV
                    </Button>
                  </div>
                </div>

                <div className="hidden sm:block h-6 w-px bg-slate-800" />

                <div className="flex items-center gap-2 bg-slate-900/60 border border-slate-700 rounded px-2 py-1">
                  <span className="uppercase tracking-wide text-slate-400">Lineups</span>
                  <Input
                    type="number"
                    min={1}
                    max={500}
                    value={numLineups}
                    onChange={(e) => setNumLineups(parseInt(e.target.value, 10) || 1)}
                    className="h-7 w-16 bg-slate-950 border border-slate-700 text-xs text-slate-100"
                  />
                  <span className="uppercase tracking-wide text-slate-400">Unique</span>
                  <Input
                    type="number"
                    min={0}
                    max={10}
                    value={minUnique}
                    onChange={(e) => setMinUnique(parseInt(e.target.value, 10) || 0)}
                    className="h-7 w-12 bg-slate-950 border border-slate-700 text-xs text-slate-100"
                  />
                </div>

                <div className="flex items-center gap-1.5 bg-slate-900/60 border border-slate-700 rounded px-2 py-1">
                  <span className="uppercase tracking-wide text-slate-400">Salary</span>
                  <Input
                    type="number"
                    min={0}
                    max={sportConfig!.maxSalary}
                    step={500}
                    value={minSalary}
                    onChange={(e) => setMinSalary(parseInt(e.target.value, 10))}
                    className="h-7 w-20 bg-slate-950 border border-slate-700 text-xs text-slate-100"
                  />
                  <span className="text-slate-500">/ {sportConfig!.maxSalary}</span>
                </div>

                <div className="flex items-center gap-1.5 bg-slate-900/60 border border-slate-700 rounded px-2 py-1">
                  <Select value={sortMethod} onValueChange={setSortMethod}>
                    <SelectTrigger className="h-7 min-w-[110px] bg-slate-950 border border-slate-700 text-xs text-slate-100">
                      <SelectValue placeholder="Sorting" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-900 border border-slate-700 text-xs">
                      <SelectItem value="points" className="text-slate-100 text-xs">
                        Sort: Points ↓
                      </SelectItem>
                      <SelectItem value="value" className="text-slate-100 text-xs">
                        Sort: Value ↓
                      </SelectItem>
                      <SelectItem value="salary" className="text-slate-100 text-xs">
                        Sort: Salary ↓
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <div className="flex items-center gap-1">
                    <Checkbox
                      id="workspace-disable-kelly"
                      checked={disableKelly}
                      onCheckedChange={(checked: boolean) => setDisableKelly(Boolean(checked))}
                      className="h-3.5 w-3.5 border-slate-600 data-[state=checked]:bg-slate-800 data-[state=checked]:border-orange-400"
                    />
                    <Label htmlFor="workspace-disable-kelly" className="text-[11px] text-slate-200 cursor-pointer">
                      Disable Kelly
                    </Label>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 bg-slate-900/60 border border-slate-700 rounded px-2 py-1">
                  <Button
                    className="h-7 px-3 text-[11px] font-semibold bg-cyan-600 text-white hover:bg-cyan-500"
                    onClick={handleRunOptimization}
                    disabled={isOptimizing || playerData.length === 0}
                  >
                    <Play className="w-3.5 h-3.5 mr-1" />
                    {isOptimizing ? 'Optimizing…' : 'Optimize'}
                  </Button>
                  <Button
                    variant="ghost"
                    className="h-7 px-3 text-[11px] font-semibold text-yellow-200 hover:text-yellow-100 hover:bg-yellow-500/10 disabled:opacity-50"
                    disabled={results.length === 0}
                    onClick={handleExportDraftKings}
                  >
                    <Save className="w-3.5 h-3.5 mr-1" />
                    Save CSV
                  </Button>
                  <Button
                    variant="ghost"
                    className="h-7 px-3 text-[11px] font-semibold text-emerald-200 hover:text-emerald-100 hover:bg-emerald-500/10 disabled:opacity-50"
                    disabled={!dkEntriesLoaded || results.length === 0}
                    onClick={handleFillEntriesWithOptimized}
                  >
                    <FileText className="w-3.5 h-3.5 mr-1" />
                    Fill Entries
                  </Button>
                </div>

                <div className="flex items-center gap-1.5 bg-slate-900/60 border border-slate-700 rounded px-2 py-1">
                  <Button
                    variant="ghost"
                    className="h-7 px-3 text-[11px] font-semibold text-slate-200 hover:text-white hover:bg-slate-800 disabled:opacity-50"
                    disabled={results.length === 0}
                    onClick={handleAddFavoritesFromResults}
                  >
                    <Plus className="w-3.5 h-3.5 mr-1" />
                    Add Favorite
                  </Button>
                  <Button
                    variant="ghost"
                    className="h-7 px-3 text-[11px] font-semibold text-slate-200 hover:text-white hover:bg-slate-800 disabled:opacity-50"
                    disabled={favorites.length === 0}
                    onClick={handleExportFavorites}
                  >
                    <Download className="w-3.5 h-3.5 mr-1" />
                    Export Fav
                  </Button>
                </div>

                <div className="flex items-center gap-3 bg-slate-900/60 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-300">
                  <span>
                    Players <span className="text-slate-100 font-semibold">{playerData.length}</span>
                  </span>
                  <span className="hidden sm:inline">
                    Selected <span className="text-slate-100 font-semibold">{selectedPlayers.length}</span>
                  </span>
                  <span>
                    Lineups <span className="text-slate-100 font-semibold">{results.length}</span>
                  </span>
                  <span className={dkEntriesLoaded ? 'text-emerald-300 font-semibold' : 'text-slate-500'}>
                    {dkEntriesLoaded ? 'Entries Loaded' : 'Entries Pending'}
                  </span>
                </div>
              </div>
            </div>
            <TabsList className="bg-slate-800 border-b border-slate-700 w-full rounded-none h-auto flex flex-nowrap">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <TabsTrigger
                    key={tab.id}
                    value={tab.id}
                    className="flex items-center gap-1.5 px-3 py-2 text-sm whitespace-nowrap flex-shrink-0 text-white hover:text-white hover:bg-slate-700/50 transition-colors"
                    style={{
                      ...(activeTab === tab.id && {
                        backgroundColor: 'rgba(56, 189, 248, 0.4)',
                        color: '#e0f2fe',
                        borderBottom: '2px solid rgb(56, 189, 248)'
                      })
                    }}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span className="font-normal">{tab.label}</span>
                  </TabsTrigger>
                );
              })}
            </TabsList>

            <div className="flex-1 overflow-auto p-3 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
              <TabsContent value="players" className="mt-0 h-full overflow-auto">
                <PlayersTab
                  playerData={playerData}
                  selectedPlayers={selectedPlayers}
                  sport={currentSport}
                  onPlayersChange={handleSelectedPlayersChange}
                  onPlayerDataChange={setPlayerData}
                />
              </TabsContent>

              <TabsContent value="team-stacks" className="mt-0 h-full overflow-auto">
                <TeamStacksTab
                  playerData={playerData}
                  teamSelections={teamSelections}
                  onTeamSelectionsChange={setTeamSelections}
                />
              </TabsContent>

              <TabsContent value="stack-exposure" className="mt-0 h-full overflow-auto">
                <StackExposureTab
                  stackSettings={stackSettings}
                  sport={currentSport}
                  onStackSettingsChange={setStackSettings}
                />
              </TabsContent>

              <TabsContent value="team-combos" className="mt-0 h-full overflow-auto">
                <TeamCombinationsTab playerData={playerData} />
              </TabsContent>

              <TabsContent value="advanced-quant" className="mt-0 h-full overflow-auto">
                <AdvancedQuantTab settings={advancedQuantSettings} onSettingsChange={setAdvancedQuantSettings} />
              </TabsContent>

              <TabsContent value="lineups" className="mt-0 h-full overflow-auto">
                <LineupsTab
                  sport={currentSport}
                  lineups={lineups}
                  isLoading={isLoadingLineups}
                  onExportLineups={async (format) => {
                    try {
                      const response = await fetch(`/api/export/${format}?sport=${currentSport}`);
                      if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${currentSport.toLowerCase()}_lineups.${format === 'draftkings' ? 'csv' : 'csv'}`;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                      }
                    } catch (error) {
                      console.error('Export failed:', error);
                    }
                  }}
                  onSaveFavorite={async (lineup) => {
                    try {
                      const response = await fetch('/api/favorites', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          lineup,
                          name: `${currentSport} Lineup ${lineup.id.slice(0, 8)}`
                        })
                      });
                      if (response.ok) {
                        console.log('Favorite saved successfully');
                      }
                    } catch (error) {
                      console.error('Save favorite failed:', error);
                    }
                  }}
                />
              </TabsContent>

              <TabsContent value="my-entries" className="mt-0 h-full overflow-auto">
                <MyEntriesTab results={results} sport={currentSport} />
              </TabsContent>
            </div>
            </Tabs>

            {/* Right Sidebar - Lineups/Favorites/Results */}
            <div className="w-80 flex-shrink-0 border-l border-slate-700 bg-slate-800 flex flex-col h-full overflow-hidden">
              {/* Tabs Header */}
              <div className="flex border-b border-slate-700 flex-shrink-0">
                <button
                  className={`flex-1 px-3 py-3 text-sm font-medium transition-colors ${
                    rightSidebarTab === 'lineups'
                      ? 'bg-slate-900 text-cyan-400 border-b-2 border-cyan-400'
                      : 'text-slate-400 hover:text-slate-300 hover:bg-slate-700'
                  }`}
                  onClick={() => setRightSidebarTab('lineups')}
                >
                  Lineups
                </button>
                <button
                  className={`flex-1 px-3 py-3 text-sm font-medium transition-colors ${
                    rightSidebarTab === 'favorites'
                      ? 'bg-slate-900 text-cyan-400 border-b-2 border-cyan-400'
                      : 'text-slate-400 hover:text-slate-300 hover:bg-slate-700'
                  }`}
                  onClick={() => setRightSidebarTab('favorites')}
                >
                  Favorites
                </button>
                <button
                  className={`flex-1 px-3 py-3 text-sm font-medium transition-colors ${
                    rightSidebarTab === 'results'
                      ? 'bg-slate-900 text-cyan-400 border-b-2 border-cyan-400'
                      : 'text-slate-400 hover:text-slate-300 hover:bg-slate-700'
                  }`}
                  onClick={() => setRightSidebarTab('results')}
                >
                  Results
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800 hover:scrollbar-thumb-slate-500">
                {rightSidebarTab === 'lineups' && (
                  <>
                    {/* Header Section */}
                    <div className="flex items-center justify-between mb-2 px-3 pt-3 flex-shrink-0">
                      <div>
                        <h3 className="text-[10px] uppercase tracking-wide text-slate-400 mb-0.5">LINEUPS</h3>
                        <div className="text-2xl font-bold text-white">{results.length}</div>
                        <div className="text-[10px] text-slate-400 mt-0.5">
                          Avg {results.length > 0 ? (results.reduce((sum, r) => sum + r.totalPoints, 0) / results.length).toFixed(1) : '0.0'} pts : ${results.length > 0 ? (results.reduce((sum, r) => sum + r.totalSalary, 0) / results.length).toFixed(0) : '0'}
                        </div>
                      </div>
                      <Button
                        size="sm"
                        className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs h-7 px-2"
                        disabled={results.length === 0}
                        onClick={() => {
                          setActiveTab('lineups');
                          toast.success('Switched to Generated Lineups tab');
                        }}
                      >
                        View
                      </Button>
                    </div>

                    {/* Select All */}
                    <div className="flex items-center justify-between py-1.5 border-b border-slate-700 mb-2 px-3 flex-shrink-0">
                      <div className="flex items-center gap-1.5">
                        <Checkbox
                          checked={selectedLineups.size > 0 && selectedLineups.size === Math.min(numLineups, results.length)}
                          onCheckedChange={(checked: boolean) => {
                            if (checked) {
                              const allIndices = new Set(Array.from({ length: Math.min(numLineups, results.length) }, (_, i) => i));
                              setSelectedLineups(allIndices);
                            } else {
                              setSelectedLineups(new Set());
                            }
                          }}
                          className="h-3.5 w-3.5"
                        />
                        <span className="text-xs text-slate-300">Select All</span>
                      </div>
                      <span className="text-xs text-slate-400">{selectedLineups.size} selected</span>
                    </div>

                    {/* Count Input */}
                    <div className="mb-2 px-3 flex-shrink-0">
                      <Label className="text-[10px] uppercase tracking-wide text-slate-400 mb-1 block">COUNT</Label>
                      <Input
                        type="number"
                        min={1}
                        max={500}
                        value={numLineups}
                        onChange={(e) => setNumLineups(parseInt(e.target.value, 10) || 1)}
                        className="bg-slate-900 border-slate-700 text-white h-7 text-xs"
                      />
                    </div>

                    {/* Add Favorites Toggle */}
                    <div className="flex items-center justify-between py-2 border-t border-slate-700 px-3 flex-shrink-0">
                      <span className="text-xs text-slate-300">Add Favorites</span>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" className="sr-only peer" />
                        <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-cyan-600"></div>
                      </label>
                    </div>

                    {/* Message or Lineups List */}
                    {results.length === 0 ? (
                      <div className="flex-1 flex items-center justify-center px-4">
                        <p className="text-sm text-slate-400 text-center">
                          Run optimizer to see lineups.
                        </p>
                      </div>
                    ) : (
                      <div className="px-3 pb-3 flex-shrink-0">
                        <div className="border-2 border-cyan-500 rounded-lg bg-slate-950">
                          <div 
                            style={{
                              height: '350px',
                              overflowY: 'scroll',
                              overflowX: 'hidden'
                            }}
                          >
                            <div className="space-y-1.5 p-2">
                          {results.slice(0, numLineups).map((result, idx) => (
                            <div
                              key={result.id || idx}
                              className={`bg-slate-900 border rounded p-2 transition-colors ${
                                selectedLineups.has(idx) 
                                  ? 'border-cyan-500 bg-cyan-900/10' 
                                  : 'border-slate-700 hover:border-cyan-500'
                              }`}
                            >
                              {/* Lineup Header */}
                              <div className="flex items-center justify-between mb-1.5">
                                <div className="flex items-center gap-1.5">
                                  <Checkbox
                                    checked={selectedLineups.has(idx)}
                                    onCheckedChange={(checked: boolean) => {
                                      const newSelected = new Set(selectedLineups);
                                      if (checked) {
                                        newSelected.add(idx);
                                      } else {
                                        newSelected.delete(idx);
                                      }
                                      setSelectedLineups(newSelected);
                                    }}
                                    onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                    className="h-3.5 w-3.5"
                                  />
                                  <span className="text-[10px] font-semibold text-slate-400">
                                    #{idx + 1}
                                  </span>
                                </div>
                                <div className="flex gap-1.5 items-center">
                                  <span className="text-[10px] font-bold text-cyan-400">
                                    {result.totalPoints?.toFixed(1) || result.points?.toFixed(1) || '0.0'}pts
                                  </span>
                                  <span className="text-[10px] text-slate-500">
                                    ${result.totalSalary || result.salary || 0}
                                  </span>
                                </div>
                              </div>

                              {/* Players List */}
                              <div className="space-y-0">
                                {(result.players || []).map((player: any, pidx: number) => (
                                  <div
                                    key={pidx}
                                    className="flex items-center justify-between text-[10px] py-0.5"
                                  >
                                    <div className="flex items-center gap-1.5 flex-1 min-w-0">
                                      <span className="text-cyan-400 font-mono w-7 flex-shrink-0 text-[9px]">
                                        {player.rosterPosition || player.position || player.pos || 'NA'}
                                      </span>
                                      <span className="text-slate-300 truncate text-[10px]">
                                        {player.name || player.player || 'Unknown'}
                                      </span>
                                    </div>
                                    <span className="text-slate-500 text-[9px] ml-1 flex-shrink-0">
                                      {player.projectedPoints?.toFixed(1) || player.points?.toFixed(1) || player.projection?.toFixed(1) || '0.0'}
                                    </span>
                                  </div>
                                ))}
                              </div>

                              {/* Lineup Footer */}
                              <div className="mt-1.5 pt-1.5 border-t border-slate-800 flex items-center justify-between">
                                <span className="text-[9px] text-slate-500">
                                  {result.players?.length || 0} players
                                </span>
                                <button
                                  className="text-[9px] text-cyan-400 hover:text-cyan-300 font-medium"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    try {
                                      const lineupText = `Lineup ${idx + 1} - ${result.totalPoints?.toFixed(1) || result.points?.toFixed(1) || '0.0'} pts\n` +
                                        `Salary: $${result.totalSalary || result.salary || 0}\n\n` +
                                        (result.players || []).map((p: any) => 
                                          `${p.position || p.pos || 'NA'}: ${p.name || p.player || 'Unknown'} (${p.projectedPoints?.toFixed(1) || p.points?.toFixed(1) || '0.0'} pts)`
                                        ).join('\n');
                                      navigator.clipboard.writeText(lineupText);
                                      toast.success('Lineup copied to clipboard!');
                                    } catch (error) {
                                      toast.error('Failed to copy lineup');
                                    }
                                  }}
                                >
                                  Copy
                                </button>
                              </div>
                            </div>
                          ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}

                {rightSidebarTab === 'favorites' && (
                  <div className="flex-1 flex items-center justify-center px-3 py-3">
                    <p className="text-sm text-slate-400 text-center">
                      No favorites yet. Add lineups from your optimizations.
                    </p>
                  </div>
                )}

                {rightSidebarTab === 'results' && (
                  <div className="px-3 py-3">
                    {/* Header Section */}
                    <div className="mb-3">
                      <h3 className="text-[10px] uppercase tracking-wide text-slate-400 mb-1.5">OPTIMIZATION RESULTS</h3>
                      <div className="bg-slate-900 border border-slate-700 rounded p-2.5 space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] text-slate-400">Total Lineups</span>
                          <span className="text-sm font-bold text-white">{results.length}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] text-slate-400">Avg Points</span>
                          <span className="text-sm font-bold text-cyan-400">
                            {results.length > 0 ? (results.reduce((sum, r) => sum + r.totalPoints, 0) / results.length).toFixed(1) : '0.0'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] text-slate-400">Avg Salary</span>
                          <span className="text-sm font-bold text-emerald-400">
                            ${results.length > 0 ? (results.reduce((sum, r) => sum + r.totalSalary, 0) / results.length).toFixed(0) : '0'}
                          </span>
                        </div>
                        {results.length > 0 && (
                          <>
                            <div className="pt-1.5 border-t border-slate-800">
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] text-slate-400">Best Lineup</span>
                                <span className="text-xs font-bold text-cyan-400">
                                  {Math.max(...results.map(r => r.totalPoints)).toFixed(1)} pts
                                </span>
                              </div>
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] text-slate-400">Worst Lineup</span>
                              <span className="text-xs font-bold text-slate-400">
                                {Math.min(...results.map(r => r.totalPoints)).toFixed(1)} pts
                              </span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] text-slate-400">Points Range</span>
                              <span className="text-xs font-bold text-yellow-400">
                                {(Math.max(...results.map(r => r.totalPoints)) - Math.min(...results.map(r => r.totalPoints))).toFixed(1)} pts
                              </span>
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Player Usage Stats */}
                    {results.length > 0 && (
                      <div className="mb-3">
                        <h3 className="text-[10px] uppercase tracking-wide text-slate-400 mb-1.5">TOP USED PLAYERS</h3>
                        <div className="bg-slate-900 border border-slate-700 rounded overflow-hidden">
                          <div className="max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-slate-900 hover:scrollbar-thumb-slate-600">
                            {(() => {
                              // Calculate player usage
                              const playerUsage = new Map<string, { count: number, avgPoints: number, totalPoints: number }>();
                              results.forEach(result => {
                                (result.players || []).forEach((player: any) => {
                                  const name = player.name || player.player || 'Unknown';
                                  const points = player.projectedPoints || player.points || 0;
                                  const existing = playerUsage.get(name) || { count: 0, avgPoints: 0, totalPoints: 0 };
                                  playerUsage.set(name, {
                                    count: existing.count + 1,
                                    totalPoints: existing.totalPoints + points,
                                    avgPoints: (existing.totalPoints + points) / (existing.count + 1)
                                  });
                                });
                              });
                              
                              // Sort by usage count
                              const sortedPlayers = Array.from(playerUsage.entries())
                                .sort((a, b) => b[1].count - a[1].count)
                                .slice(0, 10);

                              return sortedPlayers.map(([name, stats], idx) => (
                                <div 
                                  key={name}
                                  className={`flex items-center justify-between p-2 ${
                                    idx < sortedPlayers.length - 1 ? 'border-b border-slate-800' : ''
                                  }`}
                                >
                                  <div className="flex-1 min-w-0">
                                    <div className="text-[10px] font-medium text-slate-200 truncate">{name}</div>
                                    <div className="text-[9px] text-slate-500">
                                      {stats.avgPoints.toFixed(1)} pts avg
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-2 ml-1.5">
                                    <div className="text-right">
                                      <div className="text-xs font-bold text-cyan-400">{stats.count}</div>
                                      <div className="text-[9px] text-slate-500">
                                        {((stats.count / results.length) * 100).toFixed(0)}%
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ));
                            })()}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Empty State */}
                    {results.length === 0 && (
                      <div className="flex-1 flex items-center justify-center">
                        <p className="text-sm text-slate-400 text-center">
                          Run the optimizer to see detailed results and statistics.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center px-6 py-10 text-center text-slate-300">
              <div className="max-w-md space-y-3">
                <h3 className="text-xl font-semibold text-white">No Sport Selected</h3>
                <p className="text-sm text-slate-400">
                  Choose NFL, NBA, or MLB on the left to activate this optimizer build. Once selected, the sport is locked for this build.
                </p>
                <p className="text-sm text-slate-500">
                  Each build maintains its own sport and state so you can manage multiple slates at once.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

DFSOptimizer.displayName = 'DFSOptimizer';

export default DFSOptimizer;
