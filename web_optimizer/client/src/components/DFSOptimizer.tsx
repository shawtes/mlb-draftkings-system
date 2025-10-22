import React, { useState, useMemo, useEffect } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Checkbox } from './ui/checkbox';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Users, Link2, BarChart3, Target, Cpu, Star, Upload, Play, Save, Settings, FileText, Download, Plus, CheckSquare, XSquare, ChevronDown, ChevronUp } from 'lucide-react';
import { Sport, SPORT_CONFIGS, getPositionFilters, filterPlayersByPosition, getPositionCount } from './sport-config';

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

  // Update filter when sport changes
  useEffect(() => {
    setPositionFilter(sport === 'MLB' ? 'all-batters' : 'all-offense');
  }, [sport]);

  // Position counts
  const positionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    playerData.forEach(p => {
      if (sport === 'MLB') {
        const positions = p.position.split('/');
        positions.forEach(pos => {
          counts[pos] = (counts[pos] || 0) + 1;
        });
      } else {
        // NFL positions are single
        counts[p.position] = (counts[p.position] || 0) + 1;
      }
    });
    return counts;
  }, [playerData, sport]);

  // Filter players by position using sport config
  const filteredPlayers = useMemo(() => {
    const filtered = filterPlayersByPosition(playerData, positionFilter, sport);

    // Sort
    const sorted = [...filtered].sort((a, b) => {
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

    return sorted;
  }, [playerData, positionFilter, sortBy]);

  // Handle select all
  const handleSelectAll = () => {
    const currentIds = filteredPlayers.map(p => p.id);
    const newSelected = Array.from(new Set([...selectedPlayers, ...currentIds]));
    onPlayersChange(newSelected);
  };

  // Handle deselect all
  const handleDeselectAll = () => {
    const currentIds = new Set(filteredPlayers.map(p => p.id));
    const newSelected = selectedPlayers.filter(id => !currentIds.has(id));
    onPlayersChange(newSelected);
  };

  // Toggle player selection
  const togglePlayer = (playerId: string) => {
    const newSelected = selectedPlayers.includes(playerId)
      ? selectedPlayers.filter(id => id !== playerId)
      : [...selectedPlayers, playerId];
    onPlayersChange(newSelected);
  };

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
          <Button variant="outline" className="border-cyan-500/30 hover:bg-cyan-500/10 text-white">
            <Upload className="w-4 h-4 mr-2" />
            Load CSV
          </Button>
        </div>
      </div>
    );
  }

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
                <Checkbox className="border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400" />
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
                  className={`border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors ${
                    idx % 2 === 0 ? 'bg-slate-800/20' : ''
                  }`}
                >
                  <td className="px-3 py-2">
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() => togglePlayer(player.id)}
                      className="border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
                      style={{ 
                        accentColor: '#1f2937'
                      }}
                    />
                  </td>
                  <td className="px-3 py-2 text-white font-medium">{player.name}</td>
                  <td className="px-3 py-2 text-white">{player.team}</td>
                  <td className="px-3 py-2 text-white">{player.position}</td>
                  <td className="px-3 py-2 text-right text-white">${player.salary.toLocaleString()}</td>
                  <td className="px-3 py-2 text-right text-white font-medium">{player.projectedPoints.toFixed(1)}</td>
                  <td className="px-3 py-2 text-right text-cyan-400 font-medium">{value}</td>
                  <td className="px-3 py-2">
                    <Input
                      type="number"
                      min="0"
                      max="100"
                      value={player.minExp}
                      onChange={(e) => updateExposure(player.id, 'minExp', parseInt(e.target.value) || 0)}
                      className="bg-slate-700 border-slate-600 text-white text-xs h-8 w-20 text-right"
                    />
                  </td>
                  <td className="px-3 py-2">
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
  const toggleTeam = (team: string) => {
    const current = getSelectedTeams(activeStackSize);
    const updated = current.includes(team)
      ? current.filter(t => t !== team)
      : [...current, team];
    
    onTeamSelectionsChange({
      ...teamSelections,
      [activeStackSize]: updated,
    });
  };

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
                <Checkbox className="border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400" />
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
                  className={`border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors ${
                    idx % 2 === 0 ? 'bg-slate-800/20' : ''
                  } ${!canStack ? 'opacity-50' : ''}`}
                >
                  <td className="px-3 py-2">
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() => toggleTeam(team.abbr)}
                      disabled={!canStack}
                      className="border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
                      style={{ 
                        accentColor: '#1f2937'
                      }}
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
                  <td className="px-3 py-2">
                    <Input
                      type="number"
                      min="0"
                      max="100"
                      value={team.minExp}
                      className="bg-slate-700 border-slate-600 text-white text-xs h-8 w-20 text-right"
                      disabled
                    />
                  </td>
                  <td className="px-3 py-2">
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
  const toggleStackType = (id: string) => {
    const updated = stackSettings.map(s => 
      s.id === id ? { ...s, enabled: !s.enabled } : s
    );
    onStackSettingsChange(updated);
  };

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

  return (
    <div className="flex flex-col h-full p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-700">
        <div>
          <h2 className="text-2xl font-bold text-white">Stack Exposure Configuration</h2>
          <p className="text-slate-300 text-base mt-1">
            Select stack types and set exposure ranges
          </p>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right">
            <div className="text-sm text-slate-400">Active Stacks</div>
            <div className="text-3xl font-bold text-cyan-400">{enabledStacks.length}/{stackSettings.length}</div>
          </div>
          <div className="text-right">
            <div className="text-sm text-slate-400">Total Min</div>
            <div className={`text-3xl font-bold ${hasConflict ? 'text-red-400' : 'text-white'}`}>{totalMinExp}%</div>
          </div>
        </div>
      </div>

      {/* Warnings */}
      {hasNoSelection && (
        <div className="bg-red-500/10 border-l-4 border-red-500 rounded-lg p-4">
          <p className="text-red-400 text-base font-semibold">⚠ Select at least one stack type to optimize</p>
        </div>
      )}
      {hasConflict && (
        <div className="bg-yellow-500/10 border-l-4 border-yellow-500 rounded-lg p-4">
          <p className="text-yellow-400 text-base font-semibold">⚠ Total minimum exposure exceeds 100% - adjust your minimums</p>
        </div>
      )}

      {/* Stack Options - Large Vertical Cards */}
      <div className="flex-1 overflow-auto">
        <div className="space-y-4 pr-2">
          {stackSettings.map((stack) => (
            <div
              key={stack.id}
              className={`border-2 rounded-xl p-6 transition-all ${
                stack.enabled 
                  ? 'bg-slate-800/50 border-cyan-500/50' 
                  : 'bg-slate-800/20 border-slate-700'
              }`}
            >
              <div className="flex items-center gap-8">
                {/* Checkbox and Label */}
                <div className="flex items-center gap-4 min-w-[280px]">
                  <Checkbox
                    checked={stack.enabled}
                    onCheckedChange={() => toggleStackType(stack.id)}
                    className="h-6 w-6 border-2 border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
                    style={{ accentColor: '#1f2937' }}
                  />
                  <span className={`font-bold text-xl ${stack.enabled ? 'text-white' : 'text-slate-400'}`}>
                    {stack.label}
                  </span>
                </div>

                {/* Exposure Controls */}
                <div className="flex items-center gap-8 flex-1">
                  <div className="flex items-center gap-3">
                    <label className="text-base text-slate-300 w-12 font-medium">Min</label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={stack.minExp}
                      onChange={(e) => updateExposure(stack.id, 'minExp', parseInt(e.target.value) || 0)}
                      disabled={!stack.enabled}
                      className={`w-20 h-12 rounded-lg border-2 text-center text-lg font-bold ${
                        stack.enabled
                          ? 'bg-slate-700 border-slate-600 text-white'
                          : 'bg-slate-800 border-slate-700 text-slate-500 cursor-not-allowed'
                      }`}
                    />
                    <span className="text-slate-300 text-lg font-medium">%</span>
                  </div>

                  <div className="flex items-center gap-3">
                    <label className="text-base text-slate-300 w-12 font-medium">Max</label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={stack.maxExp}
                      onChange={(e) => updateExposure(stack.id, 'maxExp', parseInt(e.target.value) || 0)}
                      disabled={!stack.enabled}
                      className={`w-20 h-12 rounded-lg border-2 text-center text-lg font-bold ${
                        stack.enabled
                          ? 'bg-slate-700 border-slate-600 text-white'
                          : 'bg-slate-800 border-slate-700 text-slate-500 cursor-not-allowed'
                      }`}
                    />
                    <span className="text-slate-300 text-lg font-medium">%</span>
                  </div>
                </div>

                {/* Current Exposure Stats */}
                {(stack.lineupExp !== undefined || stack.poolExp !== undefined || stack.entryExp !== undefined) && stack.enabled && (
                  <div className="flex items-center gap-4 text-base text-slate-300">
                    {stack.lineupExp !== undefined && (
                      <div>
                        <span>Lineup: </span>
                        <span className="text-white font-bold text-lg">{stack.lineupExp.toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary Footer */}
      {enabledStacks.length > 0 && (
        <div className="border-t-2 border-slate-700 pt-5">
          <div className="flex items-center justify-between text-lg">
            <div className="flex gap-8">
              <div>
                <span className="text-slate-300 font-medium">Total Min: </span>
                <span className={`font-bold text-xl ${hasConflict ? 'text-red-400' : 'text-cyan-400'}`}>
                  {totalMinExp}%
                </span>
              </div>
              <div>
                <span className="text-slate-300 font-medium">Total Max: </span>
                <span className="font-bold text-xl text-white">{totalMaxExp}%</span>
              </div>
            </div>
            <div className="text-slate-300 text-base">
              {enabledStacks.length} of {stackSettings.length} enabled
            </div>
          </div>
        </div>
      )}

      {/* Tip */}
      <div className="bg-slate-800/40 border border-slate-600 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <div className="text-cyan-400 text-lg">💡</div>
          <p className="text-slate-200 text-base leading-relaxed">
            <span className="font-bold text-cyan-400">Tip:</span> Selected stack types will be distributed across generated lineups. 
            Exposure percentages are calculated after optimization completes.
          </p>
        </div>
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
    const teamSet = new Set(playerData.map(p => p.team));
    return Array.from(teamSet).sort();
  }, [playerData]);

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
          <div className="mb-4">
            <div className="w-16 h-16 mx-auto bg-slate-700 rounded-full flex items-center justify-center">
              <Target className="w-8 h-8 text-slate-400" />
            </div>
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">No Team Data</h3>
          <p className="text-white mb-4">Load players first to generate team combinations</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-6 p-6">
      {/* Header - Enterprise Style */}
      <div className="border-b-2 border-slate-700 pb-4">
        <h2 className="text-2xl font-bold text-white">
          Team Combination Generator
        </h2>
        <p className="text-base text-slate-300 mt-2">
          Select teams and stack type to generate all possible combinations
        </p>
        <p className="text-sm text-green-400 mt-2 font-medium">
          Loaded {playerData.length} players across {teams.length} teams
        </p>
      </div>

      {/* Controls Section - Clean Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Team Selection */}
        <div className="border-2 border-slate-700 bg-slate-800/50 p-6 rounded-xl">
          <h3 className="text-xl font-bold text-white mb-5 border-b border-slate-700 pb-3">
            Select Teams
          </h3>
          
          {/* Select All/Deselect All - Large Enterprise Buttons in Container */}
          <div className="bg-slate-700/30 border border-slate-600 rounded-lg p-3 mb-5">
            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={selectAllTeams}
                className="flex-1 h-12 border-2 border-green-400/50 bg-green-400/25 text-white text-base font-bold shadow-sm transition-none"
              >
                ✅ Select All Teams
              </Button>
              <Button
                variant="outline"
                onClick={deselectAllTeams}
                className="flex-1 h-12 border-2 border-red-500/40 bg-red-500/10 text-white text-base font-bold shadow-sm transition-none"
              >
                ❌ Deselect All
              </Button>
            </div>
          </div>

          {/* Team Checkboxes - Professional Grid */}
          <div className="max-h-80 overflow-auto mb-4 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
            <div className="grid grid-cols-3 gap-3">
              {teams.map(team => (
                <div key={team} className="flex items-center gap-3 p-4 bg-slate-700/50 border-2 border-slate-600 rounded-lg">
                  <Checkbox
                    checked={selectedTeams.includes(team)}
                    onCheckedChange={() => toggleTeam(team)}
                    className="h-5 w-5 border-2 border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
                  />
                  <Label className="text-white cursor-pointer text-base font-semibold" onClick={() => toggleTeam(team)}>
                    {team}
                  </Label>
                </div>
              ))}
            </div>
          </div>

          {/* Selection Counter */}
          <div className="text-base text-slate-200 bg-slate-700/50 border-2 border-slate-600 p-4 rounded-lg font-medium mt-4">
            <span className="font-bold text-green-400 text-lg">{selectedTeams.length}</span> of <span className="font-bold text-lg">{teams.length}</span> teams selected
          </div>
        </div>

        {/* Right: Stack Settings - Enterprise Style */}
        <div className="border-2 border-slate-700 bg-slate-800/50 p-6 rounded-xl">
          <h3 className="text-xl font-bold text-white mb-5 border-b border-slate-700 pb-3">
            Stack Settings
          </h3>

          <div className="space-y-5">
            <div>
              <Label className="text-base text-white block mb-3 font-semibold">Stack Pattern</Label>
              <Select value={stackPattern} onValueChange={setStackPattern}>
                <SelectTrigger className="w-full bg-slate-700 border-2 border-slate-600 text-white text-base h-12 font-semibold">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-700">
                  <SelectItem value="5" className="text-base">5</SelectItem>
                  <SelectItem value="4" className="text-base">4</SelectItem>
                  <SelectItem value="3" className="text-base">3</SelectItem>
                  <SelectItem value="No Stacks" className="text-base">No Stacks</SelectItem>
                  <SelectItem value="5|2" className="text-base">5|2</SelectItem>
                  <SelectItem value="4|2" className="text-base">4|2</SelectItem>
                  <SelectItem value="4|2|2" className="text-base">4|2|2</SelectItem>
                  <SelectItem value="3|3|2" className="text-base">3|3|2</SelectItem>
                  <SelectItem value="3|2|2" className="text-base">3|2|2</SelectItem>
                  <SelectItem value="2|2|2" className="text-base">2|2|2</SelectItem>
                  <SelectItem value="5|3" className="text-base">5|3</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-base text-white block mb-3 font-semibold">Lineups per Combination</Label>
              <input
                type="number"
                min="1"
                max="50"
                value={defaultLineupsPerCombo}
                onChange={(e) => setDefaultLineupsPerCombo(parseInt(e.target.value) || 5)}
                className="w-full bg-slate-700 border-2 border-slate-600 rounded-lg px-4 py-3 text-white text-base font-semibold"
                placeholder="e.g., 5"
              />
            </div>

            <Button
              onClick={generateCombinations}
              className="w-full bg-gradient-to-r from-orange-600 to-orange-500 hover:from-orange-700 hover:to-orange-600 text-white text-lg h-14 font-bold rounded-lg shadow-lg transition-none"
            >
              Generate Team Combinations
            </Button>
          </div>
        </div>
      </div>

      {/* Combinations Table - Enterprise Style */}
      {combinations.length > 0 && (
        <div className="border-2 border-slate-700 bg-slate-800/50 p-6 rounded-xl">
          <div className="flex items-center justify-between mb-6 border-b border-slate-700 pb-4">
            <h3 className="text-2xl font-bold text-white">
              Generated Combinations
            </h3>
            <div className="text-lg text-slate-200 font-medium">
              Total Lineups: <span className="font-bold text-orange-400 text-xl">{totalLineups}</span>
            </div>
          </div>

          <div className="overflow-auto max-h-96 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
            <table className="w-full text-base">
              <thead>
                <tr className="border-b-2 border-slate-700">
                  <th className="text-left py-4 px-5 text-slate-300 font-bold">Select</th>
                  <th className="text-left py-4 px-5 text-slate-300 font-bold">Team Combination</th>
                  <th className="text-left py-4 px-5 text-slate-300 font-bold">Lineups per Combo</th>
                  <th className="text-left py-4 px-5 text-slate-300 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {combinations.map(combo => (
                  <tr key={combo.id} className="border-b border-slate-700/50">
                    <td className="py-4 px-5">
                      <Checkbox
                        checked={combo.enabled}
                        onCheckedChange={() => toggleCombination(combo.id)}
                        className="h-6 w-6 border-2 border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
                      />
                    </td>
                    <td className="py-4 px-5 text-white font-semibold text-base">{combo.display}</td>
                    <td className="py-4 px-5">
                      <input
                        type="number"
                        min="1"
                        max="100"
                        value={combo.lineupsPerCombo}
                        onChange={(e) => updateLineupsPerCombo(combo.id, parseInt(e.target.value) || 5)}
                        className="w-24 h-11 bg-slate-700 border-2 border-slate-600 rounded-lg px-4 py-2 text-white text-base font-semibold"
                      />
                    </td>
                    <td className="py-4 px-5">
                      <Button
                        variant="outline"
                        className="px-5 py-2 h-11 text-base border-2 border-slate-600 bg-slate-700 text-white font-semibold transition-none"
                        onClick={() => toggleCombination(combo.id)}
                      >
                        {combo.enabled ? 'Disable' : 'Enable'}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-6 flex justify-end">
            <Button
              className="bg-gradient-to-r from-green-600 to-green-500 hover:from-green-700 hover:to-green-600 text-white text-lg h-14 px-8 font-bold rounded-lg shadow-lg transition-none"
              disabled={totalLineups === 0}
            >
              Generate All Combination Lineups ({totalLineups})
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

// Advanced Quant Tab Component
interface AdvancedQuantTabProps {
  settings: AdvancedQuantSettings;
  onSettingsChange: (settings: AdvancedQuantSettings) => void;
}

const AdvancedQuantTab: React.FC<AdvancedQuantTabProps> = ({ settings, onSettingsChange }) => {
  const updateSetting = <K extends keyof AdvancedQuantSettings>(key: K, value: AdvancedQuantSettings[K]) => {
    onSettingsChange({ ...settings, [key]: value });
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
              checked={settings.enabled}
              onCheckedChange={(checked: boolean) => updateSetting('enabled', checked as boolean)}
              className="border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
              style={{ 
                accentColor: '#1f2937'
              }}
            />
            <div>
              <Label className="text-white font-semibold text-base cursor-pointer" onClick={() => updateSetting('enabled', !settings.enabled)}>
                Enable Advanced Quantitative Optimization
              </Label>
              <p className="text-xs text-slate-400 mt-1">
                Master switch for financial-grade risk modeling
              </p>
            </div>
          </div>
          {settings.enabled && (
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
            value={settings.strategy} 
            onValueChange={(v: string) => updateSetting('strategy', v)}
            disabled={!settings.enabled}
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
            {settings.strategy === 'combined' && 'Combines multiple optimization techniques for balanced approach'}
            {settings.strategy === 'kelly_criterion' && 'Pure Kelly optimal betting strategy - maximizes long-term growth'}
            {settings.strategy === 'risk_parity' && 'Equal risk contribution - balances volatility across lineup'}
            {settings.strategy === 'mean_variance' && 'Classic Markowitz optimization - maximizes return for given risk'}
            {settings.strategy === 'equal_weight' && 'Simple equal allocation - baseline strategy'}
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
              <span className="text-cyan-400 font-medium text-sm">{settings.riskTolerance.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={settings.riskTolerance}
              onChange={(e) => updateSetting('riskTolerance', parseFloat(e.target.value))}
              disabled={!settings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">Range: 0.1 - 2.0 (1.0 = neutral, &lt;1.0 = conservative, &gt;1.0 = aggressive)</p>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">VaR Confidence Level</Label>
              <span className="text-cyan-400 font-medium text-sm">{(settings.varConfidence * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.90"
              max="0.99"
              step="0.01"
              value={settings.varConfidence}
              onChange={(e) => updateSetting('varConfidence', parseFloat(e.target.value))}
              disabled={!settings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">Range: 90% - 99% (probability level for Value-at-Risk)</p>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">Target Volatility</Label>
              <span className="text-cyan-400 font-medium text-sm">{(settings.targetVolatility * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.50"
              step="0.01"
              value={settings.targetVolatility}
              onChange={(e) => updateSetting('targetVolatility', parseFloat(e.target.value))}
              disabled={!settings.enabled}
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
              value={settings.monteCarloSims}
              onChange={(e) => updateSetting('monteCarloSims', parseInt(e.target.value) || 10000)}
              disabled={!settings.enabled}
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
              value={settings.timeHorizon}
              onChange={(e) => updateSetting('timeHorizon', parseInt(e.target.value) || 1)}
              disabled={!settings.enabled}
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
              value={settings.garchP}
              onChange={(e) => updateSetting('garchP', parseInt(e.target.value) || 1)}
              disabled={!settings.enabled}
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
              value={settings.garchQ}
              onChange={(e) => updateSetting('garchQ', parseInt(e.target.value) || 1)}
              disabled={!settings.enabled}
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
              value={settings.lookbackPeriod}
              onChange={(e) => updateSetting('lookbackPeriod', parseInt(e.target.value) || 100)}
              disabled={!settings.enabled}
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
              value={settings.copulaFamily} 
              onValueChange={(v: string) => updateSetting('copulaFamily', v)}
              disabled={!settings.enabled}
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
              {settings.copulaFamily === 'gaussian' && 'Normal distribution - symmetric, general use'}
              {settings.copulaFamily === 't' && 'Student\'s t - heavy tails, extreme events'}
              {settings.copulaFamily === 'clayton' && 'Lower tail dependence - fail together'}
              {settings.copulaFamily === 'frank' && 'Weak tail dependence - more independent'}
              {settings.copulaFamily === 'gumbel' && 'Upper tail dependence - succeed together'}
            </p>
          </div>
          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">Dependency Threshold</Label>
              <span className="text-cyan-400 font-medium text-sm">{(settings.dependencyThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={settings.dependencyThreshold}
              onChange={(e) => updateSetting('dependencyThreshold', parseFloat(e.target.value))}
              disabled={!settings.enabled}
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
              <span className="text-cyan-400 font-medium text-sm">{(settings.maxKellyFraction * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={settings.maxKellyFraction}
              onChange={(e) => updateSetting('maxKellyFraction', parseFloat(e.target.value))}
              disabled={!settings.enabled}
              className="w-full"
            />
            <p className="text-xs text-slate-400 mt-1">10% - 100% of bankroll (25% = quarter Kelly, recommended)</p>
          </div>
          <div>
            <div className="flex justify-between mb-2">
              <Label className="text-white text-sm">Expected Win Rate</Label>
              <span className="text-cyan-400 font-medium text-sm">{(settings.expectedWinRate * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={settings.expectedWinRate}
              onChange={(e) => updateSetting('expectedWinRate', parseFloat(e.target.value))}
              disabled={!settings.enabled}
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
          <div className={`flex items-center gap-2 ${settings.enabled ? 'text-white' : 'text-slate-500'}`}>
            {settings.enabled ? '✓' : '○'} 
            <span className="font-medium">
              Advanced quantitative optimization {settings.enabled ? 'ENABLED' : 'DISABLED'}
            </span>
          </div>

          {settings.enabled && (
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

interface DFSOptimizerProps {
  sport: string;
}

const DFSOptimizer = React.memo(({ sport = 'NFL' }: DFSOptimizerProps) => {
  // Use sport prop directly from Dashboard (controlled by header tabs)
  const currentSport = sport as Sport;
  const sportConfig = SPORT_CONFIGS[currentSport];
  
  const [activeTab, setActiveTab] = useState('team-combos'); // Set to team-combos for testing
  const [playerData, setPlayerData] = useState<Player[]>([
    { id: '1', name: 'Josh Allen', team: 'BUF', position: 'QB', salary: 8500, projectedPoints: 25.5, minExp: 0, maxExp: 100, selected: false },
    { id: '2', name: 'Stefon Diggs', team: 'BUF', position: 'WR', salary: 7500, projectedPoints: 18.2, minExp: 0, maxExp: 100, selected: false },
    { id: '3', name: 'Travis Kelce', team: 'KC', position: 'TE', salary: 7000, projectedPoints: 15.8, minExp: 0, maxExp: 100, selected: false },
    { id: '4', name: 'Tyreek Hill', team: 'MIA', position: 'WR', salary: 8000, projectedPoints: 20.1, minExp: 0, maxExp: 100, selected: false },
    { id: '5', name: 'Christian McCaffrey', team: 'SF', position: 'RB', salary: 9000, projectedPoints: 22.3, minExp: 0, maxExp: 100, selected: false },
    { id: '6', name: 'Lamar Jackson', team: 'BAL', position: 'QB', salary: 8200, projectedPoints: 24.1, minExp: 0, maxExp: 100, selected: false },
    { id: '7', name: 'Cooper Kupp', team: 'LAR', position: 'WR', salary: 7800, projectedPoints: 19.5, minExp: 0, maxExp: 100, selected: false },
    { id: '8', name: 'Derrick Henry', team: 'TEN', position: 'RB', salary: 7200, projectedPoints: 18.7, minExp: 0, maxExp: 100, selected: false },
    { id: '9', name: 'Davante Adams', team: 'LV', position: 'WR', salary: 7600, projectedPoints: 17.9, minExp: 0, maxExp: 100, selected: false },
    { id: '10', name: 'Patrick Mahomes', team: 'KC', position: 'QB', salary: 8800, projectedPoints: 26.2, minExp: 0, maxExp: 100, selected: false },
    { id: '11', name: 'Saquon Barkley', team: 'NYG', position: 'RB', salary: 6800, projectedPoints: 16.8, minExp: 0, maxExp: 100, selected: false },
    { id: '12', name: 'Mike Evans', team: 'TB', position: 'WR', salary: 7100, projectedPoints: 16.5, minExp: 0, maxExp: 100, selected: false },
    { id: '13', name: 'Joe Burrow', team: 'CIN', position: 'QB', salary: 8100, projectedPoints: 23.8, minExp: 0, maxExp: 100, selected: false },
    { id: '14', name: 'Ja\'Marr Chase', team: 'CIN', position: 'WR', salary: 7900, projectedPoints: 19.8, minExp: 0, maxExp: 100, selected: false },
    { id: '15', name: 'Nick Chubb', team: 'CLE', position: 'RB', salary: 7400, projectedPoints: 17.9, minExp: 0, maxExp: 100, selected: false },
    { id: '16', name: 'Amari Cooper', team: 'CLE', position: 'WR', salary: 6900, projectedPoints: 16.2, minExp: 0, maxExp: 100, selected: false },
    { id: '17', name: 'Dak Prescott', team: 'DAL', position: 'QB', salary: 8000, projectedPoints: 23.5, minExp: 0, maxExp: 100, selected: false },
    { id: '18', name: 'CeeDee Lamb', team: 'DAL', position: 'WR', salary: 8200, projectedPoints: 20.5, minExp: 0, maxExp: 100, selected: false },
    { id: '19', name: 'Tony Pollard', team: 'DAL', position: 'RB', salary: 6500, projectedPoints: 15.8, minExp: 0, maxExp: 100, selected: false },
    { id: '20', name: 'Russell Wilson', team: 'DEN', position: 'QB', salary: 7200, projectedPoints: 21.2, minExp: 0, maxExp: 100, selected: false },
    { id: '21', name: 'Courtland Sutton', team: 'DEN', position: 'WR', salary: 6300, projectedPoints: 14.9, minExp: 0, maxExp: 100, selected: false },
    { id: '22', name: 'Javonte Williams', team: 'DEN', position: 'RB', salary: 6100, projectedPoints: 14.2, minExp: 0, maxExp: 100, selected: false },
    { id: '23', name: 'Jared Goff', team: 'DET', position: 'QB', salary: 6800, projectedPoints: 20.8, minExp: 0, maxExp: 100, selected: false },
    { id: '24', name: 'Amon-Ra St. Brown', team: 'DET', position: 'WR', salary: 7700, projectedPoints: 18.6, minExp: 0, maxExp: 100, selected: false },
    { id: '25', name: 'D\'Andre Swift', team: 'DET', position: 'RB', salary: 6400, projectedPoints: 15.1, minExp: 0, maxExp: 100, selected: false },
  ]);
  const [selectedPlayers, setSelectedPlayers] = useState<string[]>([]);
  const [teamSelections, setTeamSelections] = useState<Record<number | 'all', string[]>>({
    all: [],
    2: [],
    3: [],
    4: [],
    5: [],
  });
  // Initialize stack settings based on sport
  const initializeStackSettings = (sport: Sport): StackType[] => {
    const config = SPORT_CONFIGS[sport];
    return config.stackTypes.map((stackType, index) => ({
      id: `stack-${index}`,
      label: stackType,
      minExp: 0,
      maxExp: 100,
      enabled: false
    }));
  };

  const [stackSettings, setStackSettings] = useState<StackType[]>(initializeStackSettings(currentSport));
  
  // Advanced Quant Settings
  const [advancedQuantSettings, setAdvancedQuantSettings] = useState<AdvancedQuantSettings>({
    enabled: false,
    strategy: 'combined',
    riskTolerance: 1.0,
    varConfidence: 0.95,
    targetVolatility: 0.15,
    monteCarloSims: 10000,
    timeHorizon: 1,
    garchP: 1,
    garchQ: 1,
    lookbackPeriod: 100,
    copulaFamily: 'gaussian',
    dependencyThreshold: 0.3,
    maxKellyFraction: 0.25,
    expectedWinRate: 0.2,
  });
  
  // Optimization Settings
  const [numLineups, setNumLineups] = useState(100);
  const [minUnique, setMinUnique] = useState(3);
  const [minSalary, setMinSalary] = useState(sportConfig.defaultMinSalary);
  const [disableKelly, setDisableKelly] = useState(false);
  
  // Handle sport change from Dashboard header
  useEffect(() => {
    const newConfig = SPORT_CONFIGS[currentSport];
    setMinSalary(newConfig.defaultMinSalary);
    setStackSettings(initializeStackSettings(currentSport));
    // Clear player data when switching sports
    setPlayerData([]);
    setSelectedPlayers([]);
  }, [currentSport]);
  
  // Sorting
  const [sortMethod, setSortMethod] = useState('points');
  
  // Risk Management
  const [bankroll, setBankroll] = useState(1000);
  const [riskProfile, setRiskProfile] = useState('medium');
  const [enableRiskMgmt, setEnableRiskMgmt] = useState(false);
  
  // Generated Teams
  const [generatedTeams, setGeneratedTeams] = useState<any[]>([
    {
      id: '1',
      name: 'BUF + KC Stack',
      players: ['QB: Allen', 'WR: Diggs', 'TE: Kelce'],
      type: 'stack'
    },
    {
      id: '2', 
      name: 'MIA + SF Stack',
      players: ['WR: Hill', 'RB: McCaffrey'],
      type: 'stack'
    },
    {
      id: '3',
      name: 'LAR + TEN Stack', 
      players: ['WR: Kupp', 'RB: Henry'],
      type: 'stack'
    },
    {
      id: '4',
      name: 'CIN + CLE Stack',
      players: ['QB: Burrow', 'WR: Chase', 'RB: Chubb'],
      type: 'stack'
    },
    {
      id: '5',
      name: 'DAL + DEN Stack',
      players: ['QB: Prescott', 'WR: Lamb', 'QB: Wilson'],
      type: 'stack'
    }
  ]);
  
  // Results - starts empty, filled by actual optimization
  const [results, setResults] = useState<Array<{
    id: string;
    players: Array<{ name: string; position: string; team: string; salary: number }>;
    points: number;
    salary: number;
  }>>([]);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isRunningCombinations, setIsRunningCombinations] = useState(false);
  const [dkEntriesLoaded, setDkEntriesLoaded] = useState(false);
  
  // Resizable panels
  const [controlPanelWidth, setControlPanelWidth] = useState(426); // 1/3 wider than before
  const [isResizing, setIsResizing] = useState(false);
  const [isControlPanelCollapsed, setIsControlPanelCollapsed] = useState(false);

  const tabs = [
    { id: 'players', label: 'Players', icon: Users },
    { id: 'team-stacks', label: 'Team Stacks', icon: Link2 },
    { id: 'stack-exposure', label: 'Stack Exposure', icon: BarChart3 },
    { id: 'team-combos', label: 'Team Combinations', icon: Target },
    { id: 'advanced-quant', label: 'Advanced Quant', icon: Cpu },
    { id: 'my-entries', label: 'My Entries', icon: Star },
  ];

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      // Upload file to backend
      const formData = new FormData();
      formData.append('playersFile', file);

      const response = await fetch('/api/upload-players', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        console.log(`✅ Uploaded ${result.playersCount} players to backend`);
        
        // Fetch players from backend
        const playersResponse = await fetch('/api/players');
        const playersData = await playersResponse.json();
        
        // Transform backend player format to frontend format
        const transformedPlayers: Player[] = playersData.players.map((p: any) => ({
          id: p.id,
          name: p.name,
          team: p.team,
          position: p.position,
          salary: p.salary,
          projectedPoints: p.projection || p.projectedPoints || 0,
          minExp: p.minExposure || 0,
          maxExp: p.maxExposure || 100,
          selected: p.selected || false,
        }));

        setPlayerData(transformedPlayers);
        setSelectedPlayers([]);
        alert(`✅ Loaded ${transformedPlayers.length} players successfully!`);
      } else {
        alert(`❌ Upload failed: ${result.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('❌ Upload failed. Please check the file format and try again.');
    }
  };

  const handleRunOptimization = async () => {
    // Validate inputs
    if (playerData.length === 0) {
      alert('❌ Please load player data first');
      return;
    }

    const selectedCount = selectedPlayers.length;
    const minRequired = currentSport === 'NFL' ? 9 : 10;
    
    if (selectedCount < minRequired) {
      alert(`❌ Please select at least ${minRequired} players for ${currentSport}`);
      return;
    }

    setIsOptimizing(true);
    
    try {
      // Mark selected players in backend
      await Promise.all(
        playerData.map(async (player) => {
          if (selectedPlayers.includes(player.id)) {
            await fetch(`/api/players/${player.id}`, {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ selected: true }),
            });
          }
        })
      );

      // Prepare stack settings
      const enabledStacks = stackSettings.filter(s => s.enabled);
      const stackTypes = enabledStacks.map(s => s.label);

      // Run optimization
      console.log('🚀 Starting optimization...');
      const response = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sport: currentSport,
          numLineups,
          minSalary,
          maxSalary: 50000,
          stackSettings: {
            enabled: enabledStacks.length > 0,
            types: stackTypes,
            teams: Array.from(new Set(playerData.filter(p => selectedPlayers.includes(p.id)).map(p => p.team))),
          },
          uniquePlayers: minUnique,
          maxExposure: 40,
          contestMode: 'gpp',
          riskTolerance: 'medium',
        }),
      });

      const result = await response.json();

      if (result.success) {
        console.log('✅ Optimization complete:', result.summary);
        
        // Transform lineups for display
        const transformedResults = result.lineups.map((lineup: any) => ({
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

        setResults(transformedResults);
        
        // Switch to My Entries tab to show results
        setActiveTab('my-entries');
        
        alert(`✅ Generated ${transformedResults.length} optimal lineups!\nAvg Projection: ${result.summary.avgProjection.toFixed(1)} pts`);
      } else {
        alert(`❌ Optimization failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Optimization error:', error);
      alert('❌ Optimization failed. Please try again.');
    } finally {
      setIsOptimizing(false);
    }
  };

  const handleRunCombinations = () => {
    if (generatedTeams.length === 0) {
      alert('No teams available to run combinations. Generate teams first.');
      return;
    }
    
    setIsRunningCombinations(true);
    console.log('Running combinations for teams:', generatedTeams);
    
    // TODO: Call backend API to run combinations
    setTimeout(() => {
      setIsRunningCombinations(false);
      console.log('Combinations complete');
      // TODO: Update results with combination lineups
    }, 2000);
  };

  // Function to update generated teams (called by backend)
  const updateGeneratedTeams = (teams: any[]) => {
    setGeneratedTeams(teams);
  };

  // Function to run combinations (called by backend)
  const runCombinations = () => {
    handleRunCombinations();
  };

  // Expose functions globally for backend access
  React.useEffect(() => {
    (window as any).updateGeneratedTeams = updateGeneratedTeams;
    (window as any).runCombinations = runCombinations;
    return () => {
      delete (window as any).updateGeneratedTeams;
      delete (window as any).runCombinations;
    };
  }, []);

  // Resize handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsResizing(true);
    e.preventDefault();
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isResizing) return;
    
    const newWidth = window.innerWidth - e.clientX;
    const minWidth = 200;
    const maxWidth = 400;
    
    if (newWidth >= minWidth && newWidth <= maxWidth) {
      setControlPanelWidth(newWidth);
    }
  };

  const handleMouseUp = () => {
    setIsResizing(false);
  };

  // Add event listeners
  React.useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  return (
    <div className="h-full w-full flex p-2">
      {/* MAIN CONTENT AREA - Desktop-Style Tabs (matches PyQt x.py) */}
      <div 
        className="bg-slate-900 border border-slate-700 overflow-hidden flex flex-col"
        style={{ width: `calc(100% - ${controlPanelWidth}px - 8px)` }}
      >
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col h-full">
            {/* Tab Headers - Desktop Style */}
            <TabsList className="bg-slate-800 border-b border-slate-700 w-full rounded-none h-auto flex flex-nowrap">
              {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                  <TabsTrigger
                    key={tab.id}
                    value={tab.id}
                    className="data-[state=active]:bg-slate-700 data-[state=active]:text-white data-[state=active]:border-b-2 data-[state=active]:border-blue-500 flex items-center gap-1.5 px-3 py-2 text-sm whitespace-nowrap flex-shrink-0 text-white hover:text-white hover:bg-slate-700/50 transition-colors"
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span className="font-normal">{tab.label}</span>
                  </TabsTrigger>
                  );
                })}
            </TabsList>

            {/* Tab Content - Compact Padding with Scrolling */}
            <div className="flex-1 overflow-auto p-3 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
              {/* Players Tab */}
              <TabsContent value="players" className="mt-0 h-full overflow-auto">
                <PlayersTab
                  playerData={playerData}
                  selectedPlayers={selectedPlayers}
                  sport={currentSport}
                  onPlayersChange={setSelectedPlayers}
                  onPlayerDataChange={setPlayerData}
                />
              </TabsContent>

              {/* Team Stacks Tab */}
              <TabsContent value="team-stacks" className="mt-0 h-full overflow-auto">
                <TeamStacksTab
                  playerData={playerData}
                  teamSelections={teamSelections}
                  onTeamSelectionsChange={setTeamSelections}
                />
              </TabsContent>

              {/* Stack Exposure Tab */}
              <TabsContent value="stack-exposure" className="mt-0 h-full overflow-auto">
                <StackExposureTab
                  stackSettings={stackSettings}
                  sport={currentSport}
                  onStackSettingsChange={setStackSettings}
                />
              </TabsContent>

              {/* Team Combinations Tab */}
              <TabsContent value="team-combos" className="mt-0 h-full overflow-auto">
                <TeamCombinationsTab playerData={playerData} />
              </TabsContent>

              {/* Advanced Quant Tab */}
              <TabsContent value="advanced-quant" className="mt-0 h-full overflow-auto">
                <AdvancedQuantTab
                  settings={advancedQuantSettings}
                  onSettingsChange={setAdvancedQuantSettings}
                />
              </TabsContent>

              {/* My Entries Tab */}
              <TabsContent value="my-entries" className="mt-0 h-full overflow-auto">
                <MyEntriesTab results={results} sport={currentSport} />
              </TabsContent>
            </div>
          </Tabs>
      </div>

      {/* RESIZABLE DIVIDER */}
      <div
        className={`w-1 bg-slate-600 hover:bg-slate-500 cursor-col-resize flex-shrink-0 transition-colors ${
          isResizing ? 'bg-blue-500' : ''
        }`}
        onMouseDown={handleMouseDown}
      >
        <div className="w-full h-full flex items-center justify-center">
          <div className="w-0.5 h-8 bg-slate-400 rounded-full"></div>
        </div>
      </div>

      {/* RIGHT SIDEBAR - Control Panel (Desktop Style) */}
      <div 
        className="flex-shrink-0 bg-slate-900 border border-slate-700 overflow-hidden"
        style={{ width: `${controlPanelWidth}px` }}
      >
          <div className="p-1.5 space-y-1 overflow-auto h-full">
            {/* Header - Collapsible */}
            <div 
              className="border-b border-cyan-500/30 pb-1 mb-2 cursor-pointer hover:bg-slate-800/50 rounded px-2 py-1.5 transition-colors group"
              onClick={() => setIsControlPanelCollapsed(!isControlPanelCollapsed)}
            >
              <h3 className="text-[13px] font-bold text-cyan-400 flex items-center justify-between uppercase tracking-wide">
                <div className="flex items-center gap-2">
                  <Settings className="w-4 h-4 text-cyan-400" />
                  <span>Control Panel</span>
                </div>
                {isControlPanelCollapsed ? (
                  <ChevronDown className="w-4 h-4 text-cyan-400 group-hover:text-cyan-300 transition-colors" />
                ) : (
                  <ChevronUp className="w-4 h-4 text-cyan-400 group-hover:text-cyan-300 transition-colors" />
                )}
              </h3>
            </div>

            {/* Collapsible Content */}
            {!isControlPanelCollapsed && (
            <div className="space-y-3 px-2">
            {/* Sport Info - Display Only */}
            <div className="border-2 rounded-lg p-3" style={{ borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.15)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2 mb-2" style={{ color: '#f59e0b', borderColor: '#f59e0b' }}>Current Sport</h4>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">
                  {currentSport === 'NFL' ? '🏈 NFL' : currentSport === 'NBA' ? '🏀 NBA' : '⚾ MLB'}
                </div>
                <div className="text-[10px] text-slate-300 mt-1">
                  {sportConfig.salaryCapDescription}
                </div>
                <div className="text-[9px] text-cyan-400 mt-1 italic">
                  Change sport using header tabs
                </div>
              </div>
            </div>

            {/* File Operations */}
            <div className="border-2 rounded-lg p-3" style={{ borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.15)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2 mb-2" style={{ color: '#3b82f6', borderColor: '#3b82f6' }}>File Operations</h4>
              <div className="space-y-2">
                <label htmlFor="csv-upload" className="block">
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full border-blue-500/40 bg-slate-800 hover:bg-slate-700 text-white text-[12px] h-9 justify-center px-3 font-semibold shadow-sm"
                    onClick={() => document.getElementById('csv-upload')?.click()}
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Load CSV
                  </Button>
                </label>
                <input
                  id="csv-upload"
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-blue-500/40 bg-slate-800 hover:bg-slate-700 text-white text-[12px] h-9 justify-center px-3 font-semibold shadow-sm"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Load Predictions
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-blue-500/40 bg-slate-800 hover:bg-slate-700 text-white text-[12px] h-9 justify-center px-3 font-semibold shadow-sm"
                  onClick={() => setDkEntriesLoaded(true)}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Load Entries
                </Button>
              </div>
            </div>

            {/* Optimization Settings */}
            <div className="border-2 rounded-lg p-3 space-y-2" style={{ borderColor: '#22d3ee', backgroundColor: 'rgba(34, 211, 238, 0.15)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2" style={{ color: '#22d3ee', borderColor: '#22d3ee' }}>Optimization</h4>
              <div className="space-y-2">
                <div>
                  <Label className="text-[12px] text-white block mb-1.5 font-semibold">Lineups</Label>
                  <input
                    type="number"
                    min="1"
                    max="500"
                    value={numLineups}
                    onChange={(e) => setNumLineups(parseInt(e.target.value) || 100)}
                    className="w-full bg-slate-700 border-2 border-cyan-500/30 rounded px-2 py-1.5 text-slate-200 text-[13px] font-bold"
                  />
                </div>
                
                <div>
                  <Label className="text-[12px] text-white block mb-1.5 font-semibold">Min Unique</Label>
                  <input
                    type="number"
                    min="0"
                    max="10"
                    value={minUnique}
                    onChange={(e) => setMinUnique(parseInt(e.target.value) || 3)}
                    className="w-full bg-slate-700 border-2 border-cyan-500/30 rounded px-2 py-1.5 text-slate-200 text-[13px] font-bold"
                  />
                </div>
                
                <div className="flex items-center space-x-2 pt-1">
                  <Checkbox
                    id="disable-kelly"
                    checked={disableKelly}
                    onCheckedChange={(checked: boolean) => setDisableKelly(checked)}
                    className="h-4 w-4 border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
                    style={{ 
                      accentColor: '#1f2937'
                    }}
                  />
                  <Label htmlFor="disable-kelly" className="text-[12px] text-white cursor-pointer font-semibold">
                    Disable Kelly
                  </Label>
                </div>
              </div>
            </div>

            {/* Salary Constraints */}
            <div className="border-2 rounded-lg p-3 space-y-2" style={{ borderColor: '#4ade80', backgroundColor: 'rgba(74, 222, 128, 0.15)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2" style={{ color: '#4ade80', borderColor: '#4ade80' }}>Salary</h4>
              <div className="space-y-2">
                <div>
                  <Label className="text-[12px] text-white block mb-1.5 font-semibold">Min ($)</Label>
                  <input
                    type="number"
                    min="0"
                    max="50000"
                    step="1000"
                    value={minSalary}
                    onChange={(e) => setMinSalary(parseInt(e.target.value) || 45000)}
                    className="w-full bg-slate-700 border-2 border-green-400/50 rounded px-2 py-1.5 text-white text-[13px] font-bold"
                  />
                </div>
                
                <div>
                  <Label className="text-[12px] text-white block mb-1.5 font-semibold">Max ($)</Label>
                  <input
                    type="number"
                    value={50000}
                    disabled
                    className="w-full bg-slate-600/50 border-2 border-slate-600 rounded px-2 py-1.5 text-slate-200 text-[13px] cursor-not-allowed font-bold"
                  />
                  <span className="text-[10px] text-slate-300 mt-1 block">Fixed by DK</span>
                </div>
              </div>
            </div>

            {/* Sorting */}
            <div className="border-2 rounded-lg p-3 space-y-2" style={{ borderColor: '#a78bfa', backgroundColor: 'rgba(167, 139, 250, 0.15)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2" style={{ color: '#a78bfa', borderColor: '#a78bfa' }}>Sorting</h4>
              <div>
                <Select value={sortMethod} onValueChange={setSortMethod}>
                  <SelectTrigger className="w-full bg-slate-700 border-purple-500/30 text-white text-[12px] h-9 font-semibold">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-900 border-slate-700">
                    <SelectItem value="points" className="text-white text-[12px]">Points ↓</SelectItem>
                    <SelectItem value="value" className="text-white text-[12px]">Value ↓</SelectItem>
                    <SelectItem value="salary" className="text-white text-[12px]">Salary ↓</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Risk Management - Simple Clickable Container */}
            <div 
              className="border-2 rounded-lg p-3 cursor-pointer transition-colors"
              style={enableRiskMgmt 
                ? { borderColor: '#fb923c', backgroundColor: 'rgba(251, 146, 60, 0.15)' }
                : { borderColor: '#475569', backgroundColor: 'rgba(71, 85, 105, 0.2)' }
              }
              onClick={() => setEnableRiskMgmt(!enableRiskMgmt)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="enable-risk"
                    checked={enableRiskMgmt}
                    onCheckedChange={(checked: boolean) => setEnableRiskMgmt(checked)}
                    className="h-4 w-4 border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-orange-400"
                    onClick={(e: React.MouseEvent) => e.stopPropagation()}
                  />
                  <Label className="text-[12px] text-white cursor-pointer font-semibold">
                    Enable Risk Management
                  </Label>
                </div>
                {enableRiskMgmt && (
                  <span className="text-[10px] text-orange-400 font-semibold">Active</span>
                )}
              </div>
            </div>

            {/* Actions - Yellow Color Code */}
            <div className="border-2 rounded-lg p-3 space-y-2 mt-1" style={{ borderColor: '#facc15', backgroundColor: 'rgba(250, 204, 21, 0.15)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2" style={{ color: '#facc15', borderColor: '#facc15' }}>Actions</h4>
              <div className="space-y-2">
                <Button
                  className="w-full bg-blue-600 text-white h-9 text-[12px] font-semibold px-2 transition-none"
                  onClick={handleRunOptimization}
                  disabled={isOptimizing || playerData.length === 0}
                >
                  <Play className="w-4 h-4 mr-2" />
                  {isOptimizing ? 'Optimizing...' : 'Optimize'}
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-yellow-400/40 bg-slate-800 hover:bg-slate-700 text-white text-[12px] h-9 justify-center px-2 transition-none"
                  disabled={results.length === 0}
                  onClick={async () => {
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
                      alert('❌ Export failed');
                    }
                  }}
                >
                  <Save className="w-4 h-4 mr-2" />
                  Save CSV
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-yellow-400/40 bg-slate-800 hover:bg-slate-700 text-white text-[12px] h-9 justify-center px-2 transition-none"
                  disabled={!dkEntriesLoaded || results.length === 0}
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Fill Entries
                </Button>
              </div>
            </div>

            {/* Generated Teams - Light Blue Color Code */}
            <div className="border-2 rounded-lg p-3 space-y-2" style={{ borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.15)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2" style={{ color: '#38bdf8', borderColor: '#38bdf8' }}>Generated Teams</h4>
              
              {/* Run Combinations Button */}
              <div>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-sky-400/40 bg-slate-800 hover:bg-slate-700 text-white text-[12px] h-9 justify-center px-2 transition-none"
                  onClick={handleRunCombinations}
                  disabled={generatedTeams.length === 0 || isOptimizing || isRunningCombinations}
                >
                  <Play className="w-4 h-4 mr-2" />
                  {isRunningCombinations ? 'Running...' : 'Run Combinations'}
                </Button>
              </div>

              {/* Teams List */}
              <div className="bg-slate-800/40 border border-slate-600/50 rounded p-2 max-h-28 overflow-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
                <div className="space-y-1">
                  {generatedTeams.map((team) => (
                    <div key={team.id} className="text-[10px] text-white bg-slate-700/50 rounded px-2 py-1.5">
                      <div className="font-semibold text-white">{team.name}</div>
                      <div className="text-slate-400">{team.players.join(', ')}</div>
                    </div>
                  ))}
                </div>
                <div className="mt-1 pt-1 border-t border-slate-600/50">
                  <div className="text-[9px] text-slate-400 text-center">
                    {generatedTeams.length} teams generated
                  </div>
                </div>
              </div>
            </div>

            {/* Favorites - White Color Code */}
            <div className="border-2 rounded-lg p-3 space-y-2" style={{ borderColor: '#cbd5e1', backgroundColor: 'rgba(203, 213, 225, 0.12)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2" style={{ color: '#e2e8f0', borderColor: '#cbd5e1' }}>Favorites</h4>
              <div className="space-y-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-slate-400/40 bg-slate-800 hover:bg-slate-700 text-white text-[12px] h-9 justify-center px-2 transition-none"
                  disabled={results.length === 0}
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Add Favorite
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-slate-400/40 bg-slate-800 hover:bg-slate-700 text-white text-[12px] h-9 justify-center px-2 transition-none"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
              </div>
            </div>

            {/* Results Summary - White Color Code */}
            <div className="border-2 rounded-lg p-3 space-y-2 mt-1" style={{ borderColor: '#cbd5e1', backgroundColor: 'rgba(203, 213, 225, 0.12)' }}>
              <h4 className="text-[13px] font-bold uppercase tracking-wide border-b pb-2" style={{ color: '#e2e8f0', borderColor: '#cbd5e1' }}>Results</h4>
              <div className="bg-slate-800/40 border border-slate-600/50 rounded p-2">
                {results.length > 0 ? (
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between text-white">
                      <span>Lineups Generated:</span>
                      <span className="font-semibold text-cyan-400">{results.length}</span>
                    </div>
                    <div className="flex justify-between text-white">
                      <span>Avg Points:</span>
                      <span className="font-semibold">125.3</span>
                    </div>
                    <div className="flex justify-between text-white">
                      <span>Avg Salary:</span>
                      <span className="font-semibold">$48,450</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-400 text-xs text-center py-3">
                    No results yet
                  </p>
                )}
              </div>
            </div>

            {/* Status Bar */}
            <div className="border-t border-cyan-500/20 pt-3">
              <div className="text-xs text-white space-y-1">
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span className="text-cyan-400 font-medium">
                    {isOptimizing ? 'Optimizing...' : 
                     isRunningCombinations ? 'Running Combinations...' : 'Ready'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Players Loaded:</span>
                  <span className="font-medium">{playerData.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Selected:</span>
                  <span className="font-medium">{selectedPlayers.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Lineups:</span>
                  <span className="font-medium">{results.length}</span>
                </div>
              </div>
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
