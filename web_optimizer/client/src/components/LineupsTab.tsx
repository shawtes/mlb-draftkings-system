import React, { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { 
  Trophy, 
  Download, 
  Copy, 
  Star, 
  Users, 
  Target, 
  BarChart3, 
  DollarSign,
  TrendingUp,
  Filter,
  Search,
  ChevronDown,
  ChevronUp,
  Eye,
  EyeOff
} from 'lucide-react';

// Lineup data interface
interface LineupPlayer {
  id: string;
  name: string;
  team: string;
  position: string;
  salary: number;
  projection: number;
  value: number;
}

interface Lineup {
  id: string;
  players: LineupPlayer[];
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
}

interface LineupsTabProps {
  sport: string;
  lineups: Lineup[];
  isLoading: boolean;
  onExportLineups: (format: string) => void;
  onSaveFavorite: (lineup: Lineup) => void;
}

const LineupsTab: React.FC<LineupsTabProps> = ({ 
  sport, 
  lineups, 
  isLoading, 
  onExportLineups, 
  onSaveFavorite 
}) => {
  const [sortBy, setSortBy] = useState('projection');
  const [filterBy, setFilterBy] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedLineups, setExpandedLineups] = useState<Set<string>>(new Set());
  const [selectedLineups, setSelectedLineups] = useState<Set<string>>(new Set());

  // Toggle lineup expansion
  const toggleLineupExpansion = (lineupId: string) => {
    const newExpanded = new Set(expandedLineups);
    if (newExpanded.has(lineupId)) {
      newExpanded.delete(lineupId);
    } else {
      newExpanded.add(lineupId);
    }
    setExpandedLineups(newExpanded);
  };

  // Toggle lineup selection
  const toggleLineupSelection = (lineupId: string) => {
    const newSelected = new Set(selectedLineups);
    if (newSelected.has(lineupId)) {
      newSelected.delete(lineupId);
    } else {
      newSelected.add(lineupId);
    }
    setSelectedLineups(newSelected);
  };

  // Select all lineups
  const selectAllLineups = () => {
    setSelectedLineups(new Set(lineups.map(l => l.id)));
  };

  // Deselect all lineups
  const deselectAllLineups = () => {
    setSelectedLineups(new Set());
  };

  // Filter and sort lineups
  const filteredLineups = lineups
    .filter(lineup => {
      if (filterBy === 'all') return true;
      if (filterBy === 'high-projection') return lineup.totalProjection > 120;
      if (filterBy === 'value-plays') return lineup.value > 2.5;
      if (filterBy === 'stacked') return lineup.stacks.length > 0;
      return true;
    })
    .filter(lineup => {
      if (!searchTerm) return true;
      const searchLower = searchTerm.toLowerCase();
      return lineup.players.some(p => 
        p.name.toLowerCase().includes(searchLower) ||
        p.team.toLowerCase().includes(searchLower)
      );
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'projection':
          return b.totalProjection - a.totalProjection;
        case 'value':
          return b.value - a.value;
        case 'salary':
          return b.totalSalary - a.totalSalary;
        case 'strategy':
          return a.strategy.localeCompare(b.strategy);
        default:
          return 0;
      }
    });

  // Get position requirements for sport
  const getPositionRequirements = () => {
    if (sport === 'NFL') {
      return { 'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1 };
    } else {
      return { 'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3 };
    }
  };

  // Validate lineup positions
  const validateLineup = (lineup: Lineup) => {
    const requirements = getPositionRequirements();
    const positionCounts: Record<string, number> = {};
    
    lineup.players.forEach(player => {
      const pos = player.position;
      positionCounts[pos] = (positionCounts[pos] || 0) + 1;
    });

    // Check if lineup meets position requirements
    for (const [pos, required] of Object.entries(requirements)) {
      const actual = positionCounts[pos] || 0;
      if (actual < required) {
        return { valid: false, missing: `${pos} (need ${required}, have ${actual})` };
      }
    }

    return { valid: true };
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto bg-slate-700 rounded-full flex items-center justify-center mb-4">
            <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Generating Lineups</h3>
          <p className="text-slate-400">Please wait while we optimize your lineups...</p>
        </div>
      </div>
    );
  }

  if (lineups.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto bg-slate-700 rounded-full flex items-center justify-center mb-4">
            <Trophy className="w-8 h-8 text-slate-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">No Lineups Generated</h3>
          <p className="text-slate-400 mb-4">Run optimization to generate lineups</p>
          <Button variant="outline" className="border-cyan-500/30 hover:bg-cyan-500/10 text-white">
            <Target className="w-4 h-4 mr-2" />
            Generate Lineups
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-700/40 border border-slate-600/50 rounded-lg p-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Label className="text-sm text-white">Sort by:</Label>
            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-40 bg-slate-700 border-slate-600 text-white text-sm h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-900 border-cyan-500/20">
                <SelectItem value="projection" className="text-white">Projection ↓</SelectItem>
                <SelectItem value="value" className="text-white">Value ↓</SelectItem>
                <SelectItem value="salary" className="text-white">Salary ↓</SelectItem>
                <SelectItem value="strategy" className="text-white">Strategy A-Z</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <Label className="text-sm text-white">Filter:</Label>
            <Select value={filterBy} onValueChange={setFilterBy}>
              <SelectTrigger className="w-40 bg-slate-700 border-slate-600 text-white text-sm h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-900 border-cyan-500/20">
                <SelectItem value="all" className="text-white">All Lineups</SelectItem>
                <SelectItem value="high-projection" className="text-white">High Projection</SelectItem>
                <SelectItem value="value-plays" className="text-white">Value Plays</SelectItem>
                <SelectItem value="stacked" className="text-white">Stacked</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <Search className="w-4 h-4 text-slate-400" />
            <Input
              placeholder="Search players..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-48 bg-slate-700 border-slate-600 text-white text-sm h-9"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={selectAllLineups}
            className="border-green-500/30 bg-green-500/5 text-white"
          >
            Select All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={deselectAllLineups}
            className="border-red-500/30 bg-red-500/5 text-white"
          >
            Deselect All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onExportLineups('draftkings')}
            className="border-cyan-500/30 bg-cyan-500/5 text-white"
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.location.reload()}
            className="border-blue-500/30 bg-blue-500/5 text-white"
          >
            <Target className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Lineups Grid */}
      <div className="flex-1 overflow-auto max-h-96 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {filteredLineups.map((lineup, index) => {
            const isExpanded = expandedLineups.has(lineup.id);
            const isSelected = selectedLineups.has(lineup.id);
            const validation = validateLineup(lineup);
            
            return (
              <Card
                key={lineup.id}
                className={`bg-slate-800/50 border-slate-700 hover:border-slate-600 transition-all cursor-pointer ${
                  isSelected ? 'ring-2 ring-cyan-500/50 border-cyan-500/30' : ''
                }`}
                onClick={() => toggleLineupSelection(lineup.id)}
              >
                <div className="p-4">
                  {/* Lineup Header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-slate-700 text-white border-slate-600">
                        #{index + 1}
                      </Badge>
                      <Badge 
                        variant="outline" 
                        className={`${
                          validation.valid 
                            ? 'bg-green-500/10 text-green-400 border-green-500/30' 
                            : 'bg-red-500/10 text-red-400 border-red-500/30'
                        }`}
                      >
                        {validation.valid ? 'Valid' : 'Invalid'}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSaveFavorite(lineup);
                        }}
                        className="text-slate-400 hover:text-yellow-400 hover:bg-yellow-400/10"
                      >
                        <Star className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleLineupExpansion(lineup.id);
                        }}
                        className="text-slate-400 hover:text-white"
                      >
                        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </Button>
                    </div>
                  </div>

                  {/* Lineup Stats */}
                  <div className="grid grid-cols-3 gap-3 mb-3">
                    <div className="text-center">
                      <div className="text-xs text-slate-400 uppercase tracking-wider">Projection</div>
                      <div className="text-lg font-semibold text-cyan-400">{lineup.totalProjection.toFixed(1)}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-xs text-slate-400 uppercase tracking-wider">Salary</div>
                      <div className="text-lg font-semibold text-white">${lineup.totalSalary.toLocaleString()}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-xs text-slate-400 uppercase tracking-wider">Value</div>
                      <div className="text-lg font-semibold text-green-400">{lineup.value.toFixed(2)}</div>
                    </div>
                  </div>

                  {/* Strategy & Stacks */}
                  <div className="flex items-center justify-between mb-3">
                    <Badge variant="outline" className="bg-slate-700 text-slate-300 border-slate-600">
                      {lineup.strategy}
                    </Badge>
                    {lineup.stacks.length > 0 && (
                      <div className="flex items-center gap-1">
                        <Target className="w-3 h-3 text-cyan-400" />
                        <span className="text-xs text-cyan-400">{lineup.stacks.length} stack{lineup.stacks.length > 1 ? 's' : ''}</span>
                      </div>
                    )}
                  </div>

                  {/* Players List (Collapsed) */}
                  {!isExpanded && (
                    <div className="space-y-1">
                      {lineup.players.slice(0, 4).map((player, idx) => (
                        <div key={player.id} className="flex items-center justify-between text-sm">
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-400 w-8">{player.position}</span>
                            <span className="text-white font-medium">{player.name}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-cyan-400 font-medium">{player.projection.toFixed(1)}</span>
                            <span className="text-slate-400 text-xs">${player.salary.toLocaleString()}</span>
                          </div>
                        </div>
                      ))}
                      {lineup.players.length > 4 && (
                        <div className="text-xs text-slate-400 text-center">
                          +{lineup.players.length - 4} more players
                        </div>
                      )}
                    </div>
                  )}

                  {/* Players List (Expanded) */}
                  {isExpanded && (
                    <div className="space-y-2">
                      {lineup.players.map((player, idx) => (
                        <div key={player.id} className="flex items-center justify-between p-2 bg-slate-700/30 rounded-lg">
                          <div className="flex items-center gap-3">
                            <Badge variant="outline" className="bg-slate-600 text-white text-xs">
                              {player.position}
                            </Badge>
                            <div>
                              <div className="text-white font-medium">{player.name}</div>
                              <div className="text-xs text-slate-400">{player.team}</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-cyan-400 font-medium">{player.projection.toFixed(1)} pts</div>
                            <div className="text-slate-400 text-xs">${player.salary.toLocaleString()}</div>
                            <div className="text-green-400 text-xs">{(player.projection / player.salary * 1000).toFixed(1)} value</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Stacks Detail (Expanded) */}
                  {isExpanded && lineup.stacks.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-700">
                      <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">Stacks</div>
                      <div className="space-y-1">
                        {lineup.stacks.map((stack, idx) => (
                          <div key={idx} className="flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                              <Target className="w-3 h-3 text-cyan-400" />
                              <span className="text-white">{stack.team}</span>
                              <span className="text-slate-400">({stack.players} players)</span>
                            </div>
                            <Badge variant="outline" className="bg-slate-600 text-slate-300 text-xs">
                              {stack.type}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Status Bar */}
      <div className="bg-slate-700/40 border border-slate-600/50 rounded-lg p-3">
        <div className="flex items-center justify-between text-sm">
          <div className="text-white">
            <span className="font-semibold text-cyan-400">{filteredLineups.length}</span> / {lineups.length} lineups
            {selectedLineups.size > 0 && (
              <span className="text-slate-500 ml-2">({selectedLineups.size} selected)</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="text-slate-400">
              Avg Projection: <span className="text-cyan-400 font-medium">
                {(lineups.reduce((sum, l) => sum + l.totalProjection, 0) / lineups.length).toFixed(1)}
              </span>
            </div>
            <div className="text-slate-400">|</div>
            <div className="text-slate-400">
              Avg Value: <span className="text-green-400 font-medium">
                {(lineups.reduce((sum, l) => sum + l.value, 0) / lineups.length).toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LineupsTab;
