import { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { 
  Filter, 
  X, 
  Search, 
  SlidersHorizontal,
  TrendingUp,
  Star,
  DollarSign
} from 'lucide-react';
import { Separator } from './ui/separator';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible';

interface FilterConfig {
  positions?: string[];
  teams?: string[];
  minSalary?: number;
  maxSalary?: number;
  minProjection?: number;
  minEdge?: number;
  status?: string[];
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

interface AdvancedFiltersProps {
  onFilterChange?: (filters: FilterConfig) => void;
  initialFilters?: FilterConfig;
  showPositionFilters?: boolean;
  showPriceFilters?: boolean;
  showProjectionFilters?: boolean;
  showStatusFilters?: boolean;
}

export default function AdvancedFilters({
  onFilterChange,
  initialFilters = {},
  showPositionFilters = true,
  showPriceFilters = true,
  showProjectionFilters = true,
  showStatusFilters = true,
}: AdvancedFiltersProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPositions, setSelectedPositions] = useState<string[]>(initialFilters.positions || []);
  const [selectedTeams, setSelectedTeams] = useState<string[]>(initialFilters.teams || []);
  const [salaryRange, setSalaryRange] = useState<number[]>([initialFilters.minSalary || 3000, initialFilters.maxSalary || 10000]);
  const [minProjection, setMinProjection] = useState(initialFilters.minProjection || 0);
  const [minEdge, setMinEdge] = useState(initialFilters.minEdge || 0);
  const [selectedStatus, setSelectedStatus] = useState<string[]>(initialFilters.status || ['Active']);
  const [sortBy, setSortBy] = useState(initialFilters.sortBy || 'projection');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>(initialFilters.sortOrder || 'desc');

  const positions = ['QB', 'RB', 'WR', 'TE', 'DST', 'K'];
  const teams = ['KC', 'BUF', 'SF', 'DAL', 'PHI', 'MIA', 'LAC', 'NYG', 'SEA', 'BAL', 'CIN', 'MIN', 'CHI'];
  const statusOptions = ['Active', 'Questionable', 'Doubtful', 'Out'];

  const togglePosition = (position: string) => {
    const updated = selectedPositions.includes(position)
      ? selectedPositions.filter(p => p !== position)
      : [...selectedPositions, position];
    setSelectedPositions(updated);
    applyFilters({ positions: updated });
  };

  const toggleTeam = (team: string) => {
    const updated = selectedTeams.includes(team)
      ? selectedTeams.filter(t => t !== team)
      : [...selectedTeams, team];
    setSelectedTeams(updated);
    applyFilters({ teams: updated });
  };

  const toggleStatus = (status: string) => {
    const updated = selectedStatus.includes(status)
      ? selectedStatus.filter(s => s !== status)
      : [...selectedStatus, status];
    setSelectedStatus(updated);
    applyFilters({ status: updated });
  };

  const applyFilters = (partialFilters: Partial<FilterConfig> = {}) => {
    const filters: FilterConfig = {
      positions: selectedPositions,
      teams: selectedTeams,
      minSalary: salaryRange[0],
      maxSalary: salaryRange[1],
      minProjection,
      minEdge,
      status: selectedStatus,
      sortBy,
      sortOrder,
      ...partialFilters,
    };
    onFilterChange?.(filters);
  };

  const clearAllFilters = () => {
    setSearchTerm('');
    setSelectedPositions([]);
    setSelectedTeams([]);
    setSalaryRange([3000, 10000]);
    setMinProjection(0);
    setMinEdge(0);
    setSelectedStatus(['Active']);
    setSortBy('projection');
    setSortOrder('desc');
    onFilterChange?.({});
  };

  const activeFilterCount = 
    selectedPositions.length + 
    selectedTeams.length + 
    (salaryRange[0] !== 3000 || salaryRange[1] !== 10000 ? 1 : 0) +
    (minProjection > 0 ? 1 : 0) +
    (minEdge > 0 ? 1 : 0) +
    (selectedStatus.length !== 1 || !selectedStatus.includes('Active') ? 1 : 0);

  return (
    <div className="space-y-4">
      {/* Search and Quick Filters */}
      <div className="flex items-center gap-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <Input
            type="text"
            placeholder="Search players, teams..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="bg-black/50 border-cyan-500/20 text-white pl-10"
          />
        </div>

        <Button
          variant="outline"
          onClick={() => setIsOpen(!isOpen)}
          className={`border-slate-700 text-slate-300 hover:bg-slate-800 relative ${
            isOpen ? 'bg-slate-800' : ''
          }`}
        >
          <SlidersHorizontal className="w-4 h-4 mr-2" />
          Filters
          {activeFilterCount > 0 && (
            <Badge className="ml-2 bg-cyan-500 text-white h-5 px-2">
              {activeFilterCount}
            </Badge>
          )}
        </Button>

        {activeFilterCount > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearAllFilters}
            className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
          >
            <X className="w-4 h-4 mr-1" />
            Clear
          </Button>
        )}
      </div>

      {/* Advanced Filters Panel */}
      {isOpen && (
        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Position Filter */}
            {showPositionFilters && (
              <div>
                <Label className="text-slate-300 mb-3 block">Position</Label>
                <div className="flex flex-wrap gap-2">
                  {positions.map((position) => (
                    <Button
                      key={position}
                      variant="outline"
                      size="sm"
                      onClick={() => togglePosition(position)}
                      className={`${
                        selectedPositions.includes(position)
                          ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                          : 'border-cyan-500/20 text-slate-400 hover:bg-cyan-500/5'
                      }`}
                    >
                      {position}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Team Filter */}
            <div>
              <Label className="text-slate-300 mb-3 block">Team</Label>
              <div className="max-h-[120px] overflow-auto flex flex-wrap gap-2">
                {teams.map((team) => (
                  <Button
                    key={team}
                    variant="outline"
                    size="sm"
                    onClick={() => toggleTeam(team)}
                    className={`${
                      selectedTeams.includes(team)
                        ? 'bg-blue-500/20 border-blue-500/50 text-blue-400'
                        : 'border-slate-700 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    {team}
                  </Button>
                ))}
              </div>
            </div>

            {/* Status Filter */}
            {showStatusFilters && (
              <div>
                <Label className="text-slate-300 mb-3 block">Player Status</Label>
                <div className="flex flex-wrap gap-2">
                  {statusOptions.map((status) => (
                    <Button
                      key={status}
                      variant="outline"
                      size="sm"
                      onClick={() => toggleStatus(status)}
                      className={`${
                        selectedStatus.includes(status)
                          ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                          : 'border-slate-700 text-slate-400 hover:bg-slate-800'
                      }`}
                    >
                      {status}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Salary Range */}
            {showPriceFilters && (
              <div>
                <Label className="text-slate-300 mb-3 block">
                  Salary Range: ${salaryRange[0].toLocaleString()} - ${salaryRange[1].toLocaleString()}
                </Label>
                <Slider
                  min={3000}
                  max={10000}
                  step={100}
                  value={salaryRange}
                  onValueChange={setSalaryRange}
                  onValueCommit={() => applyFilters()}
                  className="mt-2"
                />
              </div>
            )}

            {/* Min Projection */}
            {showProjectionFilters && (
              <div>
                <Label className="text-slate-300 mb-3 block">
                  Min Projection: {minProjection} pts
                </Label>
                <Slider
                  min={0}
                  max={40}
                  step={0.5}
                  value={[minProjection]}
                  onValueChange={(value) => setMinProjection(value[0])}
                  onValueCommit={() => applyFilters()}
                  className="mt-2"
                />
              </div>
            )}

            {/* Min Edge */}
            <div>
              <Label className="text-slate-300 mb-3 block">
                Min Edge: +{minEdge}%
              </Label>
              <Slider
                min={0}
                max={15}
                step={0.5}
                value={[minEdge]}
                onValueChange={(value) => setMinEdge(value[0])}
                onValueCommit={() => applyFilters()}
                className="mt-2"
              />
            </div>
          </div>

          <Separator className="bg-slate-700 my-6" />

          {/* Sort Options */}
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Label className="text-slate-300 mb-2 block text-sm">Sort By</Label>
              <Select value={sortBy} onValueChange={(value) => { setSortBy(value); applyFilters({ sortBy: value }); }}>
                <SelectTrigger className="bg-black/50 border-cyan-500/20 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="projection">Projection</SelectItem>
                  <SelectItem value="salary">Salary</SelectItem>
                  <SelectItem value="edge">Edge</SelectItem>
                  <SelectItem value="value">Value ($/pt)</SelectItem>
                  <SelectItem value="ownership">Ownership</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex-1">
              <Label className="text-slate-300 mb-2 block text-sm">Order</Label>
              <Select value={sortOrder} onValueChange={(value: 'asc' | 'desc') => { setSortOrder(value); applyFilters({ sortOrder: value }); }}>
                <SelectTrigger className="bg-black/50 border-cyan-500/20 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="desc">Highest First</SelectItem>
                  <SelectItem value="asc">Lowest First</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Apply/Reset Buttons */}
          <div className="flex justify-end gap-3 mt-6">
            <Button
              variant="outline"
              onClick={clearAllFilters}
              className="border-cyan-500/20 text-slate-400 hover:bg-cyan-500/5"
            >
              Reset All
            </Button>
            <Button
              onClick={() => applyFilters()}
              className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500"
            >
              Apply Filters
            </Button>
          </div>
        </Card>
      )}

      {/* Active Filter Tags */}
      {activeFilterCount > 0 && (
        <div className="flex flex-wrap gap-2">
          {selectedPositions.map((position) => (
            <Badge
              key={position}
              className="bg-cyan-500/20 border-cyan-500/50 text-cyan-400 cursor-pointer hover:bg-cyan-500/30"
              onClick={() => togglePosition(position)}
            >
              {position}
              <X className="w-3 h-3 ml-1" />
            </Badge>
          ))}
          {selectedTeams.map((team) => (
            <Badge
              key={team}
              className="bg-blue-500/20 border-blue-500/50 text-blue-400 cursor-pointer hover:bg-blue-500/30"
              onClick={() => toggleTeam(team)}
            >
              {team}
              <X className="w-3 h-3 ml-1" />
            </Badge>
          ))}
          {(salaryRange[0] !== 3000 || salaryRange[1] !== 10000) && (
            <Badge className="bg-purple-500/20 border-purple-500/50 text-purple-400">
              ${salaryRange[0].toLocaleString()} - ${salaryRange[1].toLocaleString()}
            </Badge>
          )}
          {minProjection > 0 && (
            <Badge className="bg-green-500/20 border-green-500/50 text-green-400">
              Proj ≥ {minProjection}
            </Badge>
          )}
          {minEdge > 0 && (
            <Badge className="bg-yellow-500/20 border-yellow-500/50 text-yellow-400">
              Edge ≥ +{minEdge}%
            </Badge>
          )}
        </div>
      )}
    </div>
  );
}

