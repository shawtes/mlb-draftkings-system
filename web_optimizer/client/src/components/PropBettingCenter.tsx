import { useState, useEffect, useCallback } from 'react';
import { Trophy, Search, Filter, TrendingUp, TrendingDown, Minus, RefreshCw, ChevronDown } from 'lucide-react';
import { bettingApi, type PropBet, type BetSelection } from '../services/betting-api';
import BettingSlip from './BettingSlip';

type Sport = 'MLB' | 'NFL' | 'NBA';
type PropFilter = 'all' | 'points' | 'rebounds' | 'assists' | 'threes' | 'strikeouts' | 'passing' | 'rushing' | 'receiving';

const sportPropFilters: Record<Sport, { id: PropFilter; label: string }[]> = {
  NBA: [
    { id: 'all', label: 'All Props' },
    { id: 'points', label: 'Points' },
    { id: 'rebounds', label: 'Rebounds' },
    { id: 'assists', label: 'Assists' },
    { id: 'threes', label: '3-Pointers' },
  ],
  MLB: [
    { id: 'all', label: 'All Props' },
    { id: 'strikeouts', label: 'Strikeouts' },
    { id: 'points', label: 'Hits' },
  ],
  NFL: [
    { id: 'all', label: 'All Props' },
    { id: 'passing', label: 'Passing' },
    { id: 'rushing', label: 'Rushing' },
    { id: 'receiving', label: 'Receiving' },
  ],
};

export default function PropBettingCenter() {
  const [sport, setSport] = useState<Sport>('NBA');
  const [props, setProps] = useState<PropBet[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [propFilter, setPropFilter] = useState<PropFilter>('all');
  const [minEdge, setMinEdge] = useState(0);
  const [selections, setSelections] = useState<BetSelection[]>([]);

  useEffect(() => {
    fetchProps();
  }, [sport]);

  async function fetchProps() {
    setLoading(true);
    try {
      const data = await bettingApi.getProps(sport);
      if (data && data.length > 0) {
        setProps(data);
      } else {
        setProps(getSampleProps(sport));
      }
    } catch {
      setProps(getSampleProps(sport));
    } finally {
      setLoading(false);
    }
  }

  const filteredProps = props.filter((p) => {
    if (searchQuery && !p.player.toLowerCase().includes(searchQuery.toLowerCase()) && !p.team.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    if (propFilter !== 'all' && !p.prop.toLowerCase().includes(propFilter)) {
      return false;
    }
    if (minEdge > 0 && (p.edge ?? 0) < minEdge) {
      return false;
    }
    return true;
  });

  const addSelection = useCallback((prop: PropBet, type: 'over' | 'under') => {
    const exists = selections.find((s) => s.id === `${prop.id}-${type}`);
    if (exists) return;

    const sel: BetSelection = {
      id: `${prop.id}-${type}`,
      player: prop.player,
      team: prop.team,
      prop: prop.prop,
      line: prop.line,
      odds: type === 'over' ? prop.overOdds : prop.underOdds,
      type,
    };
    setSelections((prev) => [...prev, sel]);
  }, [selections]);

  const removeSelection = useCallback((id: string) => {
    setSelections((prev) => prev.filter((s) => s.id !== id));
  }, []);

  const clearSelections = useCallback(() => {
    setSelections([]);
  }, []);

  const trendIcon = (trend?: 'up' | 'down' | 'neutral') => {
    if (trend === 'up') return <TrendingUp className="w-3 h-3 text-green-400" />;
    if (trend === 'down') return <TrendingDown className="w-3 h-3 text-red-400" />;
    return <Minus className="w-3 h-3 text-[var(--dfs-text-muted)]" />;
  };

  return (
    <div className="h-full flex flex-col overflow-hidden bg-[var(--dfs-bg-primary)]">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-2 border-b flex-shrink-0"
        style={{ backgroundColor: 'var(--dfs-bg-secondary)', borderColor: 'var(--dfs-border)' }}
      >
        <div className="flex items-center gap-3">
          <Trophy className="w-4 h-4 text-[var(--dfs-accent)]" />
          <h2 className="text-sm font-semibold text-white">Prop Betting Center</h2>
          <span className="text-[10px] px-2 py-0.5 rounded bg-[var(--dfs-accent)]/10 text-[var(--dfs-accent)] font-medium">
            {filteredProps.length} props
          </span>
        </div>

        <div className="flex items-center gap-2">
          {(['MLB', 'NFL', 'NBA'] as Sport[]).map((s) => (
            <button
              key={s}
              onClick={() => { setSport(s); setPropFilter('all'); }}
              className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                sport === s
                  ? 'bg-[var(--dfs-accent)] text-black'
                  : 'text-[var(--dfs-text-secondary)] hover:text-white hover:bg-[var(--dfs-bg-hover)]'
              }`}
            >
              {s}
            </button>
          ))}
          <button
            onClick={fetchProps}
            className="ml-2 p-1.5 rounded text-[var(--dfs-text-muted)] hover:text-white hover:bg-[var(--dfs-bg-hover)] transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Filters bar */}
      <div
        className="flex items-center gap-3 px-4 py-2 border-b flex-shrink-0"
        style={{ borderColor: 'var(--dfs-border)' }}
      >
        {/* Search */}
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--dfs-text-muted)]" />
          <input
            type="text"
            placeholder="Search player or team..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 rounded bg-[var(--dfs-bg-tertiary)] border border-[var(--dfs-border)] text-xs text-white placeholder-[var(--dfs-text-muted)] focus:outline-none focus:border-[var(--dfs-accent)]/50"
          />
        </div>

        {/* Prop type filters */}
        <div className="flex items-center gap-1">
          {sportPropFilters[sport].map((f) => (
            <button
              key={f.id}
              onClick={() => setPropFilter(f.id)}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                propFilter === f.id
                  ? 'bg-[var(--dfs-accent)]/15 text-[var(--dfs-accent)] border border-[var(--dfs-accent)]/30'
                  : 'text-[var(--dfs-text-muted)] hover:text-white border border-transparent'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Min edge filter */}
        <div className="flex items-center gap-1.5 ml-auto">
          <Filter className="w-3 h-3 text-[var(--dfs-text-muted)]" />
          <span className="text-[10px] text-[var(--dfs-text-muted)]">Min Edge:</span>
          <select
            value={minEdge}
            onChange={(e) => setMinEdge(Number(e.target.value))}
            className="bg-[var(--dfs-bg-tertiary)] border border-[var(--dfs-border)] rounded text-[11px] text-white px-2 py-1 focus:outline-none"
          >
            <option value={0}>Any</option>
            <option value={2}>2%+</option>
            <option value={5}>5%+</option>
            <option value={10}>10%+</option>
          </select>
        </div>
      </div>

      {/* Main content: props table + betting slip */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Props list */}
        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="w-6 h-6 text-[var(--dfs-accent)] animate-spin" />
            </div>
          ) : filteredProps.length === 0 ? (
            <div className="flex items-center justify-center h-full text-center">
              <div>
                <Trophy className="w-12 h-12 text-[var(--dfs-text-muted)] mx-auto mb-3" />
                <p className="text-[var(--dfs-text-secondary)] text-sm">No props match your filters</p>
                <p className="text-[var(--dfs-text-muted)] text-xs mt-1">Try adjusting search or filters</p>
              </div>
            </div>
          ) : (
            <table className="w-full text-xs">
              <thead className="sticky top-0 z-10">
                <tr style={{ backgroundColor: 'var(--dfs-bg-tertiary)' }}>
                  <th className="text-left px-3 py-2 text-[var(--dfs-text-muted)] font-medium">Player</th>
                  <th className="text-left px-3 py-2 text-[var(--dfs-text-muted)] font-medium">Prop</th>
                  <th className="text-center px-3 py-2 text-[var(--dfs-text-muted)] font-medium">Line</th>
                  <th className="text-center px-3 py-2 text-[var(--dfs-text-muted)] font-medium">Over</th>
                  <th className="text-center px-3 py-2 text-[var(--dfs-text-muted)] font-medium">Under</th>
                  <th className="text-center px-3 py-2 text-[var(--dfs-text-muted)] font-medium">Proj</th>
                  <th className="text-center px-3 py-2 text-[var(--dfs-text-muted)] font-medium">Edge</th>
                  <th className="text-center px-3 py-2 text-[var(--dfs-text-muted)] font-medium">Trend</th>
                </tr>
              </thead>
              <tbody>
                {filteredProps.map((prop) => {
                  const overSelected = selections.some((s) => s.id === `${prop.id}-over`);
                  const underSelected = selections.some((s) => s.id === `${prop.id}-under`);
                  const edgeColor = (prop.edge ?? 0) >= 5 ? 'text-green-400' : (prop.edge ?? 0) >= 2 ? 'text-yellow-400' : 'text-[var(--dfs-text-muted)]';

                  return (
                    <tr
                      key={prop.id}
                      className="border-b border-[var(--dfs-border)] hover:bg-[var(--dfs-bg-hover)] transition-colors"
                    >
                      <td className="px-3 py-2.5">
                        <div>
                          <p className="text-white font-medium">{prop.player}</p>
                          <p className="text-[var(--dfs-text-muted)] text-[10px]">{prop.team} vs {prop.opponent}</p>
                        </div>
                      </td>
                      <td className="px-3 py-2.5 text-[var(--dfs-text-secondary)]">{prop.prop}</td>
                      <td className="px-3 py-2.5 text-center text-white font-medium">{prop.line}</td>
                      <td className="px-3 py-2.5 text-center">
                        <button
                          onClick={() => addSelection(prop, 'over')}
                          className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                            overSelected
                              ? 'bg-[var(--dfs-accent)] text-black'
                              : 'bg-[var(--dfs-bg-tertiary)] text-[var(--dfs-text-secondary)] hover:bg-[var(--dfs-accent)]/20 hover:text-[var(--dfs-accent)]'
                          }`}
                        >
                          O {prop.overOdds > 0 ? '+' : ''}{prop.overOdds}
                        </button>
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        <button
                          onClick={() => addSelection(prop, 'under')}
                          className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                            underSelected
                              ? 'bg-[var(--dfs-accent)] text-black'
                              : 'bg-[var(--dfs-bg-tertiary)] text-[var(--dfs-text-secondary)] hover:bg-[var(--dfs-accent)]/20 hover:text-[var(--dfs-accent)]'
                          }`}
                        >
                          U {prop.underOdds > 0 ? '+' : ''}{prop.underOdds}
                        </button>
                      </td>
                      <td className="px-3 py-2.5 text-center text-[var(--dfs-accent)] font-medium">
                        {prop.projection?.toFixed(1) ?? '-'}
                      </td>
                      <td className={`px-3 py-2.5 text-center font-medium ${edgeColor}`}>
                        {prop.edge != null ? `${prop.edge > 0 ? '+' : ''}${prop.edge.toFixed(1)}%` : '-'}
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        {trendIcon(prop.trend)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Betting slip sidebar */}
        <div
          className="w-80 flex-shrink-0 border-l overflow-auto p-3"
          style={{ borderColor: 'var(--dfs-border)', backgroundColor: 'var(--dfs-bg-secondary)' }}
        >
          <BettingSlip
            selections={selections}
            onRemove={removeSelection}
            onClear={clearSelections}
          />
        </div>
      </div>
    </div>
  );
}

// Sample props when API returns nothing
function getSampleProps(sport: Sport): PropBet[] {
  const nbaProps: PropBet[] = [
    { id: '1', player: 'LeBron James', team: 'LAL', position: 'SF', opponent: 'BOS', prop: 'Points', line: 25.5, overOdds: -115, underOdds: -105, projection: 27.2, edge: 3.1, confidence: 72, trend: 'up' },
    { id: '2', player: 'Jayson Tatum', team: 'BOS', position: 'SF', opponent: 'LAL', prop: 'Points', line: 27.5, overOdds: -110, underOdds: -110, projection: 28.8, edge: 2.4, confidence: 68, trend: 'up' },
    { id: '3', player: 'Nikola Jokic', team: 'DEN', position: 'C', opponent: 'MIL', prop: 'Rebounds', line: 12.5, overOdds: -120, underOdds: +100, projection: 13.1, edge: 4.8, confidence: 75, trend: 'neutral' },
    { id: '4', player: 'Giannis Antetokounmpo', team: 'MIL', position: 'PF', opponent: 'DEN', prop: 'Points', line: 31.5, overOdds: -105, underOdds: -115, projection: 30.2, edge: -1.5, confidence: 55, trend: 'down' },
    { id: '5', player: 'Stephen Curry', team: 'GSW', position: 'PG', opponent: 'PHX', prop: '3-Pointers', line: 4.5, overOdds: +105, underOdds: -125, projection: 5.1, edge: 6.2, confidence: 70, trend: 'up' },
    { id: '6', player: 'Luka Doncic', team: 'DAL', position: 'PG', opponent: 'OKC', prop: 'Assists', line: 8.5, overOdds: -110, underOdds: -110, projection: 9.2, edge: 3.5, confidence: 65, trend: 'up' },
    { id: '7', player: 'Kevin Durant', team: 'PHX', position: 'SF', opponent: 'GSW', prop: 'Points', line: 28.5, overOdds: -110, underOdds: -110, projection: 27.1, edge: -2.1, confidence: 52, trend: 'neutral' },
    { id: '8', player: 'Anthony Davis', team: 'LAL', position: 'PF', opponent: 'BOS', prop: 'Rebounds', line: 11.5, overOdds: -105, underOdds: -115, projection: 12.3, edge: 3.8, confidence: 69, trend: 'up' },
    { id: '9', player: 'Donovan Mitchell', team: 'CLE', position: 'SG', opponent: 'MIA', prop: 'Points', line: 24.5, overOdds: -110, underOdds: -110, projection: 25.8, edge: 2.7, confidence: 66, trend: 'up' },
    { id: '10', player: 'Shai Gilgeous-Alexander', team: 'OKC', position: 'PG', opponent: 'DAL', prop: 'Points', line: 30.5, overOdds: -115, underOdds: -105, projection: 31.9, edge: 4.1, confidence: 71, trend: 'up' },
  ];

  const mlbProps: PropBet[] = [
    { id: '1', player: 'Gerrit Cole', team: 'NYY', position: 'SP', opponent: 'BOS', prop: 'Strikeouts', line: 7.5, overOdds: -120, underOdds: +100, projection: 8.2, edge: 5.1, confidence: 74, trend: 'up' },
    { id: '2', player: 'Shohei Ohtani', team: 'LAD', position: 'DH', opponent: 'SF', prop: 'Hits', line: 1.5, overOdds: +110, underOdds: -130, projection: 1.8, edge: 3.2, confidence: 62, trend: 'neutral' },
    { id: '3', player: 'Aaron Judge', team: 'NYY', position: 'OF', opponent: 'BOS', prop: 'Hits', line: 1.5, overOdds: +100, underOdds: -120, projection: 1.6, edge: 1.5, confidence: 58, trend: 'up' },
    { id: '4', player: 'Spencer Strider', team: 'ATL', position: 'SP', opponent: 'NYM', prop: 'Strikeouts', line: 8.5, overOdds: -105, underOdds: -115, projection: 9.1, edge: 4.2, confidence: 70, trend: 'up' },
  ];

  const nflProps: PropBet[] = [
    { id: '1', player: 'Patrick Mahomes', team: 'KC', position: 'QB', opponent: 'BUF', prop: 'Passing Yards', line: 275.5, overOdds: -115, underOdds: -105, projection: 289, edge: 3.8, confidence: 70, trend: 'up' },
    { id: '2', player: 'Josh Allen', team: 'BUF', position: 'QB', opponent: 'KC', prop: 'Passing Yards', line: 265.5, overOdds: -110, underOdds: -110, projection: 271, edge: 1.9, confidence: 61, trend: 'neutral' },
    { id: '3', player: 'Jalen Hurts', team: 'PHI', position: 'QB', opponent: 'DAL', prop: 'Rushing Yards', line: 44.5, overOdds: -110, underOdds: -110, projection: 52, edge: 7.5, confidence: 73, trend: 'up' },
    { id: '4', player: 'Tyreek Hill', team: 'MIA', position: 'WR', opponent: 'NE', prop: 'Receiving Yards', line: 82.5, overOdds: -105, underOdds: -115, projection: 88, edge: 3.2, confidence: 65, trend: 'up' },
  ];

  if (sport === 'MLB') return mlbProps;
  if (sport === 'NFL') return nflProps;
  return nbaProps;
}
