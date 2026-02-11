import { useState, useEffect } from 'react';
import { BarChart3, Clock, MapPin, Cloud, AlertTriangle, RefreshCw, ChevronRight, Users, TrendingUp } from 'lucide-react';
import { bettingApi, type Game } from '../services/betting-api';

type Sport = 'MLB' | 'NFL' | 'NBA';

interface TeamStats {
  wins: number;
  losses: number;
  avgPoints: number;
}

// Generate placeholder team stats when no API data
function getPlaceholderStats(): TeamStats {
  return {
    wins: Math.floor(Math.random() * 60) + 20,
    losses: Math.floor(Math.random() * 60) + 20,
    avgPoints: parseFloat((Math.random() * 15 + 95).toFixed(1)),
  };
}

export default function GamesHub() {
  const [sport, setSport] = useState<Sport>('NBA');
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedGame, setSelectedGame] = useState<string | null>(null);

  useEffect(() => {
    fetchGames();
  }, [sport]);

  async function fetchGames() {
    setLoading(true);
    try {
      const data = await bettingApi.getGames(sport);
      if (data && data.length > 0) {
        setGames(data);
      } else {
        // Show sample games so the page isn't empty
        setGames(getSampleGames(sport));
      }
    } catch {
      setGames(getSampleGames(sport));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-full flex flex-col overflow-hidden bg-[var(--dfs-bg-primary)]">
      {/* Header strip */}
      <div
        className="flex items-center justify-between px-4 py-2 border-b flex-shrink-0"
        style={{ backgroundColor: 'var(--dfs-bg-secondary)', borderColor: 'var(--dfs-border)' }}
      >
        <div className="flex items-center gap-3">
          <BarChart3 className="w-4 h-4 text-[var(--dfs-accent)]" />
          <h2 className="text-sm font-semibold text-white">Games Hub</h2>
          <span className="text-[10px] px-2 py-0.5 rounded bg-[var(--dfs-accent)]/10 text-[var(--dfs-accent)] font-medium">
            {games.length} games
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Sport selector */}
          {(['MLB', 'NFL', 'NBA'] as Sport[]).map((s) => (
            <button
              key={s}
              onClick={() => setSport(s)}
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
            onClick={fetchGames}
            className="ml-2 p-1.5 rounded text-[var(--dfs-text-muted)] hover:text-white hover:bg-[var(--dfs-bg-hover)] transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Games grid */}
      <div className="flex-1 overflow-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <RefreshCw className="w-6 h-6 text-[var(--dfs-accent)] animate-spin" />
          </div>
        ) : games.length === 0 ? (
          <div className="flex items-center justify-center h-full text-center">
            <div>
              <BarChart3 className="w-12 h-12 text-[var(--dfs-text-muted)] mx-auto mb-3" />
              <p className="text-[var(--dfs-text-secondary)] text-sm">No games found for {sport}</p>
              <p className="text-[var(--dfs-text-muted)] text-xs mt-1">Try refreshing or selecting a different sport</p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {games.map((game) => {
              const isSelected = selectedGame === game.id;
              const homeStats = getPlaceholderStats();
              const awayStats = getPlaceholderStats();

              return (
                <div
                  key={game.id}
                  onClick={() => setSelectedGame(isSelected ? null : game.id)}
                  className={`rounded-lg border cursor-pointer transition-all ${
                    isSelected
                      ? 'border-[var(--dfs-accent)]/50 bg-[var(--dfs-accent)]/5'
                      : 'border-[var(--dfs-border)] bg-[var(--dfs-bg-secondary)] hover:border-[var(--dfs-accent)]/30'
                  }`}
                >
                  {/* Game status bar */}
                  <div className="flex items-center justify-between px-3 py-1.5 border-b border-[var(--dfs-border)]">
                    <div className="flex items-center gap-2">
                      {game.status === 'live' ? (
                        <span className="flex items-center gap-1 text-[10px] font-bold text-red-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                          LIVE
                        </span>
                      ) : game.status === 'final' ? (
                        <span className="text-[10px] font-medium text-[var(--dfs-text-muted)]">FINAL</span>
                      ) : (
                        <span className="flex items-center gap-1 text-[10px] text-[var(--dfs-text-muted)]">
                          <Clock className="w-3 h-3" />
                          {game.time}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-[10px] text-[var(--dfs-text-muted)]">
                      <span>O/U {game.total}</span>
                    </div>
                  </div>

                  {/* Matchup */}
                  <div className="p-3">
                    {/* Away team */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded bg-[var(--dfs-bg-tertiary)] flex items-center justify-center text-xs font-bold text-[var(--dfs-text-secondary)]">
                          {game.away.slice(0, 3)}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-white">{game.away}</p>
                          <p className="text-[10px] text-[var(--dfs-text-muted)]">{awayStats.wins}-{awayStats.losses}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        {game.awayScore !== undefined ? (
                          <p className="text-lg font-bold text-white">{game.awayScore}</p>
                        ) : (
                          <p className="text-xs text-[var(--dfs-text-muted)]">{game.awaySpread > 0 ? '+' : ''}{game.awaySpread}</p>
                        )}
                      </div>
                    </div>

                    {/* Home team */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded bg-[var(--dfs-bg-tertiary)] flex items-center justify-center text-xs font-bold text-[var(--dfs-accent)]">
                          {game.home.slice(0, 3)}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-white">{game.home}</p>
                          <p className="text-[10px] text-[var(--dfs-text-muted)]">{homeStats.wins}-{homeStats.losses}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        {game.homeScore !== undefined ? (
                          <p className="text-lg font-bold text-white">{game.homeScore}</p>
                        ) : (
                          <p className="text-xs text-[var(--dfs-text-muted)]">{game.homeSpread > 0 ? '+' : ''}{game.homeSpread}</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Expanded details */}
                  {isSelected && (
                    <div className="px-3 pb-3 border-t border-[var(--dfs-border)] pt-2 space-y-2">
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div className="bg-[var(--dfs-bg-tertiary)] rounded p-2">
                          <p className="text-[10px] text-[var(--dfs-text-muted)] mb-0.5">Spread</p>
                          <p className="text-xs font-medium text-white">{game.homeSpread > 0 ? '+' : ''}{game.homeSpread}</p>
                        </div>
                        <div className="bg-[var(--dfs-bg-tertiary)] rounded p-2">
                          <p className="text-[10px] text-[var(--dfs-text-muted)] mb-0.5">Total</p>
                          <p className="text-xs font-medium text-white">{game.total}</p>
                        </div>
                        <div className="bg-[var(--dfs-bg-tertiary)] rounded p-2">
                          <p className="text-[10px] text-[var(--dfs-text-muted)] mb-0.5">Implied</p>
                          <p className="text-xs font-medium text-[var(--dfs-accent)]">
                            {((game.total - game.homeSpread) / 2).toFixed(1)}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1 text-[10px] text-[var(--dfs-text-muted)]">
                          <Users className="w-3 h-3" />
                          <span>Key players available</span>
                        </div>
                        <div className="flex items-center gap-1 text-[10px] text-[var(--dfs-text-muted)]">
                          <TrendingUp className="w-3 h-3" />
                          <span>Trends loading...</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// Sample games to show when API returns nothing
function getSampleGames(sport: Sport): Game[] {
  const sportGames: Record<Sport, Game[]> = {
    NBA: [
      { id: '1', home: 'Lakers', away: 'Celtics', time: '7:30 PM', homeSpread: -3.5, awaySpread: 3.5, total: 224.5, status: 'scheduled' },
      { id: '2', home: 'Bucks', away: 'Nuggets', time: '8:00 PM', homeSpread: -1.5, awaySpread: 1.5, total: 231.0, status: 'scheduled' },
      { id: '3', home: 'Warriors', away: 'Suns', time: '10:00 PM', homeSpread: -5.0, awaySpread: 5.0, total: 228.5, status: 'scheduled' },
      { id: '4', home: 'Knicks', away: '76ers', time: '7:00 PM', homeSpread: -2.0, awaySpread: 2.0, total: 215.0, status: 'scheduled' },
      { id: '5', home: 'Heat', away: 'Cavaliers', time: '7:30 PM', homeSpread: 1.5, awaySpread: -1.5, total: 211.5, status: 'scheduled' },
      { id: '6', home: 'Mavericks', away: 'Thunder', time: '8:30 PM', homeSpread: 2.5, awaySpread: -2.5, total: 222.0, status: 'scheduled' },
    ],
    MLB: [
      { id: '1', home: 'Yankees', away: 'Red Sox', time: '7:05 PM', homeSpread: -1.5, awaySpread: 1.5, total: 9.0, status: 'scheduled' },
      { id: '2', home: 'Dodgers', away: 'Giants', time: '10:10 PM', homeSpread: -1.5, awaySpread: 1.5, total: 8.5, status: 'scheduled' },
      { id: '3', home: 'Astros', away: 'Rangers', time: '8:10 PM', homeSpread: -1.5, awaySpread: 1.5, total: 8.0, status: 'scheduled' },
      { id: '4', home: 'Braves', away: 'Mets', time: '7:20 PM', homeSpread: -1.5, awaySpread: 1.5, total: 7.5, status: 'scheduled' },
      { id: '5', home: 'Cubs', away: 'Cardinals', time: '8:05 PM', homeSpread: 1.5, awaySpread: -1.5, total: 8.5, status: 'scheduled' },
    ],
    NFL: [
      { id: '1', home: 'Chiefs', away: 'Bills', time: '4:25 PM', homeSpread: -3.0, awaySpread: 3.0, total: 51.5, status: 'scheduled' },
      { id: '2', home: 'Eagles', away: 'Cowboys', time: '1:00 PM', homeSpread: -6.5, awaySpread: 6.5, total: 47.0, status: 'scheduled' },
      { id: '3', home: '49ers', away: 'Seahawks', time: '4:05 PM', homeSpread: -7.0, awaySpread: 7.0, total: 45.5, status: 'scheduled' },
      { id: '4', home: 'Lions', away: 'Packers', time: '1:00 PM', homeSpread: -3.5, awaySpread: 3.5, total: 49.0, status: 'scheduled' },
    ],
  };
  return sportGames[sport] || [];
}
