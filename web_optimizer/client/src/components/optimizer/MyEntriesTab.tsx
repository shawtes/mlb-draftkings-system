import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Checkbox } from '../ui/checkbox';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Star, Plus, XSquare, Download, BarChart3, RefreshCw } from 'lucide-react';
import { Sport } from '../sport-config';
import { FavoriteLineup } from './types';

interface MyEntriesTabProps {
  results: any[];
  sport: Sport;
}

interface ApiFavorite {
  id: string;
  name: string;
  lineup?: { players: any[]; totalProjection: number; totalSalary: number };
  players?: any[];
  totalPoints?: number;
  totalSalary?: number;
  timestamp?: string;
  createdAt?: string;
  dateAdded?: string;
  runNumber?: number;
  selected?: boolean;
}

function apiFavoriteToLocal(fav: ApiFavorite): FavoriteLineup {
  const players = fav.lineup?.players || fav.players || [];
  return {
    id: fav.id,
    players: players.map((p: any) => ({
      name: p.name || '',
      position: p.position || '',
      team: p.team || '',
      salary: p.salary || 0,
      projectedPoints: p.projection || p.projectedPoints || 0,
    })),
    totalPoints: fav.lineup?.totalProjection || fav.totalPoints || players.reduce((s: number, p: any) => s + (p.projection || p.projectedPoints || 0), 0),
    totalSalary: fav.lineup?.totalSalary || fav.totalSalary || players.reduce((s: number, p: any) => s + (p.salary || 0), 0),
    runNumber: fav.runNumber || 1,
    dateAdded: fav.timestamp || fav.createdAt || fav.dateAdded || new Date().toISOString(),
    selected: fav.selected !== false,
  };
}

const MyEntriesTab: React.FC<MyEntriesTabProps> = ({ results, sport }) => {
  const [favorites, setFavorites] = useState<FavoriteLineup[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentRun, setCurrentRun] = useState(1);
  const [sortBy, setSortBy] = useState('points-desc');
  const [filterRun, setFilterRun] = useState<number | 'all'>('all');

  // Load favorites from API on mount
  const fetchFavorites = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/favorites');
      if (response.ok) {
        const data = await response.json();
        const apiFavs: ApiFavorite[] = data.favorites || [];
        setFavorites(apiFavs.map(apiFavoriteToLocal));
        // Set currentRun based on max existing run number
        const maxRun = apiFavs.reduce((max, f) => Math.max(max, f.runNumber || 1), 0);
        if (maxRun > 0) setCurrentRun(maxRun + 1);
      }
    } catch (e) {
      console.error('Failed to load favorites:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFavorites();
  }, [fetchFavorites]);

  const handleAddCurrentPool = useCallback(async () => {
    if (results.length === 0) { alert('No lineups available. Run optimization first.'); return; }
    const countStr = prompt(`Add how many lineups?\n\nAvailable: ${results.length}\nCurrent favorites: ${favorites.length}`, Math.min(30, results.length).toString());
    const count = parseInt(countStr || '0');
    if (count <= 0 || count > results.length) return;

    const runNum = currentRun;
    let added = 0;

    for (let i = 0; i < count; i++) {
      const result = results[i];
      const players = (result.players || []).map((p: any) => ({
        name: p.name, position: p.position, team: p.team, salary: p.salary,
        projection: p.projection || p.projectedPoints || 0,
        projectedPoints: p.projection || p.projectedPoints || 0,
      }));
      const lineup = {
        players,
        totalProjection: result.points || result.totalPoints || 0,
        totalSalary: result.salary || result.totalSalary || 0,
      };

      try {
        await fetch('/api/favorites', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            lineup,
            name: `${sport} Run #${runNum} - Lineup ${i + 1}`,
            runNumber: runNum,
          }),
        });
        added++;
      } catch (e) {
        console.error('Failed to save favorite:', e);
      }
    }

    setCurrentRun(runNum + 1);
    alert(`Added ${added} lineups to favorites as Run #${runNum}`);
    fetchFavorites();
  }, [results, favorites.length, currentRun, sport, fetchFavorites]);

  const handleClearAll = useCallback(async () => {
    if (favorites.length === 0) return;
    if (!confirm(`Delete all ${favorites.length} favorite lineups?`)) return;

    for (const fav of favorites) {
      try {
        await fetch(`/api/favorites/${fav.id}`, { method: 'DELETE' });
      } catch {}
    }
    setCurrentRun(1);
    fetchFavorites();
  }, [favorites, fetchFavorites]);

  const handleExport = useCallback(async () => {
    if (favorites.length === 0) { alert('No favorites to export.'); return; }
    try {
      const response = await fetch('/api/export-dk-entries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lineups: favorites.map(f => ({
            players: f.players,
            totalProjection: f.totalPoints,
            totalSalary: f.totalSalary,
          })),
          type: 'favorites',
        }),
      });
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dk_favorites_${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert(`Export ${favorites.filter(f => f.selected).length || favorites.length} lineups to DraftKings CSV format.`);
      }
    } catch {
      alert(`Export ${favorites.filter(f => f.selected).length || favorites.length} lineups to DraftKings CSV format.`);
    }
  }, [favorites]);

  const toggleLineup = (id: string) => setFavorites(prev => prev.map(f => f.id === id ? { ...f, selected: !f.selected } : f));

  const deleteLineup = useCallback(async (id: string) => {
    if (!confirm('Delete this lineup?')) return;
    try {
      await fetch(`/api/favorites/${id}`, { method: 'DELETE' });
      fetchFavorites();
    } catch {
      setFavorites(prev => prev.filter(f => f.id !== id));
    }
  }, [fetchFavorites]);

  const runNumbers = Array.from(new Set(favorites.map(f => f.runNumber))).sort();

  const displayedFavorites = useMemo(() => {
    let filtered = filterRun !== 'all' ? favorites.filter(f => f.runNumber === filterRun) : favorites;
    return [...filtered].sort((a, b) => {
      switch (sortBy) {
        case 'points-desc': return b.totalPoints - a.totalPoints;
        case 'points-asc': return a.totalPoints - b.totalPoints;
        case 'salary-desc': return b.totalSalary - a.totalSalary;
        case 'salary-asc': return a.totalSalary - b.totalSalary;
        case 'run': return a.runNumber - b.runNumber;
        default: return 0;
      }
    });
  }, [favorites, sortBy, filterRun]);

  const totalFavorites = favorites.length;
  const selectedCount = favorites.filter(f => f.selected).length;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 text-[var(--dfs-accent)] animate-spin mx-auto mb-3" />
          <p className="text-sm text-[var(--dfs-text-muted)]">Loading favorites...</p>
        </div>
      </div>
    );
  }

  if (favorites.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <div className="mb-4"><div className="w-20 h-20 mx-auto bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-2xl flex items-center justify-center border border-cyan-500/30"><Star className="w-10 h-10 text-[var(--dfs-accent)]" /></div></div>
          <h3 className="text-2xl font-semibold text-white mb-3">No Favorites Yet</h3>
          <p className="text-[var(--dfs-text-secondary)] mb-6 leading-relaxed">Run optimizations and add your best lineups to favorites. They persist across sessions.</p>
          {results.length > 0 && (
            <Button variant="primary-action" onClick={handleAddCurrentPool}>
              <Plus className="w-4 h-4 mr-1" /> Add Current Results
            </Button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 bg-[var(--dfs-bg-tertiary)] border border-[var(--dfs-border)] rounded-lg p-3">
        <div className="flex gap-2 flex-wrap">
          <Button variant="secondary-action" size="sm" onClick={handleAddCurrentPool} className="border-green-500/30"><Plus className="w-4 h-4 mr-1" /> Add Current Pool</Button>
          <Button variant="secondary-action" size="sm" onClick={handleClearAll} disabled={favorites.length === 0} className="border-red-500/30"><XSquare className="w-4 h-4 mr-1" /> Clear All</Button>
          <Button variant="secondary-action" size="sm" onClick={handleExport} disabled={favorites.length === 0} className="border-cyan-500/30"><Download className="w-4 h-4 mr-1" /> Export</Button>
          <Button variant="secondary-action" size="sm" onClick={fetchFavorites} className="border-slate-500/30"><RefreshCw className="w-4 h-4 mr-1" /> Refresh</Button>
        </div>
        <div className="text-sm text-white"><span className="font-semibold text-[var(--dfs-accent)]">{selectedCount}</span> / {totalFavorites} selected</div>
      </div>

      <Card className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4">
        <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2"><BarChart3 className="w-5 h-5 text-[var(--dfs-accent)]" /> Portfolio Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div><div className="text-xs text-[var(--dfs-text-muted)] mb-1">Total Lineups</div><div className="text-2xl font-bold text-[var(--dfs-accent)]">{totalFavorites}</div></div>
          <div><div className="text-xs text-[var(--dfs-text-muted)] mb-1">Runs</div><div className="text-2xl font-bold text-white">{runNumbers.length}</div></div>
          <div><div className="text-xs text-[var(--dfs-text-muted)] mb-1">Avg Points</div><div className="text-2xl font-bold text-emerald-400">{totalFavorites > 0 ? (favorites.reduce((s, f) => s + f.totalPoints, 0) / totalFavorites).toFixed(1) : '0.0'}</div></div>
          <div><div className="text-xs text-[var(--dfs-text-muted)] mb-1">Avg Salary</div><div className="text-2xl font-bold text-white">${totalFavorites > 0 ? Math.round(favorites.reduce((s, f) => s + f.totalSalary, 0) / totalFavorites).toLocaleString() : '0'}</div></div>
        </div>
      </Card>

      <div className="flex flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <Label className="text-sm text-white">Sort by:</Label>
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-40 bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white text-sm h-9"><SelectValue /></SelectTrigger>
            <SelectContent className="bg-slate-900 border-cyan-500/20">
              <SelectItem value="points-desc" className="text-white">Points (High)</SelectItem>
              <SelectItem value="points-asc" className="text-white">Points (Low)</SelectItem>
              <SelectItem value="salary-desc" className="text-white">Salary (High)</SelectItem>
              <SelectItem value="salary-asc" className="text-white">Salary (Low)</SelectItem>
              <SelectItem value="run" className="text-white">Run Number</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-sm text-white">Filter:</Label>
          <Select value={filterRun.toString()} onValueChange={(v: string) => setFilterRun(v === 'all' ? 'all' : parseInt(v))}>
            <SelectTrigger className="w-32 bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white text-sm h-9"><SelectValue /></SelectTrigger>
            <SelectContent className="bg-slate-900 border-cyan-500/20">
              <SelectItem value="all" className="text-white">All Runs</SelectItem>
              {runNumbers.map(run => <SelectItem key={run} value={run.toString()} className="text-white">Run {run}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex-1 overflow-auto max-h-96 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
        <div className="space-y-3">
          {displayedFavorites.map((favorite) => (
            <Card key={favorite.id} className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4 hover:border-cyan-500/40 transition-colors">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-start gap-3">
                  <Checkbox checked={favorite.selected} onCheckedChange={() => toggleLineup(favorite.id)} className="border-slate-500 mt-1" />
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="px-2 py-1 rounded text-xs font-bold bg-blue-500/20 text-blue-400">Run #{favorite.runNumber}</span>
                      <span className="text-xs text-[var(--dfs-text-muted)]">{favorite.dateAdded}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <div><span className="text-[var(--dfs-text-muted)]">Points:</span> <span className="ml-1 font-bold text-white">{favorite.totalPoints.toFixed(1)}</span></div>
                      <div><span className="text-[var(--dfs-text-muted)]">Salary:</span> <span className="ml-1 font-bold text-white">${favorite.totalSalary.toLocaleString()}</span></div>
                    </div>
                  </div>
                </div>
                <Button variant="ghost-action" size="sm" onClick={() => deleteLineup(favorite.id)} className="text-red-400 hover:text-red-300"><XSquare className="w-4 h-4" /></Button>
              </div>
              {favorite.players.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                  {favorite.players.slice(0, 10).map((player, pidx) => (
                    <div key={pidx} className="bg-[var(--dfs-bg-secondary)] rounded px-2 py-1.5">
                      <div className="text-white font-medium truncate">{player.name}</div>
                      <div className="text-[var(--dfs-text-muted)] text-xs">{player.position} &bull; {player.team} &bull; ${(player.salary / 1000).toFixed(1)}k</div>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MyEntriesTab;
