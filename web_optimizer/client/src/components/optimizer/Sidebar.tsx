import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { Checkbox } from '../ui/checkbox';
import { Input } from '../ui/input';
import { Star, Download } from 'lucide-react';
import LineupCard from './LineupCard';

interface SidebarProps {
  results: any[];
  numLineups: number;
  onNumLineupsChange: (n: number) => void;
  onViewLineups: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  results,
  numLineups,
  onNumLineupsChange,
  onViewLineups,
}) => {
  const [activeTab, setActiveTab] = useState<'lineups' | 'favorites' | 'results'>('lineups');
  const [selectedLineups, setSelectedLineups] = useState<Set<number>>(new Set());
  const [savedFavorites, setSavedFavorites] = useState<any[]>([]);
  const [isLoadingFavorites, setIsLoadingFavorites] = useState(false);

  const fetchFavorites = async () => {
    setIsLoadingFavorites(true);
    try {
      const response = await fetch('/api/favorites');
      if (response.ok) {
        const data = await response.json();
        setSavedFavorites(data.favorites || []);
      } else { setSavedFavorites([]); }
    } catch { setSavedFavorites([]); }
    finally { setIsLoadingFavorites(false); }
  };

  useEffect(() => {
    if (activeTab === 'favorites') fetchFavorites();
  }, [activeTab]);

  const handleToggleLineup = (idx: number, selected: boolean) => {
    const newSelected = new Set(selectedLineups);
    if (selected) newSelected.add(idx); else newSelected.delete(idx);
    setSelectedLineups(newSelected);
  };

  const avgPts = results.length > 0
    ? (results.reduce((sum, r) => sum + (r.totalPoints || r.points || 0), 0) / results.length).toFixed(1)
    : '0.0';

  const tabs = [
    { id: 'lineups' as const, label: `Lineups (${results.length})` },
    { id: 'favorites' as const, label: `Favorites (${savedFavorites.length})` },
    { id: 'results' as const, label: 'Results' },
  ];

  return (
    <div className="w-72 flex-shrink-0 border-l border-[var(--dfs-border)] bg-[var(--dfs-bg-secondary)] flex flex-col h-full overflow-hidden">
      {/* Tabs Header */}
      <div className="flex border-b border-[var(--dfs-border)] flex-shrink-0">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`flex-1 px-2 py-2 text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-[var(--dfs-bg-primary)] text-[var(--dfs-accent)] border-b-2 border-[var(--dfs-accent)]'
                : 'text-[var(--dfs-text-muted)] hover:text-[var(--dfs-text-secondary)] hover:bg-[var(--dfs-bg-hover)]'
            }`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800">
        {activeTab === 'lineups' && (
          <>
            {/* Compact Header: stats + count + actions on two tight rows */}
            <div className="px-3 pt-2 pb-1 border-b border-[var(--dfs-border)] flex-shrink-0">
              <div className="flex items-center justify-between">
                <span className="text-xs text-[var(--dfs-text-secondary)]">
                  Lineups: <span className="font-bold text-white">{results.length}</span>
                  <span className="text-[var(--dfs-text-muted)] mx-1">|</span>
                  Avg: <span className="font-bold text-[var(--dfs-accent)] font-mono">{avgPts}pts</span>
                </span>
                <Button
                  variant="primary-action"
                  size="sm"
                  className="text-xs h-6 px-2"
                  disabled={results.length === 0}
                  onClick={onViewLineups}
                >
                  View All
                </Button>
              </div>
              <div className="flex items-center justify-between mt-1.5">
                <div className="flex items-center gap-2">
                  <Checkbox
                    checked={selectedLineups.size > 0 && selectedLineups.size === Math.min(numLineups, results.length)}
                    onCheckedChange={(checked: boolean) => {
                      if (checked) setSelectedLineups(new Set(Array.from({ length: Math.min(numLineups, results.length) }, (_, i) => i)));
                      else setSelectedLineups(new Set());
                    }}
                    className="h-3.5 w-3.5"
                  />
                  <span className="text-xs text-[var(--dfs-text-muted)]">All ({selectedLineups.size})</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-[var(--dfs-text-muted)]">Count:</span>
                  <Input
                    type="number"
                    min={1}
                    max={500}
                    value={numLineups}
                    onChange={(e) => onNumLineupsChange(parseInt(e.target.value, 10) || 1)}
                    className="bg-[var(--dfs-bg-primary)] border-[var(--dfs-border)] text-white h-6 text-xs w-16 px-1.5"
                  />
                </div>
              </div>
            </div>

            {/* Lineups List */}
            {results.length === 0 ? (
              <div className="flex-1 flex items-center justify-center px-3">
                <p className="text-xs text-[var(--dfs-text-muted)] text-center">Run optimizer to see lineups.</p>
              </div>
            ) : (
              <div className="px-3 pb-3 pt-2 flex-1 overflow-auto">
                <div className="space-y-1.5">
                  {results.slice(0, numLineups).map((result, idx) => (
                    <LineupCard
                      key={result.id || idx}
                      result={result}
                      idx={idx}
                      isSelected={selectedLineups.has(idx)}
                      onToggleSelect={handleToggleLineup}
                    />
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {activeTab === 'favorites' && (
          <div className="flex-1 overflow-y-auto px-3 py-3">
            {isLoadingFavorites ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="w-8 h-8 mx-auto bg-slate-700 rounded-full flex items-center justify-center mb-2">
                    <div className="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                  </div>
                  <p className="text-xs text-[var(--dfs-text-muted)]">Loading favorites...</p>
                </div>
              </div>
            ) : savedFavorites.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <Star className="w-8 h-8 text-[var(--dfs-text-muted)] mx-auto mb-2" />
                  <p className="text-xs text-[var(--dfs-text-muted)] text-center">No favorites yet.</p>
                </div>
              </div>
            ) : (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold text-[var(--dfs-text-secondary)] uppercase">Favorites ({savedFavorites.length})</h3>
                  <button
                    onClick={async () => {
                      try {
                        const response = await fetch('/api/export-dk-entries', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lineups: savedFavorites.map(f => f.lineup || { players: f.players || [] }), type: 'favorites' }) });
                        if (response.ok) {
                          const blob = await response.blob();
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url; a.download = `dk_favorites_${Date.now()}.csv`;
                          document.body.appendChild(a); a.click();
                          window.URL.revokeObjectURL(url); document.body.removeChild(a);
                        }
                      } catch (error) { console.error('Export error:', error); }
                    }}
                    className="text-xs text-[var(--dfs-accent)] hover:text-cyan-300"
                  >
                    <Download className="w-3 h-3 inline mr-1" /> Export DK
                  </button>
                </div>
                {savedFavorites.map((favorite) => (
                  <div key={favorite.id} className="bg-[var(--dfs-bg-primary)] border border-[var(--dfs-border)] rounded-md p-2 hover:border-cyan-500/30 transition-colors">
                    <div className="flex items-start justify-between mb-1">
                      <div className="flex-1">
                        <div className="text-xs font-medium text-white mb-0.5">{favorite.name}</div>
                        <div className="text-xs text-[var(--dfs-text-muted)]">
                          {(() => {
                            const dateStr = favorite.timestamp || favorite.createdAt || favorite.dateAdded;
                            if (!dateStr) return 'No date';
                            try { const date = new Date(dateStr); return isNaN(date.getTime()) ? 'Invalid date' : date.toLocaleDateString(); } catch { return 'Invalid date'; }
                          })()}
                        </div>
                      </div>
                      <button
                        onClick={async () => { if (confirm('Delete?')) { try { await fetch(`/api/favorites/${favorite.id}`, { method: 'DELETE' }); fetchFavorites(); } catch {} } }}
                        className="text-[var(--dfs-text-muted)] hover:text-red-400 text-xs"
                      >&times;</button>
                    </div>
                    {(() => {
                      const players = favorite.lineup?.players || favorite.players || [];
                      const totalProjection = favorite.lineup?.totalProjection || favorite.totalPoints || players.reduce((sum: number, p: any) => sum + (p.projection || p.projectedPoints || 0), 0);
                      const totalSalary = favorite.lineup?.totalSalary || favorite.totalSalary || players.reduce((sum: number, p: any) => sum + (p.salary || 0), 0);
                      if (players.length === 0) return null;
                      return (
                        <div className="space-y-0.5 text-xs">
                          <div className="flex justify-between"><span className="text-[var(--dfs-text-muted)]">Proj:</span><span className="text-[var(--dfs-accent)] font-medium font-mono">{totalProjection.toFixed(1)}</span></div>
                          <div className="flex justify-between"><span className="text-[var(--dfs-text-muted)]">Sal:</span><span className="text-white font-medium font-mono">${totalSalary.toLocaleString()}</span></div>
                        </div>
                      );
                    })()}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'results' && (
          <div className="px-3 py-3">
            <div className="mb-2">
              <h3 className="text-xs uppercase tracking-wider text-[var(--dfs-text-muted)] mb-1.5 font-medium">OPTIMIZATION RESULTS</h3>
              <div className="bg-[var(--dfs-bg-primary)] border border-[var(--dfs-border)] rounded-md p-2 space-y-1.5">
                <div className="flex items-center justify-between"><span className="text-xs text-[var(--dfs-text-muted)]">Total Lineups</span><span className="text-xs font-bold text-white">{results.length}</span></div>
                <div className="flex items-center justify-between"><span className="text-xs text-[var(--dfs-text-muted)]">Avg Points</span><span className="text-xs font-bold text-[var(--dfs-accent)] font-mono">{avgPts}</span></div>
                <div className="flex items-center justify-between"><span className="text-xs text-[var(--dfs-text-muted)]">Avg Salary</span><span className="text-xs font-bold text-emerald-400 font-mono">${results.length > 0 ? (results.reduce((sum, r) => sum + (r.totalSalary || r.salary || 0), 0) / results.length).toFixed(0) : '0'}</span></div>
                {results.length > 0 && (
                  <>
                    <div className="pt-1.5 border-t border-[var(--dfs-border)]">
                      <div className="flex justify-between"><span className="text-xs text-[var(--dfs-text-muted)]">Best</span><span className="text-xs font-bold text-[var(--dfs-accent)] font-mono">{Math.max(...results.map(r => r.totalPoints || r.points || 0)).toFixed(1)} pts</span></div>
                    </div>
                    <div className="flex justify-between"><span className="text-xs text-[var(--dfs-text-muted)]">Worst</span><span className="text-xs font-bold text-[var(--dfs-text-muted)] font-mono">{Math.min(...results.map(r => r.totalPoints || r.points || 0)).toFixed(1)} pts</span></div>
                  </>
                )}
              </div>
            </div>

            {/* Player Usage Stats */}
            {results.length > 0 && (
              <div className="mb-2">
                <h3 className="text-xs uppercase tracking-wider text-[var(--dfs-text-muted)] mb-1.5 font-medium">TOP USED PLAYERS</h3>
                <div className="bg-[var(--dfs-bg-primary)] border border-[var(--dfs-border)] rounded-md overflow-hidden">
                  <div className="max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-slate-900">
                    {(() => {
                      const playerUsage = new Map<string, { count: number; totalPoints: number }>();
                      results.forEach(result => {
                        (result.players || []).forEach((player: any) => {
                          const name = player.name || player.player || 'Unknown';
                          const points = player.projectedPoints || player.points || 0;
                          const existing = playerUsage.get(name) || { count: 0, totalPoints: 0 };
                          playerUsage.set(name, { count: existing.count + 1, totalPoints: existing.totalPoints + points });
                        });
                      });
                      return Array.from(playerUsage.entries()).sort((a, b) => b[1].count - a[1].count).slice(0, 10).map(([name, stats], idx, arr) => (
                        <div key={name} className={`flex items-center justify-between p-1.5 ${idx < arr.length - 1 ? 'border-b border-[var(--dfs-border)]' : ''}`}>
                          <div className="flex-1 min-w-0">
                            <div className="text-xs font-medium text-[var(--dfs-text-secondary)] truncate">{name}</div>
                            <div className="text-xs text-[var(--dfs-text-muted)]">{(stats.totalPoints / stats.count).toFixed(1)} pts avg</div>
                          </div>
                          <div className="text-right ml-2">
                            <div className="text-xs font-bold text-[var(--dfs-accent)]">{stats.count}</div>
                            <div className="text-xs text-[var(--dfs-text-muted)]">{((stats.count / results.length) * 100).toFixed(0)}%</div>
                          </div>
                        </div>
                      ));
                    })()}
                  </div>
                </div>
              </div>
            )}

            {results.length === 0 && (
              <div className="flex-1 flex items-center justify-center">
                <p className="text-xs text-[var(--dfs-text-muted)] text-center">Run the optimizer to see results.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
