import React, { useState, useEffect } from 'react';
import { Checkbox } from '../ui/checkbox';
import { Star, Download, ChevronDown, ArrowUpDown, RefreshCw } from 'lucide-react';
import LineupCard from './LineupCard';

interface SidebarProps {
  results: any[];
  numLineups: number;
  onNumLineupsChange: (n: number) => void;
  onViewLineups: () => void;
  strategy?: string;
  minUnique?: number;
  portfolioMetrics?: any;
  onOptimize?: () => void;
  isOptimizing?: boolean;
  sport?: string;
}

const Sidebar: React.FC<SidebarProps> = ({
  results,
  numLineups,
  onNumLineupsChange,
  onViewLineups,
  portfolioMetrics,
  onOptimize,
  isOptimizing,
  sport,
}) => {
  const [activeTab, setActiveTab] = useState<'lineups' | 'pool' | 'favorites' | 'trash'>('lineups');
  const [selectedLineups, setSelectedLineups] = useState<Set<number>>(new Set());
  const [savedFavorites, setSavedFavorites] = useState<any[]>([]);
  const [isLoadingFavorites, setIsLoadingFavorites] = useState(false);
  const [trashLineups, setTrashLineups] = useState<any[]>([]);
  const [method, setMethod] = useState('Unique Rank');
  const [uniques, setUniques] = useState(1);
  const [sortAsc, setSortAsc] = useState(false);

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

  const handleTrashLineup = (idx: number) => {
    const lineup = results[idx];
    if (lineup) {
      setTrashLineups(prev => [...prev, { ...lineup, trashedIdx: idx }]);
    }
  };

  const avgPts = results.length > 0
    ? (results.reduce((sum, r) => sum + (r.totalPoints || r.points || 0), 0) / results.length).toFixed(1)
    : '0.0';

  const sortedResults = React.useMemo(() => {
    const sliced = results.slice(0, numLineups);
    if (method === 'Proj Score') {
      return [...sliced].sort((a, b) => {
        const aP = a.totalPoints || a.points || 0;
        const bP = b.totalPoints || b.points || 0;
        return sortAsc ? aP - bP : bP - aP;
      });
    }
    if (method === 'ROI') {
      return [...sliced].sort((a, b) => {
        const aR = (a.totalPoints || a.points || 0) / Math.max(1, a.totalSalary || a.salary || 1);
        const bR = (b.totalPoints || b.points || 0) / Math.max(1, b.totalSalary || b.salary || 1);
        return sortAsc ? aR - bR : bR - aR;
      });
    }
    if (method === 'Own') {
      return [...sliced].sort((a, b) => {
        const aOwn = (a.players || []).reduce((s: number, p: any) => s + (p.ownership || 0), 0);
        const bOwn = (b.players || []).reduce((s: number, p: any) => s + (p.ownership || 0), 0);
        return sortAsc ? aOwn - bOwn : bOwn - aOwn;
      });
    }
    return sliced; // Unique Rank = default order
  }, [results, numLineups, method, sortAsc]);

  const trashedIds = new Set(trashLineups.map(t => t.trashedIdx));

  const tabs = [
    { id: 'lineups' as const, label: 'My Lineups', count: results.length },
    { id: 'pool' as const, label: 'Pool', count: null },
    { id: 'favorites' as const, label: 'Saved', count: savedFavorites.length },
    { id: 'trash' as const, label: 'Trash', count: trashLineups.length },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', background: 'var(--dfs-bg-secondary)' }}>
      {/* rp-r1: Fill Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 8px', borderBottom: '1px solid var(--dfs-border)', flexShrink: 0 }}>
        <button
          style={{ fontSize: 8, color: 'var(--dfs-accent)', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600, textDecoration: 'underline', textUnderlineOffset: 2 }}
        >
          Direct Fill
        </button>
        <button
          style={{ fontSize: 8, color: 'var(--dfs-text-muted)', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 500 }}
        >
          Auto Fill
        </button>
        <div style={{ flex: 1 }} />
        <button
          style={{
            fontSize: 7, padding: '2px 8px', background: 'var(--dfs-accent)', color: '#000',
            border: 'none', borderRadius: 3, cursor: 'pointer', fontWeight: 700, letterSpacing: 0.3,
          }}
          onClick={onViewLineups}
          disabled={results.length === 0}
        >
          Save to Contests
        </button>
      </div>

      {/* rp-r2: Tab Bar */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--dfs-border)', flexShrink: 0 }}>
        {tabs.map(tab => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                flex: 1, padding: '5px 4px', fontSize: 8, fontWeight: 600, cursor: 'pointer',
                background: isActive ? 'rgba(45,212,191,0.08)' : 'transparent',
                color: isActive ? 'var(--dfs-accent)' : 'var(--dfs-text-muted)',
                border: 'none', borderBottom: isActive ? '2px solid var(--dfs-accent)' : '2px solid transparent',
                transition: 'all 0.15s',
              }}
            >
              {tab.label}{tab.count !== null ? ` (${tab.count})` : ''}
            </button>
          );
        })}
      </div>

      {/* rp-r3: Method / Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '3px 8px', borderBottom: '1px solid var(--dfs-border)', flexShrink: 0 }}>
        <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontWeight: 500 }}>Method</span>
        <div style={{ position: 'relative' }}>
          <select
            value={method}
            onChange={e => setMethod(e.target.value)}
            style={{
              fontSize: 7, padding: '1px 14px 1px 4px', background: 'var(--dfs-bg-primary)', color: 'var(--dfs-text-secondary)',
              border: '1px solid var(--dfs-border)', borderRadius: 3, cursor: 'pointer', appearance: 'none',
              WebkitAppearance: 'none',
            }}
          >
            <option value="Unique Rank">Unique Rank</option>
            <option value="Proj Score">Proj Score</option>
            <option value="ROI">ROI</option>
            <option value="Own">Own</option>
          </select>
          <ChevronDown style={{ position: 'absolute', right: 2, top: '50%', transform: 'translateY(-50%)', width: 8, height: 8, color: 'var(--dfs-text-muted)', pointerEvents: 'none' }} />
        </div>
        <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontWeight: 500, marginLeft: 4 }}>Uniques</span>
        <input
          type="number"
          min={1}
          max={20}
          value={uniques}
          onChange={e => setUniques(Math.max(1, parseInt(e.target.value, 10) || 1))}
          style={{
            width: 28, fontSize: 7, padding: '1px 2px', background: 'var(--dfs-bg-primary)', color: 'var(--dfs-text-secondary)',
            border: '1px solid var(--dfs-border)', borderRadius: 3, textAlign: 'center',
          }}
        />
        <div style={{ flex: 1 }} />
        <button
          onClick={() => setSortAsc(!sortAsc)}
          title="Toggle sort direction"
          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 1, display: 'flex', alignItems: 'center' }}
        >
          <ArrowUpDown style={{ width: 9, height: 9, color: 'var(--dfs-text-muted)' }} />
        </button>
      </div>

      {/* rp-r4: Header Row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 8px', borderBottom: '1px solid var(--dfs-border)', flexShrink: 0 }}>
        <span style={{ fontSize: 8, fontWeight: 700, color: 'var(--dfs-text-secondary)' }}>Lineups</span>
        <span style={{ fontSize: 9, fontWeight: 700, color: 'var(--dfs-accent)', fontFamily: 'JetBrains Mono, monospace' }}>{results.length}</span>
        {results.length > 0 && (
          <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            avg {avgPts}pts
          </span>
        )}
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <button
            title="Refresh"
            onClick={onOptimize}
            disabled={isOptimizing}
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 1, display: 'flex', alignItems: 'center', opacity: isOptimizing ? 0.4 : 1 }}
          >
            <RefreshCw style={{ width: 9, height: 9, color: 'var(--dfs-text-muted)', animation: isOptimizing ? 'spin 1s linear infinite' : 'none' }} />
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)' }}>Count:</span>
            <input
              type="number"
              min={1}
              max={500}
              value={numLineups}
              onChange={(e) => onNumLineupsChange(parseInt(e.target.value, 10) || 1)}
              style={{
                width: 32, fontSize: 7, padding: '1px 2px', background: 'var(--dfs-bg-primary)', color: 'white',
                border: '1px solid var(--dfs-border)', borderRadius: 3, textAlign: 'center',
                fontFamily: 'JetBrains Mono, monospace',
              }}
            />
          </div>
        </div>
      </div>

      {/* Body: Scrollable Content */}
      <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>
        {activeTab === 'lineups' && (
          <>
            {results.length === 0 ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '32px 16px' }}>
                <p style={{ fontSize: 9, color: 'var(--dfs-text-muted)', textAlign: 'center' }}>
                  Run optimizer to see lineups.
                </p>
              </div>
            ) : (
              <div style={{ padding: '6px 6px 8px 6px' }}>
                {/* Select All */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 4px', marginBottom: 4 }}>
                  <Checkbox
                    checked={selectedLineups.size > 0 && selectedLineups.size === Math.min(numLineups, results.length)}
                    onCheckedChange={(checked: boolean) => {
                      if (checked) setSelectedLineups(new Set(Array.from({ length: Math.min(numLineups, results.length) }, (_, i) => i)));
                      else setSelectedLineups(new Set());
                    }}
                    className="h-3 w-3"
                  />
                  <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)' }}>Select All ({selectedLineups.size})</span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {sortedResults.map((result) => {
                    const origIdx = results.indexOf(result);
                    if (trashedIds.has(origIdx)) return null;
                    return (
                      <LineupCard
                        key={result.id || origIdx}
                        result={result}
                        idx={origIdx}
                        isSelected={selectedLineups.has(origIdx)}
                        onToggleSelect={handleToggleLineup}
                        onTrash={handleTrashLineup}
                        sport={sport}
                      />
                    );
                  })}
                </div>

                {/* Portfolio Quant Summary */}
                {(() => {
                  const lineupsWithQuant = results.filter((l: any) => l.quantMetrics);
                  if (lineupsWithQuant.length === 0) return null;
                  const avgSharpe = lineupsWithQuant.reduce((s: number, l: any) => s + (l.quantMetrics.sharpeRatio || 0), 0) / lineupsWithQuant.length;
                  const avgVaR = lineupsWithQuant.reduce((s: number, l: any) => s + (l.quantMetrics.valueAtRisk || 0), 0) / lineupsWithQuant.length;
                  const avgCeiling = lineupsWithQuant.reduce((s: number, l: any) => s + (l.quantMetrics.ceilingProbability || 0), 0) / lineupsWithQuant.length;
                  const sharpeColor = avgSharpe > 2.5 ? '#4ade80' : avgSharpe > 1.5 ? '#facc15' : '#f87171';
                  return (
                    <div style={{
                      marginTop: 6, padding: '4px 6px', borderRadius: 4,
                      background: 'var(--dfs-bg-tertiary, #161b22)', border: '1px solid var(--dfs-border)',
                    }}>
                      <div style={{ fontSize: 7, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 3, fontWeight: 600 }}>Portfolio</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 8, fontFamily: 'JetBrains Mono, monospace', color: sharpeColor }}>
                          Sharpe {avgSharpe.toFixed(2)}
                        </span>
                        <span style={{ color: 'var(--dfs-text-muted)', fontSize: 8 }}>|</span>
                        <span style={{ fontSize: 8, fontFamily: 'JetBrains Mono, monospace', color: '#60a5fa' }}>
                          VaR {avgVaR.toFixed(0)}
                        </span>
                        <span style={{ color: 'var(--dfs-text-muted)', fontSize: 8 }}>|</span>
                        <span style={{ fontSize: 8, fontFamily: 'JetBrains Mono, monospace', color: '#c084fc' }}>
                          Ceil {(avgCeiling * 100).toFixed(1)}%
                        </span>
                      </div>
                      {portfolioMetrics && (
                        <div style={{ display: 'flex', gap: 6, marginTop: 3, flexWrap: 'wrap' }}>
                          {portfolioMetrics.avgUniqueness != null && (
                            <span style={{ fontSize: 7, fontFamily: 'JetBrains Mono, monospace', color: 'var(--dfs-text-muted)' }}>
                              Uniq {(portfolioMetrics.avgUniqueness * 100).toFixed(0)}%
                            </span>
                          )}
                          {portfolioMetrics.exposureConcentration != null && (
                            <span style={{ fontSize: 7, fontFamily: 'JetBrains Mono, monospace', color: 'var(--dfs-text-muted)' }}>
                              HHI {portfolioMetrics.exposureConcentration.toFixed(3)}
                            </span>
                          )}
                        </div>
                      )}
                      <button
                        onClick={async () => {
                          try {
                            const response = await fetch('/api/export-risk-report', {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ portfolioMetrics }),
                            });
                            if (response.ok) {
                              const blob = await response.blob();
                              const url = window.URL.createObjectURL(blob);
                              const a = document.createElement('a');
                              a.href = url;
                              a.download = `risk_report_${new Date().toISOString().split('T')[0]}.csv`;
                              document.body.appendChild(a);
                              a.click();
                              window.URL.revokeObjectURL(url);
                              document.body.removeChild(a);
                            }
                          } catch (error) {
                            console.error('Risk report export error:', error);
                          }
                        }}
                        style={{
                          marginTop: 4, display: 'flex', alignItems: 'center', gap: 3, width: '100%',
                          padding: '3px 6px', fontSize: 7, fontWeight: 600, borderRadius: 3,
                          border: '1px solid rgba(96,165,250,0.3)', background: 'rgba(96,165,250,0.08)',
                          color: '#60a5fa', cursor: 'pointer',
                        }}
                      >
                        <Download style={{ width: 9, height: 9 }} />
                        Download Risk Report
                      </button>
                    </div>
                  );
                })()}
              </div>
            )}
          </>
        )}

        {activeTab === 'pool' && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '32px 16px' }}>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 9, color: 'var(--dfs-text-muted)' }}>Player pool view</p>
              <p style={{ fontSize: 8, color: 'var(--dfs-text-muted)', marginTop: 4 }}>
                {results.length > 0
                  ? `${new Set(results.flatMap(r => (r.players || []).map((p: any) => p.name || p.player))).size} unique players across ${results.length} lineups`
                  : 'No lineups generated yet'}
              </p>
            </div>
          </div>
        )}

        {activeTab === 'favorites' && (
          <div style={{ padding: '6px 8px', flex: 1 }}>
            {isLoadingFavorites ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '32px 16px' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{
                    width: 24, height: 24, margin: '0 auto 8px', background: 'var(--dfs-bg-primary)',
                    borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <div style={{
                      width: 12, height: 12, border: '2px solid var(--dfs-accent)', borderTopColor: 'transparent',
                      borderRadius: '50%', animation: 'spin 1s linear infinite',
                    }} />
                  </div>
                  <p style={{ fontSize: 8, color: 'var(--dfs-text-muted)' }}>Loading favorites...</p>
                </div>
              </div>
            ) : savedFavorites.length === 0 ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '32px 16px' }}>
                <div style={{ textAlign: 'center' }}>
                  <Star style={{ width: 20, height: 20, color: 'var(--dfs-text-muted)', margin: '0 auto 8px' }} />
                  <p style={{ fontSize: 8, color: 'var(--dfs-text-muted)' }}>No favorites yet.</p>
                </div>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 8, fontWeight: 600, color: 'var(--dfs-text-secondary)', textTransform: 'uppercase' }}>
                    Favorites ({savedFavorites.length})
                  </span>
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
                    style={{ fontSize: 7, color: 'var(--dfs-accent)', background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3 }}
                  >
                    <Download style={{ width: 9, height: 9 }} /> Export DK
                  </button>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {savedFavorites.map((favorite) => (
                    <div key={favorite.id} style={{
                      background: 'var(--dfs-bg-primary)', border: '1px solid var(--dfs-border)',
                      borderRadius: 4, padding: '5px 6px', transition: 'border-color 0.15s',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 3 }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 8, fontWeight: 600, color: 'white', marginBottom: 2 }}>{favorite.name}</div>
                          <div style={{ fontSize: 7, color: 'var(--dfs-text-muted)' }}>
                            {(() => {
                              const dateStr = favorite.timestamp || favorite.createdAt || favorite.dateAdded;
                              if (!dateStr) return 'No date';
                              try { const date = new Date(dateStr); return isNaN(date.getTime()) ? 'Invalid date' : date.toLocaleDateString(); } catch { return 'Invalid date'; }
                            })()}
                          </div>
                        </div>
                        <button
                          onClick={async () => { if (confirm('Delete?')) { try { await fetch(`/api/favorites/${favorite.id}`, { method: 'DELETE' }); fetchFavorites(); } catch {} } }}
                          style={{ fontSize: 10, color: 'var(--dfs-text-muted)', background: 'none', border: 'none', cursor: 'pointer', lineHeight: 1 }}
                        >&times;</button>
                      </div>
                      {(() => {
                        const players = favorite.lineup?.players || favorite.players || [];
                        const totalProjection = favorite.lineup?.totalProjection || favorite.totalPoints || players.reduce((sum: number, p: any) => sum + (p.projection || p.projectedPoints || 0), 0);
                        const totalSalary = favorite.lineup?.totalSalary || favorite.totalSalary || players.reduce((sum: number, p: any) => sum + (p.salary || 0), 0);
                        if (players.length === 0) return null;
                        return (
                          <div style={{ display: 'flex', gap: 12 }}>
                            <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)' }}>
                              Proj: <span style={{ color: 'var(--dfs-accent)', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace' }}>{totalProjection.toFixed(1)}</span>
                            </span>
                            <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)' }}>
                              Sal: <span style={{ color: 'white', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace' }}>${totalSalary.toLocaleString()}</span>
                            </span>
                          </div>
                        );
                      })()}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'trash' && (
          <div style={{ padding: '6px 8px' }}>
            {trashLineups.length === 0 ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '32px 16px' }}>
                <p style={{ fontSize: 9, color: 'var(--dfs-text-muted)', textAlign: 'center' }}>
                  No trashed lineups.
                </p>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 8, fontWeight: 600, color: 'var(--dfs-text-secondary)', textTransform: 'uppercase' }}>
                    Trash ({trashLineups.length})
                  </span>
                  <button
                    onClick={() => setTrashLineups([])}
                    style={{ fontSize: 7, color: '#f87171', background: 'none', border: 'none', cursor: 'pointer' }}
                  >
                    Clear All
                  </button>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {trashLineups.map((lineup, tidx) => (
                    <div key={tidx} style={{
                      background: 'var(--dfs-bg-primary)', border: '1px solid var(--dfs-border)',
                      borderRadius: 4, padding: '4px 6px', opacity: 0.6,
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: 8, color: 'var(--dfs-text-muted)' }}>
                          Lineup #{(lineup.trashedIdx || 0) + 1}
                        </span>
                        <div style={{ display: 'flex', gap: 6 }}>
                          <span style={{ fontSize: 8, fontFamily: 'JetBrains Mono, monospace', color: 'var(--dfs-accent)' }}>
                            {(lineup.totalPoints || lineup.points || 0).toFixed(1)}pts
                          </span>
                          <button
                            onClick={() => setTrashLineups(prev => prev.filter((_, i) => i !== tidx))}
                            style={{ fontSize: 7, color: 'var(--dfs-accent)', background: 'none', border: 'none', cursor: 'pointer' }}
                          >
                            Restore
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
