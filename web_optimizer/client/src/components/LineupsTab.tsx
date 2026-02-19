import React, { useState, useEffect } from 'react';
import { Checkbox } from './ui/checkbox';
import {
  Trophy,
  Download,
  Star,
  Target,
  Search,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { TeamLogo } from './ui/team-logo';

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

const PLAYER_GRID = '30px 1fr 50px 55px 40px';

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
  const [maxLineupsToShow, setMaxLineupsToShow] = useState<number>(lineups.length);

  useEffect(() => { setMaxLineupsToShow(lineups.length); }, [lineups.length]);

  const toggleLineupExpansion = (lineupId: string) => {
    const newExpanded = new Set(expandedLineups);
    if (newExpanded.has(lineupId)) newExpanded.delete(lineupId);
    else newExpanded.add(lineupId);
    setExpandedLineups(newExpanded);
  };

  const toggleLineupSelection = (lineupId: string) => {
    const newSelected = new Set(selectedLineups);
    if (newSelected.has(lineupId)) newSelected.delete(lineupId);
    else newSelected.add(lineupId);
    setSelectedLineups(newSelected);
  };

  const deselectAllLineups = () => { setSelectedLineups(new Set()); };

  const handleAddSelectedToFavorites = async () => {
    if (selectedLineups.size === 0) { alert('Please select at least one lineup to add to favorites.'); return; }
    const selectedLineupObjects = filteredLineups.filter(l => selectedLineups.has(l.id));
    try {
      let successCount = 0, failCount = 0;
      for (const lineup of selectedLineupObjects) {
        try { await onSaveFavorite(lineup); successCount++; }
        catch (error) { console.error('Failed to add lineup to favorites:', error); failCount++; }
      }
      if (successCount > 0) {
        alert(failCount > 0
          ? `Added ${successCount} lineup${successCount > 1 ? 's' : ''} to favorites\n${failCount} failed`
          : `Successfully added ${successCount} lineup${successCount > 1 ? 's' : ''} to favorites!`);
        setSelectedLineups(new Set());
      } else { alert('Failed to add lineups to favorites. Please try again.'); }
    } catch (error) { console.error('Error adding lineups to favorites:', error); alert('Error adding lineups to favorites. Please try again.'); }
  };

  const handleExportToDK = async () => {
    const lineupsToExport = selectedLineups.size > 0 ? filteredLineups.filter(l => selectedLineups.has(l.id)) : filteredLineups;
    if (lineupsToExport.length === 0) { alert('Please select at least one lineup to export.'); return; }
    try {
      let contestName, contestId, entryFee;
      try {
        const dkInfoResponse = await fetch('/api/dk-entries');
        if (dkInfoResponse.ok) {
          const dkInfo = await dkInfoResponse.json();
          if (dkInfo.loaded && dkInfo.contestInfo) {
            contestName = dkInfo.contestInfo.contest_name;
            contestId = dkInfo.contestInfo.contest_id;
            entryFee = dkInfo.contestInfo.entry_fee;
          }
        }
      } catch (error) { console.log('DK entries not loaded'); }
      if (!contestName || !contestId || !entryFee) {
        contestName = prompt('Enter Contest Name:', contestName || 'NBA Main Slate') || contestName || 'NBA Main Slate';
        contestId = prompt('Enter Contest ID:', contestId || '999999999') || contestId || '999999999';
        entryFee = prompt('Enter Entry Fee:', entryFee || '$1') || entryFee || '$1';
      }
      const response = await fetch('/api/export-dk-entries', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lineups: lineupsToExport, type: 'lineups', contestName, contestId, entryFee })
      });
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `dk_entries_${Date.now()}.csv`;
        document.body.appendChild(a); a.click();
        window.URL.revokeObjectURL(url); document.body.removeChild(a);
        alert(`Successfully exported ${lineupsToExport.length} lineup${lineupsToExport.length > 1 ? 's' : ''} to DraftKings format!`);
      } else {
        const error = await response.json();
        alert(`Export failed: ${error.error || 'Unknown error'}`);
      }
    } catch (error) { console.error('Export error:', error); alert('Error exporting lineups. Please try again.'); }
  };

  const filteredLineups = lineups
    .filter(lineup => {
      if (filterBy === 'all') return true;
      if (filterBy === 'high-projection') return lineup.totalProjection > 120;
      if (filterBy === 'value-plays') return lineup.value > 2.5;
      if (filterBy === 'stacked') return (lineup.stacks?.length ?? 0) > 0;
      return true;
    })
    .filter(lineup => {
      if (!searchTerm) return true;
      const q = searchTerm.toLowerCase();
      return lineup.players.some(p => p.name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q));
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'projection': return b.totalProjection - a.totalProjection;
        case 'value': return b.value - a.value;
        case 'salary': return b.totalSalary - a.totalSalary;
        case 'strategy': return a.strategy.localeCompare(b.strategy);
        default: return 0;
      }
    })
    .slice(0, maxLineupsToShow);

  const selectAllLineups = () => { setSelectedLineups(new Set(filteredLineups.map(l => l.id))); };

  const getPositionRequirements = () => {
    if (sport === 'NFL') return { 'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1 };
    return { 'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3 };
  };

  const validateLineup = (lineup: Lineup) => {
    const requirements = getPositionRequirements();
    const positionCounts: Record<string, number> = {};
    lineup.players.forEach(player => { positionCounts[player.position] = (positionCounts[player.position] || 0) + 1; });
    for (const [pos, required] of Object.entries(requirements)) {
      if ((positionCounts[pos] || 0) < required) return { valid: false, missing: `${pos} (need ${required}, have ${positionCounts[pos] || 0})` };
    }
    return { valid: true };
  };

  /* ── Shared inline styles ─────────────────────────────────── */
  const monoFont: React.CSSProperties = { fontFamily: "'JetBrains Mono', monospace", fontVariantNumeric: 'tabular-nums' };
  const inlineBtn = (borderColor: string, bgColor: string, textColor: string): React.CSSProperties => ({
    height: 22, fontSize: 8, padding: '0 8px', borderRadius: 3,
    border: `1px solid ${borderColor}`, background: bgColor, color: textColor,
    cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3,
    ...monoFont, fontWeight: 600, whiteSpace: 'nowrap',
  });

  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ width: 48, height: 48, margin: '0 auto 12px', border: '2px solid #00D9FF', borderTop: '2px solid transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
          <h3 style={{ fontSize: 12, fontWeight: 600, color: '#E2E8F0', marginBottom: 4 }}>Generating Lineups</h3>
          <p style={{ fontSize: 9, color: 'var(--dfs-text-muted)' }}>Optimizing...</p>
        </div>
      </div>
    );
  }

  if (lineups.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ width: 64, height: 64, margin: '0 auto 16px', background: 'var(--dfs-bg-secondary)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Trophy style={{ width: 32, height: 32, color: '#64748b' }} />
          </div>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: '#E2E8F0', marginBottom: 8 }}>No Lineups Generated</h3>
          <p style={{ fontSize: 10, color: 'var(--dfs-text-muted)', marginBottom: 16 }}>Run optimization to generate lineups</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Controls bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
        padding: '4px 8px', flexShrink: 0,
        background: 'var(--dfs-bg-secondary)',
        borderBottom: '1px solid rgba(45,212,191,0.25)',
      }}>
        {/* Sort */}
        <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Sort</span>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          style={{
            height: 22, fontSize: 9, ...monoFont,
            background: 'var(--dfs-bg-primary)', color: '#CBD5E1',
            border: '1px solid rgba(255,255,255,0.1)', borderRadius: 3,
            padding: '0 4px', outline: 'none',
          }}
        >
          <option value="projection">Projection</option>
          <option value="value">Value</option>
          <option value="salary">Salary</option>
          <option value="strategy">Strategy</option>
        </select>

        {/* Filter */}
        <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Filter</span>
        <select
          value={filterBy}
          onChange={(e) => setFilterBy(e.target.value)}
          style={{
            height: 22, fontSize: 9, ...monoFont,
            background: 'var(--dfs-bg-primary)', color: '#CBD5E1',
            border: '1px solid rgba(255,255,255,0.1)', borderRadius: 3,
            padding: '0 4px', outline: 'none',
          }}
        >
          <option value="all">All Lineups</option>
          <option value="high-projection">High Projection</option>
          <option value="value-plays">Value Plays</option>
          <option value="stacked">Stacked</option>
        </select>

        {/* Search */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 3, position: 'relative' }}>
          <Search style={{ width: 10, height: 10, color: '#5A6A7E' }} />
          <input
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              width: 100, height: 22, fontSize: 9, ...monoFont,
              background: 'transparent', color: '#CBD5E1',
              border: '1px solid rgba(255,255,255,0.1)', borderRadius: 3,
              padding: '0 6px', outline: 'none',
            }}
            onFocus={(e) => { e.target.style.borderColor = 'rgba(0,217,255,0.4)'; }}
            onBlur={(e) => { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; }}
          />
        </div>

        {/* Show count */}
        <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Show</span>
        <input
          type="number" min={1} max={lineups.length}
          value={maxLineupsToShow}
          onChange={(e) => { const val = parseInt(e.target.value) || lineups.length; setMaxLineupsToShow(Math.min(Math.max(1, val), lineups.length)); }}
          style={{
            width: 40, height: 22, fontSize: 9, ...monoFont,
            background: 'transparent', color: '#CBD5E1',
            border: '1px solid rgba(255,255,255,0.1)', borderRadius: 3,
            padding: '0 3px', outline: 'none', textAlign: 'center',
          }}
        />
        <span style={{ fontSize: 8, color: '#5A6A7E' }}>of {lineups.length}</span>

        <div style={{ flex: 1 }} />

        {/* Action buttons */}
        <button onClick={selectAllLineups} style={inlineBtn('rgba(34,197,94,0.3)', 'rgba(34,197,94,0.06)', '#4ade80')}>ALL</button>
        <button onClick={deselectAllLineups} style={inlineBtn('rgba(239,68,68,0.3)', 'rgba(239,68,68,0.06)', '#f87171')}>NONE</button>
        <button
          onClick={handleAddSelectedToFavorites}
          style={{ ...inlineBtn('rgba(234,179,8,0.3)', 'rgba(234,179,8,0.06)', '#facc15'), opacity: selectedLineups.size === 0 ? 0.4 : 1, pointerEvents: selectedLineups.size === 0 ? 'none' : 'auto' }}
        >
          <Star style={{ width: 9, height: 9 }} /> FAV ({selectedLineups.size})
        </button>
        <button
          onClick={handleExportToDK}
          style={{ ...inlineBtn('rgba(34,197,94,0.3)', 'rgba(34,197,94,0.06)', '#4ade80'), opacity: filteredLineups.length === 0 ? 0.4 : 1 }}
        >
          <Download style={{ width: 9, height: 9 }} /> DK ({selectedLineups.size > 0 ? selectedLineups.size : filteredLineups.length})
        </button>
      </div>

      {/* Lineups List */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, padding: '4px 6px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 4 }}>
          {filteredLineups.map((lineup, index) => {
            const isExpanded = expandedLineups.has(lineup.id);
            const isSelected = selectedLineups.has(lineup.id);
            const validation = validateLineup(lineup);
            const lineupValue = lineup.value ?? (lineup.totalSalary > 0 ? lineup.totalProjection / lineup.totalSalary * 1000 : 0);

            return (
              <div
                key={lineup.id}
                onClick={() => toggleLineupSelection(lineup.id)}
                style={{
                  background: 'var(--dfs-bg-secondary)',
                  border: `1px solid ${isSelected ? 'rgba(0,217,255,0.3)' : 'var(--dfs-border)'}`,
                  borderRadius: 4,
                  cursor: 'pointer',
                  transition: 'border-color 0.1s',
                  boxShadow: isSelected ? '0 0 6px rgba(0,217,255,0.15)' : 'none',
                }}
              >
                {/* Card header */}
                <div style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '4px 6px', borderBottom: '1px solid var(--dfs-border)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <div onClick={(e) => e.stopPropagation()}>
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={(checked) => {
                          if (checked) setSelectedLineups(new Set([...selectedLineups, lineup.id]));
                          else { const ns = new Set(selectedLineups); ns.delete(lineup.id); setSelectedLineups(ns); }
                        }}
                        className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
                        style={{ width: 12, height: 12 }}
                      />
                    </div>
                    <span style={{
                      fontSize: 8, fontWeight: 700, ...monoFont,
                      background: 'rgba(255,255,255,0.06)', color: '#8899AA',
                      padding: '1px 5px', borderRadius: 2,
                    }}>#{index + 1}</span>
                    <span style={{
                      fontSize: 7, fontWeight: 600, padding: '1px 4px', borderRadius: 2,
                      background: validation.valid ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                      color: validation.valid ? '#4ade80' : '#f87171',
                    }}>{validation.valid ? 'Valid' : 'Invalid'}</span>
                    <span style={{
                      fontSize: 7, color: '#5A6A7E', ...monoFont,
                    }}>{lineup.strategy}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                    {(lineup.stacks?.length ?? 0) > 0 && (
                      <span style={{ fontSize: 7, color: '#00D9FF', ...monoFont }}>
                        <Target style={{ width: 8, height: 8, display: 'inline', verticalAlign: 'middle' }} /> {lineup.stacks.length}stk
                      </span>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); onSaveFavorite(lineup); }}
                      style={{ padding: 2, background: 'transparent', border: 'none', color: '#5A6A7E', cursor: 'pointer' }}
                    >
                      <Star style={{ width: 10, height: 10 }} />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); toggleLineupExpansion(lineup.id); }}
                      style={{ padding: 2, background: 'transparent', border: 'none', color: '#5A6A7E', cursor: 'pointer' }}
                    >
                      {isExpanded ? <ChevronUp style={{ width: 10, height: 10 }} /> : <ChevronDown style={{ width: 10, height: 10 }} />}
                    </button>
                  </div>
                </div>

                {/* Stats row */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', padding: '3px 6px', borderBottom: '1px solid var(--dfs-border)' }}>
                  {[
                    { label: 'PROJ', value: lineup.totalProjection.toFixed(1), color: '#00D9FF' },
                    { label: 'SAL', value: `$${lineup.totalSalary.toLocaleString()}`, color: '#E2E8F0' },
                    { label: 'VAL', value: lineupValue.toFixed(2), color: '#4ade80' },
                  ].map(stat => (
                    <div key={stat.label} style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 6, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{stat.label}</div>
                      <div style={{ fontSize: 10, fontWeight: 700, ...monoFont, color: stat.color }}>{stat.value}</div>
                    </div>
                  ))}
                </div>

                {/* Players — collapsed (first 4) */}
                {!isExpanded && (
                  <div style={{ padding: '2px 0' }}>
                    {lineup.players.slice(0, 4).map((player) => (
                      <div key={player.id} style={{
                        display: 'grid', gridTemplateColumns: PLAYER_GRID,
                        height: 22, alignItems: 'center', padding: '0 6px',
                        ...monoFont, fontSize: '8.5px',
                      }}>
                        <div style={{ color: '#5A6A7E', fontSize: 7 }}>{player.position}</div>
                        <div style={{ color: '#E2E8F0', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{player.name}</div>
                        <div style={{ textAlign: 'right', color: '#00D9FF', fontWeight: 500 }}>{player.projection.toFixed(1)}</div>
                        <div style={{ textAlign: 'right', color: '#8899AA' }}>${player.salary.toLocaleString()}</div>
                        <div style={{ textAlign: 'right', color: '#4ade80', fontSize: 7 }}>{(player.projection / player.salary * 1000).toFixed(1)}</div>
                      </div>
                    ))}
                    {lineup.players.length > 4 && (
                      <div style={{ fontSize: 7, color: '#475569', textAlign: 'center', padding: '1px 0' }}>
                        +{lineup.players.length - 4} more
                      </div>
                    )}
                  </div>
                )}

                {/* Players — expanded (all) */}
                {isExpanded && (
                  <div style={{ padding: '2px 0' }}>
                    {/* Mini header */}
                    <div style={{
                      display: 'grid', gridTemplateColumns: PLAYER_GRID,
                      height: 18, alignItems: 'center', padding: '0 6px',
                      fontSize: 6, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', letterSpacing: '0.3px',
                    }}>
                      <div>POS</div>
                      <div>PLAYER</div>
                      <div style={{ textAlign: 'right' }}>PROJ</div>
                      <div style={{ textAlign: 'right' }}>SAL</div>
                      <div style={{ textAlign: 'right' }}>VAL</div>
                    </div>
                    {lineup.players.map((player) => (
                      <div key={player.id} style={{
                        display: 'grid', gridTemplateColumns: PLAYER_GRID,
                        height: 22, alignItems: 'center', padding: '0 6px',
                        borderBottom: '1px solid var(--dfs-border)',
                        ...monoFont, fontSize: '8.5px',
                      }}>
                        <div style={{ color: '#5A6A7E', fontSize: 7, fontWeight: 600 }}>{player.position}</div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 3, overflow: 'hidden' }}>
                          <TeamLogo team={player.team} sport={sport} size={12} />
                          <span style={{ color: '#E2E8F0', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{player.name}</span>
                        </div>
                        <div style={{ textAlign: 'right', color: '#00D9FF', fontWeight: 500 }}>{player.projection.toFixed(1)}</div>
                        <div style={{ textAlign: 'right', color: '#8899AA' }}>${player.salary.toLocaleString()}</div>
                        <div style={{ textAlign: 'right', color: '#4ade80', fontSize: 7 }}>{(player.projection / player.salary * 1000).toFixed(1)}</div>
                      </div>
                    ))}

                    {/* Stacks detail */}
                    {(lineup.stacks?.length ?? 0) > 0 && (
                      <div style={{ padding: '4px 6px', borderTop: '1px solid var(--dfs-border)' }}>
                        <div style={{ fontSize: 6, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', letterSpacing: '0.3px', marginBottom: 3 }}>Stacks</div>
                        {lineup.stacks.map((stack, idx) => (
                          <div key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 18, ...monoFont, fontSize: '8.5px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                              <TeamLogo team={stack.team} sport={sport} size={12} />
                              <span style={{ color: '#E2E8F0' }}>{stack.team}</span>
                              <span style={{ color: '#5A6A7E', fontSize: 7 }}>({stack.players} plrs)</span>
                            </div>
                            <span style={{
                              fontSize: 7, padding: '1px 4px', borderRadius: 2,
                              background: 'rgba(255,255,255,0.06)', color: '#8899AA',
                            }}>{stack.type}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer bar */}
      <div style={{
        height: 22,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 10px',
        fontSize: 8,
        color: 'var(--dfs-text-muted, #5A6A7E)',
        borderTop: '1px solid var(--dfs-border)',
        flexShrink: 0,
        ...monoFont,
      }}>
        <span>
          <span style={{ color: '#00D9FF', fontWeight: 600 }}>{filteredLineups.length}</span> / {lineups.length} lineups
          {selectedLineups.size > 0 && (
            <span style={{ color: '#5A6A7E' }}> ({selectedLineups.size} selected)</span>
          )}
        </span>
        <span>
          Avg Proj: <span style={{ color: '#00D9FF', fontWeight: 600 }}>
            {(lineups.reduce((sum, l) => sum + l.totalProjection, 0) / lineups.length).toFixed(1)}
          </span>
          {' | '}
          Avg Val: <span style={{ color: '#4ade80', fontWeight: 600 }}>
            {(lineups.reduce((sum, l) => sum + l.value, 0) / lineups.length).toFixed(2)}
          </span>
        </span>
      </div>
    </div>
  );
};

export default LineupsTab;
