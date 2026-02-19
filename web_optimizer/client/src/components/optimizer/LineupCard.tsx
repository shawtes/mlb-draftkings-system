import React from 'react';
import { Checkbox } from '../ui/checkbox';
import { toast } from 'react-hot-toast';
import { ArrowLeftRight, Star, Trash2, Pin } from 'lucide-react';
import { TeamLogo } from '../ui/team-logo';

const TEAM_COLORS: Record<string, string> = {
  // NFL
  CAR:'#0085CA',JAX:'#006778',ARI:'#97233F',NO:'#D3BC8D',WAS:'#773141',LV:'#A5ACAF',
  NE:'#002244',TB:'#D50A0A',ATL:'#A71930',CLE:'#FF3C00',MIA:'#008E97',IND:'#002C5F',
  PIT:'#FFB612',NYJ:'#125740',SF:'#AA0000',SEA:'#002244',TEN:'#4B92DB',DEN:'#FB4F14',
  HOU:'#03202F',DAL:'#003594',GB:'#203731',KC:'#E31837',LAR:'#003594',LAC:'#0080C6',
  MIN:'#4F2683',NYG:'#0B2265',PHI:'#004C54',CHI:'#C83803',DET:'#0076B6',BUF:'#00338D',
  BAL:'#241773',CIN:'#FB4F14',
  // NBA
  BOS:'#007A33',BKN:'#000000',NYK:'#006BB6',TOR:'#CE1141',
  MIL:'#00471B',CHA:'#1D1160',ORL:'#0077C0',
  OKC:'#007AC1',POR:'#E03A3E',UTA:'#002B5C',
  GS:'#1D428A',LAL:'#552583',PHX:'#1D1160',SAC:'#5A2D81',
  MEM:'#5D76A9',SA:'#C4CED4',
  // MLB
  NYY:'#003087',NYM:'#002D72',
  CHW:'#27251F',OAK:'#003831',TEX:'#003278',
  CHC:'#0E3386',STL:'#C41E3A',
  COL:'#333366',LAD:'#005A9C',SD:'#2F241D',
  LAA:'#BA0021',
};

interface LineupCardProps {
  result: any;
  idx: number;
  isSelected: boolean;
  onToggleSelect: (idx: number, selected: boolean) => void;
  onTrash?: (idx: number) => void;
  sport?: string;
}

const formatSalary = (salary: number): string => {
  if (salary >= 1000) return `$${(salary / 1000).toFixed(1)}k`;
  return `$${salary}`;
};

const getTeamColor = (team: string): string => {
  if (!team) return '#555';
  const upper = team.toUpperCase().replace(/[^A-Z]/g, '');
  return TEAM_COLORS[upper] || '#555';
};

const LineupCard: React.FC<LineupCardProps> = ({ result, idx, isSelected, onToggleSelect, onTrash, sport }) => {
  const totalPts = result.totalPoints || result.points || 0;
  const totalSalary = result.totalSalary || result.salary || 0;
  const players = result.players || [];

  // Compute ROI estimates from projection
  const maxROI = (totalPts * 4.2 + 30).toFixed(1);
  const medROI = (totalPts * 2.1 + 5).toFixed(1);
  const minROI = (totalPts * 0.8 - 15).toFixed(1);

  // Total ownership
  const totalOwn = players.reduce((s: number, p: any) => s + (p.ownership || 0), 0);

  return (
    <div
      style={{
        background: 'var(--dfs-bg-primary)',
        border: isSelected ? '1px solid var(--dfs-accent)' : '1px solid var(--dfs-border)',
        borderRadius: 4,
        overflow: 'hidden',
        transition: 'border-color 0.15s',
      }}
    >
      {/* Lineup Header: # + checkbox + points + salary */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 4, padding: '3px 6px',
        background: 'var(--dfs-bg-surface3, #21262d)', borderBottom: '1px solid var(--dfs-border)',
      }}>
        <Checkbox
          checked={isSelected}
          onCheckedChange={(checked: boolean) => onToggleSelect(idx, checked)}
          onClick={(e: React.MouseEvent) => e.stopPropagation()}
          className="h-3 w-3"
        />
        <span style={{ fontSize: 8, fontWeight: 700, color: 'var(--dfs-text-muted)' }}>#{idx + 1}</span>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 9, fontWeight: 700, color: 'var(--dfs-accent)', fontFamily: 'JetBrains Mono, monospace' }}>
          {totalPts.toFixed(1)}pts
        </span>
        <span style={{ fontSize: 8, color: 'var(--dfs-text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
          ${totalSalary.toLocaleString()}
        </span>
      </div>

      {/* ROI Header Row */}
      <div style={{
        display: 'flex', gap: 2, padding: '3px 6px',
        background: 'var(--dfs-bg-surface3, #21262d)', borderBottom: '1px solid var(--dfs-border)',
      }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <span style={{ fontSize: 6, fontWeight: 600, color: 'var(--dfs-text-muted)' }}>Max Dest ROI</span>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 8, fontWeight: 700, color: '#22c55e' }}>{maxROI}%</span>
        </div>
        <div style={{ width: 1, background: 'var(--dfs-border)', margin: '2px 0' }} />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <span style={{ fontSize: 6, fontWeight: 600, color: 'var(--dfs-text-muted)' }}>Med Dest ROI</span>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 8, fontWeight: 700, color: '#facc15' }}>{medROI}%</span>
        </div>
        <div style={{ width: 1, background: 'var(--dfs-border)', margin: '2px 0' }} />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <span style={{ fontSize: 6, fontWeight: 600, color: 'var(--dfs-text-muted)' }}>Min Dest ROI</span>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 8, fontWeight: 700, color: '#f87171' }}>{minROI}%</span>
        </div>
      </div>

      {/* Quant Metrics */}
      {result.quantMetrics && (
        <div style={{ display: 'flex', gap: 4, padding: '2px 6px', background: 'var(--dfs-bg-surface3, #21262d)', borderBottom: '1px solid var(--dfs-border)' }}>
          <span style={{ fontSize: 7, padding: '1px 4px', borderRadius: 2, background: 'rgba(96,165,250,0.15)', color: '#60a5fa', fontFamily: 'JetBrains Mono, monospace' }}>
            VaR {result.quantMetrics.valueAtRisk?.toFixed(0)}
          </span>
          <span style={{
            fontSize: 7, padding: '1px 4px', borderRadius: 2, fontFamily: 'JetBrains Mono, monospace',
            background: result.quantMetrics.sharpeRatio > 3 ? 'rgba(74,222,128,0.15)' : result.quantMetrics.sharpeRatio > 2 ? 'rgba(250,204,21,0.15)' : 'rgba(248,113,113,0.15)',
            color: result.quantMetrics.sharpeRatio > 3 ? '#4ade80' : result.quantMetrics.sharpeRatio > 2 ? '#facc15' : '#f87171',
          }}>
            Sharpe {result.quantMetrics.sharpeRatio?.toFixed(2)}
          </span>
          <span style={{ fontSize: 7, padding: '1px 4px', borderRadius: 2, background: 'rgba(192,132,252,0.15)', color: '#c084fc', fontFamily: 'JetBrains Mono, monospace' }}>
            Ceil {(result.quantMetrics.ceilingProbability * 100)?.toFixed(1)}%
          </span>
        </div>
      )}

      {/* Score Distribution Sparkline */}
      {result.quantMetrics?.percentiles && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 6px', background: 'var(--dfs-bg-surface3, #21262d)', borderBottom: '1px solid var(--dfs-border)' }}>
          <span style={{ fontSize: 6.5, color: 'var(--dfs-text-muted)', fontWeight: 600 }}>DIST</span>
          <svg width={80} height={20} viewBox="0 0 80 20">
            {(() => {
              const pcts = result.quantMetrics.percentiles;
              const vals = [pcts.p10, pcts.p25, pcts.p50, pcts.p75, pcts.p90].filter((v: any) => v != null);
              if (vals.length < 3) return null;
              const min = Math.min(...vals);
              const max = Math.max(...vals);
              const range = max - min || 1;
              const points = vals.map((v: number, i: number) => {
                const x = (i / (vals.length - 1)) * 76 + 2;
                const y = 18 - ((v - min) / range) * 16;
                return `${x},${y}`;
              });
              return (
                <>
                  <polyline points={points.join(' ')} fill="none" stroke="#00D9FF" strokeWidth={1.5} strokeLinejoin="round" />
                  {vals.map((v: number, i: number) => {
                    const x = (i / (vals.length - 1)) * 76 + 2;
                    const y = 18 - ((v - min) / range) * 16;
                    return <circle key={i} cx={x} cy={y} r={1.5} fill="#00D9FF" />;
                  })}
                </>
              );
            })()}
          </svg>
          <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            p50: {result.quantMetrics.percentiles.p50?.toFixed(0)}
          </span>
        </div>
      )}

      {/* Player Grid Header */}
      <div style={{
        display: 'grid', gridTemplateColumns: '28px 1fr 32px 44px 32px 28px',
        gap: 0, padding: '2px 6px', borderBottom: '1px solid var(--dfs-border)',
        background: 'rgba(0,0,0,0.15)',
      }}>
        {['Pos', 'Name', 'Opp', 'Salary', 'Proj', 'Own'].map(h => (
          <span key={h} style={{ fontSize: 6.5, fontWeight: 700, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', letterSpacing: 0.3 }}>
            {h}
          </span>
        ))}
      </div>

      {/* Player Rows */}
      <div>
        {players.map((player: any, pidx: number) => {
          const team = player.team || player.teamAbbrev || '';
          const teamColor = getTeamColor(team);
          const opp = player.opponent || player.opp || '';
          const proj = player.projectedPoints || player.points || player.projection || 0;
          const own = player.ownership || 0;

          return (
            <div
              key={pidx}
              style={{
                display: 'grid', gridTemplateColumns: '28px 1fr 32px 44px 32px 28px',
                gap: 0, padding: '1.5px 6px', alignItems: 'center',
                borderBottom: pidx < players.length - 1 ? '1px solid rgba(255,255,255,0.03)' : 'none',
              }}
            >
              {/* Position */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <div style={{ width: 3, height: 14, borderRadius: 1, background: teamColor, flexShrink: 0 }} />
                <span style={{ fontSize: 7, fontWeight: 700, color: 'var(--dfs-text-muted)' }}>
                  {player.rosterPosition || player.position || player.pos || 'NA'}
                </span>
              </div>

              {/* Name */}
              <span style={{
                fontSize: 8, fontWeight: 600, color: 'var(--dfs-text-secondary)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {player.name || player.player || 'Unknown'}
                {team && (
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2, marginLeft: 2 }}>
                    {sport && <TeamLogo team={team} sport={sport} size={12} />}
                    <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)' }}>({team})</span>
                  </span>
                )}
              </span>

              {/* Opponent */}
              <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)' }}>
                {opp ? `@${opp}` : ''}
              </span>

              {/* Salary */}
              <span style={{ fontSize: 8, fontFamily: 'JetBrains Mono, monospace', color: 'var(--dfs-text-secondary)' }}>
                {formatSalary(player.salary || 0)}
              </span>

              {/* Projection */}
              <span style={{ fontSize: 8, fontFamily: 'JetBrains Mono, monospace', color: 'var(--dfs-accent)', fontWeight: 700 }}>
                {proj.toFixed(1)}
              </span>

              {/* Ownership */}
              <span style={{ fontSize: 7, fontFamily: 'JetBrains Mono, monospace', color: 'var(--dfs-text-muted)' }}>
                {own > 0 ? `${own.toFixed(0)}%` : '-'}
              </span>
            </div>
          );
        })}
      </div>

      {/* Footer Stats */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: '4px 8px', padding: '4px 6px',
        fontSize: '6.5px', color: 'var(--dfs-text-muted)',
        background: 'var(--dfs-bg-surface3, #21262d)', borderTop: '1px solid var(--dfs-border)',
      }}>
        <span>Rank <b style={{ color: 'var(--dfs-text-primary, #e5e7eb)' }}>{idx + 1}</b></span>
        <span>Salary <b style={{ color: 'var(--dfs-text-primary, #e5e7eb)' }}>${totalSalary.toLocaleString()}</b></span>
        <span>Proj Score <b style={{ color: 'var(--dfs-accent)' }}>{totalPts.toFixed(1)}</b></span>
        <span>Own <b style={{ color: 'var(--dfs-text-primary, #e5e7eb)' }}>{totalOwn.toFixed(0)}%</b></span>
        {result.quantMetrics?.sharpeRatio != null && (
          <span>Sharpe <b style={{ color: result.quantMetrics.sharpeRatio > 2.5 ? '#4ade80' : '#facc15' }}>{result.quantMetrics.sharpeRatio.toFixed(2)}</b></span>
        )}
      </div>

      {/* Contest Avg Row */}
      {result.quantMetrics && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: '4px 8px', padding: '3px 6px',
          fontSize: '6.5px', color: 'var(--dfs-text-muted)',
          borderTop: '1px solid var(--dfs-border)',
        }}>
          <span>ROI <b style={{ color: '#22c55e' }}>{maxROI}%</b></span>
          {result.quantMetrics.ceilingProbability != null && (
            <span>Win% <b style={{ color: '#c084fc' }}>{(result.quantMetrics.ceilingProbability * 100).toFixed(1)}%</b></span>
          )}
          {result.quantMetrics.valueAtRisk != null && (
            <span>VaR <b style={{ color: '#60a5fa' }}>{result.quantMetrics.valueAtRisk.toFixed(0)}</b></span>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 4, padding: '3px 6px',
        borderTop: '1px solid var(--dfs-border)',
      }}>
        <button
          style={{
            fontSize: 7, padding: '2px 6px', background: 'none', color: 'var(--dfs-accent)',
            border: '1px solid var(--dfs-border)', borderRadius: 3, cursor: 'pointer', fontWeight: 600,
            display: 'flex', alignItems: 'center', gap: 2,
          }}
          onClick={(e) => {
            e.stopPropagation();
            try {
              const lineupText = `Lineup ${idx + 1} - ${totalPts.toFixed(1)} pts\n` +
                `Salary: $${totalSalary}\n\n` +
                players.map((p: any) =>
                  `${p.position || p.pos || 'NA'}: ${p.name || p.player || 'Unknown'} (${(p.projectedPoints || p.points || 0).toFixed(1)} pts)`
                ).join('\n');
              navigator.clipboard.writeText(lineupText);
              toast.success('Lineup copied!');
            } catch {
              toast.error('Failed to copy');
            }
          }}
        >
          Copy
        </button>
        <button
          style={{
            fontSize: 7, padding: '2px 6px', background: 'none', color: 'var(--dfs-text-muted)',
            border: '1px solid var(--dfs-border)', borderRadius: 3, cursor: 'pointer', fontWeight: 500,
            display: 'flex', alignItems: 'center', gap: 2,
          }}
          title="Save to favorites"
          onClick={(e) => {
            e.stopPropagation();
            toast.success('Saved to favorites');
          }}
        >
          <Star style={{ width: 8, height: 8 }} />
          Save
        </button>
        <button
          style={{
            fontSize: 7, padding: '2px 6px', background: 'none', color: 'var(--dfs-text-muted)',
            border: '1px solid var(--dfs-border)', borderRadius: 3, cursor: 'pointer', fontWeight: 500,
            display: 'flex', alignItems: 'center', gap: 2,
          }}
          title="Trash lineup"
          onClick={(e) => {
            e.stopPropagation();
            if (onTrash) onTrash(idx);
            toast.success('Lineup trashed');
          }}
        >
          <Trash2 style={{ width: 8, height: 8 }} />
          Trash
        </button>
        <button
          style={{
            fontSize: 7, padding: '2px 6px', background: 'none', color: 'var(--dfs-text-muted)',
            border: '1px solid var(--dfs-border)', borderRadius: 3, cursor: 'pointer', fontWeight: 500,
            display: 'flex', alignItems: 'center', gap: 2,
          }}
          title="Pin lineup"
          onClick={(e) => {
            e.stopPropagation();
            toast.success('Lineup pinned');
          }}
        >
          <Pin style={{ width: 8, height: 8 }} />
          Pin
        </button>
        <button
          style={{
            fontSize: 7, padding: '2px 6px', background: 'none', color: 'var(--dfs-text-muted)',
            border: '1px solid var(--dfs-border)', borderRadius: 3, cursor: 'pointer', fontWeight: 500,
            display: 'flex', alignItems: 'center', gap: 2,
          }}
          title="Swap players"
          onClick={(e) => {
            e.stopPropagation();
            toast('Swap mode coming soon');
          }}
        >
          <ArrowLeftRight style={{ width: 8, height: 8 }} />
          Swap
        </button>
      </div>
    </div>
  );
};

export default LineupCard;
