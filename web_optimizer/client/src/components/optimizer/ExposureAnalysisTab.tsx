import React, { useMemo } from 'react';
import { Player } from './types';

interface ExposureAnalysisTabProps {
  playerData: Player[];
  results: any[];
}

const ExposureAnalysisTab: React.FC<ExposureAnalysisTabProps> = ({ playerData, results }) => {
  const exposureData = useMemo(() => {
    if (results.length === 0) return [];

    const totalLineups = results.length;
    const playerCounts = new Map<string, number>();
    results.forEach((r: any) => {
      (r.players || []).forEach((p: any) => {
        const name = (p.name || '').toLowerCase();
        playerCounts.set(name, (playerCounts.get(name) || 0) + 1);
      });
    });

    return playerData
      .filter(p => p.minExp > 0 || p.maxExp < 100 || (playerCounts.get(p.name.toLowerCase()) || 0) > 0)
      .map(p => {
        const count = playerCounts.get(p.name.toLowerCase()) || 0;
        const actual = parseFloat(((count / totalLineups) * 100).toFixed(1));
        const inRange = actual >= p.minExp && actual <= p.maxExp;
        const delta = actual < p.minExp ? actual - p.minExp : actual > p.maxExp ? actual - p.maxExp : 0;
        return { ...p, actual, inRange, delta };
      })
      .sort((a, b) => b.actual - a.actual);
  }, [playerData, results]);

  // Summary stats
  const summary = useMemo(() => {
    if (exposureData.length === 0) return null;
    const actuals = exposureData.map(e => e.actual).filter(a => a > 0);
    if (actuals.length === 0) return null;
    const avg = actuals.reduce((s, v) => s + v, 0) / actuals.length;
    const max = Math.max(...actuals);
    const sumSq = actuals.reduce((s, v) => s + (v / 100) ** 2, 0);
    const hhi = parseFloat(sumSq.toFixed(4));
    const violations = exposureData.filter(e => !e.inRange && e.actual > 0).length;
    return { avg, max, hhi, violations, total: actuals.length };
  }, [exposureData]);

  if (results.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', padding: 32 }}>
        <p style={{ fontSize: 10, color: 'var(--dfs-text-muted)' }}>Run optimization to see exposure analysis.</p>
      </div>
    );
  }

  const GRID = '1fr 50px 50px 50px 50px';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Summary row */}
      {summary && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, padding: 8, borderBottom: '1px solid var(--dfs-border)' }}>
          {[
            { label: 'Avg Exposure', value: `${summary.avg.toFixed(1)}%`, color: 'var(--dfs-accent)' },
            { label: 'Max Exposure', value: `${summary.max.toFixed(1)}%`, color: summary.max > 60 ? '#f87171' : '#4ade80' },
            { label: 'HHI Concentration', value: summary.hhi.toFixed(4), color: summary.hhi < 0.1 ? '#4ade80' : summary.hhi < 0.2 ? '#facc15' : '#f87171' },
            { label: 'Violations', value: String(summary.violations), color: summary.violations === 0 ? '#4ade80' : '#f87171' },
          ].map((m, i) => (
            <div key={i} style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 7, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{m.label}</div>
              <div style={{ fontSize: 12, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: m.color }}>{m.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Table header */}
      <div style={{ display: 'grid', gridTemplateColumns: GRID, height: 26, alignItems: 'center', padding: '0 8px', background: 'var(--dfs-bg-secondary)', borderBottom: '1px solid rgba(45,212,191,0.25)', fontSize: 7, textTransform: 'uppercase', letterSpacing: '0.3px', color: 'var(--dfs-text-muted)', fontWeight: 600, flexShrink: 0, userSelect: 'none' }}>
        <div>Player</div>
        <div style={{ textAlign: 'right' }}>Min</div>
        <div style={{ textAlign: 'right' }}>Max</div>
        <div style={{ textAlign: 'right' }}>Actual</div>
        <div style={{ textAlign: 'right' }}>Delta</div>
      </div>

      {/* Table body */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {exposureData.map(row => (
          <div
            key={row.id}
            style={{
              display: 'grid', gridTemplateColumns: GRID, height: 28, alignItems: 'center', padding: '0 8px',
              borderBottom: '1px solid var(--dfs-border)',
              background: !row.inRange && row.actual > 0 ? 'rgba(239,68,68,0.06)' : 'transparent',
            }}
          >
            <div style={{ fontSize: 9, fontWeight: 500, color: 'var(--dfs-text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {row.name}
              <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', marginLeft: 4 }}>{row.position} {row.team}</span>
            </div>
            <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: 'var(--dfs-text-secondary)' }}>{row.minExp}%</div>
            <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: 'var(--dfs-text-secondary)' }}>{row.maxExp}%</div>
            <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: 8, fontWeight: 600, color: row.inRange ? '#4ade80' : '#f87171' }}>{row.actual.toFixed(1)}%</div>
            <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: 8, fontWeight: 600, color: row.delta === 0 ? 'var(--dfs-text-muted)' : row.delta > 0 ? '#f87171' : '#60a5fa' }}>
              {row.delta === 0 ? '\u2014' : `${row.delta > 0 ? '+' : ''}${row.delta.toFixed(1)}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExposureAnalysisTab;
