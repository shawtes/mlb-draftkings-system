import React, { useCallback } from 'react';
import { Button } from '../ui/button';
import { Checkbox } from '../ui/checkbox';
import { Input } from '../ui/input';
import { CheckSquare, XSquare, RefreshCcw } from 'lucide-react';
import { Sport, getStackDescription } from '../sport-config';
import { StackType } from './types';

interface StackExposureTabProps {
  stackSettings: StackType[];
  sport: Sport;
  onStackSettingsChange: (settings: StackType[]) => void;
}

const StackExposureTab: React.FC<StackExposureTabProps> = ({ stackSettings, sport, onStackSettingsChange }) => {
  const toggleStackType = useCallback((id: string, event?: React.MouseEvent) => {
    if (event) event.stopPropagation();
    const updated = stackSettings.map(s => s.id === id ? { ...s, enabled: !s.enabled } : s);
    onStackSettingsChange(updated);
  }, [stackSettings, onStackSettingsChange]);

  const updateExposure = (id: string, field: 'minExp' | 'maxExp', value: number) => {
    const updated = stackSettings.map(s => {
      if (s.id === id) {
        const newValue = Math.max(0, Math.min(100, value));
        if (field === 'minExp' && newValue > s.maxExp) return { ...s, minExp: newValue, maxExp: newValue };
        else if (field === 'maxExp' && newValue < s.minExp) return { ...s, maxExp: newValue, minExp: newValue };
        return { ...s, [field]: newValue };
      }
      return s;
    });
    onStackSettingsChange(updated);
  };

  const enabledStacks = stackSettings.filter(s => s.enabled);
  const totalMinExp = enabledStacks.reduce((sum, s) => sum + s.minExp, 0);
  const totalMaxExp = enabledStacks.reduce((sum, s) => sum + s.maxExp, 0);
  const hasConflict = totalMinExp > 100;
  const hasNoSelection = enabledStacks.length === 0;
  const minRemaining = Math.max(0, 100 - totalMinExp);
  const maxHeadroom = Math.max(0, 100 - totalMaxExp);

  const handleEnableAll = () => {
    if (stackSettings.every(s => s.enabled)) return;
    onStackSettingsChange(stackSettings.map(s => ({ ...s, enabled: true })));
  };

  const handleDisableAll = () => {
    if (enabledStacks.length === 0) return;
    onStackSettingsChange(stackSettings.map(s => ({ ...s, enabled: false })));
  };

  const handleResetRanges = () => {
    onStackSettingsChange(stackSettings.map(s => ({ ...s, minExp: 0, maxExp: 100 })));
  };

  const getExposureBadges = (stack: StackType) => {
    const badges: Array<{ label: string; value: number }> = [];
    if (typeof stack.lineupExp === 'number') badges.push({ label: 'Lineup', value: stack.lineupExp });
    if (typeof stack.poolExp === 'number') badges.push({ label: 'Pool', value: stack.poolExp });
    if (typeof stack.entryExp === 'number') badges.push({ label: 'Entries', value: stack.entryExp });
    return badges;
  };

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Stack Types</h2>
          <p className="text-sm text-[var(--dfs-text-secondary)]">Configure lineup correlation structures with min/max exposure targets per stack type.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary-action" onClick={handleEnableAll} disabled={stackSettings.length === 0 || stackSettings.every(s => s.enabled)} className="gap-2 border-cyan-500/40 bg-[var(--dfs-bg-primary)]">
            <CheckSquare className="h-4 w-4" /> Enable All
          </Button>
          <Button variant="secondary-action" onClick={handleDisableAll} disabled={enabledStacks.length === 0} className="gap-2 bg-[var(--dfs-bg-primary)]">
            <XSquare className="h-4 w-4" /> Disable All
          </Button>
          <Button variant="secondary-action" onClick={handleResetRanges} disabled={stackSettings.length === 0} className="gap-2 border-amber-500/40 bg-[var(--dfs-bg-primary)]">
            <RefreshCcw className="h-4 w-4" /> Reset Ranges
          </Button>
        </div>
      </div>

      {(hasNoSelection || hasConflict) && (
        <div className="space-y-2">
          {hasNoSelection && <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">Select at least one stack to include in generated lineups.</div>}
          {hasConflict && <div className="rounded-md border border-yellow-400/40 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-100">Total minimum exposure exceeds 100%. Lower a few minimums to proceed.</div>}
        </div>
      )}

      <div className="flex-1">
        {stackSettings.length === 0 ? (
          <div className="flex h-full items-center justify-center px-6 py-10 text-sm text-[var(--dfs-text-muted)]">No stack types available for {sport}.</div>
        ) : (
          <div className="mx-auto h-full w-full max-w-4xl overflow-hidden rounded-lg border border-[var(--dfs-border)]">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-[var(--dfs-bg-secondary)]/90 backdrop-blur">
                <tr className="text-left text-[var(--dfs-text-secondary)]">
                  <th className="w-14 px-2 py-2 font-semibold">Use</th>
                  <th className="px-2 py-2 font-semibold">Stack Type</th>
                  <th className="w-20 px-2 py-2 font-semibold">Corr.</th>
                  <th className="w-28 px-2 py-2 text-right font-semibold">Min %</th>
                  <th className="w-28 px-2 py-2 text-right font-semibold">Max %</th>
                  <th className="px-2 py-2 text-right font-semibold">Live Exposure</th>
                </tr>
              </thead>
              <tbody>
                {stackSettings.map((stack, index) => {
                  const exposureBadges = getExposureBadges(stack);
                  const exposureSummary = exposureBadges.length ? exposureBadges.map(b => `${b.label}: ${b.value.toFixed(1)}%`).join(' \u2022 ') : '\u2014';
                  return (
                    <tr
                      key={stack.id}
                      onClick={(e) => { const target = e.target as HTMLElement; if (target.tagName === 'INPUT' || target.closest('input')) return; toggleStackType(stack.id, e); }}
                      className={`border-t border-[var(--dfs-border)] cursor-pointer transition-colors hover:bg-[var(--dfs-bg-hover)] ${index % 2 === 0 ? 'bg-slate-900/70' : 'bg-slate-900/50'} ${stack.enabled ? 'bg-cyan-500/10 hover:bg-cyan-500/15' : ''}`}
                    >
                      <td className="px-2 py-2" onClick={(e) => e.stopPropagation()}>
                        <Checkbox checked={stack.enabled} onCheckedChange={(checked: boolean | 'indeterminate') => { if (checked !== stack.enabled) toggleStackType(stack.id); }} className="cursor-pointer" />
                      </td>
                      <td className="px-2 py-2 align-middle">
                        <div className="flex items-center gap-1">
                          <span className={`font-medium ${stack.enabled ? 'text-white' : 'text-[var(--dfs-text-muted)]'}`}>{stack.label}</span>
                          <span className="cursor-help text-[var(--dfs-text-muted)]" title={getStackDescription(stack.label, sport)}>
                            <svg xmlns="http://www.w3.org/2000/svg" className="inline h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" /></svg>
                          </span>
                        </div>
                        <div className="text-xs text-[var(--dfs-text-muted)]">{getStackDescription(stack.label, sport)}</div>
                      </td>
                      <td className="px-2 py-2 text-center">
                        {(() => {
                          const desc = getStackDescription(stack.label, sport);
                          const corrMatch = desc.match(/r=\+?([\d.]+)/);
                          const isAvoid = desc.toLowerCase().includes('avoid') || desc.toLowerCase().includes('fade');
                          if (isAvoid) return <span className="inline-block rounded-full bg-red-500/20 px-2 py-0.5 text-[10px] font-bold text-red-400">Fade</span>;
                          if (corrMatch) {
                            const corr = parseFloat(corrMatch[1]);
                            const color = corr >= 0.35 ? 'bg-green-500/20 text-green-400' : corr >= 0.20 ? 'bg-yellow-500/20 text-yellow-400' : 'bg-blue-500/20 text-blue-400';
                            return <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-bold ${color}`}>r={corrMatch[0].replace('r=', '')}</span>;
                          }
                          return <span className="text-[10px] text-[var(--dfs-text-muted)]">{'\u2014'}</span>;
                        })()}
                      </td>
                      <td className="px-2 py-2 text-right" onClick={(e) => e.stopPropagation()}>
                        <Input type="number" min="0" max="100" value={stack.minExp} onChange={(e) => updateExposure(stack.id, 'minExp', parseInt(e.target.value) || 0)} disabled={!stack.enabled} className={`h-9 w-full text-right ${stack.enabled ? 'bg-[var(--dfs-bg-secondary)] text-white' : 'bg-[var(--dfs-bg-primary)] text-[var(--dfs-text-muted)]'}`} />
                      </td>
                      <td className="px-2 py-2 text-right" onClick={(e) => e.stopPropagation()}>
                        <Input type="number" min="0" max="100" value={stack.maxExp} onChange={(e) => updateExposure(stack.id, 'maxExp', parseInt(e.target.value) || 0)} disabled={!stack.enabled} className={`h-9 w-full text-right ${stack.enabled ? 'bg-[var(--dfs-bg-secondary)] text-white' : 'bg-[var(--dfs-bg-primary)] text-[var(--dfs-text-muted)]'}`} />
                      </td>
                      <td className="px-2 py-2 text-right text-[var(--dfs-text-secondary)]">{stack.enabled ? exposureSummary : '\u2014'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="mx-auto mt-auto flex w-full max-w-4xl flex-wrap items-center justify-between gap-3 rounded-md border border-[var(--dfs-border)] bg-[var(--dfs-bg-primary)] px-4 py-3 text-sm text-[var(--dfs-text-secondary)]">
        <span>Active stacks: <span className="font-semibold text-white">{enabledStacks.length}</span> / {stackSettings.length}</span>
        <span>Total min: <span className={`font-semibold ${hasConflict ? 'text-red-300' : 'text-white'}`}>{totalMinExp}%</span><span className="ml-3">Total max: <span className="font-semibold text-white">{totalMaxExp}%</span></span></span>
        <span className="text-xs text-[var(--dfs-text-muted)]">Min remaining: {minRemaining}% \u2022 Max headroom: {maxHeadroom}%</span>
      </div>
    </div>
  );
};

export default StackExposureTab;
