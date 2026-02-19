import React, { useCallback, useMemo, useState, useRef, useEffect } from 'react';
import { Checkbox } from '../ui/checkbox';
import { CheckSquare, XSquare, RefreshCcw } from 'lucide-react';
import { Sport, SPORT_CONFIGS, StackTypeDefinition } from '../sport-config';
import { StackType } from './types';

/* ── Debounced numeric input (mirrors PlayerTable pattern) ──── */
const DebouncedInput: React.FC<{
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  disabled?: boolean;
}> = React.memo(({ value, onChange, min = 0, max = 100, disabled }) => {
  const [local, setLocal] = useState(String(value));
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => { setLocal(String(value)); }, [value]);

  const flush = useCallback((v: string) => {
    const n = Math.max(min, Math.min(max, parseInt(v) || 0));
    if (n !== value) onChange(n);
  }, [value, onChange, min, max]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value;
    setLocal(v);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => flush(v), 500);
  }, [flush]);

  const handleBlur = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    flush(local);
  }, [flush, local]);

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  return (
    <input
      type="number" min={min} max={max}
      value={local}
      onChange={handleChange}
      onBlur={handleBlur}
      disabled={disabled}
      style={{
        width: 38, height: 20, textAlign: 'right' as const,
        fontSize: '8.5px', fontVariantNumeric: 'tabular-nums', fontFamily: "'JetBrains Mono', monospace",
        background: 'transparent', color: disabled ? '#475569' : '#CBD5E1',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 3, outline: 'none', padding: '0 3px',
        opacity: disabled ? 0.4 : 1,
      }}
      onFocus={(e) => { e.target.style.borderColor = 'rgba(0,217,255,0.4)'; e.target.style.background = 'rgba(255,255,255,0.04)'; }}
    />
  );
});

/** Correlation badge colors based on coefficient magnitude */
function getCorrelationBadge(def: StackTypeDefinition | undefined): { text: string; bg: string; fg: string } | null {
  if (!def) return null;
  if (def.category === 'fade') return { text: 'Fade', bg: 'rgba(239,68,68,0.15)', fg: '#f87171' };
  if (!def.correlationLabel || def.correlation === null) return null;
  const abs = Math.abs(def.correlation);
  if (abs >= 0.40) return { text: def.correlationLabel, bg: 'rgba(34,197,94,0.15)', fg: '#4ade80' };
  if (abs >= 0.25) return { text: def.correlationLabel, bg: 'rgba(16,185,129,0.15)', fg: '#34d399' };
  if (abs >= 0.15) return { text: def.correlationLabel, bg: 'rgba(234,179,8,0.15)', fg: '#facc15' };
  return { text: def.correlationLabel, bg: 'rgba(59,130,246,0.15)', fg: '#60a5fa' };
}

/** GPP impact badge */
function getGppBadge(def: StackTypeDefinition | undefined): { text: string; bg: string; fg: string } | null {
  if (!def) return null;
  switch (def.gppImpact) {
    case 'high': return { text: 'High GPP', bg: 'rgba(168,85,247,0.12)', fg: '#c084fc' };
    case 'moderate': return { text: 'Moderate', bg: 'rgba(100,116,139,0.12)', fg: '#94a3b8' };
    case 'low': return { text: 'Low', bg: 'rgba(71,85,105,0.12)', fg: '#64748b' };
    case 'negative': return { text: 'Risk Mgmt', bg: 'rgba(249,115,22,0.12)', fg: '#fb923c' };
    default: return null;
  }
}

/** Category label for grouping */
function getCategoryLabel(category: string): string {
  switch (category) {
    case 'primary': return 'Primary Stacks';
    case 'secondary': return 'Secondary Stacks';
    case 'game-stack': return 'Game Stacks';
    case 'structural': return 'Structural';
    case 'fade': return 'Fades';
    case 'none': return 'Other';
    default: return category;
  }
}

const ROW_GRID = '22px 1fr 38px 38px';

interface StackExposureTabProps {
  stackSettings: StackType[];
  sport: Sport;
  onStackSettingsChange: (settings: StackType[]) => void;
}

const StackExposureTab: React.FC<StackExposureTabProps> = ({ stackSettings, sport, onStackSettingsChange }) => {
  const config = SPORT_CONFIGS[sport];
  const definitions = config.stackTypeDefinitions || [];

  const defMap = useMemo(() => {
    const map = new Map<string, StackTypeDefinition>();
    definitions.forEach(d => { map.set(d.id, d); map.set(d.label, d); });
    return map;
  }, [definitions]);

  const groupedStacks = useMemo(() => {
    const groups = new Map<string, Array<{ stack: StackType; def: StackTypeDefinition | undefined }>>();
    const categoryOrder = ['primary', 'game-stack', 'secondary', 'structural', 'fade', 'none'];
    stackSettings.forEach(stack => {
      const def = defMap.get(stack.id) || defMap.get(stack.label);
      const cat = def?.category || 'none';
      if (!groups.has(cat)) groups.set(cat, []);
      groups.get(cat)!.push({ stack, def });
    });
    return categoryOrder.filter(cat => groups.has(cat)).map(cat => ({ category: cat, label: getCategoryLabel(cat), items: groups.get(cat)! }));
  }, [stackSettings, defMap]);

  const toggleStackType = useCallback((id: string, event?: React.MouseEvent) => {
    if (event) event.stopPropagation();
    onStackSettingsChange(stackSettings.map(s => s.id === id ? { ...s, enabled: !s.enabled } : s));
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

  const handleEnableAll = () => { if (!stackSettings.every(s => s.enabled)) onStackSettingsChange(stackSettings.map(s => ({ ...s, enabled: true }))); };
  const handleDisableAll = () => { if (enabledStacks.length > 0) onStackSettingsChange(stackSettings.map(s => ({ ...s, enabled: false }))); };
  const handleResetDefaults = () => { onStackSettingsChange(stackSettings.map(s => { const def = defMap.get(s.id) || defMap.get(s.label); return { ...s, minExp: 0, maxExp: 100, enabled: def?.defaultEnabled ?? false }; })); };
  const handleResetRanges = () => { onStackSettingsChange(stackSettings.map(s => ({ ...s, minExp: 0, maxExp: 100 }))); };

  /* ── Shared inline styles ─────────────────────────────────── */
  const inlineBtn = (borderColor: string, color: string, disabled?: boolean): React.CSSProperties => ({
    height: 22, fontSize: 8, padding: '0 8px', borderRadius: 3,
    border: `1px solid ${borderColor}`, background: 'transparent', color,
    cursor: disabled ? 'not-allowed' : 'pointer',
    display: 'flex', alignItems: 'center', gap: 3,
    fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
    opacity: disabled ? 0.4 : 1,
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap',
        padding: '4px 8px', flexShrink: 0, gap: 6,
        background: 'var(--dfs-bg-secondary)',
        borderBottom: '1px solid rgba(45,212,191,0.25)',
      }}>
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: '#E2E8F0', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Stack Types</div>
          <div style={{ fontSize: 8, color: 'var(--dfs-text-muted)' }}>{sport} correlation-backed lineup structures</div>
        </div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <button onClick={handleEnableAll} disabled={stackSettings.length === 0 || stackSettings.every(s => s.enabled)} style={inlineBtn('rgba(0,217,255,0.3)', '#00D9FF', stackSettings.every(s => s.enabled))}>
            <CheckSquare style={{ width: 9, height: 9 }} /> Enable All
          </button>
          <button onClick={handleDisableAll} disabled={enabledStacks.length === 0} style={inlineBtn('rgba(255,255,255,0.12)', '#CBD5E1', enabledStacks.length === 0)}>
            <XSquare style={{ width: 9, height: 9 }} /> Disable All
          </button>
          <button onClick={handleResetDefaults} disabled={stackSettings.length === 0} style={inlineBtn('rgba(234,179,8,0.3)', '#facc15', stackSettings.length === 0)}>
            <RefreshCcw style={{ width: 9, height: 9 }} /> Defaults
          </button>
          <button onClick={handleResetRanges} disabled={stackSettings.length === 0} style={inlineBtn('rgba(255,255,255,0.12)', '#CBD5E1', stackSettings.length === 0)}>
            <RefreshCcw style={{ width: 9, height: 9 }} /> Reset
          </button>
        </div>
      </div>

      {/* Warnings */}
      {(hasNoSelection || hasConflict) && (
        <div style={{ padding: '0 8px 0', flexShrink: 0 }}>
          {hasNoSelection && (
            <div style={{ padding: '3px 8px', fontSize: 8, background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 3, color: '#fca5a5', marginTop: 4 }}>
              Select at least one stack type to include in generated lineups.
            </div>
          )}
          {hasConflict && (
            <div style={{ padding: '3px 8px', fontSize: 8, background: 'rgba(234,179,8,0.06)', border: '1px solid rgba(234,179,8,0.2)', borderRadius: 3, color: '#fde68a', marginTop: 4 }}>
              Total minimum exposure exceeds 100%. Lower a few minimums to proceed.
            </div>
          )}
        </div>
      )}

      {/* Stack Types grouped list */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, padding: '4px 8px' }}>
        {stackSettings.length === 0 ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', fontSize: 9, color: 'var(--dfs-text-muted)' }}>
            No stack types available for {sport}.
          </div>
        ) : (
          groupedStacks.map(group => (
            <div key={group.category} style={{ marginBottom: 8 }}>
              {/* Category Header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0', marginBottom: 2 }}>
                <span style={{
                  fontSize: 8, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em',
                  color: 'var(--dfs-text-muted)', fontFamily: "'JetBrains Mono', monospace",
                }}>{group.label}</span>
                <div style={{ flex: 1, height: 1, background: 'var(--dfs-border)' }} />
                <span style={{ fontSize: 7, color: 'var(--dfs-text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>
                  {group.items.filter(i => i.stack.enabled).length}/{group.items.length}
                </span>
              </div>

              {/* Stack Rows */}
              <div style={{ border: '1px solid var(--dfs-border)', borderRadius: 4, overflow: 'hidden' }}>
                {group.items.map(({ stack, def }, index) => {
                  const corrBadge = getCorrelationBadge(def);
                  const gppBadge = getGppBadge(def);
                  const description = def?.description || stack.label;

                  let rowBg = index % 2 === 0 ? 'rgba(15,23,42,0.7)' : 'rgba(15,23,42,0.5)';
                  if (stack.enabled) rowBg = 'rgba(0,217,255,0.04)';

                  return (
                    <div
                      key={stack.id}
                      onClick={(e) => { const target = e.target as HTMLElement; if (target.tagName === 'INPUT' || target.closest('input')) return; toggleStackType(stack.id, e); }}
                      style={{
                        display: 'grid', gridTemplateColumns: ROW_GRID,
                        minHeight: 32, alignItems: 'center', padding: '0 6px',
                        borderBottom: index < group.items.length - 1 ? '1px solid var(--dfs-border)' : 'none',
                        background: rowBg, cursor: 'pointer',
                        transition: 'background 0.04s',
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = stack.enabled ? 'rgba(0,217,255,0.08)' : 'rgba(255,255,255,0.015)'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = rowBg; }}
                    >
                      {/* Checkbox */}
                      <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Checkbox
                          checked={stack.enabled}
                          onCheckedChange={(checked: boolean | 'indeterminate') => { if (checked !== stack.enabled) toggleStackType(stack.id); }}
                          className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
                          style={{ width: 12, height: 12 }}
                        />
                      </div>

                      {/* Name + description + badges */}
                      <div style={{ minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
                          <span style={{
                            fontSize: 9, fontWeight: 600,
                            color: stack.enabled ? '#E2E8F0' : '#5A6A7E',
                          }}>{stack.label}</span>
                          {corrBadge && (
                            <span style={{
                              fontSize: 7, fontWeight: 700, padding: '1px 5px', borderRadius: 8,
                              background: corrBadge.bg, color: corrBadge.fg,
                            }}>{corrBadge.text}</span>
                          )}
                          {gppBadge && (
                            <span style={{
                              fontSize: 7, fontWeight: 600, padding: '1px 5px', borderRadius: 8,
                              background: gppBadge.bg, color: gppBadge.fg,
                            }}>{gppBadge.text}</span>
                          )}
                        </div>
                        <div style={{ fontSize: 7, color: 'var(--dfs-text-muted)', marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {description}
                        </div>
                      </div>

                      {/* Min exposure */}
                      <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
                        <span style={{ fontSize: 6, color: 'var(--dfs-text-muted)', textTransform: 'uppercase' }}>Min</span>
                        <DebouncedInput value={stack.minExp} onChange={(v) => updateExposure(stack.id, 'minExp', v)} disabled={!stack.enabled} />
                      </div>

                      {/* Max exposure */}
                      <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
                        <span style={{ fontSize: 6, color: 'var(--dfs-text-muted)', textTransform: 'uppercase' }}>Max</span>
                        <DebouncedInput value={stack.maxExp} onChange={(v) => updateExposure(stack.id, 'maxExp', v)} disabled={!stack.enabled} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))
        )}
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
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        <span>
          Active: <span style={{ color: '#E2E8F0', fontWeight: 600 }}>{enabledStacks.length}</span> / {stackSettings.length}
        </span>
        <span>
          Min: <span style={{ color: hasConflict ? '#f87171' : '#E2E8F0', fontWeight: 600 }}>{totalMinExp}%</span>
          {' | '}
          Max: <span style={{ color: '#E2E8F0', fontWeight: 600 }}>{totalMaxExp}%</span>
          {' | '}
          Remaining: <span style={{ color: 'var(--dfs-text-muted)' }}>{Math.max(0, 100 - totalMinExp)}%</span>
        </span>
      </div>
    </div>
  );
};

export default StackExposureTab;
