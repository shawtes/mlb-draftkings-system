import React, { useCallback } from 'react';
import { Checkbox } from '../ui/checkbox';
import { Cpu } from 'lucide-react';
import { AdvancedQuantSettings, DEFAULT_ADVANCED_QUANT_SETTINGS } from './types';

/* ── Strategy one-line descriptions ────────────────────────────── */
const STRATEGY_DESC: Record<string, string> = {
  combined: 'Blended Sharpe + leverage + Kelly signals',
  kelly_criterion: 'Maximize long-term bankroll growth via Kelly',
  risk_parity: 'Equal risk contribution across all positions',
  mean_variance: 'Markowitz efficient frontier optimization',
  equal_weight: 'Baseline equal allocation strategy',
};

/* ── Scoped CSS for custom range inputs ────────────────────────── */
const SLIDER_CSS = `
  .qt-range { -webkit-appearance: none; appearance: none; width: 100%; height: 3px; outline: none; border-radius: 2px; background: var(--dfs-bg-primary, #0B1120); cursor: pointer; }
  .qt-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 12px; height: 12px; border-radius: 50%; background: var(--dfs-bg-secondary, #111827); border: 2px solid #00D9FF; cursor: pointer; box-shadow: 0 0 4px rgba(0,217,255,0.35); transition: box-shadow 0.15s; }
  .qt-range::-webkit-slider-thumb:hover { box-shadow: 0 0 8px rgba(0,217,255,0.6); }
  .qt-range::-moz-range-thumb { width: 12px; height: 12px; border-radius: 50%; background: var(--dfs-bg-secondary, #111827); border: 2px solid #00D9FF; cursor: pointer; box-shadow: 0 0 4px rgba(0,217,255,0.35); }
  .qt-range::-moz-range-track { height: 3px; border-radius: 2px; background: var(--dfs-bg-primary, #0B1120); }
  .qt-range:disabled { opacity: 0.3; cursor: not-allowed; }
  .qt-range:disabled::-webkit-slider-thumb { box-shadow: none; border-color: #475569; }
`;

/* ── Reusable components ───────────────────────────────────────── */

/** Section header: tiny cyan label with trailing rule line */
const SectionHeader: React.FC<{ label: string }> = ({ label }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '10px 0 6px' }}>
    <span style={{
      fontSize: 7.5, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase',
      color: '#00D9FF', whiteSpace: 'nowrap', fontFamily: "'JetBrains Mono', monospace",
    }}>
      {label}
    </span>
    <div style={{ flex: 1, height: 1, background: 'var(--dfs-border, rgba(255,255,255,0.08))' }} />
  </div>
);

/** Compact slider row: [label] [---slider---] [value] */
const SliderRow: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}> = ({ label, value, min, max, step, disabled, onChange, format }) => {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, height: 26 }}>
      <span style={{
        fontSize: 9, color: 'var(--dfs-text-muted, #5A6A7E)', whiteSpace: 'nowrap', minWidth: 100,
        fontWeight: 500,
      }}>
        {label}
      </span>
      <div style={{ flex: 1, position: 'relative' }}>
        {/* Filled track overlay */}
        <div style={{
          position: 'absolute', top: '50%', left: 0, height: 3, borderRadius: 2,
          width: `${pct}%`, background: 'rgba(0,217,255,0.45)',
          transform: 'translateY(-50%)', pointerEvents: 'none', zIndex: 0,
        }} />
        <input
          type="range" className="qt-range" min={min} max={max} step={step}
          value={value} disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={{ position: 'relative', zIndex: 1 }}
        />
      </div>
      <span style={{
        fontSize: 10, fontFamily: "'JetBrains Mono', monospace", fontVariantNumeric: 'tabular-nums',
        color: '#00D9FF', minWidth: 42, textAlign: 'right', fontWeight: 600,
      }}>
        {format ? format(value) : value.toFixed(2)}
      </span>
    </div>
  );
};

/** Compact number input */
const CompactNumberInput: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled: boolean;
  onChange: (v: number) => void;
}> = ({ label, value, min, max, step = 1, disabled, onChange }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 6, height: 26 }}>
    <span style={{ fontSize: 9, color: 'var(--dfs-text-muted, #5A6A7E)', whiteSpace: 'nowrap', fontWeight: 500 }}>
      {label}
    </span>
    <input
      type="number" min={min} max={max} step={step} value={value} disabled={disabled}
      onChange={(e) => {
        const parsed = step >= 1 ? parseInt(e.target.value) : parseFloat(e.target.value);
        if (!isNaN(parsed)) onChange(Math.max(min, Math.min(max, parsed)));
      }}
      style={{
        width: 60, height: 24, textAlign: 'right',
        fontSize: 10, fontFamily: "'JetBrains Mono', monospace", fontVariantNumeric: 'tabular-nums',
        background: 'var(--dfs-bg-primary, #0B1120)', color: '#CBD5E1',
        border: '1px solid var(--dfs-border, rgba(255,255,255,0.08))', borderRadius: 3,
        outline: 'none', padding: '0 4px',
      }}
      onFocus={(e) => { e.target.style.borderColor = 'rgba(0,217,255,0.4)'; }}
      onBlur={(e) => { e.target.style.borderColor = 'var(--dfs-border, rgba(255,255,255,0.08))'; }}
    />
  </div>
);

/* ── Main Component ────────────────────────────────────────────── */

interface AdvancedQuantTabProps {
  settings: AdvancedQuantSettings;
  onSettingsChange: (settings: AdvancedQuantSettings) => void;
}

const AdvancedQuantTab: React.FC<AdvancedQuantTabProps> = ({ settings, onSettingsChange }) => {
  const safeSettings: AdvancedQuantSettings = { ...DEFAULT_ADVANCED_QUANT_SETTINGS, ...settings };

  const updateSetting = useCallback(<K extends keyof AdvancedQuantSettings>(key: K, value: AdvancedQuantSettings[K]) => {
    onSettingsChange({ ...safeSettings, [key]: value });
  }, [safeSettings, onSettingsChange]);

  const disabled = !safeSettings.enabled;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'auto' }}>
      {/* Inject scoped slider CSS */}
      <style>{SLIDER_CSS}</style>

      {/* ── Master enable bar ─────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 38, padding: '0 12px', flexShrink: 0,
        background: 'var(--dfs-bg-secondary, #111827)',
        borderBottom: '1px solid var(--dfs-border, rgba(255,255,255,0.08))',
      }}>
        {/* Left: checkbox + label + status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Checkbox
            checked={safeSettings.enabled}
            onCheckedChange={(checked: boolean) => updateSetting('enabled', checked)}
            className="border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400"
            style={{ width: 14, height: 14 }}
          />
          <Cpu style={{ width: 13, height: 13, color: '#00D9FF', opacity: safeSettings.enabled ? 1 : 0.4 }} />
          <span
            style={{
              fontSize: 10, fontWeight: 700, color: '#E2E8F0', letterSpacing: '0.02em',
              cursor: 'pointer', userSelect: 'none',
            }}
            onClick={() => updateSetting('enabled', !safeSettings.enabled)}
          >
            Advanced Quant Engine
          </span>
          {safeSettings.enabled && (
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              fontSize: 8, fontWeight: 700, color: '#4ade80', letterSpacing: '0.08em',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%', background: '#4ade80',
                boxShadow: '0 0 6px rgba(74,222,128,0.6)',
                animation: 'pulse 2s infinite',
              }} />
              ACTIVE
            </span>
          )}
        </div>

        {/* Right: strategy dropdown */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 8, color: 'var(--dfs-text-muted, #5A6A7E)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            STRATEGY
          </span>
          <select
            value={safeSettings.strategy}
            onChange={(e) => updateSetting('strategy', e.target.value)}
            disabled={disabled}
            style={{
              height: 24, fontSize: 9, fontFamily: "'JetBrains Mono', monospace",
              minWidth: 150, padding: '0 8px',
              background: 'var(--dfs-bg-primary, #0B1120)', color: '#CBD5E1',
              border: '1px solid var(--dfs-border, rgba(255,255,255,0.08))',
              borderRadius: 3, outline: 'none',
            }}
          >
            <option value="combined">Combined</option>
            <option value="kelly_criterion">Kelly Criterion</option>
            <option value="risk_parity">Risk Parity</option>
            <option value="mean_variance">Mean-Variance</option>
            <option value="equal_weight">Equal Weight</option>
          </select>
        </div>
      </div>

      {/* Strategy description subtitle */}
      <div style={{
        height: 18, padding: '0 12px', display: 'flex', alignItems: 'center',
        background: 'var(--dfs-bg-secondary, #111827)',
        borderBottom: '1px solid var(--dfs-border, rgba(255,255,255,0.08))',
      }}>
        <span style={{
          fontSize: 8, color: 'var(--dfs-text-muted, #5A6A7E)',
          fontStyle: 'italic', fontFamily: "'JetBrains Mono', monospace",
        }}>
          {STRATEGY_DESC[safeSettings.strategy] || ''}
        </span>
      </div>

      {/* ── Disabled overlay message ──────────────────────────── */}
      {disabled && (
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexDirection: 'column', gap: 12, padding: 20,
        }}>
          <Cpu style={{ width: 28, height: 28, color: '#475569' }} />
          <span style={{ fontSize: 10, color: '#5A6A7E', textAlign: 'center', lineHeight: 1.5 }}>
            Enable quant engine for financial-grade optimization
          </span>
          <button
            onClick={() => updateSetting('enabled', true)}
            style={{
              fontSize: 9, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
              color: '#00D9FF', background: 'rgba(0,217,255,0.08)',
              border: '1px solid rgba(0,217,255,0.25)', borderRadius: 4,
              padding: '4px 14px', cursor: 'pointer', letterSpacing: '0.04em',
            }}
          >
            ENABLE
          </button>
        </div>
      )}

      {/* ── Parameter sections (grayed when disabled) ─────────── */}
      <div style={{
        flex: 1, padding: '0 12px 12px',
        opacity: disabled ? 0.3 : 1,
        pointerEvents: disabled ? 'none' : 'auto',
        transition: 'opacity 0.2s',
        display: disabled ? 'none' : 'block',
      }}>

        {/* ── RISK PARAMETERS ──────────────────────────────────── */}
        <SectionHeader label="Risk Parameters" />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <SliderRow
            label="Risk Tolerance" value={safeSettings.riskTolerance ?? 1.0}
            min={0.1} max={2.0} step={0.1} disabled={disabled}
            onChange={(v) => updateSetting('riskTolerance', v)}
          />
          <SliderRow
            label="VaR Confidence" value={safeSettings.varConfidence ?? 0.95}
            min={0.90} max={0.99} step={0.01} disabled={disabled}
            onChange={(v) => updateSetting('varConfidence', v)}
            format={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <SliderRow
            label="Target Volatility" value={safeSettings.targetVolatility ?? 0.20}
            min={0.05} max={0.50} step={0.01} disabled={disabled}
            onChange={(v) => updateSetting('targetVolatility', v)}
            format={(v) => `${(v * 100).toFixed(0)}%`}
          />
        </div>

        {/* ── MONTE CARLO ──────────────────────────────────────── */}
        <SectionHeader label="Monte Carlo Simulation" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <CompactNumberInput
            label="Simulations" value={safeSettings.monteCarloSims}
            min={1000} max={50000} step={1000} disabled={disabled}
            onChange={(v) => updateSetting('monteCarloSims', v)}
          />
          <CompactNumberInput
            label="Horizon (d)" value={safeSettings.timeHorizon}
            min={1} max={30} step={1} disabled={disabled}
            onChange={(v) => updateSetting('timeHorizon', v)}
          />
        </div>

        {/* ── GARCH ────────────────────────────────────────────── */}
        <SectionHeader label="GARCH Volatility" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          <CompactNumberInput
            label="p" value={safeSettings.garchP}
            min={1} max={5} step={1} disabled={disabled}
            onChange={(v) => updateSetting('garchP', v)}
          />
          <CompactNumberInput
            label="q" value={safeSettings.garchQ}
            min={1} max={5} step={1} disabled={disabled}
            onChange={(v) => updateSetting('garchQ', v)}
          />
          <CompactNumberInput
            label="Lookback" value={safeSettings.lookbackPeriod}
            min={30} max={365} step={10} disabled={disabled}
            onChange={(v) => updateSetting('lookbackPeriod', v)}
          />
        </div>

        {/* ── COPULA ───────────────────────────────────────────── */}
        <SectionHeader label="Copula Dependency" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, height: 26 }}>
            <span style={{ fontSize: 9, color: 'var(--dfs-text-muted, #5A6A7E)', whiteSpace: 'nowrap', fontWeight: 500 }}>
              Family
            </span>
            <select
              value={safeSettings.copulaFamily}
              onChange={(e) => updateSetting('copulaFamily', e.target.value)}
              disabled={disabled}
              style={{
                height: 24, fontSize: 9, fontFamily: "'JetBrains Mono', monospace",
                flex: 1, padding: '0 6px',
                background: 'var(--dfs-bg-primary, #0B1120)', color: '#CBD5E1',
                border: '1px solid var(--dfs-border, rgba(255,255,255,0.08))',
                borderRadius: 3, outline: 'none',
              }}
            >
              <option value="gaussian">Gaussian</option>
              <option value="t">t-Copula</option>
              <option value="clayton">Clayton</option>
              <option value="frank">Frank</option>
              <option value="gumbel">Gumbel</option>
            </select>
          </div>
          <SliderRow
            label="Dep. Threshold" value={safeSettings.dependencyThreshold ?? 0.3}
            min={0.1} max={0.9} step={0.05} disabled={disabled}
            onChange={(v) => updateSetting('dependencyThreshold', v)}
            format={(v) => `${(v * 100).toFixed(0)}%`}
          />
        </div>

        {/* ── KELLY CRITERION ──────────────────────────────────── */}
        <SectionHeader label="Kelly Criterion" />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <SliderRow
            label="Max Kelly Frac." value={safeSettings.maxKellyFraction ?? 0.25}
            min={0.1} max={1.0} step={0.05} disabled={disabled}
            onChange={(v) => updateSetting('maxKellyFraction', v)}
            format={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <SliderRow
            label="Expected Win Rate" value={safeSettings.expectedWinRate ?? 0.20}
            min={0.1} max={0.9} step={0.05} disabled={disabled}
            onChange={(v) => updateSetting('expectedWinRate', v)}
            format={(v) => `${(v * 100).toFixed(0)}%`}
          />
        </div>

        {/* ── Footer status ────────────────────────────────────── */}
        <div style={{
          marginTop: 14, padding: '6px 0',
          borderTop: '1px solid var(--dfs-border, rgba(255,255,255,0.08))',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{ fontSize: 8, color: 'var(--dfs-text-muted, #5A6A7E)', fontFamily: "'JetBrains Mono', monospace" }}>
            {safeSettings.monteCarloSims.toLocaleString()} sims | GARCH({safeSettings.garchP},{safeSettings.garchQ}) | {safeSettings.copulaFamily} copula
          </span>
          <span style={{
            fontSize: 8, fontWeight: 700, letterSpacing: '0.06em',
            fontFamily: "'JetBrains Mono', monospace",
            color: safeSettings.enabled ? '#a78bfa' : '#475569',
          }}>
            QUANT {safeSettings.enabled ? 'ON' : 'OFF'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default AdvancedQuantTab;
