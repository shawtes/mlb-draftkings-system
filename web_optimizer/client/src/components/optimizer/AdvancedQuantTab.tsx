import React from 'react';
import { Card } from '../ui/card';
import { Checkbox } from '../ui/checkbox';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Cpu } from 'lucide-react';
import { AdvancedQuantSettings, DEFAULT_ADVANCED_QUANT_SETTINGS } from './types';

interface AdvancedQuantTabProps {
  settings: AdvancedQuantSettings;
  onSettingsChange: (settings: AdvancedQuantSettings) => void;
}

const AdvancedQuantTab: React.FC<AdvancedQuantTabProps> = ({ settings, onSettingsChange }) => {
  const safeSettings: AdvancedQuantSettings = { ...DEFAULT_ADVANCED_QUANT_SETTINGS, ...settings };

  const updateSetting = <K extends keyof AdvancedQuantSettings>(key: K, value: AdvancedQuantSettings[K]) => {
    onSettingsChange({ ...safeSettings, [key]: value });
  };

  return (
    <div className="flex flex-col h-full space-y-4 overflow-auto">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2"><Cpu className="w-6 h-6 text-[var(--dfs-accent)]" /> Advanced Quant</h2>
          <p className="text-[var(--dfs-text-muted)] text-sm mt-1">Financial-grade quantitative optimization settings</p>
        </div>
      </div>

      <Card className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Checkbox checked={safeSettings.enabled} onCheckedChange={(checked: boolean) => updateSetting('enabled', checked)} className="border-slate-500 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400" />
            <div>
              <Label className="text-white font-semibold text-base cursor-pointer" onClick={() => updateSetting('enabled', !safeSettings.enabled)}>Enable Advanced Quantitative Optimization</Label>
              <p className="text-xs text-[var(--dfs-text-muted)] mt-1">Master switch for financial-grade risk modeling</p>
            </div>
          </div>
          {safeSettings.enabled && <div className="flex items-center gap-2 text-white"><div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" /><span className="text-xs font-medium">ENABLED</span></div>}
        </div>
      </Card>

      <Card className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4">
        <h3 className="text-sm font-bold text-[var(--dfs-accent)] uppercase tracking-wider mb-3">Optimization Strategy</h3>
        <div>
          <Label className="text-white block mb-2 text-sm">Strategy</Label>
          <Select value={safeSettings.strategy} onValueChange={(v: string) => updateSetting('strategy', v)} disabled={!safeSettings.enabled}>
            <SelectTrigger className="bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white"><SelectValue /></SelectTrigger>
            <SelectContent className="bg-slate-900 border-cyan-500/20">
              <SelectItem value="combined" className="text-white">Combined (Recommended)</SelectItem>
              <SelectItem value="kelly_criterion" className="text-white">Kelly Criterion</SelectItem>
              <SelectItem value="risk_parity" className="text-white">Risk Parity</SelectItem>
              <SelectItem value="mean_variance" className="text-white">Mean-Variance</SelectItem>
              <SelectItem value="equal_weight" className="text-white">Equal Weight</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-[var(--dfs-text-muted)] mt-1">
            {safeSettings.strategy === 'combined' && 'Combines multiple optimization techniques for balanced approach'}
            {safeSettings.strategy === 'kelly_criterion' && 'Pure Kelly optimal betting strategy - maximizes long-term growth'}
            {safeSettings.strategy === 'risk_parity' && 'Equal risk contribution - balances volatility across lineup'}
            {safeSettings.strategy === 'mean_variance' && 'Classic Markowitz optimization - maximizes return for given risk'}
            {safeSettings.strategy === 'equal_weight' && 'Simple equal allocation - baseline strategy'}
          </p>
        </div>
      </Card>

      <Card className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4">
        <h3 className="text-sm font-bold text-[var(--dfs-accent)] uppercase tracking-wider mb-3">Risk Parameters</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between mb-2"><Label className="text-white text-sm">Risk Tolerance</Label><span className="text-[var(--dfs-accent)] font-medium text-sm">{(safeSettings.riskTolerance ?? 1.0).toFixed(2)}</span></div>
            <input type="range" min="0.1" max="2.0" step="0.1" value={safeSettings.riskTolerance} onChange={(e) => updateSetting('riskTolerance', parseFloat(e.target.value))} disabled={!safeSettings.enabled} className="w-full" />
            <p className="text-xs text-[var(--dfs-text-muted)] mt-1">Range: 0.1 - 2.0</p>
          </div>
          <div>
            <div className="flex justify-between mb-2"><Label className="text-white text-sm">VaR Confidence Level</Label><span className="text-[var(--dfs-accent)] font-medium text-sm">{((safeSettings.varConfidence ?? 0.95) * 100).toFixed(0)}%</span></div>
            <input type="range" min="0.90" max="0.99" step="0.01" value={safeSettings.varConfidence} onChange={(e) => updateSetting('varConfidence', parseFloat(e.target.value))} disabled={!safeSettings.enabled} className="w-full" />
          </div>
          <div>
            <div className="flex justify-between mb-2"><Label className="text-white text-sm">Target Volatility</Label><span className="text-[var(--dfs-accent)] font-medium text-sm">{((safeSettings.targetVolatility ?? 0.20) * 100).toFixed(0)}%</span></div>
            <input type="range" min="0.05" max="0.50" step="0.01" value={safeSettings.targetVolatility} onChange={(e) => updateSetting('targetVolatility', parseFloat(e.target.value))} disabled={!safeSettings.enabled} className="w-full" />
          </div>
        </div>
      </Card>

      <Card className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4">
        <h3 className="text-sm font-bold text-[var(--dfs-accent)] uppercase tracking-wider mb-3">Monte Carlo Simulation</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label className="text-white block mb-2 text-sm">Simulations</Label>
            <Input type="number" min="1000" max="50000" step="1000" value={safeSettings.monteCarloSims} onChange={(e) => updateSetting('monteCarloSims', parseInt(e.target.value) || 10000)} disabled={!safeSettings.enabled} className="bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white" />
            <p className="text-xs text-[var(--dfs-text-muted)] mt-1">1K - 50K (10K recommended)</p>
          </div>
          <div>
            <Label className="text-white block mb-2 text-sm">Time Horizon (days)</Label>
            <Input type="number" min="1" max="30" value={safeSettings.timeHorizon} onChange={(e) => updateSetting('timeHorizon', parseInt(e.target.value) || 1)} disabled={!safeSettings.enabled} className="bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white" />
          </div>
        </div>
      </Card>

      <Card className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4">
        <h3 className="text-sm font-bold text-[var(--dfs-accent)] uppercase tracking-wider mb-3">GARCH Volatility Modeling</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <Label className="text-white block mb-2 text-sm">GARCH p</Label>
            <Input type="number" min="1" max="5" value={safeSettings.garchP} onChange={(e) => updateSetting('garchP', parseInt(e.target.value) || 1)} disabled={!safeSettings.enabled} className="bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white" />
          </div>
          <div>
            <Label className="text-white block mb-2 text-sm">GARCH q</Label>
            <Input type="number" min="1" max="5" value={safeSettings.garchQ} onChange={(e) => updateSetting('garchQ', parseInt(e.target.value) || 1)} disabled={!safeSettings.enabled} className="bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white" />
          </div>
          <div>
            <Label className="text-white block mb-2 text-sm">Lookback Period</Label>
            <Input type="number" min="30" max="365" step="10" value={safeSettings.lookbackPeriod} onChange={(e) => updateSetting('lookbackPeriod', parseInt(e.target.value) || 100)} disabled={!safeSettings.enabled} className="bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white" />
          </div>
        </div>
      </Card>

      <Card className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4">
        <h3 className="text-sm font-bold text-[var(--dfs-accent)] uppercase tracking-wider mb-3">Copula Dependency Modeling</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label className="text-white block mb-2 text-sm">Copula Family</Label>
            <Select value={safeSettings.copulaFamily} onValueChange={(v: string) => updateSetting('copulaFamily', v)} disabled={!safeSettings.enabled}>
              <SelectTrigger className="bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-slate-900 border-cyan-500/20">
                <SelectItem value="gaussian" className="text-white">Gaussian</SelectItem>
                <SelectItem value="t" className="text-white">t-Copula</SelectItem>
                <SelectItem value="clayton" className="text-white">Clayton</SelectItem>
                <SelectItem value="frank" className="text-white">Frank</SelectItem>
                <SelectItem value="gumbel" className="text-white">Gumbel</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="flex justify-between mb-2"><Label className="text-white text-sm">Dependency Threshold</Label><span className="text-[var(--dfs-accent)] font-medium text-sm">{((safeSettings.dependencyThreshold ?? 0.3) * 100).toFixed(0)}%</span></div>
            <input type="range" min="0.1" max="0.9" step="0.05" value={safeSettings.dependencyThreshold} onChange={(e) => updateSetting('dependencyThreshold', parseFloat(e.target.value))} disabled={!safeSettings.enabled} className="w-full" />
          </div>
        </div>
      </Card>

      <Card className="bg-[var(--dfs-bg-tertiary)] border-[var(--dfs-border)] p-4">
        <h3 className="text-sm font-bold text-[var(--dfs-accent)] uppercase tracking-wider mb-3">Kelly Criterion Position Sizing</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="flex justify-between mb-2"><Label className="text-white text-sm">Max Kelly Fraction</Label><span className="text-[var(--dfs-accent)] font-medium text-sm">{((safeSettings.maxKellyFraction ?? 0.25) * 100).toFixed(0)}%</span></div>
            <input type="range" min="0.1" max="1.0" step="0.05" value={safeSettings.maxKellyFraction} onChange={(e) => updateSetting('maxKellyFraction', parseFloat(e.target.value))} disabled={!safeSettings.enabled} className="w-full" />
          </div>
          <div>
            <div className="flex justify-between mb-2"><Label className="text-white text-sm">Expected Win Rate</Label><span className="text-[var(--dfs-accent)] font-medium text-sm">{((safeSettings.expectedWinRate ?? 0.20) * 100).toFixed(0)}%</span></div>
            <input type="range" min="0.1" max="0.9" step="0.05" value={safeSettings.expectedWinRate} onChange={(e) => updateSetting('expectedWinRate', parseFloat(e.target.value))} disabled={!safeSettings.enabled} className="w-full" />
          </div>
        </div>
      </Card>
    </div>
  );
};

export default AdvancedQuantTab;
