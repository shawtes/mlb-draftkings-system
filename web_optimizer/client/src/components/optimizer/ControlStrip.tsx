import React from 'react';
import { Upload, FileText, Download, Play, Save } from 'lucide-react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Checkbox } from '../ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';

interface ControlStripProps {
  // File actions
  onUploadCsv: () => void;
  onOpenBlending: () => void;
  onUploadOwnership: () => void;
  onLoadEntries: () => void;
  // Lineup config
  numLineups: number;
  onNumLineupsChange: (n: number) => void;
  minUnique: number;
  onMinUniqueChange: (n: number) => void;
  minSalary: number;
  onMinSalaryChange: (n: number) => void;
  maxSalary: number;
  sortMethod: string;
  onSortMethodChange: (s: string) => void;
  disableKelly: boolean;
  onDisableKellyChange: (v: boolean) => void;
  // Actions
  onOptimize: () => void;
  onExport: () => void;
  isOptimizing: boolean;
  hasPlayerData: boolean;
  hasResults: boolean;
}

const ControlStrip: React.FC<ControlStripProps> = ({
  onUploadCsv,
  onOpenBlending,
  onUploadOwnership,
  onLoadEntries,
  numLineups,
  onNumLineupsChange,
  minUnique,
  onMinUniqueChange,
  minSalary,
  onMinSalaryChange,
  maxSalary,
  sortMethod,
  onSortMethodChange,
  disableKelly,
  onDisableKellyChange,
  onOptimize,
  onExport,
  isOptimizing,
  hasPlayerData,
  hasResults,
}) => {
  return (
    <div className="bg-[var(--dfs-bg-secondary)] border-b border-[var(--dfs-border)] flex flex-wrap items-center gap-3 px-4 py-2">
      {/* Left: File Buttons */}
      <div className="flex items-center gap-1.5">
        <span className="text-xs uppercase tracking-wider text-[var(--dfs-text-muted)] font-medium mr-1">Files</span>
        <Button
          variant="secondary-action"
          size="sm"
          className="text-xs"
          onClick={onUploadCsv}
        >
          <Upload className="w-3.5 h-3.5" />
          CSV
        </Button>
        <Button
          variant="ghost-action"
          size="sm"
          className="text-xs"
          onClick={onOpenBlending}
        >
          <FileText className="w-3.5 h-3.5" />
          Predictions
        </Button>
        <Button
          variant="ghost-action"
          size="sm"
          className="text-xs text-orange-300 hover:text-orange-200"
          onClick={onUploadOwnership}
        >
          <Upload className="w-3.5 h-3.5" />
          Own%
        </Button>
        <Button
          variant="ghost-action"
          size="sm"
          className="text-xs text-emerald-300 hover:text-emerald-200"
          onClick={onLoadEntries}
        >
          <Download className="w-3.5 h-3.5" />
          Entries
        </Button>
      </div>

      <div className="h-6 w-px bg-[var(--dfs-border)]" />

      {/* Center: Lineup Config */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1.5">
          <Label className="text-xs text-[var(--dfs-text-muted)] font-medium">Lineups</Label>
          <Input
            type="number"
            min={1}
            max={500}
            value={numLineups}
            onChange={(e) => onNumLineupsChange(parseInt(e.target.value, 10) || 1)}
            className="h-8 w-16 bg-[var(--dfs-bg-primary)] border-[var(--dfs-border)] text-xs text-[var(--dfs-text-primary)]"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <Label className="text-xs text-[var(--dfs-text-muted)] font-medium">Unique</Label>
          <Input
            type="number"
            min={0}
            max={10}
            value={minUnique}
            onChange={(e) => onMinUniqueChange(parseInt(e.target.value, 10) || 0)}
            className="h-8 w-14 bg-[var(--dfs-bg-primary)] border-[var(--dfs-border)] text-xs text-[var(--dfs-text-primary)]"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <Label className="text-xs text-[var(--dfs-text-muted)] font-medium">Min Salary</Label>
          <Input
            type="number"
            min={0}
            max={maxSalary}
            step={500}
            value={minSalary}
            onChange={(e) => onMinSalaryChange(parseInt(e.target.value, 10))}
            className="h-8 w-20 bg-[var(--dfs-bg-primary)] border-[var(--dfs-border)] text-xs text-[var(--dfs-text-primary)]"
          />
          <span className="text-xs text-[var(--dfs-text-muted)]">/ {maxSalary.toLocaleString()}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Select value={sortMethod} onValueChange={onSortMethodChange}>
            <SelectTrigger className="h-8 min-w-[110px] bg-[var(--dfs-bg-primary)] border-[var(--dfs-border)] text-xs text-[var(--dfs-text-primary)]">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border border-slate-700 text-xs">
              <SelectItem value="points" className="text-slate-100 text-xs">Sort: Points</SelectItem>
              <SelectItem value="value" className="text-slate-100 text-xs">Sort: Value</SelectItem>
              <SelectItem value="salary" className="text-slate-100 text-xs">Sort: Salary</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-1.5">
          <Checkbox
            id="ctrl-disable-kelly"
            checked={disableKelly}
            onCheckedChange={(checked: boolean) => onDisableKellyChange(Boolean(checked))}
            className="h-4 w-4 border-slate-600 data-[state=checked]:bg-slate-800 data-[state=checked]:border-orange-400"
          />
          <Label htmlFor="ctrl-disable-kelly" className="text-xs text-[var(--dfs-text-secondary)] cursor-pointer">
            Disable Kelly
          </Label>
        </div>
      </div>

      <div className="h-6 w-px bg-[var(--dfs-border)]" />

      {/* Right: Optimize + Save */}
      <div className="flex items-center gap-2">
        <Button
          variant="primary-action"
          size="lg"
          className="h-10 px-5 text-sm font-semibold"
          onClick={onOptimize}
          disabled={isOptimizing || !hasPlayerData}
        >
          <Play className="w-4 h-4" />
          {isOptimizing ? 'Optimizing...' : 'Optimize'}
        </Button>
        <Button
          variant="ghost-action"
          size="sm"
          className="text-xs text-yellow-300 hover:text-yellow-200 hover:bg-yellow-500/10"
          disabled={!hasResults}
          onClick={onExport}
        >
          <Save className="w-3.5 h-3.5" />
          Save CSV
        </Button>
      </div>
    </div>
  );
};

export default ControlStrip;
