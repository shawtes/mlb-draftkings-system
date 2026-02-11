import React from 'react';
import { Checkbox } from '../ui/checkbox';
import { toast } from 'react-hot-toast';

interface LineupCardProps {
  result: any;
  idx: number;
  isSelected: boolean;
  onToggleSelect: (idx: number, selected: boolean) => void;
}

const formatSalary = (salary: number): string => {
  if (salary >= 1000) return `$${(salary / 1000).toFixed(1)}k`;
  return `$${salary}`;
};

const LineupCard: React.FC<LineupCardProps> = ({ result, idx, isSelected, onToggleSelect }) => {
  return (
    <div
      className={`bg-[var(--dfs-bg-primary)] border rounded-md p-2 transition-colors ${
        isSelected
          ? 'border-cyan-500 bg-cyan-900/10'
          : 'border-[var(--dfs-border)] hover:border-cyan-500/50'
      }`}
    >
      {/* Lineup Header */}
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-1.5">
          <Checkbox
            checked={isSelected}
            onCheckedChange={(checked: boolean) => onToggleSelect(idx, checked)}
            onClick={(e: React.MouseEvent) => e.stopPropagation()}
            className="h-3.5 w-3.5"
          />
          <span className="text-xs font-semibold text-[var(--dfs-text-muted)]">
            #{idx + 1}
          </span>
        </div>
        <div className="flex gap-2 items-center">
          <span className="text-xs font-bold text-[var(--dfs-accent)] font-mono">
            {result.totalPoints?.toFixed(1) || result.points?.toFixed(1) || '0.0'}pts
          </span>
          <span className="text-xs text-[var(--dfs-text-muted)] font-mono">
            ${result.totalSalary || result.salary || 0}
          </span>
        </div>
      </div>

      {/* Players List */}
      <div className="space-y-0">
        {(result.players || []).map((player: any, pidx: number) => (
          <div
            key={pidx}
            className="flex items-center justify-between text-xs py-px"
          >
            <div className="flex items-center gap-1 flex-1 min-w-0">
              <span className="text-[var(--dfs-accent)] font-mono w-6 flex-shrink-0 text-xs">
                {player.rosterPosition || player.position || player.pos || 'NA'}
              </span>
              <span className="text-[var(--dfs-text-secondary)] truncate">
                {player.name || player.player || 'Unknown'}
                {(player.team || player.teamAbbrev) && (
                  <span className="text-[var(--dfs-text-muted)]"> ({player.team || player.teamAbbrev})</span>
                )}
              </span>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0 ml-1">
              <span className="text-[var(--dfs-text-muted)] text-xs font-mono">
                {formatSalary(player.salary || 0)}
              </span>
              <span className="text-[var(--dfs-text-secondary)] text-xs font-mono w-8 text-right">
                {player.projectedPoints?.toFixed(1) || player.points?.toFixed(1) || player.projection?.toFixed(1) || '0.0'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Compact Footer: just Copy aligned right */}
      <div className="mt-1 pt-1 border-t border-[var(--dfs-border)] flex items-center justify-end">
        <button
          className="text-xs text-[var(--dfs-accent)] hover:text-cyan-300 font-medium transition-colors"
          onClick={(e) => {
            e.stopPropagation();
            try {
              const lineupText = `Lineup ${idx + 1} - ${result.totalPoints?.toFixed(1) || result.points?.toFixed(1) || '0.0'} pts\n` +
                `Salary: $${result.totalSalary || result.salary || 0}\n\n` +
                (result.players || []).map((p: any) =>
                  `${p.position || p.pos || 'NA'}: ${p.name || p.player || 'Unknown'} (${p.projectedPoints?.toFixed(1) || p.points?.toFixed(1) || '0.0'} pts)`
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
      </div>
    </div>
  );
};

export default LineupCard;
