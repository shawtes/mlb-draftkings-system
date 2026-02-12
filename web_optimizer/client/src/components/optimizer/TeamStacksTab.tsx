import React, { useState, useMemo, useCallback } from 'react';
import { Button } from '../ui/button';
import { Checkbox } from '../ui/checkbox';
import { Input } from '../ui/input';
import { Link2, CheckSquare, XSquare } from 'lucide-react';
import { Player, Team } from './types';
import { Sport } from '../sport-config';

/** ESPN CDN logo mapping — DK abbreviation → ESPN slug */
const DK_TO_ESPN: Record<string, Record<string, string>> = {
  NBA: {
    ATL:'atl',BOS:'bos',BKN:'bkn',CHA:'cha',CHI:'chi',CLE:'cle',DAL:'dal',DEN:'den',DET:'det',
    GS:'gs',GSW:'gs',HOU:'hou',IND:'ind',LAC:'lac',LAL:'lal',MEM:'mem',MIA:'mia',MIL:'mil',
    MIN:'min',NO:'no',NOP:'no',NY:'ny',NYK:'ny',OKC:'okc',ORL:'orl',PHI:'phi',PHO:'phx',PHX:'phx',
    POR:'por',SAC:'sac',SA:'sa',SAS:'sa',TOR:'tor',UTA:'utah',UTAH:'utah',WAS:'wsh',WSH:'wsh',
  },
  NFL: {
    ARI:'ari',ATL:'atl',BAL:'bal',BUF:'buf',CAR:'car',CHI:'chi',CIN:'cin',CLE:'cle',DAL:'dal',
    DEN:'den',DET:'det',GB:'gb',HOU:'hou',IND:'ind',JAX:'jax',JAC:'jax',KC:'kc',LV:'lv',
    LAR:'lar',LAC:'lac',MIA:'mia',MIN:'min',NE:'ne',NO:'no',NYG:'nyg',NYJ:'nyj',PHI:'phi',
    PIT:'pit',SF:'sf',SEA:'sea',TB:'tb',TEN:'ten',WAS:'wsh',WSH:'wsh',
  },
  MLB: {
    ARI:'ari',ATL:'atl',BAL:'bal',BOS:'bos',CHC:'chc',CHW:'chw',CWS:'chw',CIN:'cin',CLE:'cle',
    COL:'col',DET:'det',HOU:'hou',KC:'kc',LAA:'laa',LAD:'lad',MIA:'mia',MIL:'mil',MIN:'min',
    NYM:'nym',NYY:'nyy',OAK:'oak',PHI:'phi',PIT:'pit',SD:'sd',SF:'sf',SEA:'sea',STL:'stl',
    TB:'tb',TEX:'tex',TOR:'tor',WAS:'wsh',WSH:'wsh',
  },
};

const TeamLogo: React.FC<{ team: string; sport: string; size?: number }> = ({ team, sport, size = 20 }) => {
  const [failed, setFailed] = React.useState(false);
  if (failed) return null;
  const s = sport === 'NBA' ? 'nba' : sport === 'NFL' ? 'nfl' : 'mlb';
  const slug = DK_TO_ESPN[sport]?.[team.toUpperCase()] || team.toLowerCase();
  return (
    <img
      src={`https://a.espncdn.com/i/teamlogos/${s}/500/${slug}.png`}
      alt={team} width={size} height={size}
      className="object-contain flex-shrink-0"
      onError={() => setFailed(true)}
      loading="lazy"
    />
  );
};

interface TeamStacksTabProps {
  playerData: Player[];
  teamSelections: Record<number | 'all', string[]>;
  onTeamSelectionsChange: (selections: Record<number | 'all', string[]>) => void;
  onTeamExposuresChange?: (exposures: Record<string, { minExp: number; maxExp: number }>) => void;
  sport?: string;
}

const TeamStacksTab: React.FC<TeamStacksTabProps> = ({ playerData, teamSelections, onTeamSelectionsChange, onTeamExposuresChange, sport }) => {
  const isNBA = sport === 'NBA';
  const [activeStackSize, setActiveStackSize] = useState<'all' | number>('all');
  const [teamExposures, setTeamExposures] = useState<Record<string, { minExp: number; maxExp: number }>>({});

  const teams = useMemo(() => {
    const teamMap = new Map<string, Team>();
    playerData.forEach(player => {
      if (!teamMap.has(player.team)) {
        teamMap.set(player.team, {
          abbr: player.team, status: 'Active', gameTime: '7:00 PM', projRuns: 0, minExp: 0, maxExp: 100, playerCount: 0, batterCount: 0,
        });
      }
      const team = teamMap.get(player.team)!;
      team.playerCount++;
      if (!player.position.includes('P')) team.batterCount++;
      team.projRuns += player.projectedPoints / 10;
    });
    // Apply stored exposure settings
    const result = Array.from(teamMap.values()).map(team => {
      const exp = teamExposures[team.abbr];
      if (exp) {
        return { ...team, minExp: exp.minExp, maxExp: exp.maxExp };
      }
      return team;
    });
    return result.sort((a, b) => a.abbr.localeCompare(b.abbr));
  }, [playerData, teamExposures]);

  const getSelectedTeams = (stackSize: 'all' | number): string[] => teamSelections[stackSize] || [];
  const getTeamCount = useCallback((t: Team) => isNBA ? t.playerCount : t.batterCount, [isNBA]);

  const updateTeamExposure = useCallback((abbr: string, field: 'minExp' | 'maxExp', value: number) => {
    const clamped = Math.max(0, Math.min(100, value));
    setTeamExposures(prev => {
      const current = prev[abbr] || { minExp: 0, maxExp: 100 };
      const updated = { ...current, [field]: clamped };
      // Ensure min <= max
      if (field === 'minExp' && clamped > updated.maxExp) updated.maxExp = clamped;
      if (field === 'maxExp' && clamped < updated.minExp) updated.minExp = clamped;
      const newState = { ...prev, [abbr]: updated };
      onTeamExposuresChange?.(newState);
      return newState;
    });
  }, [onTeamExposuresChange]);

  const toggleTeam = useCallback((team: string, event?: React.MouseEvent) => {
    if (event) event.stopPropagation();
    const current = getSelectedTeams(activeStackSize);
    const updated = current.includes(team) ? current.filter(t => t !== team) : [...current, team];
    onTeamSelectionsChange({ ...teamSelections, [activeStackSize]: updated });
  }, [activeStackSize, teamSelections, onTeamSelectionsChange]);

  const handleSelectAll = () => {
    const allTeams = teams.filter(t => activeStackSize === 'all' || getTeamCount(t) >= (activeStackSize as number)).map(t => t.abbr);
    onTeamSelectionsChange({ ...teamSelections, [activeStackSize]: allTeams });
  };

  const handleDeselectAll = () => {
    onTeamSelectionsChange({ ...teamSelections, [activeStackSize]: [] });
  };

  if (teams.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="mb-4"><div className="w-16 h-16 mx-auto bg-slate-700 rounded-full flex items-center justify-center"><Link2 className="w-8 h-8 text-slate-400" /></div></div>
          <h3 className="text-xl font-semibold text-white mb-2">No Team Data</h3>
          <p className="text-[var(--dfs-text-secondary)] mb-4">Load players first to configure team stacks</p>
        </div>
      </div>
    );
  }

  const selectedCount = getSelectedTeams(activeStackSize).length;

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Stack Size Sub-Tabs */}
      <div className="flex gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent pb-2">
        {[
          { id: 'all', label: 'All Stacks', count: getSelectedTeams('all').length },
          { id: 2, label: '2 Stack', count: getSelectedTeams(2).length },
          { id: 3, label: '3 Stack', count: getSelectedTeams(3).length },
          { id: 4, label: '4 Stack', count: getSelectedTeams(4).length },
          { id: 5, label: '5 Stack', count: getSelectedTeams(5).length },
        ].map((stack) => (
          <button
            key={stack.id}
            onClick={() => setActiveStackSize(stack.id as 'all' | number)}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition-all text-sm ${
              activeStackSize === stack.id
                ? 'bg-cyan-500/20 text-[var(--dfs-accent)] border border-cyan-500/40'
                : 'bg-[var(--dfs-bg-tertiary)] text-[var(--dfs-text-secondary)] border border-[var(--dfs-border)] hover:bg-[var(--dfs-bg-hover)]'
            }`}
          >
            {stack.label}
            {stack.count > 0 ? (
              <span className="ml-1.5 inline-flex items-center justify-center px-1.5 py-0.5 text-xs font-bold rounded-full bg-cyan-500 text-white">{stack.count}</span>
            ) : (
              <span className="text-xs opacity-70 ml-1">(0)</span>
            )}
          </button>
        ))}
      </div>

      {/* Action Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-[var(--dfs-bg-tertiary)] border border-[var(--dfs-border)] rounded-lg p-3">
        <div className="flex gap-2 flex-wrap">
          <Button variant="secondary-action" size="sm" onClick={handleSelectAll} className="border-green-500/30 bg-green-500/5">
            <CheckSquare className="w-4 h-4 mr-1" /> Select All
          </Button>
          <Button variant="secondary-action" size="sm" onClick={handleDeselectAll} className="border-red-500/30 bg-red-500/5">
            <XSquare className="w-4 h-4 mr-1" /> Deselect All
          </Button>
        </div>
        {activeStackSize !== 'all' && (
          <div className="text-xs text-[var(--dfs-text-muted)]">
            Teams with {activeStackSize}+ {isNBA ? 'players' : 'batters'}: {teams.filter(t => getTeamCount(t) >= (activeStackSize as number)).length}
          </div>
        )}
      </div>

      {/* Warning */}
      {activeStackSize !== 'all' && (() => {
        const selectedTeams = getSelectedTeams(activeStackSize);
        const insufficientTeams = selectedTeams.map(abbr => teams.find(t => t.abbr === abbr)).filter(t => t && getTeamCount(t) < (activeStackSize as number));
        return insufficientTeams.length > 0 ? (
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-3 py-2 text-xs text-yellow-400">
            {insufficientTeams.map(t => t!.abbr).join(', ')} {insufficientTeams.length === 1 ? 'has' : 'have'} fewer than {activeStackSize as number} {isNBA ? 'players' : 'batters'} and will be skipped during optimization.
          </div>
        ) : null;
      })()}

      {/* Team Stack Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="bg-[var(--dfs-bg-tertiary)] sticky top-0 z-10">
            <tr className="border-b border-[var(--dfs-border)]">
              <th className="px-3 py-3 text-left text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-12">
                <Checkbox
                  checked={teams.length > 0 && (getSelectedTeams(activeStackSize).length === (activeStackSize === 'all' ? teams.length : teams.filter(t => getTeamCount(t) >= (activeStackSize as number)).length))}
                  onCheckedChange={(checked: boolean | 'indeterminate') => { if (checked) handleSelectAll(); else handleDeselectAll(); }}
                  className="border-cyan-400 data-[state=checked]:bg-slate-900 data-[state=checked]:border-cyan-400 cursor-pointer"
                />
              </th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-20">Team</th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-24">Status</th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-24">Time</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-24">Proj Runs</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-24">{isNBA ? 'Players' : 'Batters'}</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-24">Min Exp</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-24">Max Exp</th>
              <th className="px-3 py-3 text-right text-xs font-semibold text-[var(--dfs-accent)] uppercase tracking-wider w-24">Actual</th>
            </tr>
          </thead>
          <tbody>
            {teams.map((team, idx) => {
              const isSelected = getSelectedTeams(activeStackSize).includes(team.abbr);
              const canStack = activeStackSize === 'all' || getTeamCount(team) >= (activeStackSize as number);
              return (
                <tr
                  key={team.abbr}
                  onClick={(e) => {
                    if (!canStack) return;
                    const target = e.target as HTMLElement;
                    if (target.tagName === 'INPUT' || target.closest('input')) return;
                    toggleTeam(team.abbr, e);
                  }}
                  className={`border-b border-slate-700/50 hover:bg-[var(--dfs-bg-hover)] transition-colors ${idx % 2 === 0 ? 'bg-slate-800/20' : ''} ${!canStack ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'} ${isSelected && canStack ? 'bg-cyan-500/10 hover:bg-cyan-500/15' : ''}`}
                >
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <Checkbox checked={isSelected} onCheckedChange={(checked: boolean | 'indeterminate') => { if (!canStack) return; if (checked !== isSelected) toggleTeam(team.abbr); }} disabled={!canStack} className="cursor-pointer" />
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <TeamLogo team={team.abbr} sport={sport || 'MLB'} size={22} />
                      <span className="text-white font-bold">{team.abbr}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <span className={`px-2 py-1 rounded text-xs ${team.status === 'Active' ? 'bg-green-500/20 text-white' : team.status === 'Postponed' ? 'bg-red-500/20 text-red-400' : 'bg-slate-500/20 text-slate-400'}`}>{team.status}</span>
                  </td>
                  <td className="px-3 py-2 text-white">{team.gameTime}</td>
                  <td className="px-3 py-2 text-right"><span className={`font-medium ${team.projRuns > 5.0 ? 'text-white' : team.projRuns > 4.0 ? 'text-yellow-400' : 'text-slate-400'}`}>{team.projRuns.toFixed(1)}</span></td>
                  <td className="px-3 py-2 text-right text-cyan-400 font-medium">{isNBA ? team.playerCount : team.batterCount}</td>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}><Input type="number" min="0" max="100" value={team.minExp} onChange={(e) => updateTeamExposure(team.abbr, 'minExp', parseInt(e.target.value) || 0)} className="bg-slate-700 border-slate-600 text-white text-xs h-8 w-20 text-right" /></td>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}><Input type="number" min="0" max="100" value={team.maxExp} onChange={(e) => updateTeamExposure(team.abbr, 'maxExp', parseInt(e.target.value) || 0)} className="bg-slate-700 border-slate-600 text-white text-xs h-8 w-20 text-right" /></td>
                  <td className="px-3 py-2 text-right text-slate-400 text-xs">{team.actualExp !== undefined ? `${team.actualExp.toFixed(1)}%` : '\u2014'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Status Bar */}
      <div className="bg-[var(--dfs-bg-tertiary)] border border-[var(--dfs-border)] rounded-lg p-3">
        <div className="flex items-center justify-between text-sm flex-wrap gap-2">
          <div className="text-white">
            <span className="font-semibold text-[var(--dfs-accent)]">{activeStackSize === 'all' ? 'All Stacks' : `${activeStackSize}-Stack`}:</span>{' '}
            {selectedCount > 0 ? <span className="text-white">{getSelectedTeams(activeStackSize).join(', ')}</span> : <span className="text-[var(--dfs-text-muted)]">No teams selected</span>}
            <span className="text-[var(--dfs-text-muted)] ml-2">({selectedCount}/{teams.length})</span>
          </div>
          {activeStackSize !== 'all' && selectedCount > 0 && (
            <div className="text-xs text-[var(--dfs-accent)]">{selectedCount} team{selectedCount !== 1 ? 's' : ''} configured for {activeStackSize}-stacks</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TeamStacksTab;
