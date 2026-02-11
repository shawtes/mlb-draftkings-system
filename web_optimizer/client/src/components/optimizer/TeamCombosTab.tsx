import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Button } from '../ui/button';
import { Checkbox } from '../ui/checkbox';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Target, Check, X } from 'lucide-react';
import { Player, TeamCombination } from './types';

interface TeamCombosTabProps {
  playerData: Player[];
}

const TeamCombosTab: React.FC<TeamCombosTabProps> = ({ playerData }) => {
  const [selectedTeams, setSelectedTeams] = useState<string[]>([]);
  const [stackPattern, setStackPattern] = useState('4');
  const [defaultLineupsPerCombo, setDefaultLineupsPerCombo] = useState(5);
  const [combinations, setCombinations] = useState<TeamCombination[]>([]);

  const teams = useMemo(() => {
    const teamSet = new Set((playerData || []).map(p => p.team));
    return Array.from(teamSet).sort();
  }, [playerData]);

  const prevTeamsRef = useRef<string[]>([]);
  useEffect(() => {
    if (teams.length > 0) {
      const teamsChanged = teams.length !== prevTeamsRef.current.length || teams.some(team => !prevTeamsRef.current.includes(team));
      if (teamsChanged) { setSelectedTeams([...teams]); prevTeamsRef.current = [...teams]; }
    } else { prevTeamsRef.current = []; }
  }, [teams]);

  const toggleTeam = (team: string) => setSelectedTeams(prev => prev.includes(team) ? prev.filter(t => t !== team) : [...prev, team]);
  const selectAllTeams = () => setSelectedTeams([...teams]);
  const deselectAllTeams = () => setSelectedTeams([]);

  const getCombinations = (arr: string[], k: number): string[][] => {
    if (k === 0) return [[]];
    if (arr.length === 0) return [];
    const [first, ...rest] = arr;
    return [...getCombinations(rest, k - 1).map(c => [first, ...c]), ...getCombinations(rest, k)];
  };

  const getPermutations = (arr: string[]): string[][] => {
    if (arr.length <= 1) return [arr];
    const result: string[][] = [];
    for (let i = 0; i < arr.length; i++) {
      const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
      getPermutations(rest).forEach(p => result.push([arr[i], ...p]));
    }
    return result;
  };

  const generateCombinations = () => {
    const stackSizes = stackPattern.split('|').map(s => parseInt(s));
    const teamsNeeded = stackSizes.length;
    if (selectedTeams.length < teamsNeeded) { alert(`Pattern "${stackPattern}" requires ${teamsNeeded} teams. Only ${selectedTeams.length} selected.`); return; }
    const combos: TeamCombination[] = [];
    getCombinations(selectedTeams, teamsNeeded).forEach(teamCombo => {
      getPermutations(teamCombo).forEach(perm => {
        const display = perm.map((t, i) => `${t}(${stackSizes[i]})`).join(' + ');
        combos.push({ id: `combo-${combos.length}`, teams: perm, stackSizes, display, lineupsPerCombo: defaultLineupsPerCombo, status: 'ready', enabled: true });
      });
    });
    if (combos.length > 50 && !confirm(`This will create ${combos.length} combinations (${combos.length * defaultLineupsPerCombo} total lineups). Continue?`)) return;
    setCombinations(combos);
  };

  const toggleCombination = (id: string) => setCombinations(prev => prev.map(c => c.id === id ? { ...c, enabled: !c.enabled } : c));
  const updateLineupsPerCombo = (id: string, value: number) => setCombinations(prev => prev.map(c => c.id === id ? { ...c, lineupsPerCombo: Math.max(1, Math.min(100, value)) } : c));

  const enabledCombos = combinations.filter(c => c.enabled);
  const totalLineups = enabledCombos.reduce((sum, c) => sum + c.lineupsPerCombo, 0);

  if (teams.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Target className="w-10 h-10 mx-auto text-[var(--dfs-text-muted)] mb-3" />
          <h3 className="text-sm font-medium text-[var(--dfs-text-secondary)] mb-1">No Team Data</h3>
          <p className="text-xs text-[var(--dfs-text-muted)]">Load players first to generate team combinations</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-8 p-8">
      <div className="pb-2">
        <h2 className="text-xl font-medium text-white mb-1">Team Combinations</h2>
        <p className="text-sm text-[var(--dfs-text-muted)]">{playerData.length} players \u00B7 {teams.length} teams</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-[var(--dfs-text-secondary)]">Select Teams</h3>
            <div className="flex gap-2">
              <Button variant="ghost-action" size="sm" onClick={selectAllTeams}><Check className="w-3 h-3 mr-1.5" /> All</Button>
              <Button variant="ghost-action" size="sm" onClick={deselectAllTeams}><X className="w-3 h-3 mr-1.5" /> None</Button>
            </div>
          </div>
          <div className="max-h-80 overflow-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
            <div className="grid grid-cols-3 gap-2">
              {teams.map(team => (
                <div key={team} className="flex items-center gap-2 p-2.5 rounded-md hover:bg-[var(--dfs-bg-hover)] transition-colors cursor-pointer" onClick={() => toggleTeam(team)}>
                  <Checkbox checked={selectedTeams.includes(team)} onCheckedChange={() => toggleTeam(team)} className="h-4 w-4 border-slate-600 data-[state=checked]:bg-slate-700" />
                  <Label className="text-sm text-[var(--dfs-text-secondary)] cursor-pointer font-normal">{team}</Label>
                </div>
              ))}
            </div>
          </div>
          <div className="text-xs text-[var(--dfs-text-muted)] pt-2"><span className="text-[var(--dfs-text-secondary)]">{selectedTeams.length}</span> of <span className="text-[var(--dfs-text-secondary)]">{teams.length}</span> selected</div>
        </div>

        <div className="space-y-5">
          <h3 className="text-sm font-medium text-[var(--dfs-text-secondary)]">Stack Settings</h3>
          <div className="space-y-4">
            <div>
              <Label className="text-xs text-[var(--dfs-text-muted)] block mb-2">Stack Pattern</Label>
              <Select value={stackPattern} onValueChange={setStackPattern}>
                <SelectTrigger className="w-full bg-[var(--dfs-bg-secondary)] border border-[var(--dfs-border)] text-white text-sm h-10"><SelectValue /></SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-700">
                  {['5','4','3','No Stacks','5|2','4|2','4|2|2','3|3|2','3|2|2','2|2|2','5|3'].map(v => <SelectItem key={v} value={v} className="text-sm">{v}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs text-[var(--dfs-text-muted)] block mb-2">Lineups per Combination</Label>
              <Input type="number" min="1" max="50" value={defaultLineupsPerCombo} onChange={(e) => setDefaultLineupsPerCombo(parseInt(e.target.value) || 5)} className="w-full bg-[var(--dfs-bg-secondary)] border border-[var(--dfs-border)] text-white text-sm h-10" />
            </div>
            <Button onClick={generateCombinations} variant="secondary-action" className="w-full h-10">Generate Combinations</Button>
          </div>
        </div>
      </div>

      {combinations.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-[var(--dfs-text-secondary)]">Generated Combinations</h3>
            <div className="text-xs text-[var(--dfs-text-muted)]">Total: <span className="text-[var(--dfs-text-secondary)] font-medium">{totalLineups}</span> lineups</div>
          </div>
          <div className="overflow-auto max-h-96 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent space-y-2">
            {combinations.map(combo => (
              <div key={combo.id} className="flex items-center gap-4 p-4 bg-[var(--dfs-bg-secondary)] border border-[var(--dfs-border)] rounded-lg hover:bg-[var(--dfs-bg-hover)] transition-colors">
                <Checkbox checked={combo.enabled} onCheckedChange={() => toggleCombination(combo.id)} className="h-4 w-4 border-slate-600" />
                <div className="flex-1 min-w-0"><div className="text-sm text-white font-medium truncate">{combo.display}</div></div>
                <div className="flex items-center gap-2">
                  <Label className="text-xs text-[var(--dfs-text-muted)]">Lineups:</Label>
                  <Input type="number" min="1" max="100" value={combo.lineupsPerCombo} onChange={(e) => updateLineupsPerCombo(combo.id, parseInt(e.target.value) || 5)} className="w-16 h-8 bg-[var(--dfs-bg-secondary)] border border-[var(--dfs-border)] text-white text-xs text-center" onClick={(e) => e.stopPropagation()} />
                </div>
              </div>
            ))}
          </div>
          <div className="flex justify-end pt-2">
            <Button variant="secondary-action" className="h-10 px-6" disabled={totalLineups === 0}>Generate Lineups ({totalLineups})</Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeamCombosTab;
