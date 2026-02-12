import { useState, useCallback } from 'react';
import { Sport, SPORT_CONFIGS } from '../../sport-config';
import { Player, StackType, DEFAULT_ADVANCED_QUANT_SETTINGS } from '../types';
import { dfsApi } from '../../../services/dfs-api';

interface UseOptimizerProps {
  currentSport: Sport | null;
  playerData: Player[];
  selectedPlayers: string[];
  stackSettings: StackType[];
  teamSelections: Record<number | 'all', string[]>;
  teamExposures?: Record<string, { minExp: number; maxExp: number }>;
  advancedQuantSettings: any;
  numLineups: number;
  minUnique: number;
  minSalary: number;
  sortMethod: string;
  disableKelly: boolean;
  onResultsChange: (results: any[]) => void;
  onLineupsChange: (lineups: any[]) => void;
  onActiveTabChange: (tab: string) => void;
  onPlayerDataChange?: (data: Player[]) => void;
  onPortfolioMetricsChange?: (metrics: any) => void;
}

export function useOptimizer(props: UseOptimizerProps) {
  const {
    currentSport, playerData, selectedPlayers, stackSettings,
    teamSelections, teamExposures, advancedQuantSettings, numLineups, minUnique,
    minSalary, sortMethod, disableKelly,
    onResultsChange, onLineupsChange, onActiveTabChange, onPlayerDataChange,
    onPortfolioMetricsChange,
  } = props;

  const [isOptimizing, setIsOptimizing] = useState(false);
  const [lastOptimizationTime, setLastOptimizationTime] = useState<string | null>(null);

  const syncSelectionsWithBackend = useCallback(async (selectedIds: string[]) => {
    try {
      await dfsApi.bulkUpdatePlayers({ action: 'deselect', filters: {} });
      if (selectedIds.length === 0) return;
      await Promise.all(selectedIds.map((id) => dfsApi.updatePlayer(id, { selected: true })));
    } catch (error) { console.error('Failed to sync', error); throw error; }
  }, []);

  const handleRunOptimization = useCallback(async () => {
    if (!currentSport) { alert('Select a sport before running the optimizer.'); return; }
    const sportConfig = SPORT_CONFIGS[currentSport];
    if (!sportConfig) return;
    if (playerData.length === 0) { alert('Please load player data first'); return; }
    if (selectedPlayers.length < sportConfig.lineupSize) { alert(`Please select at least ${sportConfig.lineupSize} players`); return; }

    setIsOptimizing(true);
    try {
      await dfsApi.setSport(currentSport);
      await syncSelectionsWithBackend(selectedPlayers);

      const enabledStacks = stackSettings.filter(s => s.enabled);
      const stackSizeEntries = Object.entries(teamSelections).filter(([sizeKey, teams]) => sizeKey !== 'all' && Array.isArray(teams) && teams.length > 0);
      const stackTeamsSet = new Set<string>();
      stackSizeEntries.forEach(([, teams]) => { teams.forEach(team => stackTeamsSet.add(team)); });
      if (stackTeamsSet.size === 0) { playerData.filter(p => selectedPlayers.includes(p.id)).forEach(p => stackTeamsSet.add(p.team)); }

      const stackTeams = Array.from(stackTeamsSet);
      const stackSizeValues = stackSizeEntries.map(([sk]) => Number(sk)).filter(s => !Number.isNaN(s) && s > 0);
      const minPlayersPerTeam = stackSizeValues.length > 0 ? Math.min(...stackSizeValues) : 2;
      const maxPlayersPerTeam = stackSizeValues.length > 0 ? Math.max(...stackSizeValues) : 4;

      const sortingMethodMap: Record<string, string> = { points: 'Points', value: 'Value', salary: 'Salary' };
      const exposureSettingsPayload = enabledStacks.reduce<Record<string, { min: number; max: number }>>((acc, stack) => { acc[stack.label] = { min: stack.minExp, max: stack.maxExp }; return acc; }, {});
      const globalMaxExposure = enabledStacks.length > 0 ? Math.max(...enabledStacks.map(s => s.maxExp)) : 100;

      const requestedStackSizes = Object.keys(teamSelections)
        .filter(k => k !== 'all')
        .map(Number)
        .filter(n => !isNaN(n) && n > 0);

      const teamExposuresPayload = teamExposures
        ? Object.fromEntries(
            Object.entries(teamExposures).map(([k, v]) => [k, { min: v.minExp, max: v.maxExp }])
          )
        : undefined;

      const optimizationResponse = await dfsApi.optimizeLineups({
        sport: currentSport, numLineups, minSalary, maxSalary: sportConfig.maxSalary,
        stackSettings: { enabled: enabledStacks.length > 0 && stackTeams.length > 0, teams: stackTeams, requestedStackSizes, minPlayersPerTeam, maxPlayersPerTeam },
        teamSelections, teamExposures: teamExposuresPayload, uniquePlayers: minUnique, maxExposure: globalMaxExposure,
        sortingMethod: sortingMethodMap[sortMethod] ?? 'Points', minUniquePlayersBetweenLineups: minUnique,
        disableKellySizing: disableKelly, exposureSettings: exposureSettingsPayload, contestMode: 'gpp',
        monteCarloIterations: 100, advancedQuantSettings: { ...DEFAULT_ADVANCED_QUANT_SETTINGS, ...advancedQuantSettings },
      });

      if (optimizationResponse.success) {
        const transformedResults = optimizationResponse.lineups.map((lineup: any) => ({
          id: lineup.id,
          players: lineup.players.map((p: any) => ({
            name: p.name, position: p.position, team: p.team, salary: p.salary,
            projection: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
            projectedPoints: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
          })),
          points: lineup.totalProjection,
          salary: lineup.totalSalary,
          totalPoints: lineup.totalProjection,
          totalSalary: lineup.totalSalary,
          quantMetrics: lineup.quantMetrics || null,
        }));
        onResultsChange(transformedResults);

        // Compute actual exposure % per player
        if (onPlayerDataChange && transformedResults.length > 0) {
          const totalLineups = transformedResults.length;
          const playerCounts = new Map<string, number>();
          transformedResults.forEach((result: any) => {
            (result.players || []).forEach((p: any) => {
              const name = (p.name || '').toLowerCase();
              playerCounts.set(name, (playerCounts.get(name) || 0) + 1);
            });
          });
          const updatedPlayerData = playerData.map(p => {
            const count = playerCounts.get(p.name.toLowerCase()) || 0;
            return { ...p, actualExp: parseFloat(((count / totalLineups) * 100).toFixed(1)) };
          });
          onPlayerDataChange(updatedPlayerData);
        }

        setLastOptimizationTime(new Date().toLocaleTimeString());

        const transformedLineups = (optimizationResponse.lineups || []).map((lineup: any) => ({
          ...lineup,
          quantMetrics: lineup.quantMetrics || null,
          players: lineup.players.map((p: any) => ({
            ...p,
            projection: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
            projectedPoints: p.projection || p.Predicted_DK_Points || p.projectedPoints || 0,
          }))
        }));
        onLineupsChange(transformedLineups);
        onActiveTabChange('lineups');

        const portfolioMetrics = optimizationResponse.summary?.portfolioMetrics || optimizationResponse.portfolioMetrics || null;
        if (portfolioMetrics && onPortfolioMetricsChange) {
          onPortfolioMetricsChange(portfolioMetrics);
        }

        const warningsText = optimizationResponse.warnings?.length ? `\n\nWarnings:\n${optimizationResponse.warnings.join('\n')}` : '';
        alert(`Generated ${transformedResults.length} optimal lineups!\nAvg Projection: ${optimizationResponse.summary.avgProjection.toFixed(1)} pts${warningsText}`);
      } else {
        alert(`Optimization failed: ${optimizationResponse.error || 'Unknown error'}`);
      }
    } catch (error) {
      const apiError = dfsApi.handleApiError(error);
      alert(`Optimization failed: ${apiError.message}`);
    } finally {
      setIsOptimizing(false);
    }
  }, [currentSport, playerData, selectedPlayers, stackSettings, teamSelections, teamExposures, advancedQuantSettings, numLineups, minUnique, minSalary, sortMethod, disableKelly, onResultsChange, onLineupsChange, onActiveTabChange, onPlayerDataChange, onPortfolioMetricsChange, syncSelectionsWithBackend]);

  const handleExportDraftKings = useCallback(async () => {
    if (!currentSport) return;
    try {
      const response = await fetch(`/api/export/draftkings?sport=${currentSport}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentSport.toLowerCase()}_lineups_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a); a.click();
      window.URL.revokeObjectURL(url); document.body.removeChild(a);
      alert(`Exported lineups`);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed');
    }
  }, [currentSport]);

  return {
    isOptimizing,
    lastOptimizationTime,
    handleRunOptimization,
    handleExportDraftKings,
    syncSelectionsWithBackend,
  };
}
