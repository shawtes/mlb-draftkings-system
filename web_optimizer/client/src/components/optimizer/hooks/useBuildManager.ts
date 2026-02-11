import { useState, useCallback } from 'react';
import { Sport, SPORT_CONFIGS } from '../../sport-config';
import { BuildState, BuildSport, StackType, DEFAULT_ADVANCED_QUANT_SETTINGS } from '../types';

function createEmptyTeamSelections(): Record<number | 'all', string[]> {
  return { all: [], 2: [], 3: [], 4: [], 5: [] };
}

function initializeStackSettings(sport: Sport): StackType[] {
  const config = SPORT_CONFIGS[sport];
  return config.stackTypes.map((stackType, index) => ({
    id: `stack-${index}`,
    label: stackType,
    minExp: 0,
    maxExp: 100,
    enabled: false,
  }));
}

function createBuildState(id: string, name: string, sport: BuildSport): BuildState {
  return {
    id,
    name,
    sport,
    activeTab: 'team-combos',
    playerData: [],
    selectedPlayers: [],
    teamSelections: createEmptyTeamSelections(),
    teamExposures: {},
    stackSettings: sport ? initializeStackSettings(sport) : [],
    advancedQuantSettings: { ...DEFAULT_ADVANCED_QUANT_SETTINGS },
    results: [],
    minSalary: sport ? SPORT_CONFIGS[sport].defaultMinSalary : null,
  };
}

export function useBuildManager() {
  const [builds, setBuilds] = useState<BuildState[]>([
    createBuildState('build-1', 'Build 1', null),
  ]);
  const [activeBuildId, setActiveBuildId] = useState<string>('build-1');

  const currentBuild = builds.find(build => build.id === activeBuildId) || builds[0];
  const currentSport = currentBuild?.sport ?? null;
  const sportConfig = currentSport ? SPORT_CONFIGS[currentSport] : undefined;
  const sportLocked = Boolean(currentSport);

  const getHighestBuildNumber = (buildList: BuildState[]) =>
    buildList.reduce((max, build) => {
      const match = build.id.match(/build-(\d+)/);
      if (!match) return max;
      const parsed = parseInt(match[1], 10);
      return Number.isNaN(parsed) ? max : Math.max(max, parsed);
    }, 0);

  const addNewBuild = useCallback(() => {
    if (builds.length >= 5) return;
    const newBuildNumber = getHighestBuildNumber(builds) + 1;
    const newBuild = createBuildState(`build-${newBuildNumber}`, `Build ${newBuildNumber}`, null);
    setBuilds(prev => [...prev, newBuild]);
    setActiveBuildId(newBuild.id);
  }, [builds]);

  const removeBuild = useCallback((buildId: string) => {
    if (builds.length <= 1) return;
    setBuilds(prev => {
      const newBuilds = prev.filter(build => build.id !== buildId);
      if (buildId === activeBuildId) setActiveBuildId(newBuilds[0].id);
      return newBuilds;
    });
  }, [builds.length, activeBuildId]);

  const switchBuild = useCallback((buildId: string) => {
    setActiveBuildId(buildId);
  }, []);

  const updateCurrentBuild = useCallback((updates: Partial<BuildState>) => {
    setBuilds(prev => prev.map(build =>
      build.id === activeBuildId ? { ...build, ...updates } : build
    ));
  }, [activeBuildId]);

  const handleBuildSportChange = useCallback((sport: Sport) => {
    if (!currentBuild || currentBuild.sport) return;
    updateCurrentBuild({
      sport,
      activeTab: 'team-combos',
      playerData: [],
      selectedPlayers: [],
      teamSelections: createEmptyTeamSelections(),
      teamExposures: {},
      stackSettings: initializeStackSettings(sport),
      advancedQuantSettings: { ...DEFAULT_ADVANCED_QUANT_SETTINGS },
      results: [],
      minSalary: SPORT_CONFIGS[sport].defaultMinSalary,
    });
  }, [currentBuild, updateCurrentBuild]);

  return {
    builds,
    activeBuildId,
    currentBuild,
    currentSport,
    sportConfig,
    sportLocked,
    addNewBuild,
    removeBuild,
    switchBuild,
    updateCurrentBuild,
    handleBuildSportChange,
    initializeStackSettings,
    createEmptyTeamSelections,
  };
}
