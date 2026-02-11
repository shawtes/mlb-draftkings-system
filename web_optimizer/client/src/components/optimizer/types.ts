import { Sport } from '../sport-config';

export interface Player {
  id: string;
  name: string;
  team: string;
  position: string;
  salary: number;
  projectedPoints: number;
  minExp: number;
  maxExp: number;
  actualExp?: number;
  selected: boolean;
  ownership: number;
  locked: boolean;
  excluded: boolean;
  originalProjection?: number;
  projectionEdited?: boolean;
  ceiling?: number;
  floor?: number;
  stdDev?: number;
  leverageScore?: number;
  opponent?: string;
  projectionSources?: Array<{ source: string; value: number; weight: number }>;
}

export interface Team {
  abbr: string;
  status: 'Active' | 'Postponed' | 'Final';
  gameTime: string;
  projRuns: number;
  minExp: number;
  maxExp: number;
  actualExp?: number;
  playerCount: number;
  batterCount: number;
}

export interface StackType {
  id: string;
  label: string;
  minExp: number;
  maxExp: number;
  lineupExp?: number;
  poolExp?: number;
  entryExp?: number;
  enabled: boolean;
}

export interface TeamCombination {
  id: string;
  teams: string[];
  stackSizes: number[];
  display: string;
  lineupsPerCombo: number;
  status: 'ready' | 'generating' | 'complete' | 'error';
  enabled: boolean;
}

export interface AdvancedQuantSettings {
  enabled: boolean;
  strategy: string;
  riskTolerance: number;
  varConfidence: number;
  targetVolatility: number;
  monteCarloSims: number;
  timeHorizon: number;
  garchP: number;
  garchQ: number;
  lookbackPeriod: number;
  copulaFamily: string;
  dependencyThreshold: number;
  maxKellyFraction: number;
  expectedWinRate: number;
}

export const DEFAULT_ADVANCED_QUANT_SETTINGS: AdvancedQuantSettings = {
  enabled: false,
  strategy: 'combined',
  riskTolerance: 1.0,
  varConfidence: 0.95,
  targetVolatility: 0.20,
  monteCarloSims: 10000,
  timeHorizon: 1,
  garchP: 1,
  garchQ: 1,
  lookbackPeriod: 100,
  copulaFamily: 'gaussian',
  dependencyThreshold: 0.3,
  maxKellyFraction: 0.25,
  expectedWinRate: 0.20,
};

export interface FavoriteLineup {
  id: string;
  players: Player[];
  totalPoints: number;
  totalSalary: number;
  runNumber: number;
  dateAdded: string;
  selected: boolean;
}

export type BuildSport = Sport | null;

export interface BuildState {
  id: string;
  name: string;
  sport: BuildSport;
  activeTab: string;
  playerData: Player[];
  selectedPlayers: string[];
  teamSelections: Record<number | 'all', string[]>;
  teamExposures: Record<string, { minExp: number; maxExp: number }>;
  stackSettings: StackType[];
  advancedQuantSettings: any;
  results: any[];
  minSalary: number | null;
}

// Verbose logging
const DEBUG_LOG = true;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const vlog = (...args: any[]) => {
  if (DEBUG_LOG) {
    // eslint-disable-next-line no-console
    console.log('[DFSOptimizer]', ...args);
  }
};
