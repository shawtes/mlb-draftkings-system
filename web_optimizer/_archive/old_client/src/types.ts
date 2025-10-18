// Types for the DFS Optimizer Application

export interface Player {
  id: string;
  name: string;
  position: string;
  team: string;
  salary: number;
  projectedPoints: number;
  ownership: number;
  value: number;
  injured: boolean;
  confirmed: boolean;
  news?: string;
  gameInfo?: string;
  opponent?: string;
  isHome?: boolean;
  weatherInfo?: string;
  ceiling?: number;
  floor?: number;
  stdDev?: number;
}

export interface TeamStack {
  team: string;
  stackSize: number;
  players: Player[];
  minExposure: number;
  maxExposure: number;
  actualExposure: number;
  selected: boolean;
}

export interface StackExposure {
  stackId: string;
  team: string;
  stackSize: number;
  minExposure: number;
  maxExposure: number;
  actualExposure: number;
  priority: number;
}

export interface OptimizationSettings {
  maxSalary: number;
  minProjectedPoints: number;
  maxOwnership: number;
  uniqueness: number;
  stackingEnabled: boolean;
  teamStacks: TeamStack[];
  stackExposures: StackExposure[];
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  diversification: boolean;
}

export interface LineupResult {
  id: string;
  players: Player[];
  totalSalary: number;
  projectedPoints: number;
  totalOwnership: number;
  uniqueness: number;
  stacks: string[];
  value: number;
  rank: number;
}

export interface Contest {
  id: string;
  name: string;
  sport: string;
  site: string;
  entryFee: number;
  totalPrizes: number;
  entries: number;
  maxEntries: number;
  startTime: string;
  positions: ContestPosition[];
}

export interface ContestPosition {
  position: string;
  minSalary: number;
  maxSalary: number;
  count: number;
}

export interface FavoriteGroup {
  id: string;
  name: string;
  players: Player[];
  description?: string;
  createdAt: string;
  updatedAt: string;
}

export interface SystemStatus {
  connected: boolean;
  playersLoaded: number;
  optimizationProgress: number;
  lastUpdate: string;
  errors: string[];
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export type PlayerPosition = 'QB' | 'RB' | 'WR' | 'TE' | 'K' | 'DST';
export type ContestType = 'cash' | 'gpp' | 'tournament';
export type OptimizationStatus = 'idle' | 'running' | 'completed' | 'error';
