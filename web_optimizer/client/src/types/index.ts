// Player and Game Types
export interface Player {
  id: number | string;
  position: 'QB' | 'RB' | 'WR' | 'TE' | 'DST' | 'K';
  name: string;
  team: string;
  opponent: string;
  salary: number;
  projection: number;
  status: 'Active' | 'Questionable' | 'Doubtful' | 'Out' | 'IR';
}

export interface Game {
  id: number | string;
  home: string;
  away: string;
  time: string;
  spread: string;
  total: string;
  edge: string;
  status: 'live' | 'upcoming' | 'final';
}

export interface PropBet {
  player: string;
  team: string;
  position: string;
  prop: string;
  edge: string;
  confidence: number;
  matchup: string;
}

// Lineup Types
export interface Lineup {
  id: string;
  players: Player[];
  totalSalary: number;
  projectedPoints: number;
  variance?: number;
}

// Form Data Types
export interface LoginFormData {
  email: string;
  password: string;
}

export interface RegisterFormData {
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  confirmPassword: string;
}

// Firebase Error Type
export interface FirebaseError extends Error {
  code: string;
  message: string;
}

// Sport and Slate Types
export type Sport = 'NFL' | 'NBA' | 'MLB' | 'NHL' | 'ESPORTS';
export type Slate = 'main' | 'early' | 'showdown' | 'afternoon' | 'prime';
export type DFSSite = 'draftkings' | 'fanduel' | 'yahoo';

// Component Props Types
export interface DashboardProps {
  onLogout: () => void;
}

export interface HomepageProps {
  onLogin: () => void;
  onSignUp: () => void;
}

export interface LoginPageProps {
  onLogin: () => void;
  onSwitchToRegister: () => void;
}

export interface RegisterPageProps {
  onRegister: () => void;
  onSwitchToLogin: () => void;
}

export interface DashboardOverviewProps {
  sport: string;
}

export interface LineupBuilderProps {
  sport: string;
  slate: string;
}

export interface PropBetFinderProps {
  sport: string;
}

export interface GameAnalysisProps {
  sport: string;
}

export interface PopularParlaysProps {
  sport: string;
}

