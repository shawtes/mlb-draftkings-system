import axios from 'axios';

const API_BASE_URL = '/api';

export interface BulkUpdatePlayersPayload {
  action: 'select' | 'deselect' | 'update_exposure';
  filters?: {
    position?: string;
    team?: string;
    salaryMin?: number;
    salaryMax?: number;
  };
  settings?: {
    minExposure?: number;
    maxExposure?: number;
  };
}

export interface OptimizeRequest {
  sport: 'MLB' | 'NFL' | 'NBA';
  numLineups: number;
  minSalary: number;
  maxSalary: number;
  stackSettings: {
    enabled: boolean;
    teams: string[];
    requestedStackSizes?: number[];
    minPlayersPerTeam?: number;
    maxPlayersPerTeam?: number;
  };
  uniquePlayers: number;
  maxExposure: number;
  sortingMethod?: string;
  minUniquePlayersBetweenLineups?: number;
  enableRiskManagement?: boolean;
  disableKellySizing?: boolean;
  stackTypes?: Record<string, boolean>;
  exposureSettings?: Record<string, { min: number; max: number }>;
  riskTolerance?: string;
  bankroll?: number;
  contestMode?: string;
  monteCarloIterations?: number;
  teamSelections?: Record<number | 'all', string[]>;
  teamExposures?: Record<string, { min: number; max: number }>;
  advancedQuantSettings?: Record<string, unknown>;
  playerExposures?: Record<string, { min: number; max: number }>;
}

export interface OptimizeResponse {
  success: boolean;
  sport: string;
  optimizationId: string;
  lineups: any[];
  error?: string;
  warnings?: string[];
  summary: {
    totalLineups: number;
    avgProjection: number;
    avgSalary: number;
    topProjection: number;
    strategies: string[];
  };
}

// Player endpoints
const uploadPlayers = async (file: File) => {
  const formData = new FormData();
  formData.append('playersFile', file);
  const response = await axios.post(`${API_BASE_URL}/upload-players`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

const getPlayers = async () => {
  const response = await axios.get(`${API_BASE_URL}/players`);
  return response.data;
};

const updatePlayer = async (playerId: string, updates: Record<string, unknown>) => {
  const response = await axios.put(`${API_BASE_URL}/players/${playerId}`, updates);
  return response.data;
};

const bulkUpdatePlayers = async (payload: BulkUpdatePlayersPayload) => {
  const response = await axios.put(`${API_BASE_URL}/players/bulk`, payload);
  return response.data;
};

const setSport = async (sport: 'MLB' | 'NFL' | 'NBA') => {
  const response = await axios.post(`${API_BASE_URL}/set-sport`, { sport });
  return response.data;
};

// Optimization endpoints
const optimizeLineups = async (settings: OptimizeRequest): Promise<OptimizeResponse> => {
  const response = await axios.post(`${API_BASE_URL}/optimize`, settings);
  return response.data;
};

const getOptimizationStatus = async () => {
  const response = await axios.get(`${API_BASE_URL}/optimize/status`);
  return response.data;
};

const cancelOptimization = async () => {
  const response = await axios.post(`${API_BASE_URL}/optimize/cancel`);
  return response.data;
};

// Export endpoints
const exportLineups = async (lineups: any[], format: 'csv' | 'draftkings' | 'fanduel') => {
  const response = await axios.post(`${API_BASE_URL}/export/${format}`, { lineups }, {
    responseType: 'blob',
  });
  return response.data;
};

// Projection endpoints
const getProjections = async (sport: string) => {
  const response = await axios.get(`${API_BASE_URL}/projections/${sport}`);
  return response.data;
};

const uploadProjections = async (file: File, sport: string) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('sport', sport);
  const response = await axios.post(`${API_BASE_URL}/projections/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

const getTeams = async (): Promise<string[]> => {
  const response = await axios.get(`${API_BASE_URL}/teams`);
  return response.data;
};

// Favorites endpoints
const saveFavorites = async (name: string, playerIds: string[]) => {
  const response = await axios.post(`${API_BASE_URL}/favorites`, { name, playerIds });
  return response.data;
};

const getFavorites = async () => {
  const response = await axios.get(`${API_BASE_URL}/favorites`);
  return response.data;
};

// Get contest formats
const getContestFormats = async () => {
  const response = await axios.get(`${API_BASE_URL}/contest-formats`);
  return response.data;
};

// Stack analysis
const getStackAnalysis = async (teamStacks: any[], players: any[]) => {
  const response = await axios.get(`${API_BASE_URL}/stack-analysis`, {
    params: { teamStacks: JSON.stringify(teamStacks), players: JSON.stringify(players) },
  });
  return response.data;
};

// Get results
const getResults = async () => {
  const response = await axios.get(`${API_BASE_URL}/results`);
  return response.data;
};

// Advanced export with custom settings
const advancedExport = async (lineups: any[], options: {
  format: string;
  includeProjections?: boolean;
  includeOwnership?: boolean;
  customFields?: string[];
}) => {
  const response = await axios.post(`${API_BASE_URL}/export-advanced`, {
    lineups,
    ...options,
  }, {
    responseType: 'blob',
  });
  return response.data;
};

// Upload ownership CSV
const uploadOwnership = async (file: File) => {
  const formData = new FormData();
  formData.append('ownershipFile', file);
  const response = await axios.post(`${API_BASE_URL}/upload-ownership`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// Upload projection source for blending
const uploadProjectionSource = async (file: File, sourceName: string) => {
  const formData = new FormData();
  formData.append('projectionFile', file);
  formData.append('sourceName', sourceName);
  const response = await axios.post(`${API_BASE_URL}/upload-projection-source`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// Error handling wrapper
const handleApiError = (error: unknown) => {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      console.error('API Error:', error.response.status, error.response.data);
      return {
        message: error.response.data.message || 'An error occurred',
        status: error.response.status,
      };
    } else if (error.request) {
      console.error('Network Error:', error.request);
      return {
        message: 'Network error - please check your connection',
        status: 0,
      };
    }
  }
  console.error('Unknown Error:', error);
  return {
    message: 'An unexpected error occurred',
    status: -1,
  };
};

export const dfsApi = {
  uploadPlayers,
  getPlayers,
  updatePlayer,
  bulkUpdatePlayers,
  setSport,
  optimizeLineups,
  getOptimizationStatus,
  cancelOptimization,
  exportLineups,
  advancedExport,
  getProjections,
  uploadProjections,
  getTeams,
  saveFavorites,
  getFavorites,
  getContestFormats,
  getStackAnalysis,
  getResults,
  uploadOwnership,
  uploadProjectionSource,
  handleApiError,
};
