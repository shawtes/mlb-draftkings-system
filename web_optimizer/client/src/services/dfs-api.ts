import axios from 'axios';
import type { Player, LineupResult, OptimizationSettings } from '../types/dfs-types';

const API_BASE_URL = '/api';

// Player endpoints
export const uploadPlayers = async (file: File) => {
  const formData = new FormData();
  formData.append('playersFile', file); // Backend expects 'playersFile'
  const response = await axios.post(`${API_BASE_URL}/upload-players`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const getPlayers = async (): Promise<Player[]> => {
  const response = await axios.get(`${API_BASE_URL}/players`);
  return response.data;
};

export const updatePlayer = async (playerId: string, updates: Partial<Player>) => {
  const response = await axios.patch(`${API_BASE_URL}/players/${playerId}`, updates);
  return response.data;
};

// Optimization endpoints
export const optimizeLineups = async (settings: OptimizationSettings): Promise<LineupResult[]> => {
  const response = await axios.post(`${API_BASE_URL}/optimize`, settings);
  return response.data;
};

export const getOptimizationStatus = async () => {
  const response = await axios.get(`${API_BASE_URL}/optimize/status`);
  return response.data;
};

export const cancelOptimization = async () => {
  const response = await axios.post(`${API_BASE_URL}/optimize/cancel`);
  return response.data;
};

// Export endpoints
export const exportLineups = async (lineups: LineupResult[], format: 'csv' | 'draftkings' | 'fanduel') => {
  const response = await axios.post(`${API_BASE_URL}/export/${format}`, { lineups }, {
    responseType: 'blob',
  });
  return response.data;
};

// Projection endpoints
export const getProjections = async (sport: string) => {
  const response = await axios.get(`${API_BASE_URL}/projections/${sport}`);
  return response.data;
};

export const uploadProjections = async (file: File, sport: string) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('sport', sport);
  const response = await axios.post(`${API_BASE_URL}/projections/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// Bulk update players
export const bulkUpdatePlayers = async (updates: Array<{ id: string; updates: Partial<Player> }>) => {
  const response = await axios.put(`${API_BASE_URL}/players/bulk`, { updates });
  return response.data;
};

// Get teams
export const getTeams = async (): Promise<string[]> => {
  const response = await axios.get(`${API_BASE_URL}/teams`);
  return response.data;
};

// Favorites endpoints
export const saveFavorites = async (name: string, playerIds: string[]) => {
  const response = await axios.post(`${API_BASE_URL}/favorites`, { name, playerIds });
  return response.data;
};

export const getFavorites = async () => {
  const response = await axios.get(`${API_BASE_URL}/favorites`);
  return response.data;
};

// Get contest formats
export const getContestFormats = async () => {
  const response = await axios.get(`${API_BASE_URL}/contest-formats`);
  return response.data;
};

// Stack analysis
export const getStackAnalysis = async (teamStacks: any[], players: Player[]) => {
  const response = await axios.get(`${API_BASE_URL}/stack-analysis`, {
    params: { teamStacks: JSON.stringify(teamStacks), players: JSON.stringify(players) },
  });
  return response.data;
};

// Get results
export const getResults = async (): Promise<LineupResult[]> => {
  const response = await axios.get(`${API_BASE_URL}/results`);
  return response.data;
};

// Advanced export with custom settings
export const advancedExport = async (lineups: LineupResult[], options: {
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

// Error handling wrapper
export const handleApiError = (error: any) => {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.status, error.response.data);
      return {
        message: error.response.data.message || 'An error occurred',
        status: error.response.status,
      };
    } else if (error.request) {
      // Request made but no response
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
  handleApiError,
};


