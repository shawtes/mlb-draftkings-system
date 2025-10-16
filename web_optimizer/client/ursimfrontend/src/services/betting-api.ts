import axios from 'axios';

const API_BASE_URL = '/api';

export interface BetSelection {
  id: string;
  player: string;
  team: string;
  prop: string;
  line: number;
  odds: number;
  type: 'over' | 'under';
}

export interface PayoutCalculation {
  totalOdds: string;
  potentialPayout: number;
  profit: number;
  probability: number;
  kellyRecommendation?: {
    recommendedStake: number;
    fractionalStake: number;
    kellyPercentage: number;
  };
}

export interface PropBet {
  id: string;
  player: string;
  team: string;
  position: string;
  opponent: string;
  prop: string;
  line: number;
  overOdds: number;
  underOdds: number;
  projection?: number;
  edge?: number;
  confidence?: number;
  hitRate?: number;
  trend?: 'up' | 'down' | 'neutral';
}

export interface Game {
  id: string;
  home: string;
  away: string;
  time: string;
  homeSpread: number;
  awaySpread: number;
  total: number;
  status: 'scheduled' | 'live' | 'final';
  homeScore?: number;
  awayScore?: number;
}

// Calculate parlay or straight bet payout
export const calculatePayout = async (
  selections: BetSelection[],
  betType: 'straight' | 'parlay',
  stake: number
): Promise<PayoutCalculation> => {
  try {
    const response = await axios.post(`${API_BASE_URL}/bets/calculate-payout`, {
      selections,
      betType,
      stake,
    });
    return response.data;
  } catch (error) {
    console.error('Error calculating payout:', error);
    // Fallback to client-side calculation
    return calculatePayoutFallback(selections, betType, stake);
  }
};

// Fallback client-side calculation
function calculatePayoutFallback(
  selections: BetSelection[],
  betType: 'straight' | 'parlay',
  stake: number
): PayoutCalculation {
  if (betType === 'straight' && selections.length === 1) {
    const odds = selections[0].odds;
    const multiplier = odds > 0 ? odds / 100 : 100 / Math.abs(odds);
    const profit = stake * multiplier;
    const probability = odds > 0 ? 100 / (odds + 100) * 100 : Math.abs(odds) / (Math.abs(odds) + 100) * 100;
    
    return {
      totalOdds: odds > 0 ? `+${odds}` : `${odds}`,
      potentialPayout: stake + profit,
      profit,
      probability,
    };
  }
  
  // Parlay calculation
  let decimalOdds = 1;
  let totalProbability = 1;
  
  selections.forEach(sel => {
    const odds = sel.odds;
    const decimal = odds > 0 ? (odds / 100) + 1 : (100 / Math.abs(odds)) + 1;
    decimalOdds *= decimal;
    
    const prob = odds > 0 ? 100 / (odds + 100) : Math.abs(odds) / (Math.abs(odds) + 100);
    totalProbability *= prob;
  });
  
  const potentialPayout = stake * decimalOdds;
  const profit = potentialPayout - stake;
  const americanOdds = decimalOdds >= 2 ? Math.round((decimalOdds - 1) * 100) : Math.round(-100 / (decimalOdds - 1));
  
  return {
    totalOdds: americanOdds > 0 ? `+${americanOdds}` : `${americanOdds}`,
    potentialPayout,
    profit,
    probability: totalProbability * 100,
  };
}

// Place a bet
export const placeBet = async (betData: {
  selections: BetSelection[];
  betType: 'straight' | 'parlay';
  stake: number;
  oddsFormat: 'american' | 'decimal' | 'fractional';
}) => {
  const response = await axios.post(`${API_BASE_URL}/bets/place`, betData);
  return response.data;
};

// Get props for a sport with filters
export const getProps = async (
  sport: string,
  filters?: {
    date?: string;
    team?: string;
    player?: string;
    propType?: string;
    minEdge?: number;
  }
): Promise<PropBet[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/odds/props`, {
      params: { sport, ...filters },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching props:', error);
    return [];
  }
};

// Get games for a sport
export const getGames = async (
  sport: string,
  slate?: string
): Promise<Game[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/odds/games`, {
      params: { sport, slate },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching games:', error);
    return [];
  }
};

// Convert odds between formats
export const convertOdds = async (
  odds: number,
  toFormat: 'american' | 'decimal' | 'fractional'
) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/odds/convert`, {
      params: { odds, toFormat },
    });
    return response.data;
  } catch (error) {
    console.error('Error converting odds:', error);
    // Fallback conversion
    return convertOddsFallback(odds, toFormat);
  }
};

function convertOddsFallback(
  americanOdds: number,
  toFormat: 'american' | 'decimal' | 'fractional'
) {
  const decimal = americanOdds > 0 
    ? (americanOdds / 100) + 1 
    : (100 / Math.abs(americanOdds)) + 1;
  
  const impliedProbability = americanOdds > 0
    ? 100 / (americanOdds + 100) * 100
    : Math.abs(americanOdds) / (Math.abs(americanOdds) + 100) * 100;
  
  let fractional = '';
  if (americanOdds > 0) {
    fractional = `${americanOdds}/100`;
  } else {
    fractional = `100/${Math.abs(americanOdds)}`;
  }
  
  return {
    american: americanOdds > 0 ? `+${americanOdds}` : `${americanOdds}`,
    decimal: decimal.toFixed(2),
    fractional,
    impliedProbability: impliedProbability.toFixed(1),
  };
}

export const bettingApi = {
  calculatePayout,
  placeBet,
  getProps,
  getGames,
  convertOdds,
};

