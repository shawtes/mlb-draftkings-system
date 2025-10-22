/**
 * Sport Configuration for DFS Optimizer
 * Defines positions, stack types, and defaults for MLB and NFL
 */

export type Sport = 'MLB' | 'NFL' | 'NBA';

export interface SportConfig {
  positions: string[];
  positionLabels: Record<string, string>;
  positionCounts: Record<string, number>;
  stackTypes: string[];
  defaultMinSalary: number;
  maxSalary: number;
  lineupSize: number;
  salaryCapDescription: string;
}

export const SPORT_CONFIGS: Record<Sport, SportConfig> = {
  MLB: {
    positions: ['P', 'C', '1B', '2B', '3B', 'SS', 'OF'],
    positionLabels: {
      'P': 'Pitcher',
      'C': 'Catcher',
      '1B': 'First Base',
      '2B': 'Second Base',
      '3B': 'Third Base',
      'SS': 'Shortstop',
      'OF': 'Outfield'
    },
    positionCounts: {
      'P': 2,
      'C': 1,
      '1B': 1,
      '2B': 1,
      '3B': 1,
      'SS': 1,
      'OF': 3
    },
    stackTypes: [
      'Same Team (2+)',
      'Same Team (3+)',
      'Same Team (4+)',
      'Pitcher + Batter',
      'Batter Stack',
      'No Stacks'
    ],
    defaultMinSalary: 45000,
    maxSalary: 50000,
    lineupSize: 10,
    salaryCapDescription: 'DraftKings MLB Classic'
  },
  NFL: {
    positions: ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST'],
    positionLabels: {
      'QB': 'Quarterback',
      'RB': 'Running Back',
      'WR': 'Wide Receiver',
      'TE': 'Tight End',
      'FLEX': 'RB/WR/TE',
      'DST': 'Defense/Special Teams'
    },
    positionCounts: {
      'QB': 1,
      'RB': 2,
      'WR': 3,
      'TE': 1,
      'FLEX': 1,
      'DST': 1
    },
    stackTypes: [
      'QB + WR',
      'QB + 2 WR',
      'QB + WR + TE',
      'QB + WR + RB',
      'QB + 2 WR + TE',
      'Game Stack',
      'Bring-Back',
      'No Stack'
    ],
    defaultMinSalary: 48000,
    maxSalary: 50000,
    lineupSize: 9,
    salaryCapDescription: 'DraftKings NFL Classic'
  },
  NBA: {
    positions: ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'],
    positionLabels: {
      'PG': 'Point Guard',
      'SG': 'Shooting Guard',
      'SF': 'Small Forward',
      'PF': 'Power Forward',
      'C': 'Center',
      'G': 'Guard',
      'F': 'Forward',
      'UTIL': 'Utility'
    },
    positionCounts: {
      'PG': 1,
      'SG': 1,
      'SF': 1,
      'PF': 1,
      'C': 1,
      'G': 1,
      'F': 1,
      'UTIL': 1
    },
    stackTypes: [
      'Same Team (2+)',
      'Same Team (3+)',
      'Same Team (4+)',
      'Same Team (5+)',
      'PG + SG',
      'Stars + Value',
      'Balanced',
      'No Stack'
    ],
    defaultMinSalary: 48000,
    maxSalary: 50000,
    lineupSize: 8,
    salaryCapDescription: 'DraftKings NBA Classic'
  }
};

/**
 * Get position filter options for a sport
 */
export function getPositionFilters(sport: Sport): Array<{ id: string; label: string }> {
  const config = SPORT_CONFIGS[sport];
  
  if (sport === 'MLB') {
    return [
      { id: 'all-batters', label: 'All Batters' },
      ...config.positions.map(pos => ({ id: pos, label: pos }))
    ];
  } else if (sport === 'NFL') {
    return [
      { id: 'all-offense', label: 'All Offense' },
      ...config.positions.map(pos => ({ id: pos, label: pos }))
    ];
  } else {
    // NBA
    return [
      { id: 'all', label: 'All Players' },
      ...config.positions.map(pos => ({ id: pos, label: pos }))
    ];
  }
}

/**
 * Filter players by position based on sport
 */
export function filterPlayersByPosition(players: any[], position: string, sport: Sport): any[] {
  if (position === 'all-batters' && sport === 'MLB') {
    return players.filter(p => !p.position.includes('P'));
  }
  if (position === 'all-offense' && sport === 'NFL') {
    return players.filter(p => p.position !== 'DST');
  }
  if (position === 'all') {
    return players;
  }
  
  return players.filter(p => {
    if (sport === 'MLB') {
      // MLB positions can have multi-position eligibility (e.g., "1B/OF")
      return p.position.includes(position);
    } else {
      // NFL positions are single (e.g., "QB")
      return p.position === position;
    }
  });
}

/**
 * Get position count for filtering
 */
export function getPositionCount(players: any[], position: string, sport: Sport): number {
  const filtered = filterPlayersByPosition(players, position, sport);
  return filtered.length;
}

/**
 * Get stack type descriptions
 */
export function getStackDescription(stackType: string, sport: Sport): string {
  const descriptions: Record<string, string> = {
    // MLB
    'Same Team (2+)': 'At least 2 players from same team',
    'Same Team (3+)': 'At least 3 players from same team',
    'Same Team (4+)': 'At least 4 players from same team',
    'Pitcher + Batter': 'Pitcher with batter from opposing team',
    'Batter Stack': 'Multiple batters from same team',
    
    // NFL
    'QB + WR': 'QB with 1 WR from same team',
    'QB + 2 WR': 'QB with 2 WRs from same team',
    'QB + WR + TE': 'QB with WR and TE from same team',
    'QB + WR + RB': 'QB with WR and RB from same team',
    'QB + 2 WR + TE': 'QB with 2 WRs and TE from same team',
    'Game Stack': 'QB + WR (Team A) + WR (Team B)',
    'Bring-Back': 'QB + WR (Team A) + RB (Team B)',
    'No Stack': 'No correlation requirements'
  };
  
  return descriptions[stackType] || stackType;
}

/**
 * Get sport icon emoji
 */
export function getSportIcon(sport: Sport): string {
  if (sport === 'MLB') return '⚾';
  if (sport === 'NFL') return '🏈';
  return '🏀';
}

/**
 * Get sport name
 */
export function getSportName(sport: Sport): string {
  if (sport === 'MLB') return 'Baseball';
  if (sport === 'NFL') return 'Football';
  return 'Basketball';
}

/**
 * Validate lineup based on sport rules
 */
export function validateLineup(players: any[], sport: Sport): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];
  const config = SPORT_CONFIGS[sport];
  
  // Check lineup size
  if (players.length !== config.lineupSize) {
    errors.push(`Lineup must have exactly ${config.lineupSize} players`);
  }
  
  // Check position requirements
  const positionCounts: Record<string, number> = {};
  players.forEach(player => {
    const pos = player.position;
    positionCounts[pos] = (positionCounts[pos] || 0) + 1;
  });
  
  // Validate each position requirement
  Object.entries(config.positionCounts).forEach(([pos, required]) => {
    const actual = positionCounts[pos] || 0;
    if (actual < required) {
      errors.push(`Need ${required} ${pos}, have ${actual}`);
    }
  });
  
  // Check salary cap
  const totalSalary = players.reduce((sum, p) => sum + (p.salary || 0), 0);
  if (totalSalary > config.maxSalary) {
    errors.push(`Total salary ${totalSalary} exceeds cap of ${config.maxSalary}`);
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Format lineup for DraftKings CSV export
 */
export function formatLineupForExport(lineup: any[], sport: Sport): Record<string, string> {
  const config = SPORT_CONFIGS[sport];
  const formatted: Record<string, string> = {};
  
  if (sport === 'MLB') {
    formatted['P'] = lineup.find(p => p.position.includes('P'))?.name || '';
    formatted['P/UTIL'] = lineup.filter(p => p.position.includes('P'))[1]?.name || '';
    formatted['C'] = lineup.find(p => p.position.includes('C'))?.name || '';
    formatted['1B'] = lineup.find(p => p.position.includes('1B'))?.name || '';
    formatted['2B'] = lineup.find(p => p.position.includes('2B'))?.name || '';
    formatted['3B'] = lineup.find(p => p.position.includes('3B'))?.name || '';
    formatted['SS'] = lineup.find(p => p.position.includes('SS'))?.name || '';
    formatted['OF'] = lineup.filter(p => p.position.includes('OF'))[0]?.name || '';
    formatted['OF '] = lineup.filter(p => p.position.includes('OF'))[1]?.name || '';
    formatted['OF  '] = lineup.filter(p => p.position.includes('OF'))[2]?.name || '';
  } else {
    // NFL
    formatted['QB'] = lineup.find(p => p.position === 'QB')?.name || '';
    formatted['RB'] = lineup.filter(p => p.position === 'RB')[0]?.name || '';
    formatted['RB '] = lineup.filter(p => p.position === 'RB')[1]?.name || '';
    formatted['WR'] = lineup.filter(p => p.position === 'WR')[0]?.name || '';
    formatted['WR '] = lineup.filter(p => p.position === 'WR')[1]?.name || '';
    formatted['WR  '] = lineup.filter(p => p.position === 'WR')[2]?.name || '';
    formatted['TE'] = lineup.find(p => p.position === 'TE')?.name || '';
    formatted['FLEX'] = lineup[7]?.name || ''; // 8th player
    formatted['DST'] = lineup.find(p => p.position === 'DST')?.name || '';
  }
  
  return formatted;
}

