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
      '5/4',
      '5/3',
      '5/2',
      '4/4',
      '4/3',
      '4/2',
      '3/3/2',
      '5-Man Batter Stack',
      '4-Man Batter Stack',
      'Wrap-Around Stack',
      'Pitcher vs Weak Lineup',
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
      'QB + 2WR',
      'QB + WR + TE',
      'QB + 2WR + TE',
      '4/3 Game Stack',
      '3/2 Game Stack',
      '3/2/1',
      'Bring-Back',
      'Run-Back',
      'Mini Stack (WR+WR)',
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
      '4/2 Game Stack',
      '3/3 Game Stack',
      '3/2/2',
      '2/2/2/2',
      'Game Environment (2+)',
      'Game Environment (3+)',
      'Pace Stack',
      'Stars + Value',
      'Mini Stack (PG+Wing)',
      'Blowout Fade',
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
  
  // NBA flex positions
  if (sport === 'NBA') {
    if (position === 'G') {
      // Guard flex: any PG or SG
      return players.filter(p => p.position.includes('PG') || p.position.includes('SG'));
    }
    if (position === 'F') {
      // Forward flex: any SF or PF
      return players.filter(p => p.position.includes('SF') || p.position.includes('PF'));
    }
    if (position === 'UTIL') {
      // Utility: any position
      return players;
    }
  }
  
  return players.filter(p => {
    if (sport === 'MLB') {
      // MLB positions can have multi-position eligibility (e.g., "1B/OF")
      return p.position.includes(position);
    } else if (sport === 'NBA') {
      // NBA positions can have multi-position eligibility (e.g., "PG/SG")
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
    // NFL — correlation-backed stacking
    'QB + WR': 'QB with 1 WR from same team (r=+0.54)',
    'QB + 2WR': 'QB with 2 WRs from same team — double stack for ceiling (r=+0.54/+0.41)',
    'QB + WR + TE': 'QB + WR + TE same team — red zone correlation (r=+0.54/+0.38)',
    'QB + 2WR + TE': 'QB + 2 WRs + TE — max same-team ceiling (r=+0.54/+0.41/+0.38)',
    '4/3 Game Stack': '4 from Team A (QB+3) + 3 from Team B — full shootout capture (r=+0.37)',
    '3/2 Game Stack': '3 from Team A (QB+2) + 2 from Team B — standard game stack (r=+0.37)',
    '3/2/1': '3 from Team A + 2 from Team B + 1 from Team C — diversified game stack',
    'Bring-Back': 'Opposing pass catcher paired with QB stack (r=+0.37)',
    'Run-Back': 'Opposing RB/WR paired with QB stack — shootout hedge',
    'Mini Stack (WR+WR)': 'Same-team WR pair without QB — moderate correlation (r=+0.22)',
    'No Stack': 'No correlation requirements — pure projection-driven',

    // NBA — game environment + numeric constructions
    '4/2 Game Stack': '4 from Game A + 2 from Game B — concentrated game environment (r=+0.30)',
    '3/3 Game Stack': '3 from Game A + 3 from Game B — balanced two-game exposure (r=+0.25)',
    '3/2/2': '3 from Game A + 2 from Game B + 2 from Game C — three-game spread',
    '2/2/2/2': '2 from each of 4 games — max diversification, lower ceiling',
    'Game Environment (2+)': '2+ players from same high-total game (O/U 230+)',
    'Game Environment (3+)': '3+ from same game — aggressive game environment capture',
    'Pace Stack': 'Players from top-pace teams (pace > 100) — more possessions',
    'Stars + Value': 'Salary structure: 2-3 studs ($8K+) + value plays ($3.5-5K)',
    'Mini Stack (PG+Wing)': 'PG + SG/SF — assist-to-score correlation (r=+0.15)',
    'Blowout Fade': 'Avoid starters from heavy favorites (spread > 10)',

    // MLB — numeric construction formats (batters only, excludes pitcher slots)
    '5/4': '5 batters Team A + 4 batters Team B — max correlated, GPP ceiling (r=+0.45)',
    '5/3': '5 batters Team A + 3 batters Team B — dominant GPP format, 64% of winners (r=+0.40)',
    '5/2': '5 batters Team A + 2 batters Team B — heavy primary stack (r=+0.38)',
    '4/4': '4 batters Team A + 4 batters Team B — balanced two-team (r=+0.35)',
    '4/3': '4 batters Team A + 3 batters Team B — standard GPP construction (r=+0.35)',
    '4/2': '4 batters Team A + 2 batters Team B + 2 individuals (r=+0.30)',
    '3/3/2': '3 batters each from 2 teams + 2 from a 3rd — three-team diversified',
    '5-Man Batter Stack': '5 consecutive batters from same team — big inning capture (r=+0.45)',
    '4-Man Batter Stack': '4 consecutive batters from same team (r=+0.35)',
    'Wrap-Around Stack': 'Bottom + top of order (8-9-1-2-3) — extra ABs (r=+0.35)',
    'Pitcher vs Weak Lineup': 'Pitcher facing bottom-5 offense — high floor',
    'No Stacks': 'No correlation requirements — uncorrelated lineup',

    // Legacy fallbacks
    'Same Team (2+)': 'At least 2 players from same team',
    'Same Team (3+)': 'At least 3 players from same team',
    'Same Team (4+)': 'At least 4 players from same team',
    'Same Team (5+)': 'At least 5 players from same team',
    'Pitcher + Batter': 'Pitcher with batter from opposing team',
    'Batter Stack': 'Multiple batters from same team',
    '5+3 Structure': '5-man primary stack + 3-man secondary stack',
    'Secondary Stack (3-man)': '3-man complementary stack from different team',
    'PG + SG': 'Point guard + shooting guard from same team',
    'Balanced': 'No structural constraint — projection-driven',
    'Game Stack': 'QB+WR (Team A) + WR/TE (Team B) — full game environment',
    'QB + WR + RB': 'QB with WR and RB from same team',
    'QB + 2 WR + TE': 'QB with 2 WRs and TE from same team',
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

