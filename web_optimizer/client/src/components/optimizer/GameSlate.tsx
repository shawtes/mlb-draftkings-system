import React, { useMemo, useState, useCallback } from 'react';
import { Player } from './types';
import { Sport } from '../sport-config';
import { Users } from 'lucide-react';

interface GameSlateProps {
  playerData: Player[];
  sport: Sport;
  onGameFilter?: (teams: string[]) => void;
}

interface GameCard {
  id: string;
  teamA: string;
  teamB: string;
  teamAProjection: number;
  teamBProjection: number;
  impliedTotal: number;
  playerCount: number;
}

/**
 * Map DraftKings team abbreviations → ESPN CDN slugs (lowercase)
 * ESPN CDN pattern: https://a.espncdn.com/i/teamlogos/{sport}/500/{slug}.png
 */
const DK_TO_ESPN_SLUG: Record<string, Record<string, string>> = {
  NBA: {
    ATL: 'atl', BOS: 'bos', BKN: 'bkn', CHA: 'cha', CHI: 'chi',
    CLE: 'cle', DAL: 'dal', DEN: 'den', DET: 'det', GS: 'gs',
    GSW: 'gs', HOU: 'hou', IND: 'ind', LAC: 'lac', LAL: 'lal',
    MEM: 'mem', MIA: 'mia', MIL: 'mil', MIN: 'min', NO: 'no',
    NOP: 'no', NY: 'ny', NYK: 'ny', OKC: 'okc', ORL: 'orl',
    PHI: 'phi', PHO: 'phx', PHX: 'phx', POR: 'por', SAC: 'sac',
    SA: 'sa', SAS: 'sa', TOR: 'tor', UTA: 'utah', UTAH: 'utah',
    WAS: 'wsh', WSH: 'wsh',
  },
  NFL: {
    ARI: 'ari', ATL: 'atl', BAL: 'bal', BUF: 'buf', CAR: 'car',
    CHI: 'chi', CIN: 'cin', CLE: 'cle', DAL: 'dal', DEN: 'den',
    DET: 'det', GB: 'gb', HOU: 'hou', IND: 'ind', JAX: 'jax',
    JAC: 'jax', KC: 'kc', LV: 'lv', LAR: 'lar', LAC: 'lac',
    MIA: 'mia', MIN: 'min', NE: 'ne', NO: 'no', NYG: 'nyg',
    NYJ: 'nyj', PHI: 'phi', PIT: 'pit', SF: 'sf', SEA: 'sea',
    TB: 'tb', TEN: 'ten', WAS: 'wsh', WSH: 'wsh',
  },
  MLB: {
    ARI: 'ari', ATL: 'atl', BAL: 'bal', BOS: 'bos', CHC: 'chc',
    CHW: 'chw', CWS: 'chw', CIN: 'cin', CLE: 'cle', COL: 'col',
    DET: 'det', HOU: 'hou', KC: 'kc', LAA: 'laa', LAD: 'lad',
    MIA: 'mia', MIL: 'mil', MIN: 'min', NYM: 'nym', NYY: 'nyy',
    OAK: 'oak', PHI: 'phi', PIT: 'pit', SD: 'sd', SF: 'sf',
    SEA: 'sea', STL: 'stl', TB: 'tb', TEX: 'tex', TOR: 'tor',
    WAS: 'wsh', WSH: 'wsh',
  },
};

function getTeamLogoUrl(team: string, sport: Sport): string {
  const slug = DK_TO_ESPN_SLUG[sport]?.[team.toUpperCase()] || team.toLowerCase();
  const espnSport = sport === 'NBA' ? 'nba' : sport === 'NFL' ? 'nfl' : 'mlb';
  return `https://a.espncdn.com/i/teamlogos/${espnSport}/500/${slug}.png`;
}

/** Tiny logo with graceful fallback to text */
const TeamLogo: React.FC<{ team: string; sport: Sport; size?: number }> = ({ team, sport, size = 20 }) => {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return (
    <img
      src={getTeamLogoUrl(team, sport)}
      alt={team}
      width={size}
      height={size}
      className="object-contain"
      onError={() => setFailed(true)}
      loading="lazy"
    />
  );
};

/**
 * Normalize a matchup key so that "NYY vs BOS" and "BOS vs NYY"
 * resolve to the same game regardless of which team the player is on.
 */
function makeGameKey(teamA: string, teamB: string): string {
  const sorted = [teamA, teamB].sort();
  return `${sorted[0]}|${sorted[1]}`;
}

const GameSlate: React.FC<GameSlateProps> = ({ playerData, sport, onGameFilter }) => {
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);

  // Derive unique games from player data
  const games: GameCard[] = useMemo(() => {
    // Only include players that have an opponent field
    const playersWithOpponent = playerData.filter(
      (p) => p.opponent && p.opponent.trim() !== ''
    );

    if (playersWithOpponent.length === 0) return [];

    // Accumulate stats per game
    const gameMap = new Map<
      string,
      {
        teamA: string;
        teamB: string;
        teamAProjection: number;
        teamBProjection: number;
        playerCount: number;
        seenPlayers: Set<string>;
      }
    >();

    for (const player of playersWithOpponent) {
      const team = player.team.trim().toUpperCase();
      const opp = (player.opponent as string).trim().toUpperCase();
      const key = makeGameKey(team, opp);

      if (!gameMap.has(key)) {
        // Use alphabetical order for consistent display
        const [first, second] = [team, opp].sort();
        gameMap.set(key, {
          teamA: first,
          teamB: second,
          teamAProjection: 0,
          teamBProjection: 0,
          playerCount: 0,
          seenPlayers: new Set(),
        });
      }

      const game = gameMap.get(key)!;
      if (!game.seenPlayers.has(player.id)) {
        game.seenPlayers.add(player.id);
        game.playerCount += 1;

        if (team === game.teamA) {
          game.teamAProjection += player.projectedPoints;
        } else {
          game.teamBProjection += player.projectedPoints;
        }
      }
    }

    return Array.from(gameMap.entries())
      .map(([key, g]) => ({
        id: key,
        teamA: g.teamA,
        teamB: g.teamB,
        teamAProjection: g.teamAProjection,
        teamBProjection: g.teamBProjection,
        impliedTotal: parseFloat(
          ((g.teamAProjection + g.teamBProjection) / Math.max(g.playerCount, 1) * 4).toFixed(1)
        ),
        playerCount: g.playerCount,
      }))
      .sort((a, b) => b.impliedTotal - a.impliedTotal);
  }, [playerData]);

  const handleCardClick = useCallback(
    (game: GameCard) => {
      if (selectedGameId === game.id) {
        // Deselect
        setSelectedGameId(null);
        onGameFilter?.([]);
      } else {
        setSelectedGameId(game.id);
        onGameFilter?.([game.teamA, game.teamB]);
      }
    },
    [selectedGameId, onGameFilter]
  );

  // If no games derived, render nothing
  if (games.length === 0) return null;

  return (
    <div className="overflow-x-auto flex gap-2 px-4 py-2 border-b border-[var(--dfs-border)] bg-[var(--dfs-bg-primary)] flex-shrink-0 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-transparent">
      {games.map((game) => {
        const isSelected = selectedGameId === game.id;
        return (
          <button
            key={game.id}
            type="button"
            onClick={() => handleCardClick(game)}
            className={`flex-shrink-0 rounded border px-3 py-1.5 cursor-pointer transition-colors text-left ${
              isSelected
                ? 'border-[var(--dfs-accent)] bg-[var(--dfs-accent)]/10'
                : 'border-[var(--dfs-border)] bg-[var(--dfs-bg-secondary)] hover:border-[var(--dfs-accent)]/40'
            }`}
            style={{ width: 150, height: 60 }}
          >
            {/* Top row: logos + team matchup */}
            <div className="flex items-center justify-center gap-1.5 text-xs font-medium leading-tight">
              <div className="flex items-center gap-1">
                <TeamLogo team={game.teamA} sport={sport} size={18} />
                <span
                  className="inline-flex items-center justify-center rounded px-1 py-0.5 text-[10px] font-semibold leading-none"
                  style={{
                    backgroundColor: isSelected ? 'var(--dfs-accent)' : 'var(--dfs-bg-primary)',
                    color: isSelected ? 'var(--dfs-bg-primary)' : 'var(--dfs-text-secondary)',
                    minWidth: 24,
                  }}
                >
                  {game.teamA}
                </span>
              </div>
              <span className="text-[var(--dfs-text-muted)] text-[10px]">vs</span>
              <div className="flex items-center gap-1">
                <span
                  className="inline-flex items-center justify-center rounded px-1 py-0.5 text-[10px] font-semibold leading-none"
                  style={{
                    backgroundColor: isSelected ? 'var(--dfs-accent)' : 'var(--dfs-bg-primary)',
                    color: isSelected ? 'var(--dfs-bg-primary)' : 'var(--dfs-text-secondary)',
                    minWidth: 24,
                  }}
                >
                  {game.teamB}
                </span>
                <TeamLogo team={game.teamB} sport={sport} size={18} />
              </div>
            </div>

            {/* Bottom row: implied total + player count */}
            <div className="flex items-center justify-center gap-2 mt-1.5">
              <span className="text-[10px] font-medium text-[var(--dfs-text-secondary)]">
                O/U {game.impliedTotal.toFixed(1)}
              </span>
              <span className="flex items-center gap-0.5 text-[10px] text-[var(--dfs-text-muted)]">
                <Users className="w-2.5 h-2.5" />
                {game.playerCount}
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
};

export default React.memo(GameSlate);
