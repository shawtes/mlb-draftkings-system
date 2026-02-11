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
            style={{ width: 120, height: 52 }}
          >
            {/* Top row: team matchup */}
            <div className="flex items-center justify-center gap-1 text-xs font-medium leading-tight">
              <span
                className="inline-flex items-center justify-center rounded px-1 py-0.5 text-[10px] font-semibold leading-none"
                style={{
                  backgroundColor: isSelected ? 'var(--dfs-accent)' : 'var(--dfs-bg-primary)',
                  color: isSelected ? 'var(--dfs-bg-primary)' : 'var(--dfs-text-secondary)',
                  minWidth: 28,
                }}
              >
                {game.teamA}
              </span>
              <span className="text-[var(--dfs-text-muted)] text-[10px]">vs</span>
              <span
                className="inline-flex items-center justify-center rounded px-1 py-0.5 text-[10px] font-semibold leading-none"
                style={{
                  backgroundColor: isSelected ? 'var(--dfs-accent)' : 'var(--dfs-bg-primary)',
                  color: isSelected ? 'var(--dfs-bg-primary)' : 'var(--dfs-text-secondary)',
                  minWidth: 28,
                }}
              >
                {game.teamB}
              </span>
            </div>

            {/* Bottom row: implied total + player count */}
            <div className="flex items-center justify-center gap-2 mt-1">
              <span className="text-[10px] text-[var(--dfs-text-muted)]">
                {game.impliedTotal.toFixed(1)}
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
