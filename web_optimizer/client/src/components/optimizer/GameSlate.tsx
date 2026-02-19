import React, { useMemo, useState, useCallback } from 'react';
import { Player } from './types';
import { Sport } from '../sport-config';
import { TeamLogo } from '../ui/team-logo';

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

/* Team color dots — compact replacement for logos */
const TEAM_COLORS: Record<string, string> = {
  // NBA
  ATL: '#e03a3e', BOS: '#007a33', BKN: '#000', CHA: '#1d1160', CHI: '#ce1141',
  CLE: '#860038', DAL: '#00538c', DEN: '#0e2240', DET: '#c8102e', GSW: '#1d428a',
  GS: '#1d428a', HOU: '#ce1141', IND: '#002d62', LAC: '#c8102e', LAL: '#552583',
  MEM: '#5d76a9', MIA: '#98002e', MIL: '#00471b', MIN: '#0c2340', NOP: '#0c2340',
  NO: '#0c2340', NYK: '#006bb6', NY: '#006bb6', OKC: '#007ac1', ORL: '#0077c0',
  PHI: '#006bb6', PHX: '#1d1160', PHO: '#1d1160', POR: '#e03a3e', SAC: '#5a2d81',
  SAS: '#c4ced4', SA: '#c4ced4', TOR: '#ce1141', UTA: '#002b5c', UTAH: '#002b5c',
  WAS: '#002b5c', WSH: '#002b5c',
  // NFL extras
  ARI: '#97233f', BAL: '#241773', BUF: '#00338d', CAR: '#0085ca', CIN: '#fb4f14',
  GB: '#203731', JAX: '#006778', JAC: '#006778', KC: '#e31837', LV: '#a5acaf',
  LAR: '#003594', NE: '#002244', NYG: '#0b2265', NYJ: '#125740', PIT: '#ffb612',
  SF: '#aa0000', SEA: '#002244', TB: '#d50a0a', TEN: '#0c2340',
  // MLB extras
  CHC: '#0e3386', CHW: '#27251f', CWS: '#27251f', COL: '#33006f', KAN: '#004687',
  LAA: '#ba0021', LAD: '#005a9c', NYM: '#002d72', NYY: '#003087',
  OAK: '#003831', SD: '#2f241d', STL: '#c41e3a', TEX: '#003278',
};

function getTeamColor(team: string): string {
  return TEAM_COLORS[team.toUpperCase()] || '#6b7280';
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
    const playersWithOpponent = playerData.filter(
      (p) => p.opponent && p.opponent.trim() !== ''
    );

    if (playersWithOpponent.length === 0) return [];

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
        setSelectedGameId(null);
        onGameFilter?.([]);
      } else {
        setSelectedGameId(game.id);
        onGameFilter?.([game.teamA, game.teamB]);
      }
    },
    [selectedGameId, onGameFilter]
  );

  if (games.length === 0) return null;

  return (
    <div
      style={{
        height: 54,
        display: 'flex',
        alignItems: 'center',
        padding: '0 8px',
        gap: 2,
        background: 'var(--dfs-bg-secondary)',
        borderBottom: '1px solid var(--dfs-border)',
        overflowX: 'auto',
        flexShrink: 0,
      }}
      className="game-slate-scroll"
    >
      {/* Entry Fees summary on the left */}
      <div
        style={{
          padding: '4px 8px',
          textAlign: 'center',
          flexShrink: 0,
          marginRight: 4,
        }}
      >
        <div
          style={{
            fontSize: 7,
            color: 'var(--dfs-text-muted)',
            fontWeight: 600,
            letterSpacing: '0.3px',
            textTransform: 'uppercase',
          }}
        >
          Entry Fees
        </div>
        <div
          style={{
            fontSize: 13,
            fontWeight: 700,
            color: 'var(--dfs-text-primary)',
          }}
        >
          $0.00
        </div>
      </div>

      {/* Game cards */}
      {games.map((game) => {
        const isSelected = selectedGameId === game.id;
        return (
          <button
            key={game.id}
            type="button"
            onClick={() => handleCardClick(game)}
            style={{
              flexShrink: 0,
              borderRadius: 4,
              border: isSelected
                ? '1px solid var(--dfs-accent)'
                : '1px solid var(--dfs-border)',
              background: isSelected
                ? 'rgba(6,182,212,0.1)'
                : 'var(--dfs-bg-primary)',
              cursor: 'pointer',
              padding: '4px 8px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 2,
              height: 44,
              minWidth: 100,
            }}
          >
            {/* Team matchup row */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                lineHeight: 1,
              }}
            >
              {/* Team A */}
              <TeamLogo team={game.teamA} sport={sport} size={16} />
              <span
                style={{
                  fontSize: 8,
                  fontWeight: 700,
                  color: isSelected ? 'var(--dfs-text-primary)' : 'var(--dfs-text-secondary)',
                }}
              >
                {game.teamA}
              </span>

              {/* Lightning bolt separator */}
              <span
                style={{
                  fontSize: 7,
                  color: 'var(--dfs-text-muted)',
                  opacity: 0.5,
                }}
              >
                ⚡
              </span>

              {/* Team B */}
              <span
                style={{
                  fontSize: 8,
                  fontWeight: 700,
                  color: isSelected ? 'var(--dfs-text-primary)' : 'var(--dfs-text-secondary)',
                }}
              >
                {game.teamB}
              </span>
              <TeamLogo team={game.teamB} sport={sport} size={16} />
            </div>

            {/* Implied total + time */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                lineHeight: 1,
              }}
            >
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  fontFamily: 'JetBrains Mono, monospace',
                  color: isSelected ? 'var(--dfs-accent)' : 'var(--dfs-text-primary)',
                }}
              >
                {game.impliedTotal.toFixed(1)}
              </span>
              <span
                style={{
                  fontSize: 6,
                  color: 'var(--dfs-text-muted)',
                  opacity: 0.6,
                }}
              >
                7:05 PM
              </span>
            </div>
          </button>
        );
      })}

      {/* CSS to hide scrollbar */}
      <style>{`
        .game-slate-scroll::-webkit-scrollbar { height: 0; display: none; }
        .game-slate-scroll { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  );
};

export default React.memo(GameSlate);
