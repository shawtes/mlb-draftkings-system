import React from 'react';

/** ESPN CDN logo mapping — DK abbreviation → ESPN slug */
export const DK_TO_ESPN: Record<string, Record<string, string>> = {
  NBA: {
    ATL:'atl',BOS:'bos',BKN:'bkn',CHA:'cha',CHI:'chi',CLE:'cle',DAL:'dal',DEN:'den',DET:'det',
    GS:'gs',GSW:'gs',HOU:'hou',IND:'ind',LAC:'lac',LAL:'lal',MEM:'mem',MIA:'mia',MIL:'mil',
    MIN:'min',NO:'no',NOP:'no',NY:'ny',NYK:'ny',OKC:'okc',ORL:'orl',PHI:'phi',PHO:'phx',PHX:'phx',
    POR:'por',SAC:'sac',SA:'sa',SAS:'sa',TOR:'tor',UTA:'utah',UTAH:'utah',WAS:'wsh',WSH:'wsh',
  },
  NFL: {
    ARI:'ari',ATL:'atl',BAL:'bal',BUF:'buf',CAR:'car',CHI:'chi',CIN:'cin',CLE:'cle',DAL:'dal',
    DEN:'den',DET:'det',GB:'gb',HOU:'hou',IND:'ind',JAX:'jax',JAC:'jax',KC:'kc',LV:'lv',
    LAR:'lar',LAC:'lac',MIA:'mia',MIN:'min',NE:'ne',NO:'no',NYG:'nyg',NYJ:'nyj',PHI:'phi',
    PIT:'pit',SF:'sf',SEA:'sea',TB:'tb',TEN:'ten',WAS:'wsh',WSH:'wsh',
  },
  MLB: {
    ARI:'ari',ATL:'atl',BAL:'bal',BOS:'bos',CHC:'chc',CHW:'chw',CWS:'chw',CIN:'cin',CLE:'cle',
    COL:'col',DET:'det',HOU:'hou',KC:'kc',LAA:'laa',LAD:'lad',MIA:'mia',MIL:'mil',MIN:'min',
    NYM:'nym',NYY:'nyy',OAK:'oak',PHI:'phi',PIT:'pit',SD:'sd',SF:'sf',SEA:'sea',STL:'stl',
    TB:'tb',TEX:'tex',TOR:'tor',WAS:'wsh',WSH:'wsh',
  },
};

export const TeamLogo: React.FC<{ team: string; sport: string; size?: number }> = ({ team, sport, size = 20 }) => {
  const [failed, setFailed] = React.useState(false);
  if (failed) return null;
  const s = sport === 'NBA' ? 'nba' : sport === 'NFL' ? 'nfl' : 'mlb';
  const slug = DK_TO_ESPN[sport]?.[team.toUpperCase()] || team.toLowerCase();
  return (
    <img
      src={`https://a.espncdn.com/i/teamlogos/${s}/500/${slug}.png`}
      alt={team} width={size} height={size}
      className="object-contain flex-shrink-0"
      onError={() => setFailed(true)}
      loading="lazy"
    />
  );
};
