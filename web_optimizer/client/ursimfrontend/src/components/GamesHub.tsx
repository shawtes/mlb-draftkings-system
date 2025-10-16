import { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  Calendar, 
  Clock, 
  TrendingUp, 
  AlertCircle, 
  CloudRain,
  Wind,
  Activity,
  BarChart3,
  Target,
  Shield,
  Swords
} from 'lucide-react';
import { bettingApi } from '../services/betting-api';

interface GamesHubProps {
  sport: string;
}

export default function GamesHub({ sport }: GamesHubProps) {
  const mockGames = [
    {
      id: '1',
      home: 'Kansas City Chiefs',
      away: 'Los Angeles Chargers',
      homeScore: 0,
      awayScore: 0,
      time: '4:25 PM ET',
      status: 'scheduled',
      homeSpread: -3.5,
      awaySpread: 3.5,
      total: 52.5,
      weather: { condition: 'clear', temp: 72, wind: 8 },
      homeStats: { offense: 1, defense: 5, record: '11-6' },
      awayStats: { offense: 7, defense: 10, record: '10-7' },
      injuries: ['WR Marquise Brown (KC) - OUT', 'S Derwin James (LAC) - Questionable'],
      keyMatchups: ['Mahomes vs LAC Defense', 'Herbert vs KC Secondary'],
      propCount: 145,
    },
    {
      id: '2',
      home: 'San Francisco 49ers',
      away: 'Dallas Cowboys',
      homeScore: 0,
      awayScore: 0,
      time: '4:25 PM ET',
      status: 'scheduled',
      homeSpread: -4.5,
      awaySpread: 4.5,
      total: 49.5,
      weather: { condition: 'clear', temp: 68, wind: 5 },
      homeStats: { offense: 3, defense: 2, record: '12-5' },
      awayStats: { offense: 4, defense: 6, record: '12-5' },
      injuries: ['RB Christian McCaffrey (SF) - Probable'],
      keyMatchups: ['CMC vs Dallas Front 7', 'Purdy vs Dallas Secondary'],
      propCount: 138,
    },
    {
      id: '3',
      home: 'Buffalo Bills',
      away: 'Miami Dolphins',
      homeScore: 0,
      awayScore: 0,
      time: '1:00 PM ET',
      status: 'scheduled',
      homeSpread: -7.5,
      awaySpread: 7.5,
      total: 51.5,
      weather: { condition: 'snow', temp: 28, wind: 15 },
      homeStats: { offense: 2, defense: 3, record: '14-3' },
      awayStats: { offense: 5, defense: 15, record: '11-6' },
      injuries: ['QB Tua Tagovailoa (MIA) - Questionable'],
      keyMatchups: ['Allen in Snow', 'Miami Offense in Cold'],
      propCount: 132,
    },
  ];

  const [games, setGames] = useState<any[]>(mockGames); // Initialize with mock data
  const [selectedGame, setSelectedGame] = useState<any>(mockGames[0]);
  const [view, setView] = useState('grid');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadGames();
  }, [sport]);

  const loadGames = async () => {
    setLoading(true);
    try {
      const data = await bettingApi.getGames(sport);
      if (data && data.length > 0) {
        setGames(data);
        setSelectedGame(data[0]);
      }
    } catch (error) {
      console.error('Error loading games, using mock data:', error);
      // Already using mock data from initialization
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-black relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0a0a0a_1px,transparent_1px),linear-gradient(to_bottom,#0a0a0a_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40 pointer-events-none" />
      <div className="absolute top-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />

      {/* Content */}
      <div className="relative z-10 flex flex-col h-full p-6">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                Games Hub
              </h1>
              <p className="text-slate-400">Research games, matchups, and prop markets</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge className="bg-cyan-500/20 border-cyan-500">
                <Activity className="w-3 h-3 mr-1" />
                {games.length} Games Today
              </Badge>
              <Badge className="bg-blue-500/20 border-blue-500">
                <Calendar className="w-3 h-3 mr-1" />
                {sport.toUpperCase()}
              </Badge>
            </div>
          </div>
        </div>

        {/* Games Grid */}
        <div className="flex-1 overflow-auto space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Games List */}
            <div className="space-y-3">
              {games.map((game) => (
                <Card 
                  key={game.id}
                  onClick={() => setSelectedGame(game)}
                  className={`p-4 cursor-pointer transition-all ${
                    selectedGame?.id === game.id 
                      ? 'bg-cyan-500/10 border-cyan-500' 
                      : 'bg-black/60 border-cyan-500/20 hover:bg-black/80 hover:border-cyan-500/40'
                  }`}
                >
                  <div className="space-y-3">
                    {/* Game Header */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-300">{game.time}</span>
                        {game.weather?.condition === 'snow' && (
                          <CloudRain className="w-4 h-4 text-blue-400" />
                        )}
                        {game.weather?.wind > 12 && (
                          <Wind className="w-4 h-4 text-slate-400" />
                        )}
                      </div>
                      <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
                        {game.propCount} Props
                      </Badge>
                    </div>

                    {/* Teams */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <p className="text-white font-semibold">{game.away}</p>
                          <p className="text-sm text-slate-400">{game.awayStats.record}</p>
                        </div>
                        <div className="text-right">
                          <Badge variant="outline" className="border-slate-500/30">
                            {game.awaySpread > 0 ? '+' : ''}{game.awaySpread}
                          </Badge>
                        </div>
                      </div>
                      <div className="flex items-center justify-between border-t border-cyan-500/10 pt-2">
                        <div className="flex-1">
                          <p className="text-white font-semibold">{game.home}</p>
                          <p className="text-sm text-slate-400">{game.homeStats.record}</p>
                        </div>
                        <div className="text-right">
                          <Badge variant="outline" className="border-slate-500/30">
                            {game.homeSpread > 0 ? '+' : ''}{game.homeSpread}
                          </Badge>
                        </div>
                      </div>
                    </div>

                    {/* Game Info */}
                    <div className="flex items-center gap-4 text-sm">
                      <div className="flex items-center gap-1">
                        <Target className="w-4 h-4 text-cyan-400" />
                        <span className="text-slate-300">O/U {game.total}</span>
                      </div>
                      {game.weather && (
                        <div className="flex items-center gap-1">
                          <Activity className="w-4 h-4 text-slate-400" />
                          <span className="text-slate-300">{game.weather.temp}°F</span>
                        </div>
                      )}
                    </div>

                    {/* Injuries */}
                    {game.injuries.length > 0 && (
                      <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/30 rounded p-2">
                        <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                        <p className="text-xs text-red-300">{game.injuries[0]}</p>
                      </div>
                    )}
                  </div>
                </Card>
              ))}
            </div>

            {/* Selected Game Details */}
            {selectedGame && (
              <Card className="p-6 bg-black/60 border-cyan-500/20 sticky top-0">
                <Tabs defaultValue="matchup" className="h-full">
                  <TabsList className="mb-4">
                    <TabsTrigger value="matchup">Matchup Analysis</TabsTrigger>
                    <TabsTrigger value="props">Top Props</TabsTrigger>
                  </TabsList>

                  <TabsContent value="matchup" className="space-y-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-3">
                        {selectedGame.away} @ {selectedGame.home}
                      </h3>
                      
                      {/* Rankings */}
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <Card className="p-4 bg-black/40 border-cyan-500/10">
                          <div className="flex items-center gap-2 mb-2">
                            <Swords className="w-5 h-5 text-cyan-400" />
                            <h4 className="font-semibold text-white">Offense Rankings</h4>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-slate-300">{selectedGame.away}</span>
                              <Badge className="bg-cyan-500/20 border-cyan-500">
                                #{selectedGame.awayStats.offense}
                              </Badge>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-slate-300">{selectedGame.home}</span>
                              <Badge className="bg-cyan-500/20 border-cyan-500">
                                #{selectedGame.homeStats.offense}
                              </Badge>
                            </div>
                          </div>
                        </Card>

                        <Card className="p-4 bg-black/40 border-cyan-500/10">
                          <div className="flex items-center gap-2 mb-2">
                            <Shield className="w-5 h-5 text-blue-400" />
                            <h4 className="font-semibold text-white">Defense Rankings</h4>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-slate-300">{selectedGame.away}</span>
                              <Badge className="bg-blue-500/20 border-blue-500">
                                #{selectedGame.awayStats.defense}
                              </Badge>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-slate-300">{selectedGame.home}</span>
                              <Badge className="bg-blue-500/20 border-blue-500">
                                #{selectedGame.homeStats.defense}
                              </Badge>
                            </div>
                          </div>
                        </Card>
                      </div>

                      {/* Weather */}
                      {selectedGame.weather && (
                        <Card className="p-4 bg-black/40 border-cyan-500/10 mb-4">
                          <h4 className="font-semibold text-white mb-3">Weather Conditions</h4>
                          <div className="flex items-center gap-6">
                            <div className="flex items-center gap-2">
                              <Activity className="w-5 h-5 text-cyan-400" />
                              <div>
                                <p className="text-sm text-slate-400">Temperature</p>
                                <p className="text-white font-semibold">{selectedGame.weather.temp}°F</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <Wind className="w-5 h-5 text-slate-400" />
                              <div>
                                <p className="text-sm text-slate-400">Wind</p>
                                <p className="text-white font-semibold">{selectedGame.weather.wind} mph</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              {selectedGame.weather.condition === 'snow' ? (
                                <CloudRain className="w-5 h-5 text-blue-400" />
                              ) : (
                                <Activity className="w-5 h-5 text-yellow-400" />
                              )}
                              <div>
                                <p className="text-sm text-slate-400">Condition</p>
                                <p className="text-white font-semibold capitalize">{selectedGame.weather.condition}</p>
                              </div>
                            </div>
                          </div>
                        </Card>
                      )}

                      {/* Key Matchups */}
                      <Card className="p-4 bg-black/40 border-cyan-500/10 mb-4">
                        <h4 className="font-semibold text-white mb-3">Key Matchups to Watch</h4>
                        <ul className="space-y-2">
                          {selectedGame.keyMatchups.map((matchup: string, i: number) => (
                            <li key={i} className="flex items-center gap-2">
                              <TrendingUp className="w-4 h-4 text-cyan-400" />
                              <span className="text-slate-300">{matchup}</span>
                            </li>
                          ))}
                        </ul>
                      </Card>

                      {/* Injuries */}
                      <Card className="p-4 bg-red-500/10 border-red-500/30">
                        <h4 className="font-semibold text-red-300 mb-3">Injury Report</h4>
                        <ul className="space-y-2">
                          {selectedGame.injuries.map((injury: string, i: number) => (
                            <li key={i} className="flex items-center gap-2">
                              <AlertCircle className="w-4 h-4 text-red-400" />
                              <span className="text-red-200 text-sm">{injury}</span>
                            </li>
                          ))}
                        </ul>
                      </Card>
                    </div>
                  </TabsContent>

                  <TabsContent value="props" className="space-y-3">
                    <p className="text-slate-400 text-sm mb-4">
                      Top {selectedGame.propCount} prop opportunities for this game
                    </p>
                    
                    <Card className="p-4 bg-black/40 border-cyan-500/10">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <p className="text-white font-semibold">Patrick Mahomes</p>
                          <p className="text-sm text-slate-400">Passing Yards</p>
                        </div>
                        <Badge className="bg-green-500/20 border-green-500">+8.2% Edge</Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">O 287.5 (-110)</span>
                        <span className="text-cyan-400">Proj: 312.4</span>
                      </div>
                    </Card>

                    <Card className="p-4 bg-black/40 border-cyan-500/10">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <p className="text-white font-semibold">Travis Kelce</p>
                          <p className="text-sm text-slate-400">Receptions</p>
                        </div>
                        <Badge className="bg-green-500/20 border-green-500">+6.5% Edge</Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">O 5.5 (-105)</span>
                        <span className="text-cyan-400">Proj: 7.2</span>
                      </div>
                    </Card>

                    <Card className="p-4 bg-black/40 border-cyan-500/10">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <p className="text-white font-semibold">Austin Ekeler</p>
                          <p className="text-sm text-slate-400">Rushing + Receiving Yards</p>
                        </div>
                        <Badge className="bg-green-500/20 border-green-500">+5.3% Edge</Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">O 95.5 (-115)</span>
                        <span className="text-cyan-400">Proj: 108.1</span>
                      </div>
                    </Card>

                    <Button className="w-full mt-4 bg-gradient-to-r from-cyan-500 to-blue-600">
                      <BarChart3 className="w-4 h-4 mr-2" />
                      View All {selectedGame.propCount} Props
                    </Button>
                  </TabsContent>
                </Tabs>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

