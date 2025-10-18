import { Card } from './ui/card';
import { TrendingUp, Activity, Zap, Calendar, Users, Flame } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import type { DashboardOverviewProps, Game, PropBet } from '../types';

const todaysGames: Game[] = [
  { id: 1, home: 'Kansas City Chiefs', away: 'LA Chargers', time: '8:20 PM ET', spread: 'KC -3.5', total: 'O/U 52.5', edge: '+8.2%', status: 'live' },
  { id: 2, home: 'Buffalo Bills', away: 'Miami Dolphins', time: '1:00 PM ET', spread: 'BUF -7', total: 'O/U 48.5', edge: '+6.5%', status: 'upcoming' },
  { id: 3, home: 'San Francisco 49ers', away: 'Dallas Cowboys', time: '4:25 PM ET', spread: 'SF -5.5', total: 'O/U 45.5', edge: '+5.1%', status: 'upcoming' },
  { id: 4, home: 'Philadelphia Eagles', away: 'NY Giants', time: '1:00 PM ET', spread: 'PHI -10', total: 'O/U 44.5', edge: '+4.8%', status: 'upcoming' },
  { id: 5, home: 'Seattle Seahawks', away: 'Arizona Cardinals', time: '4:05 PM ET', spread: 'SEA -3', total: 'O/U 50.5', edge: '+3.9%', status: 'upcoming' },
];

const topPlayerProps: PropBet[] = [
  { player: 'Patrick Mahomes', team: 'KC', position: 'QB', prop: 'Passing Yards O287.5', edge: '+8.2%', confidence: 95, matchup: 'vs LAC' },
  { player: 'Christian McCaffrey', team: 'SF', position: 'RB', prop: 'Rush Yards O95.5', edge: '+7.5%', confidence: 92, matchup: '@ DAL' },
  { player: 'Tyreek Hill', team: 'MIA', position: 'WR', prop: 'Rec Yards O82.5', edge: '+6.8%', confidence: 88, matchup: '@ BUF' },
  { player: 'Travis Kelce', team: 'KC', position: 'TE', prop: 'Receptions O5.5', edge: '+6.3%', confidence: 90, matchup: 'vs LAC' },
  { player: 'Josh Allen', team: 'BUF', position: 'QB', prop: 'Pass TDs O2.5', edge: '+5.9%', confidence: 85, matchup: 'vs MIA' },
];

export default function DashboardOverview({ sport }: DashboardOverviewProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-white mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            {sport} Games Dashboard
          </h1>
          <p className="text-slate-400">Today's games, matchups, and top betting opportunities</p>
        </div>
        <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30">
          <Activity className="w-3 h-3 mr-1" />
          Live Updates
        </Badge>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 relative overflow-hidden group hover:border-cyan-400/40 transition-all hover:shadow-[0_0_30px_rgba(6,182,212,0.15)]">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">Games Today</span>
              <div className="p-2 bg-cyan-500/10 rounded-lg">
                <Calendar className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div className="text-white text-3xl">{todaysGames.length}</div>
            <p className="text-cyan-400 text-sm mt-2">1 Live Now</p>
          </div>
        </Card>

        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 relative overflow-hidden group hover:border-cyan-400/40 transition-all hover:shadow-[0_0_30px_rgba(59,130,246,0.15)]">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">High Value Props</span>
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Flame className="w-5 h-5 text-blue-400" />
              </div>
            </div>
            <div className="text-white text-3xl">47</div>
            <p className="text-blue-400 text-sm mt-2">Edge &gt; 5%</p>
          </div>
        </Card>

        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 relative overflow-hidden group hover:border-cyan-400/40 transition-all hover:shadow-[0_0_30px_rgba(124,58,237,0.15)]">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">Avg Edge</span>
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <Zap className="w-5 h-5 text-purple-400" />
              </div>
            </div>
            <div className="text-white text-3xl">+6.2%</div>
            <p className="text-purple-400 text-sm mt-2">Top opportunities</p>
          </div>
        </Card>

        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 relative overflow-hidden group hover:border-cyan-400/40 transition-all hover:shadow-[0_0_30px_rgba(236,72,153,0.15)]">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">Popular Parlays</span>
              <div className="p-2 bg-pink-500/10 rounded-lg">
                <Users className="w-5 h-5 text-pink-400" />
              </div>
            </div>
            <div className="text-white text-3xl">24</div>
            <p className="text-pink-400 text-sm mt-2">Trending now</p>
          </div>
        </Card>
      </div>

      {/* Today's Games Table */}
      <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-white">Today's {sport} Games</h3>
          <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
            {todaysGames.length} Games
          </Badge>
        </div>
        <div className="overflow-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-slate-700">
                <TableHead className="text-slate-300">Status</TableHead>
                <TableHead className="text-slate-300">Matchup</TableHead>
                <TableHead className="text-slate-300">Time</TableHead>
                <TableHead className="text-slate-300">Spread</TableHead>
                <TableHead className="text-slate-300">Total</TableHead>
                <TableHead className="text-slate-300">Best Edge</TableHead>
                <TableHead className="text-slate-300">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {todaysGames.map((game) => (
                <TableRow key={game.id} className="border-slate-700 hover:bg-slate-800/50 transition-colors">
                  <TableCell>
                    <Badge 
                      className={game.status === 'live' 
                        ? 'bg-red-500/20 text-red-400 border-red-500/30' 
                        : 'bg-slate-700/50 text-slate-400 border-slate-600'
                      }
                    >
                      {game.status === 'live' ? (
                        <><Activity className="w-3 h-3 mr-1 animate-pulse" />Live</>
                      ) : (
                        'Upcoming'
                      )}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="text-white">{game.away}</div>
                    <div className="text-slate-400 text-sm">@ {game.home}</div>
                  </TableCell>
                  <TableCell className="text-slate-300">{game.time}</TableCell>
                  <TableCell className="text-cyan-400">{game.spread}</TableCell>
                  <TableCell className="text-blue-400">{game.total}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1 text-cyan-400">
                      <TrendingUp className="w-4 h-4" />
                      {game.edge}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Button size="sm" className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500">
                      Analyze
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Top Player Props */}
      <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-white">Top Value Player Props</h3>
          <Button variant="outline" className="border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10">
            View All Props
          </Button>
        </div>
        <div className="overflow-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-slate-700">
                <TableHead className="text-slate-300">Player</TableHead>
                <TableHead className="text-slate-300">Position</TableHead>
                <TableHead className="text-slate-300">Matchup</TableHead>
                <TableHead className="text-slate-300">Prop</TableHead>
                <TableHead className="text-slate-300">Edge</TableHead>
                <TableHead className="text-slate-300">Confidence</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {topPlayerProps.map((prop, i) => (
                <TableRow key={i} className="border-slate-700 hover:bg-slate-800/50 cursor-pointer transition-colors">
                  <TableCell>
                    <div className="text-white">{prop.player}</div>
                    <div className="text-slate-400 text-sm">{prop.team}</div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="border-slate-600 text-slate-400">
                      {prop.position}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-slate-400">{prop.matchup}</TableCell>
                  <TableCell className="text-slate-300">{prop.prop}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1 text-cyan-400">
                      <TrendingUp className="w-4 h-4" />
                      {prop.edge}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-slate-700 rounded-full h-2 w-20">
                        <div 
                          className="bg-gradient-to-r from-cyan-500 to-blue-600 h-2 rounded-full"
                          style={{ width: `${prop.confidence}%` }}
                        />
                      </div>
                      <span className="text-slate-400 text-sm w-8">{prop.confidence}%</span>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>
    </div>
  );
}
