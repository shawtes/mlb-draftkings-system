import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { TrendingUp, TrendingDown, Activity, Target, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from './ui/chart';

interface GameAnalysisProps {
  sport: string;
}

const teamStats = [
  { metric: 'Offense', home: 92, away: 78 },
  { metric: 'Defense', home: 85, away: 88 },
  { metric: 'Recent Form', home: 88, away: 82 },
  { metric: 'Home/Away', home: 90, away: 75 },
  { metric: 'Injuries', home: 70, away: 85 },
];

const playerProps = [
  { player: 'Patrick Mahomes', position: 'QB', prop: 'Passing Yards', line: 287.5, projection: 312.4, edge: 8.2, confidence: 95 },
  { player: 'Travis Kelce', position: 'TE', prop: 'Receptions', line: 5.5, projection: 6.8, edge: 7.5, confidence: 92 },
  { player: 'Christian McCaffrey', position: 'RB', prop: 'Rush + Rec Yards', line: 125.5, projection: 138.2, edge: 6.8, confidence: 88 },
  { player: 'Tyreek Hill', position: 'WR', prop: 'Receiving Yards', line: 82.5, projection: 94.1, edge: 5.3, confidence: 85 },
  { player: 'Josh Allen', position: 'QB', prop: 'Passing TDs', line: 2.5, projection: 2.9, edge: 4.1, confidence: 82 },
];

const matchupHistory = [
  { date: 'Week 1 2025', homeScore: 24, awayScore: 21, totalPoints: 45 },
  { date: 'Week 15 2024', homeScore: 31, awayScore: 28, totalPoints: 59 },
  { date: 'Week 8 2024', homeScore: 27, awayScore: 24, totalPoints: 51 },
  { date: 'Week 2 2024', homeScore: 20, awayScore: 17, totalPoints: 37 },
  { date: 'Week 12 2023', homeScore: 35, awayScore: 31, totalPoints: 66 },
];

const chartConfig = {
  home: {
    label: "Home",
    color: "#06b6d4",
  },
  away: {
    label: "Away",
    color: "#3b82f6",
  },
  homeScore: {
    label: "Home Score",
    color: "#06b6d4",
  },
  awayScore: {
    label: "Away Score",
    color: "#3b82f6",
  },
} satisfies ChartConfig;

export default function GameAnalysis({ sport }: GameAnalysisProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-white mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Game Analysis
          </h1>
          <p className="text-slate-400">Deep dive into matchups, trends, and betting opportunities</p>
        </div>
        <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30">
          <Activity className="w-3 h-3 mr-1" />
          Live Updates
        </Badge>
      </div>

      {/* Featured Game */}
      <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <Badge className="bg-red-500/10 text-red-400 border-red-500/30 mb-2">
              FEATURED GAME
            </Badge>
            <h2 className="text-white text-2xl">Kansas City Chiefs vs LA Chargers</h2>
            <p className="text-slate-400">Sunday, 8:20 PM ET • Arrowhead Stadium</p>
          </div>
          <div className="text-right">
            <div className="text-slate-400 text-sm mb-1">Recommended Edge</div>
            <div className="text-4xl bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              +8.2%
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-black/30 border border-cyan-500/20 rounded-lg p-4">
            <div className="text-slate-400 text-sm mb-1">Spread</div>
            <div className="text-white text-xl">KC -3.5</div>
            <div className="text-cyan-400 text-sm mt-1">59% betting KC</div>
          </div>
          <div className="bg-black/30 border border-cyan-500/20 rounded-lg p-4">
            <div className="text-slate-400 text-sm mb-1">Total</div>
            <div className="text-white text-xl">O/U 52.5</div>
            <div className="text-blue-400 text-sm mt-1">Model: 54.8 pts</div>
          </div>
          <div className="bg-black/30 border border-cyan-500/20 rounded-lg p-4">
            <div className="text-slate-400 text-sm mb-1">Moneyline</div>
            <div className="text-white text-xl">KC -165</div>
            <div className="text-purple-400 text-sm mt-1">LAC +140</div>
          </div>
        </div>
      </Card>

      <Tabs defaultValue="props" className="w-full">
        <TabsList className="bg-black/50 border-cyan-500/20">
          <TabsTrigger value="props">Player Props</TabsTrigger>
          <TabsTrigger value="matchup">Team Matchup</TabsTrigger>
          <TabsTrigger value="trends">Trends & History</TabsTrigger>
        </TabsList>

        <TabsContent value="props" className="space-y-4">
          <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-white">Top Value Props</h3>
              <Button variant="outline" className="border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10">
                <Target className="w-4 h-4 mr-2" />
                Export Selections
              </Button>
            </div>
            <div className="overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-slate-700">
                    <TableHead className="text-slate-300">Player</TableHead>
                    <TableHead className="text-slate-300">Prop Type</TableHead>
                    <TableHead className="text-slate-300">Line</TableHead>
                    <TableHead className="text-slate-300">Projection</TableHead>
                    <TableHead className="text-slate-300">Edge</TableHead>
                    <TableHead className="text-slate-300">Confidence</TableHead>
                    <TableHead className="text-slate-300">Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {playerProps.map((prop, i) => (
                    <TableRow key={i} className="border-slate-700 hover:bg-slate-800/50">
                      <TableCell>
                        <div>
                          <div className="text-white">{prop.player}</div>
                          <Badge variant="outline" className="border-slate-600 text-slate-400 text-xs mt-1">
                            {prop.position}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell className="text-slate-300">{prop.prop}</TableCell>
                      <TableCell className="text-slate-300">{prop.line}</TableCell>
                      <TableCell className="text-cyan-400">{prop.projection}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <TrendingUp className="w-4 h-4 text-cyan-400" />
                          <span className="text-cyan-400">+{prop.edge}%</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-slate-700 rounded-full h-2">
                            <div 
                              className="bg-gradient-to-r from-cyan-500 to-blue-600 h-2 rounded-full" 
                              style={{ width: `${prop.confidence}%` }}
                            />
                          </div>
                          <span className="text-slate-400 text-sm">{prop.confidence}%</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Button size="sm" className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500">
                          View Details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="matchup" className="space-y-4">
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
              <h3 className="text-white mb-6">Team Comparison Radar</h3>
              <ChartContainer config={chartConfig} className="h-[300px] w-full">
                <RadarChart data={teamStats}>
                  <PolarGrid stroke="#334155" />
                  <PolarAngleAxis dataKey="metric" stroke="#64748b" />
                  <PolarRadiusAxis stroke="#64748b" />
                  <Radar name="Home" dataKey="home" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.3} />
                  <Radar name="Away" dataKey="away" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                </RadarChart>
              </ChartContainer>
            </Card>

            <Card className="bg-black/40 backdrop-blur-sm border-blue-500/20 p-6">
              <h3 className="text-white mb-6">Key Insights</h3>
              <div className="space-y-4">
                <div className="flex items-start gap-3 p-3 bg-cyan-500/5 border border-cyan-500/20 rounded-lg">
                  <TrendingUp className="w-5 h-5 text-cyan-400 mt-0.5" />
                  <div>
                    <div className="text-white mb-1">Strong Home Advantage</div>
                    <div className="text-slate-400 text-sm">Chiefs are 7-1 at home this season with avg margin of +12 pts</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg">
                  <Activity className="w-5 h-5 text-blue-400 mt-0.5" />
                  <div>
                    <div className="text-white mb-1">High Scoring Matchup</div>
                    <div className="text-slate-400 text-sm">Last 5 meetings averaged 56.2 total points</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-yellow-400 mt-0.5" />
                  <div>
                    <div className="text-white mb-1">Weather Factor</div>
                    <div className="text-slate-400 text-sm">15 mph winds expected - may impact passing game</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-purple-500/5 border border-purple-500/20 rounded-lg">
                  <Target className="w-5 h-5 text-purple-400 mt-0.5" />
                  <div>
                    <div className="text-white mb-1">Injury Impact</div>
                    <div className="text-slate-400 text-sm">LAC missing starting CB - exploit through air attack</div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="trends" className="space-y-4">
          <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-white">Head-to-Head History</h3>
              <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
                Last 5 Games
              </Badge>
            </div>
            <ChartContainer config={chartConfig} className="h-[250px] w-full">
              <BarChart data={matchupHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="homeScore" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="awayScore" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </Card>

          <div className="grid md:grid-cols-3 gap-6">
            <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
              <h3 className="text-white mb-4">Chiefs Trends</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">ATS Record</span>
                  <span className="text-cyan-400">8-4-1</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">O/U Record</span>
                  <span className="text-cyan-400">7-6</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Home ATS</span>
                  <span className="text-cyan-400">5-2</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">As Favorite</span>
                  <span className="text-cyan-400">6-3</span>
                </div>
              </div>
            </Card>

            <Card className="bg-black/40 backdrop-blur-sm border-blue-500/20 p-6">
              <h3 className="text-white mb-4">Chargers Trends</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">ATS Record</span>
                  <span className="text-blue-400">6-7</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">O/U Record</span>
                  <span className="text-blue-400">8-5</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Away ATS</span>
                  <span className="text-blue-400">3-4</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">As Underdog</span>
                  <span className="text-blue-400">4-5</span>
                </div>
              </div>
            </Card>

            <Card className="bg-black/40 backdrop-blur-sm border-purple-500/20 p-6">
              <h3 className="text-white mb-4">Betting Trends</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-400 text-sm">Public Money</span>
                    <span className="text-purple-400">59% KC</span>
                  </div>
                  <div className="bg-slate-700 rounded-full h-2">
                    <div className="bg-purple-500 h-2 rounded-full" style={{ width: '59%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-400 text-sm">Sharp Money</span>
                    <span className="text-purple-400">72% KC</span>
                  </div>
                  <div className="bg-slate-700 rounded-full h-2">
                    <div className="bg-gradient-to-r from-cyan-500 to-purple-500 h-2 rounded-full" style={{ width: '72%' }} />
                  </div>
                </div>
                <div className="pt-2 border-t border-slate-700">
                  <div className="text-slate-400 text-sm mb-1">Line Movement</div>
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-red-400" />
                    <span className="text-slate-300">-4.5 → -3.5</span>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
