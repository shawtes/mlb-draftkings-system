import { useState } from 'react';
import { Card } from './ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { TrendingUp, Download, Activity, Star, AlertCircle, Users, Target, Copy, Flame } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from './ui/chart';
import AdvancedFilters from './AdvancedFilters';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface PropBetFinderProps {
  sport: string;
}

const mockPropBets = [
  { 
    id: 1, 
    player: 'Patrick Mahomes', 
    team: 'KC',
    opponent: '@LAC',
    prop: 'Passing Yards', 
    line: 287.5, 
    odds: '-110', 
    projection: 312.4,
    confidence: 95,
    edge: 8.2,
    hitRate: 68,
    trend: 'up'
  },
  { 
    id: 2, 
    player: 'Christian McCaffrey', 
    team: 'SF',
    opponent: '@DAL',
    prop: 'Rushing Yards', 
    line: 95.5, 
    odds: '-115', 
    projection: 108.3,
    confidence: 92,
    edge: 6.7,
    hitRate: 72,
    trend: 'up'
  },
  { 
    id: 3, 
    player: 'Tyreek Hill', 
    team: 'MIA',
    opponent: '@BUF',
    prop: 'Receiving Yards', 
    line: 82.5, 
    odds: '+105', 
    projection: 94.1,
    confidence: 85,
    edge: 5.3,
    hitRate: 64,
    trend: 'up'
  },
  { 
    id: 4, 
    player: 'Josh Allen', 
    team: 'BUF',
    opponent: 'vs MIA',
    prop: 'Passing TDs', 
    line: 2.5, 
    odds: '-120', 
    projection: 2.9,
    confidence: 82,
    edge: 4.1,
    hitRate: 58,
    trend: 'neutral'
  },
  { 
    id: 5, 
    player: 'Travis Kelce', 
    team: 'KC',
    opponent: '@LAC',
    prop: 'Receptions', 
    line: 5.5, 
    odds: '-105', 
    projection: 6.8,
    confidence: 88,
    edge: 7.5,
    hitRate: 70,
    trend: 'up'
  },
  { 
    id: 6, 
    player: 'Saquon Barkley', 
    team: 'NYG',
    opponent: '@SEA',
    prop: 'Rush + Rec Yards', 
    line: 110.5, 
    odds: '-110', 
    projection: 122.8,
    confidence: 87,
    edge: 6.1,
    hitRate: 66,
    trend: 'up'
  },
];

const edgeDistribution = [
  { range: '0-2%', count: 12 },
  { range: '2-4%', count: 18 },
  { range: '4-6%', count: 24 },
  { range: '6-8%', count: 15 },
  { range: '8%+', count: 8 },
];

const chartConfig = {
  count: {
    label: "Props",
    color: "#06b6d4",
  },
} satisfies ChartConfig;

// Popular Stacks Data (from PopularParlays)
const parlaysByPosition = {
  QB: [
    { id: 1, players: ['Patrick Mahomes O287.5 Pass Yds', 'Josh Allen O2.5 Pass TDs'], odds: '+265', popularity: 92, edge: '+8.2%', legs: 2 },
    { id: 2, players: ['Patrick Mahomes O287.5 Pass Yds', 'Josh Allen O2.5 Pass TDs', 'Lamar Jackson O45.5 Rush Yds'], odds: '+485', popularity: 78, edge: '+6.5%', legs: 3 },
  ],
  RB: [
    { id: 3, players: ['Christian McCaffrey O95.5 Rush Yds', 'Saquon Barkley O75.5 Rush Yds'], odds: '+310', popularity: 88, edge: '+7.1%', legs: 2 },
    { id: 4, players: ['Christian McCaffrey O95.5 Rush Yds', 'Derrick Henry O85.5 Rush Yds', 'Nick Chubb O70.5 Rush Yds'], odds: '+520', popularity: 75, edge: '+5.8%', legs: 3 },
  ],
  WR: [
    { id: 5, players: ['Tyreek Hill O82.5 Rec Yds', 'Justin Jefferson O75.5 Rec Yds'], odds: '+280', popularity: 85, edge: '+6.9%', legs: 2 },
    { id: 6, players: ['Tyreek Hill O82.5 Rec Yds', 'CeeDee Lamb O70.5 Rec Yds', 'Amon-Ra St. Brown O65.5 Rec Yds'], odds: '+495', popularity: 71, edge: '+5.2%', legs: 3 },
  ],
  TE: [
    { id: 7, players: ['Travis Kelce O5.5 Rec', 'Mark Andrews O4.5 Rec'], odds: '+245', popularity: 80, edge: '+7.5%', legs: 2 },
  ],
};

const teamStacks = [
  { id: 1, team: 'Kansas City Chiefs', players: ['Patrick Mahomes O287.5 Pass Yds', 'Travis Kelce O5.5 Rec', 'Isiah Pacheco O60.5 Rush Yds'], odds: '+425', popularity: 94, edge: '+8.8%', legs: 3 },
  { id: 2, team: 'Buffalo Bills', players: ['Josh Allen O2.5 Pass TDs', 'Stefon Diggs O75.5 Rec Yds'], odds: '+290', popularity: 87, edge: '+7.2%', legs: 2 },
  { id: 3, team: 'San Francisco 49ers', players: ['Christian McCaffrey O95.5 Rush Yds', 'Deebo Samuel O65.5 Rec Yds', 'George Kittle O50.5 Rec Yds'], odds: '+510', popularity: 82, edge: '+6.5%', legs: 3 },
  { id: 4, team: 'Philadelphia Eagles', players: ['Jalen Hurts O1.5 Rush TDs', 'AJ Brown O80.5 Rec Yds'], odds: '+315', popularity: 79, edge: '+6.1%', legs: 2 },
];

export default function PropBetFinder({ sport }: PropBetFinderProps) {
  const [mainTab, setMainTab] = useState<'props' | 'stacks'>('props');
  const [selectedPosition, setSelectedPosition] = useState<keyof typeof parlaysByPosition>('QB');
  const [filterLegs, setFilterLegs] = useState('all');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-white mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Player Props & Stacks
          </h1>
          <p className="text-slate-400">AI-powered analysis with edge detection and popular stacking strategies</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30">
            <Activity className="w-3 h-3 mr-1" />
            {mockPropBets.length} Props Analyzed
          </Badge>
        </div>
      </div>

      {/* Main Tabs - Props vs Stacks */}
      <Tabs value={mainTab} onValueChange={(v) => setMainTab(v as 'props' | 'stacks')} className="w-full">
        <TabsList className="bg-black/50 border-cyan-500/20 grid w-full grid-cols-2">
          <TabsTrigger value="props">Individual Props</TabsTrigger>
          <TabsTrigger value="stacks">Popular Stacks</TabsTrigger>
        </TabsList>

        {/* INDIVIDUAL PROPS TAB */}
        <TabsContent value="props" className="space-y-6 mt-6">

      {/* Stats Cards */}
      <div className="grid md:grid-cols-4 gap-6">
        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 relative overflow-hidden group hover:border-cyan-400/40 transition-all hover:shadow-[0_0_30px_rgba(6,182,212,0.15)]">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">High Value Props</span>
              <div className="p-2 bg-cyan-500/10 rounded-lg">
                <Star className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div className="text-white text-3xl">24</div>
            <p className="text-cyan-400 text-sm mt-2">Edge &gt; 5%</p>
          </div>
        </Card>

        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 relative overflow-hidden group hover:border-cyan-400/40 transition-all hover:shadow-[0_0_30px_rgba(59,130,246,0.15)]">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">Avg Edge Found</span>
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <TrendingUp className="w-5 h-5 text-blue-400" />
              </div>
            </div>
            <div className="text-white text-3xl">+6.2%</div>
            <p className="text-blue-400 text-sm mt-2">vs market lines</p>
          </div>
        </Card>

        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 relative overflow-hidden group hover:border-cyan-400/40 transition-all hover:shadow-[0_0_30px_rgba(147,51,234,0.15)]">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">Model Accuracy</span>
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <Activity className="w-5 h-5 text-purple-400" />
              </div>
            </div>
            <div className="text-white text-3xl">68%</div>
            <p className="text-purple-400 text-sm mt-2">Last 30 days</p>
          </div>
        </Card>

        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 relative overflow-hidden group hover:border-cyan-400/40 transition-all hover:shadow-[0_0_30px_rgba(236,72,153,0.15)]">
          <div className="absolute inset-0 bg-gradient-to-br from-pink-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">Sharp Plays</span>
              <div className="p-2 bg-pink-500/10 rounded-lg">
                <AlertCircle className="w-5 h-5 text-pink-400" />
              </div>
            </div>
            <div className="text-white text-3xl">12</div>
            <p className="text-pink-400 text-sm mt-2">Confidence &gt; 90%</p>
          </div>
        </Card>
      </div>

      {/* Edge Distribution Chart */}
      <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
        <h3 className="text-white mb-6">Edge Distribution</h3>
        <ChartContainer config={chartConfig} className="h-[200px] w-full">
          <BarChart data={edgeDistribution}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="range" stroke="#64748b" />
            <YAxis stroke="#64748b" />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="count" fill="url(#colorGradient)" radius={[8, 8, 0, 0]} />
            <defs>
              <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.8}/>
              </linearGradient>
            </defs>
          </BarChart>
        </ChartContainer>
      </Card>

      {/* Advanced Filters */}
      <AdvancedFilters
        showPositionFilters={false}
        showPriceFilters={false}
        onFilterChange={(filters) => console.log('Filters changed:', filters)}
      />

      {/* Props Table */}
      <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
        <div className="mb-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <h3 className="text-white">Player Props Analysis</h3>
          <Button variant="outline" className="border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>

        <div className="overflow-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-slate-700">
                <TableHead className="text-slate-300">Player</TableHead>
                <TableHead className="text-slate-300">Matchup</TableHead>
                <TableHead className="text-slate-300">Prop Type</TableHead>
                <TableHead className="text-slate-300">Line</TableHead>
                <TableHead className="text-slate-300">Projection</TableHead>
                <TableHead className="text-slate-300">Edge</TableHead>
                <TableHead className="text-slate-300">Confidence</TableHead>
                <TableHead className="text-slate-300">Hit Rate</TableHead>
                <TableHead className="text-slate-300">Trend</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mockPropBets.map((bet) => (
                <TableRow key={bet.id} className="border-slate-700 hover:bg-slate-800/50 cursor-pointer transition-colors">
                  <TableCell>
                    <div>
                      <div className="text-white mb-1">{bet.player}</div>
                      <Badge variant="outline" className="border-slate-600 text-slate-400 text-xs">
                        {bet.team}
                      </Badge>
                    </div>
                  </TableCell>
                  <TableCell className="text-slate-400">{bet.opponent}</TableCell>
                  <TableCell className="text-slate-300">{bet.prop}</TableCell>
                  <TableCell className="text-slate-300">O {bet.line}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <span className="text-cyan-400">{bet.projection}</span>
                      <span className="text-slate-500 text-xs">({bet.odds})</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <TrendingUp className="w-4 h-4 text-cyan-400" />
                      <span className="text-cyan-400">+{bet.edge}%</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-slate-700 rounded-full h-2 w-20">
                        <div 
                          className={`h-2 rounded-full ${
                            bet.confidence >= 90 ? 'bg-gradient-to-r from-cyan-500 to-blue-600' :
                            bet.confidence >= 80 ? 'bg-gradient-to-r from-blue-500 to-purple-600' :
                            'bg-gradient-to-r from-purple-500 to-pink-600'
                          }`}
                          style={{ width: `${bet.confidence}%` }}
                        />
                      </div>
                      <span className="text-slate-400 text-sm w-8">{bet.confidence}%</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge 
                      variant="outline"
                      className={`${
                        bet.hitRate >= 70 ? 'border-cyan-500/30 text-cyan-400' :
                        bet.hitRate >= 60 ? 'border-blue-500/30 text-blue-400' :
                        'border-yellow-500/30 text-yellow-400'
                      }`}
                    >
                      {bet.hitRate}%
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className={`inline-flex items-center gap-1 px-2 py-1 rounded ${
                      bet.trend === 'up' ? 'bg-cyan-500/10 text-cyan-400' :
                      bet.trend === 'down' ? 'bg-red-500/10 text-red-400' :
                      'bg-slate-700/50 text-slate-400'
                    }`}>
                      <TrendingUp className={`w-3 h-3 ${bet.trend === 'down' ? 'rotate-180' : ''}`} />
                      <span className="text-xs">{bet.trend === 'up' ? 'Hot' : bet.trend === 'down' ? 'Cold' : 'Neutral'}</span>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Analysis Insights */}
      <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
        <h3 className="text-white mb-6">AI Insights & Alerts</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-cyan-500/5 border border-cyan-500/20 p-4 rounded-lg">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-cyan-400 mt-0.5" />
              <div>
                <h4 className="text-cyan-400 mb-2">Weather Alert</h4>
                <p className="text-slate-300 text-sm">
                  High winds expected in Buffalo (20+ mph). Consider under bets for passing props.
                </p>
              </div>
            </div>
          </div>
          <div className="bg-blue-500/5 border border-blue-500/20 p-4 rounded-lg">
            <div className="flex items-start gap-3">
              <Activity className="w-5 h-5 text-blue-400 mt-0.5" />
              <div>
                <h4 className="text-blue-400 mb-2">Injury Updates</h4>
                <p className="text-slate-300 text-sm">
                  WR1 out for Miami - increased target share for Tyreek Hill. Line value detected.
                </p>
              </div>
            </div>
          </div>
          <div className="bg-purple-500/5 border border-purple-500/20 p-4 rounded-lg">
            <div className="flex items-start gap-3">
              <TrendingUp className="w-5 h-5 text-purple-400 mt-0.5" />
              <div>
                <h4 className="text-purple-400 mb-2">Matchup Advantage</h4>
                <p className="text-slate-300 text-sm">
                  SF running backs averaging 135+ yards vs DAL defense. Strong rushing prop value.
                </p>
              </div>
            </div>
          </div>
          <div className="bg-pink-500/5 border border-pink-500/20 p-4 rounded-lg">
            <div className="flex items-start gap-3">
              <Star className="w-5 h-5 text-pink-400 mt-0.5" />
              <div>
                <h4 className="text-pink-400 mb-2">Line Movement</h4>
                <p className="text-slate-300 text-sm">
                  Mahomes passing yards line moved from 275.5 to 287.5. Sharp money on over.
                </p>
              </div>
            </div>
          </div>
        </div>
      </Card>

      </TabsContent>

      {/* POPULAR STACKS TAB */}
      <TabsContent value="stacks" className="space-y-6 mt-6">
        
        {/* Quick Stats for Stacks */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2.5 bg-gradient-to-br from-cyan-500/20 to-cyan-600/10 rounded-lg ring-2 ring-cyan-500/30">
                <Target className="w-5 h-5 text-cyan-400" />
              </div>
              <div className="text-xs text-slate-400 uppercase tracking-wider">Total Stacks</div>
            </div>
            <div className="text-4xl bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-1">156</div>
            <div className="text-xs text-cyan-400/80">Tracked today</div>
          </Card>

          <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2.5 bg-gradient-to-br from-blue-500/20 to-blue-600/10 rounded-lg ring-2 ring-blue-500/30">
                <Users className="w-5 h-5 text-blue-400" />
              </div>
              <div className="text-xs text-slate-400 uppercase tracking-wider">Avg Popularity</div>
            </div>
            <div className="text-4xl bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent mb-1">83%</div>
            <div className="text-xs text-blue-400/80">User interest</div>
          </Card>

          <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2.5 bg-gradient-to-br from-purple-500/20 to-purple-600/10 rounded-lg ring-2 ring-purple-500/30">
                <TrendingUp className="w-5 h-5 text-purple-400" />
              </div>
              <div className="text-xs text-slate-400 uppercase tracking-wider">Best Edge</div>
            </div>
            <div className="text-4xl bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent mb-1">+8.8%</div>
            <div className="text-xs text-purple-400/80">KC stack</div>
          </Card>

          <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2.5 bg-gradient-to-br from-pink-500/20 to-pink-600/10 rounded-lg ring-2 ring-pink-500/30 relative">
                <Flame className="w-5 h-5 text-pink-400" />
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-pink-500 rounded-full animate-pulse" />
              </div>
              <div className="text-xs text-slate-400 uppercase tracking-wider">Hot Stacks</div>
            </div>
            <div className="text-4xl bg-gradient-to-r from-pink-400 to-red-500 bg-clip-text text-transparent mb-1">24</div>
            <div className="text-xs text-pink-400/80">Edge &gt; 7%</div>
          </Card>
        </div>

        {/* Stack Tabs */}
        <Tabs defaultValue="position" className="w-full">
          <TabsList className="bg-black/50 border-cyan-500/20">
            <TabsTrigger value="position">By Position</TabsTrigger>
            <TabsTrigger value="team">Team Stacks</TabsTrigger>
          </TabsList>

          <TabsContent value="position" className="space-y-4 mt-6">
            <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6">
                <h3 className="text-white">Position-Based Stacks</h3>
                <div className="flex items-center gap-3">
                  <select
                    value={selectedPosition}
                    onChange={(e) => setSelectedPosition(e.target.value as keyof typeof parlaysByPosition)}
                    className="bg-black/50 border border-cyan-500/20 rounded-lg px-3 py-2 text-white text-sm"
                  >
                    <option value="QB">QB</option>
                    <option value="RB">RB</option>
                    <option value="WR">WR</option>
                    <option value="TE">TE</option>
                  </select>
                  <select
                    value={filterLegs}
                    onChange={(e) => setFilterLegs(e.target.value)}
                    className="bg-black/50 border border-cyan-500/20 rounded-lg px-3 py-2 text-white text-sm"
                  >
                    <option value="all">All Legs</option>
                    <option value="2">2-Leg Stacks</option>
                    <option value="3">3-Leg Stacks</option>
                    <option value="4">4+ Leg Stacks</option>
                  </select>
                </div>
              </div>

              <div className="grid gap-3">
                {parlaysByPosition[selectedPosition]
                  .filter(stack => filterLegs === 'all' || stack.legs.toString() === filterLegs)
                  .map((stack) => (
                  <div key={stack.id} className="bg-black/60 border border-cyan-500/20 rounded-lg p-4 hover:border-cyan-400/40 transition-all hover:shadow-[0_0_20px_rgba(6,182,212,0.15)]">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-3">
                          <Badge className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border-cyan-500/40 text-xs px-2 py-0.5">
                            {stack.legs} Legs
                          </Badge>
                          <Badge className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-400 border-purple-500/40 text-xs px-2 py-0.5">
                            {stack.odds}
                          </Badge>
                          <div className="flex items-center gap-1.5 ml-auto">
                            <Users className="w-3 h-3 text-slate-500" />
                            <span className="text-xs text-slate-400">{stack.popularity}%</span>
                          </div>
                        </div>
                        <div className="space-y-1.5">
                          {stack.players.map((player, i) => (
                            <div key={i} className="flex items-center gap-2 text-sm">
                              <div className="w-1 h-1 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500" />
                              <span className="text-slate-300">{player}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex flex-col items-end gap-3 flex-shrink-0">
                        <div className="text-right">
                          <div className="text-xs text-slate-500 mb-0.5">Edge</div>
                          <div className="text-2xl bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                            {stack.edge}
                          </div>
                        </div>
                        <Button size="sm" className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-xs h-8 px-3 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
                          <Copy className="w-3 h-3 mr-1.5" />
                          Copy
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="team" className="space-y-4 mt-6">
            <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-white">Team Stacking Strategies</h3>
                <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
                  {teamStacks.length} Teams
                </Badge>
              </div>

              <div className="grid gap-3">
                {teamStacks.map((stack) => (
                  <div key={stack.id} className="bg-black/60 border border-cyan-500/20 rounded-lg p-4 hover:border-cyan-400/40 transition-all hover:shadow-[0_0_20px_rgba(6,182,212,0.15)]">
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <div className="flex items-center gap-2 flex-1">
                        <h4 className="text-white font-semibold">{stack.team}</h4>
                        <Badge className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border-cyan-500/40 text-xs px-2 py-0.5">
                          {stack.legs} Legs
                        </Badge>
                        <Badge className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-400 border-purple-500/40 text-xs px-2 py-0.5">
                          {stack.odds}
                        </Badge>
                      </div>
                      
                      <div className="flex items-center gap-1.5 bg-black/50 px-2.5 py-1 rounded-md">
                        <Users className="w-3 h-3 text-pink-400" />
                        <span className="text-xs text-pink-400">{stack.popularity}%</span>
                      </div>
                    </div>
                    
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 space-y-1.5">
                        {stack.players.map((player, i) => (
                          <div key={i} className="flex items-center gap-2 text-sm">
                            <div className="w-1 h-1 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500" />
                            <span className="text-slate-300">{player}</span>
                          </div>
                        ))}
                      </div>
                      
                      <div className="flex flex-col items-end gap-3 flex-shrink-0">
                        <div className="text-right">
                          <div className="text-xs text-slate-500 mb-0.5">Edge</div>
                          <div className="text-2xl bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                            {stack.edge}
                          </div>
                        </div>
                        <Button size="sm" className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-xs h-8 px-3 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
                          <Copy className="w-3 h-3 mr-1.5" />
                          Copy
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>
        </Tabs>

      </TabsContent>
      </Tabs>
    </div>
  );
}
