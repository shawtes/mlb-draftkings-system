import { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Slider } from './ui/slider';
import { Lock, X, Download, Save, Play } from 'lucide-react';
import { Progress } from './ui/progress';
import AdvancedFilters from './AdvancedFilters';
import type { LineupBuilderProps, Player, Lineup } from '../types';

// Mock player data
const mockPlayers: Player[] = [
  { id: 1, position: 'QB', name: 'Patrick Mahomes', team: 'KC', opponent: '@LAC', salary: 8500, projection: 24.8, status: 'Active' },
  { id: 2, position: 'QB', name: 'Josh Allen', team: 'BUF', opponent: 'vs MIA', salary: 8200, projection: 23.4, status: 'Active' },
  { id: 3, position: 'RB', name: 'Christian McCaffrey', team: 'SF', opponent: '@DAL', salary: 9000, projection: 22.1, status: 'Active' },
  { id: 4, position: 'RB', name: 'Austin Ekeler', team: 'LAC', opponent: 'vs KC', salary: 7500, projection: 18.3, status: 'Active' },
  { id: 5, position: 'RB', name: 'Saquon Barkley', team: 'NYG', opponent: '@SEA', salary: 7800, projection: 17.9, status: 'Questionable' },
  { id: 6, position: 'WR', name: 'Tyreek Hill', team: 'MIA', opponent: '@BUF', salary: 8700, projection: 19.6, status: 'Active' },
  { id: 7, position: 'WR', name: 'Justin Jefferson', team: 'MIN', opponent: 'vs CHI', salary: 8400, projection: 18.8, status: 'Active' },
  { id: 8, position: 'WR', name: 'CeeDee Lamb', team: 'DAL', opponent: 'vs SF', salary: 8100, projection: 17.4, status: 'Active' },
  { id: 9, position: 'TE', name: 'Travis Kelce', team: 'KC', opponent: '@LAC', salary: 6800, projection: 15.2, status: 'Active' },
  { id: 10, position: 'TE', name: 'Mark Andrews', team: 'BAL', opponent: 'vs CIN', salary: 6200, projection: 13.7, status: 'Active' },
  { id: 11, position: 'DST', name: 'Buffalo DST', team: 'BUF', opponent: 'vs MIA', salary: 3500, projection: 9.2, status: 'Active' },
  { id: 12, position: 'DST', name: 'San Francisco DST', team: 'SF', opponent: '@DAL', salary: 3200, projection: 8.4, status: 'Active' },
];

export default function LineupBuilder({ sport, slate }: LineupBuilderProps) {
  const [site, setSite] = useState('draftkings');
  const [numLineups, setNumLineups] = useState('3');
  const [lockedPlayers, setLockedPlayers] = useState<Set<number>>(new Set());
  const [excludedPlayers, setExcludedPlayers] = useState<Set<number>>(new Set());
  const [generatedLineups, setGeneratedLineups] = useState<Lineup[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [minSalary, setMinSalary] = useState([48000]);
  const [maxExposure, setMaxExposure] = useState([50]);

  const toggleLock = (playerId: number) => {
    const newLocked = new Set(lockedPlayers);
    if (newLocked.has(playerId)) {
      newLocked.delete(playerId);
    } else {
      newLocked.add(playerId);
      excludedPlayers.delete(playerId);
    }
    setLockedPlayers(newLocked);
  };

  const toggleExclude = (playerId: number) => {
    const newExcluded = new Set(excludedPlayers);
    if (newExcluded.has(playerId)) {
      newExcluded.delete(playerId);
    } else {
      newExcluded.add(playerId);
      lockedPlayers.delete(playerId);
    }
    setExcludedPlayers(newExcluded);
  };

  const generateLineups = () => {
    setIsGenerating(true);
    setProgress(0);
    
    // Simulate lineup generation
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsGenerating(false);
          
          // Generate mock lineups
          const lineups = Array.from({ length: parseInt(numLineups) }, (_, i) => ({
            id: i + 1,
            totalSalary: 49500 + Math.floor(Math.random() * 500),
            projectedPoints: 145.6 + Math.random() * 20,
            players: [
              mockPlayers[0],
              mockPlayers[2],
              mockPlayers[4],
              mockPlayers[6],
              mockPlayers[7],
              mockPlayers[8],
              mockPlayers[9],
              mockPlayers[11],
            ]
          }));
          
          setGeneratedLineups(lineups);
          return 100;
        }
        return prev + 10;
      });
    }, 200);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-white mb-2">DFS Lineup Builder</h1>
          <p className="text-slate-400">{sport} - {slate.charAt(0).toUpperCase() + slate.slice(1)} Slate</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="border-slate-600 text-slate-300">
            <Save className="w-4 h-4 mr-2" />
            Save Rules
          </Button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 lg:col-span-1">
          <h3 className="text-white mb-4">Optimization Settings</h3>
          
          <div className="space-y-4">
            <div>
              <Label className="text-slate-300">DFS Site</Label>
              <Select value={site} onValueChange={setSite}>
                <SelectTrigger className="bg-black/50 border-cyan-500/20 text-white mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="draftkings">DraftKings</SelectItem>
                  <SelectItem value="fanduel">FanDuel</SelectItem>
                  <SelectItem value="yahoo">Yahoo</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-slate-300">Number of Lineups</Label>
              <Input
                type="number"
                value={numLineups}
                onChange={(e) => setNumLineups(e.target.value)}
                className="bg-black/50 border-cyan-500/20 text-white mt-1"
                min="1"
                max="150"
              />
            </div>

            <div>
              <Label className="text-slate-300">Optimization Goal</Label>
              <Select defaultValue="maximize">
                <SelectTrigger className="bg-black/50 border-cyan-500/20 text-white mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="maximize">Maximize Raw Score</SelectItem>
                  <SelectItem value="balance">Balance Roster</SelectItem>
                  <SelectItem value="risk">Minimize Risk</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-slate-300">Min Salary Used: ${minSalary[0].toLocaleString()}</Label>
              <Slider
                value={minSalary}
                onValueChange={setMinSalary}
                min={45000}
                max={50000}
                step={100}
                className="mt-2"
              />
            </div>

            <div>
              <Label className="text-slate-300">Max Player Exposure: {maxExposure[0]}%</Label>
              <Slider
                value={maxExposure}
                onValueChange={setMaxExposure}
                min={0}
                max={100}
                step={5}
                className="mt-2"
              />
            </div>

            <div>
              <Label className="text-slate-300">Stacking Rules</Label>
              <Select defaultValue="qb-wr">
                <SelectTrigger className="bg-black/50 border-cyan-500/20 text-white mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No Stacking</SelectItem>
                  <SelectItem value="qb-wr">QB + WR/TE</SelectItem>
                  <SelectItem value="rb-dst">RB + DST</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button 
              onClick={generateLineups} 
              className="w-full bg-blue-500 hover:bg-blue-600 text-white"
              disabled={isGenerating}
            >
              {isGenerating ? (
                <>Generating...</>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Generate Lineups
                </>
              )}
            </Button>

            {isGenerating && (
              <Progress value={progress} className="w-full" />
            )}
          </div>
        </Card>

        {/* Player Selection Grid */}
        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6 lg:col-span-2">
          <div className="mb-4">
            <h3 className="text-white mb-2">Player Pool</h3>
            <AdvancedFilters
              onFilterChange={(filters) => console.log('Filters changed:', filters)}
            />
          </div>

          <div className="overflow-auto max-h-[600px]">
            <Table>
              <TableHeader>
                <TableRow className="border-slate-700">
                  <TableHead className="text-slate-300">Pos</TableHead>
                  <TableHead className="text-slate-300">Player</TableHead>
                  <TableHead className="text-slate-300">Team</TableHead>
                  <TableHead className="text-slate-300">Opp</TableHead>
                  <TableHead className="text-slate-300">Salary</TableHead>
                  <TableHead className="text-slate-300">Proj</TableHead>
                  <TableHead className="text-slate-300">Status</TableHead>
                  <TableHead className="text-slate-300">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {mockPlayers.map((player) => (
                  <TableRow key={player.id} className="border-slate-700">
                    <TableCell>
                      <Badge variant="outline" className="border-slate-600 text-slate-300">
                        {player.position}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-white">{player.name}</TableCell>
                    <TableCell className="text-slate-400">{player.team}</TableCell>
                    <TableCell className="text-slate-400">{player.opponent}</TableCell>
                    <TableCell className="text-slate-300">${player.salary.toLocaleString()}</TableCell>
                    <TableCell className="text-blue-400">{player.projection}</TableCell>
                    <TableCell>
                      <Badge 
                        variant={player.status === 'Active' ? 'default' : 'secondary'}
                        className={player.status === 'Active' ? 'bg-blue-500/20 text-blue-400' : 'bg-yellow-500/20 text-yellow-400'}
                      >
                        {player.status}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => toggleLock(player.id)}
                          className={lockedPlayers.has(player.id) ? 'text-blue-400' : 'text-slate-400'}
                        >
                          <Lock className="w-4 h-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => toggleExclude(player.id)}
                          className={excludedPlayers.has(player.id) ? 'text-red-400' : 'text-slate-400'}
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Card>
      </div>

      {/* Generated Lineups */}
      {generatedLineups.length > 0 && (
        <Card className="bg-black/40 backdrop-blur-sm border-cyan-500/20 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-white">Generated Lineups ({generatedLineups.length})</h3>
            <Button variant="outline" className="border-slate-600 text-slate-300">
              <Download className="w-4 h-4 mr-2" />
              Export All
            </Button>
          </div>

          <Tabs defaultValue="lineup-1" className="w-full">
            <TabsList className="bg-black/50 border border-cyan-500/20">
              {generatedLineups.map((lineup) => (
                <TabsTrigger key={lineup.id} value={`lineup-${lineup.id}`}>
                  Lineup {lineup.id}
                </TabsTrigger>
              ))}
            </TabsList>
            {generatedLineups.map((lineup) => (
              <TabsContent key={lineup.id} value={`lineup-${lineup.id}`}>
                <div className="grid md:grid-cols-2 gap-4 mb-4">
                  <Card className="bg-black/50 border-cyan-500/20 p-4">
                    <div className="text-slate-400 text-sm">Total Salary</div>
                    <div className="text-white text-2xl">${lineup.totalSalary.toLocaleString()}</div>
                  </Card>
                  <Card className="bg-black/50 border-cyan-500/20 p-4">
                    <div className="text-slate-400 text-sm">Projected Points</div>
                    <div className="text-blue-400 text-2xl">{lineup.projectedPoints.toFixed(1)}</div>
                  </Card>
                </div>

                <Table>
                  <TableHeader>
                    <TableRow className="border-slate-700">
                      <TableHead className="text-slate-300">Pos</TableHead>
                      <TableHead className="text-slate-300">Player</TableHead>
                      <TableHead className="text-slate-300">Team</TableHead>
                      <TableHead className="text-slate-300">Salary</TableHead>
                      <TableHead className="text-slate-300">Projection</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {lineup.players.map((player: any) => (
                      <TableRow key={player.id} className="border-slate-700">
                        <TableCell>
                          <Badge variant="outline" className="border-slate-600 text-slate-300">
                            {player.position}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-white">{player.name}</TableCell>
                        <TableCell className="text-slate-400">{player.team}</TableCell>
                        <TableCell className="text-slate-300">${player.salary.toLocaleString()}</TableCell>
                        <TableCell className="text-blue-400">{player.projection}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>

                <div className="flex gap-2 mt-4">
                  <Button variant="outline" className="border-slate-600 text-slate-300">
                    <Save className="w-4 h-4 mr-2" />
                    Save Lineup
                  </Button>
                  <Button variant="outline" className="border-slate-600 text-slate-300">
                    <Download className="w-4 h-4 mr-2" />
                    Export CSV
                  </Button>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </Card>
      )}
    </div>
  );
}
