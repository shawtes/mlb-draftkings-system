import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material';
import { toast } from 'react-hot-toast';
import { Users, Settings, Play, Trophy, Upload, Save, Download, Lock, X, Star } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { dfsApi } from '../services/dfs-api';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#06b6d4' },
    secondary: { main: '#3b82f6' },
    background: { default: '#0a0a0a', paper: '#1a1a1a' },
  },
});

interface Player {
  id: string;
  name: string;
  position: string;
  team: string;
  salary: number;
  projection: number;
  locked?: boolean;
  excluded?: boolean;
  favorite?: boolean;
}

interface DFSOptimizerUnifiedProps {
  sport: string;
}

export default function DFSOptimizerUnified({ sport }: DFSOptimizerUnifiedProps) {
  const [currentTab, setCurrentTab] = useState('players');
  const [players, setPlayers] = useState<Player[]>([]);
  const [uploading, setUploading] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<any[]>([]);

  // Optimization settings
  const [numLineups, setNumLineups] = useState(20);
  const [minSalary, setMinSalary] = useState(45000);
  const [maxSalary, setMaxSalary] = useState(50000);
  const [uniqueness, setUniqueness] = useState(7);
  const [contestType, setContestType] = useState('gpp');
  const [monteCarloIterations, setMonteCarloIterations] = useState(100);
  const [riskTolerance, setRiskTolerance] = useState<'conservative' | 'balanced' | 'aggressive'>('balanced');
  const [enableKelly, setEnableKelly] = useState(false);

  // Stacking settings
  const [twoStack, setTwoStack] = useState<string[]>([]);
  const [threeStack, setThreeStack] = useState<string[]>([]);
  const [teamExposure, setTeamExposure] = useState<{[team: string]: {min: number, max: number}}>({});

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      const result = await dfsApi.uploadPlayers(file);
      setPlayers(result.players || []);
      toast.success(`Uploaded ${result.players?.length || 0} players`);
    } catch (error) {
      toast.error('Failed to upload players');
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  const toggleLock = (playerId: string) => {
    setPlayers(prev => prev.map(p => 
      p.id === playerId ? { ...p, locked: !p.locked, excluded: false } : p
    ));
  };

  const toggleExclude = (playerId: string) => {
    setPlayers(prev => prev.map(p => 
      p.id === playerId ? { ...p, excluded: !p.excluded, locked: false } : p
    ));
  };

  const toggleFavorite = (playerId: string) => {
    setPlayers(prev => prev.map(p => 
      p.id === playerId ? { ...p, favorite: !p.favorite } : p
    ));
  };

  const runOptimization = async () => {
    if (players.length === 0) {
      toast.error('Please upload players first');
      return;
    }

    setOptimizing(true);
    setProgress(0);

    // Simulate progress
    const interval = setInterval(() => {
      setProgress(prev => Math.min(prev + 10, 90));
    }, 500);

    try {
      const settings = {
        players,
        numLineups,
        minSalary,
        maxSalary,
        uniquePlayers: uniqueness,
        monteCarloIterations,
        riskTolerance,
        disableKellySizing: !enableKelly,
        contestType,
        lockedPlayers: players.filter(p => p.locked).map(p => p.id),
        excludedPlayers: players.filter(p => p.excluded).map(p => p.id),
        stackSettings: {
          twoStack,
          threeStack,
        },
        exposureSettings: teamExposure,
      };

      const result = await dfsApi.optimizeLineups(settings);
      setResults(result);
      setProgress(100);
      setCurrentTab('results');
      toast.success(`Generated ${result.length} lineups!`);
    } catch (error) {
      toast.error('Optimization failed');
      console.error(error);
    } finally {
      clearInterval(interval);
      setOptimizing(false);
    }
  };

  const exportLineups = async (format: 'csv' | 'draftkings' | 'fanduel') => {
    try {
      const blob = await dfsApi.exportLineups(results, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `lineups_${format}.${format === 'csv' ? 'csv' : 'csv'}`;
      a.click();
      toast.success('Lineups exported!');
    } catch (error) {
      toast.error('Export failed');
      console.error(error);
    }
  };

  const tabs = [
    { id: 'players', label: 'Players & Projections', icon: Users },
    { id: 'strategy', label: 'Strategy & Settings', icon: Settings },
    { id: 'optimization', label: 'Optimization Control', icon: Play },
    { id: 'results', label: 'Results & Export', icon: Trophy },
  ];

  return (
    <ThemeProvider theme={darkTheme}>
      <div className="h-full flex flex-col bg-black relative overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#0a0a0a_1px,transparent_1px),linear-gradient(to_bottom,#0a0a0a_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40 pointer-events-none" />
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />

        {/* Content */}
        <div className="relative z-10 flex flex-col h-full">
          {/* Header Stats */}
          <div className="p-4 bg-black/60 backdrop-blur-sm border-b border-cyan-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Users className="w-5 h-5 text-cyan-400" />
                  <span className="text-white font-medium">{players.length} Players</span>
                </div>
                <div className="flex items-center gap-2">
                  <Lock className="w-5 h-5 text-green-400" />
                  <span className="text-white font-medium">{players.filter(p => p.locked).length} Locked</span>
                </div>
                <div className="flex items-center gap-2">
                  <X className="w-5 h-5 text-red-400" />
                  <span className="text-white font-medium">{players.filter(p => p.excluded).length} Excluded</span>
                </div>
                <div className="flex items-center gap-2">
                  <Trophy className="w-5 h-5 text-yellow-400" />
                  <span className="text-white font-medium">{results.length} Lineups</span>
                </div>
              </div>
              <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
                {sport.toUpperCase()}
              </Badge>
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1 overflow-hidden">
            <Tabs value={currentTab} onValueChange={setCurrentTab} className="h-full flex flex-col">
              <TabsList className="bg-gradient-to-r from-black via-cyan-950/10 to-black border-b border-cyan-500/20 justify-start px-4">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <TabsTrigger key={tab.id} value={tab.id} className="data-[state=active]:bg-cyan-500/10 data-[state=active]:text-cyan-400">
                      <Icon className="w-4 h-4 mr-2" />
                      {tab.label}
                    </TabsTrigger>
                  );
                })}
              </TabsList>

              <div className="flex-1 overflow-auto p-6">
                {/* Tab 1: Players & Projections */}
                <TabsContent value="players" className="mt-0 h-full">
                  <div className="space-y-4">
                    <Card className="p-4 bg-black/40 border-cyan-500/20">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-white">Upload Players</h3>
                        <div className="flex gap-2">
                          <label htmlFor="file-upload">
                            <Button asChild disabled={uploading}>
                              <span className="cursor-pointer">
                                <Upload className="w-4 h-4 mr-2" />
                                {uploading ? 'Uploading...' : 'Upload CSV'}
                              </span>
                            </Button>
                          </label>
                          <input
                            id="file-upload"
                            type="file"
                            accept=".csv"
                            onChange={handleFileUpload}
                            className="hidden"
                          />
                        </div>
                      </div>
                    </Card>

                    {players.length > 0 && (
                      <Card className="bg-black/40 border-cyan-500/20">
                        <div className="p-4 border-b border-cyan-500/10">
                          <Input 
                            placeholder="Search players..." 
                            className="bg-black/60 border-cyan-500/20"
                          />
                        </div>
                        <div className="max-h-[500px] overflow-auto">
                          <Table>
                            <TableHeader>
                              <TableRow className="border-cyan-500/10">
                                <TableHead>Actions</TableHead>
                                <TableHead>Name</TableHead>
                                <TableHead>Pos</TableHead>
                                <TableHead>Team</TableHead>
                                <TableHead>Salary</TableHead>
                                <TableHead>Projection</TableHead>
                                <TableHead>Value</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {players.slice(0, 50).map((player) => (
                                <TableRow key={player.id} className="border-cyan-500/10">
                                  <TableCell>
                                    <div className="flex gap-1">
                                      <Button
                                        size="sm"
                                        variant={player.locked ? "default" : "ghost"}
                                        onClick={() => toggleLock(player.id)}
                                        className={player.locked ? "bg-green-500/20 hover:bg-green-500/30" : ""}
                                      >
                                        <Lock className="w-3 h-3" />
                                      </Button>
                                      <Button
                                        size="sm"
                                        variant={player.excluded ? "default" : "ghost"}
                                        onClick={() => toggleExclude(player.id)}
                                        className={player.excluded ? "bg-red-500/20 hover:bg-red-500/30" : ""}
                                      >
                                        <X className="w-3 h-3" />
                                      </Button>
                                      <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => toggleFavorite(player.id)}
                                        className={player.favorite ? "text-yellow-400" : ""}
                                      >
                                        <Star className="w-3 h-3" />
                                      </Button>
                                    </div>
                                  </TableCell>
                                  <TableCell className="font-medium text-white">{player.name}</TableCell>
                                  <TableCell>
                                    <Badge variant="outline" className="border-cyan-500/30">
                                      {player.position}
                                    </Badge>
                                  </TableCell>
                                  <TableCell className="text-slate-300">{player.team}</TableCell>
                                  <TableCell className="text-slate-300">${player.salary.toLocaleString()}</TableCell>
                                  <TableCell className="text-cyan-400">{player.projection.toFixed(1)}</TableCell>
                                  <TableCell className="text-green-400">
                                    {((player.projection / player.salary) * 1000).toFixed(2)}x
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </Card>
                    )}
                  </div>
                </TabsContent>

                {/* Tab 2: Strategy & Settings */}
                <TabsContent value="strategy" className="mt-0">
                  <div className="space-y-4">
                    <Card className="p-6 bg-black/40 border-cyan-500/20">
                      <h3 className="text-lg font-semibold text-white mb-4">Team Stacking</h3>
                      <div className="space-y-4">
                        <div>
                          <Label className="text-white">2-Stack Teams</Label>
                          <p className="text-sm text-slate-400 mb-2">Select teams for 2-player stacks</p>
                          <div className="flex flex-wrap gap-2">
                            {[...new Set(players.map(p => p.team))].map(team => (
                              <Badge
                                key={team}
                                variant={twoStack.includes(team) ? "default" : "outline"}
                                className={`cursor-pointer ${twoStack.includes(team) ? 'bg-cyan-500/20 border-cyan-500' : 'border-cyan-500/30'}`}
                                onClick={() => setTwoStack(prev => 
                                  prev.includes(team) ? prev.filter(t => t !== team) : [...prev, team]
                                )}
                              >
                                {team}
                              </Badge>
                            ))}
                          </div>
                        </div>

                        <div>
                          <Label className="text-white">3-Stack Teams</Label>
                          <p className="text-sm text-slate-400 mb-2">Select teams for 3-player stacks</p>
                          <div className="flex flex-wrap gap-2">
                            {[...new Set(players.map(p => p.team))].map(team => (
                              <Badge
                                key={team}
                                variant={threeStack.includes(team) ? "default" : "outline"}
                                className={`cursor-pointer ${threeStack.includes(team) ? 'bg-blue-500/20 border-blue-500' : 'border-cyan-500/30'}`}
                                onClick={() => setThreeStack(prev => 
                                  prev.includes(team) ? prev.filter(t => t !== team) : [...prev, team]
                                )}
                              >
                                {team}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </Card>

                    <Card className="p-6 bg-black/40 border-cyan-500/20">
                      <h3 className="text-lg font-semibold text-white mb-4">Exposure Limits</h3>
                      <p className="text-sm text-slate-400 mb-4">
                        Set min/max exposure for each team across all lineups
                      </p>
                      <div className="grid grid-cols-2 gap-4">
                        {[...new Set(players.map(p => p.team))].slice(0, 8).map(team => (
                          <div key={team} className="space-y-2">
                            <Label className="text-white">{team}</Label>
                            <div className="flex gap-2">
                              <Input
                                type="number"
                                placeholder="Min %"
                                className="bg-black/60 border-cyan-500/20"
                                onChange={(e) => setTeamExposure(prev => ({
                                  ...prev,
                                  [team]: { ...prev[team], min: Number(e.target.value) }
                                }))}
                              />
                              <Input
                                type="number"
                                placeholder="Max %"
                                className="bg-black/60 border-cyan-500/20"
                                onChange={(e) => setTeamExposure(prev => ({
                                  ...prev,
                                  [team]: { ...prev[team], max: Number(e.target.value) }
                                }))}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </Card>
                  </div>
                </TabsContent>

                {/* Tab 3: Optimization Control */}
                <TabsContent value="optimization" className="mt-0">
                  <div className="space-y-4">
                    <Card className="p-6 bg-black/40 border-cyan-500/20">
                      <h3 className="text-lg font-semibold text-white mb-4">Basic Settings</h3>
                      <div className="grid grid-cols-2 gap-6">
                        <div className="space-y-2">
                          <Label className="text-white">Number of Lineups</Label>
                          <Input
                            type="number"
                            value={numLineups}
                            onChange={(e) => setNumLineups(Number(e.target.value))}
                            min={1}
                            max={150}
                            className="bg-black/60 border-cyan-500/20"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-white">Contest Type</Label>
                          <Select value={contestType} onValueChange={setContestType}>
                            <SelectTrigger className="bg-black/60 border-cyan-500/20">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="cash">Cash Game (50/50, Double-up)</SelectItem>
                              <SelectItem value="gpp">GPP (Tournament)</SelectItem>
                              <SelectItem value="showdown">Showdown</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <Label className="text-white">Min Salary: ${minSalary.toLocaleString()}</Label>
                          <Slider
                            value={[minSalary]}
                            onValueChange={(v) => setMinSalary(v[0])}
                            min={40000}
                            max={50000}
                            step={500}
                            className="cursor-pointer"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-white">Max Salary: ${maxSalary.toLocaleString()}</Label>
                          <Slider
                            value={[maxSalary]}
                            onValueChange={(v) => setMaxSalary(v[0])}
                            min={45000}
                            max={50000}
                            step={100}
                            className="cursor-pointer"
                          />
                        </div>

                        <div className="space-y-2 col-span-2">
                          <Label className="text-white">Uniqueness: {uniqueness} players different between lineups</Label>
                          <Slider
                            value={[uniqueness]}
                            onValueChange={(v) => setUniqueness(v[0])}
                            min={1}
                            max={9}
                            step={1}
                            className="cursor-pointer"
                          />
                        </div>
                      </div>
                    </Card>

                    <Card className="p-6 bg-black/40 border-cyan-500/20">
                      <h3 className="text-lg font-semibold text-white mb-4">Advanced Settings</h3>
                      <div className="space-y-6">
                        <div className="space-y-2">
                          <Label className="text-white">Monte Carlo Iterations: {monteCarloIterations}</Label>
                          <p className="text-sm text-slate-400">Higher values = more robust lineups, longer processing</p>
                          <Slider
                            value={[monteCarloIterations]}
                            onValueChange={(v) => setMonteCarloIterations(v[0])}
                            min={10}
                            max={1000}
                            step={10}
                            className="cursor-pointer"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-white">Risk Tolerance</Label>
                          <Select value={riskTolerance} onValueChange={(v: any) => setRiskTolerance(v)}>
                            <SelectTrigger className="bg-black/60 border-cyan-500/20">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="conservative">Conservative (High Floor)</SelectItem>
                              <SelectItem value="balanced">Balanced</SelectItem>
                              <SelectItem value="aggressive">Aggressive (High Ceiling)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="flex items-center justify-between">
                          <div>
                            <Label className="text-white">Kelly Criterion Bankroll Sizing</Label>
                            <p className="text-sm text-slate-400">Optimize lineup distribution based on bankroll management</p>
                          </div>
                          <Switch checked={enableKelly} onCheckedChange={setEnableKelly} />
                        </div>
                      </div>
                    </Card>

                    <Button
                      onClick={runOptimization}
                      disabled={optimizing || players.length === 0}
                      className="w-full h-12 text-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
                    >
                      {optimizing ? (
                        <>
                          <Play className="w-5 h-5 mr-2 animate-spin" />
                          Optimizing... {progress}%
                        </>
                      ) : (
                        <>
                          <Play className="w-5 h-5 mr-2" />
                          Start Optimization
                        </>
                      )}
                    </Button>

                    {optimizing && (
                      <Progress value={progress} className="h-2" />
                    )}
                  </div>
                </TabsContent>

                {/* Tab 4: Results & Export */}
                <TabsContent value="results" className="mt-0">
                  {results.length === 0 ? (
                    <Card className="p-12 bg-black/40 border-cyan-500/20 text-center">
                      <Trophy className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                      <h3 className="text-xl font-semibold text-slate-400 mb-2">No Results Yet</h3>
                      <p className="text-slate-500">Run an optimization to see your lineups here</p>
                    </Card>
                  ) : (
                    <div className="space-y-4">
                      <Card className="p-4 bg-black/40 border-cyan-500/20">
                        <div className="flex items-center justify-between">
                          <h3 className="text-lg font-semibold text-white">
                            Generated {results.length} Lineups
                          </h3>
                          <div className="flex gap-2">
                            <Button onClick={() => exportLineups('draftkings')} variant="outline">
                              <Download className="w-4 h-4 mr-2" />
                              DraftKings
                            </Button>
                            <Button onClick={() => exportLineups('fanduel')} variant="outline">
                              <Download className="w-4 h-4 mr-2" />
                              FanDuel
                            </Button>
                            <Button onClick={() => exportLineups('csv')} variant="outline">
                              <Download className="w-4 h-4 mr-2" />
                              CSV
                            </Button>
                          </div>
                        </div>
                      </Card>

                      <Card className="bg-black/40 border-cyan-500/20">
                        <Table>
                          <TableHeader>
                            <TableRow className="border-cyan-500/10">
                              <TableHead>Rank</TableHead>
                              <TableHead>Projected Points</TableHead>
                              <TableHead>Salary</TableHead>
                              <TableHead>Players</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {results.slice(0, 20).map((lineup, index) => (
                              <TableRow key={lineup.id || index} className="border-cyan-500/10">
                                <TableCell>
                                  <Badge className="bg-cyan-500/20 border-cyan-500">#{index + 1}</Badge>
                                </TableCell>
                                <TableCell className="text-cyan-400 font-semibold">
                                  {lineup.projectedPoints?.toFixed(1) || 'N/A'}
                                </TableCell>
                                <TableCell className="text-white">
                                  ${lineup.totalSalary?.toLocaleString() || 'N/A'}
                                </TableCell>
                                <TableCell>
                                  <div className="flex flex-wrap gap-1">
                                    {lineup.players?.slice(0, 3).map((p: any, i: number) => (
                                      <Badge key={i} variant="outline" className="border-cyan-500/30">
                                        {p.name || p}
                                      </Badge>
                                    ))}
                                    {lineup.players?.length > 3 && (
                                      <Badge variant="outline" className="border-slate-500/30">
                                        +{lineup.players.length - 3} more
                                      </Badge>
                                    )}
                                  </div>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </Card>
                    </div>
                  )}
                </TabsContent>
              </div>
            </Tabs>
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}

