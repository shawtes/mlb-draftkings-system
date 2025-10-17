import { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import { 
  TrendingUp, 
  Target, 
  Plus, 
  Flame, 
  Filter,
  Search,
  Activity,
  Star,
  ArrowUp,
  ArrowDown,
  Minus
} from 'lucide-react';
import BettingSlip from './BettingSlip';
import { bettingApi, type BetSelection, type PropBet } from '../services/betting-api';
import { toast } from 'react-hot-toast';

interface PropBettingCenterProps {
  sport: string;
}

export default function PropBettingCenter({ sport }: PropBettingCenterProps) {
  const [props, setProps] = useState<PropBet[]>([]);
  const [selections, setSelections] = useState<BetSelection[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [minEdge, setMinEdge] = useState([0]);
  const [propTypeFilter, setPropTypeFilter] = useState('all');

  useEffect(() => {
    loadProps();
  }, [sport]);

  const loadProps = async () => {
    setLoading(true);
    try {
      const data = await bettingApi.getProps(sport, {
        minEdge: minEdge[0],
      });
      setProps(data);
    } catch (error) {
      console.error('Error loading props:', error);
      // Use mock data
      setProps(mockProps);
    } finally {
      setLoading(false);
    }
  };

  const mockProps: PropBet[] = [
    { id: '1', player: 'Patrick Mahomes', team: 'KC', position: 'QB', opponent: '@LAC', prop: 'Passing Yards', line: 287.5, overOdds: -110, underOdds: -110, projection: 312.4, edge: 8.2, confidence: 95, hitRate: 68, trend: 'up' },
    { id: '2', player: 'Christian McCaffrey', team: 'SF', position: 'RB', opponent: '@DAL', prop: 'Rushing Yards', line: 95.5, overOdds: -115, underOdds: -105, projection: 108.3, edge: 6.7, confidence: 92, hitRate: 72, trend: 'up' },
    { id: '3', player: 'Tyreek Hill', team: 'MIA', position: 'WR', opponent: '@BUF', prop: 'Receiving Yards', line: 82.5, overOdds: 105, underOdds: -125, projection: 94.1, edge: 5.3, confidence: 85, hitRate: 64, trend: 'up' },
    { id: '4', player: 'Travis Kelce', team: 'KC', position: 'TE', opponent: '@LAC', prop: 'Receptions', line: 5.5, overOdds: -105, underOdds: -115, projection: 7.2, edge: 7.1, confidence: 88, hitRate: 71, trend: 'up' },
    { id: '5', player: 'Josh Allen', team: 'BUF', position: 'QB', opponent: 'vs MIA', prop: 'Passing TDs', line: 2.5, overOdds: -120, underOdds: 100, projection: 2.9, edge: 4.1, confidence: 82, hitRate: 58, trend: 'neutral' },
    { id: '6', player: 'Austin Ekeler', team: 'LAC', position: 'RB', opponent: 'vs KC', prop: 'Rush + Rec Yards', line: 95.5, overOdds: -110, underOdds: -110, projection: 108.3, edge: 5.8, confidence: 87, hitRate: 66, trend: 'up' },
    { id: '7', player: 'Justin Jefferson', team: 'MIN', position: 'WR', opponent: 'vs CHI', prop: 'Receiving Yards', line: 85.5, overOdds: -115, underOdds: -105, projection: 96.2, edge: 6.2, confidence: 90, hitRate: 69, trend: 'up' },
    { id: '8', player: 'CeeDee Lamb', team: 'DAL', position: 'WR', opponent: 'vs SF', prop: 'Receptions', line: 6.5, overOdds: -110, underOdds: -110, projection: 7.8, edge: 5.1, confidence: 86, hitRate: 67, trend: 'neutral' },
  ];

  const addToBetSlip = (prop: PropBet, type: 'over' | 'under') => {
    const odds = type === 'over' ? prop.overOdds : prop.underOdds;
    const selection: BetSelection = {
      id: `${prop.id}-${type}`,
      player: prop.player,
      team: prop.team,
      prop: prop.prop,
      line: prop.line,
      odds,
      type,
    };

    if (selections.find(s => s.id === selection.id)) {
      toast.error('Already in bet slip');
      return;
    }

    setSelections([...selections, selection]);
    toast.success(`Added ${prop.player} to bet slip`);
  };

  const removeFromBetSlip = (id: string) => {
    setSelections(selections.filter(s => s.id !== id));
  };

  const filteredProps = props.filter(prop => {
    const matchesSearch = prop.player.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          prop.team.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesEdge = !prop.edge || prop.edge >= minEdge[0];
    const matchesType = propTypeFilter === 'all' || prop.prop.toLowerCase().includes(propTypeFilter.toLowerCase());
    return matchesSearch && matchesEdge && matchesType;
  });

  return (
    <div className="h-full overflow-auto p-6">
      {/* Main Card Container */}
      <div className="bg-slate-800 backdrop-blur-sm rounded-2xl border border-cyan-500/20 shadow-2xl relative overflow-hidden min-h-full flex flex-col">
        
        {/* Content */}
        <div className="relative z-10 flex flex-col h-full p-6">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                Prop Betting Center
              </h1>
              <p className="text-white">Find edges, build parlays, and maximize value</p>
            </div>
            <Badge className="bg-cyan-500/20 border-cyan-500">
              <Flame className="w-3 h-3 mr-1" />
              {filteredProps.length} Hot Props
            </Badge>
          </div>

          {/* Filters */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
              <Input
                placeholder="Search players or teams..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 bg-black/60 border-cyan-500/20"
              />
            </div>

            <Select value={propTypeFilter} onValueChange={setPropTypeFilter}>
              <SelectTrigger className="bg-black/60 border-cyan-500/20 text-white">
                <SelectValue placeholder="Prop Type" />
              </SelectTrigger>
              <SelectContent className="bg-slate-900 border-cyan-500/20 text-white">
                <SelectItem value="all" className="text-white focus:text-white">All Props</SelectItem>
                <SelectItem value="passing" className="text-white focus:text-white">Passing</SelectItem>
                <SelectItem value="rushing" className="text-white focus:text-white">Rushing</SelectItem>
                <SelectItem value="receiving" className="text-white focus:text-white">Receiving</SelectItem>
                <SelectItem value="yards" className="text-white focus:text-white">Yards</SelectItem>
                <SelectItem value="tds" className="text-white focus:text-white">Touchdowns</SelectItem>
              </SelectContent>
            </Select>

            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-300" />
              <span className="text-sm text-white whitespace-nowrap">Min Edge: {minEdge[0]}%</span>
            </div>

            <Slider
              value={minEdge}
              onValueChange={setMinEdge}
              min={0}
              max={15}
              step={0.5}
              className="cursor-pointer"
            />
          </div>
        </div>

        {/* Main Content - Side by Side */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 overflow-hidden">
          {/* Props List - 2/3 width */}
          <div className="lg:col-span-2 overflow-auto space-y-3">
            {filteredProps.map((prop) => (
              <Card key={prop.id} className="p-4 bg-black/60 border-cyan-500/20 hover:bg-black/80 transition-all">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="text-white font-semibold">{prop.player}</h4>
                      <Badge variant="outline" className="border-cyan-500/30 text-xs">
                        {prop.position}
                      </Badge>
                      <span className="text-slate-200 text-sm">{prop.team} {prop.opponent}</span>
                    </div>
                    <p className="text-slate-200 text-sm">{prop.prop}</p>
                  </div>
                  {prop.trend && (
                    <Badge className={
                      prop.trend === 'up' ? 'bg-green-500/20 border-green-500' :
                      prop.trend === 'down' ? 'bg-red-500/20 border-red-500' :
                      'bg-slate-500/20 border-slate-500'
                    }>
                      {prop.trend === 'up' && <ArrowUp className="w-3 h-3" />}
                      {prop.trend === 'down' && <ArrowDown className="w-3 h-3" />}
                      {prop.trend === 'neutral' && <Minus className="w-3 h-3" />}
                    </Badge>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3 mb-3">
                  <Button
                    onClick={() => addToBetSlip(prop, 'over')}
                    variant="outline"
                    className="border-green-500/30 hover:bg-green-500/10 hover:border-green-500"
                  >
                    <div className="flex flex-col items-start w-full">
                      <span className="text-xs text-slate-200">OVER {prop.line}</span>
                      <span className="text-green-400 font-semibold">
                        {prop.overOdds > 0 ? '+' : ''}{prop.overOdds}
                      </span>
                    </div>
                  </Button>

                  <Button
                    onClick={() => addToBetSlip(prop, 'under')}
                    variant="outline"
                    className="border-red-500/30 hover:bg-red-500/10 hover:border-red-500"
                  >
                    <div className="flex flex-col items-start w-full">
                      <span className="text-xs text-slate-200">UNDER {prop.line}</span>
                      <span className="text-red-400 font-semibold">
                        {prop.underOdds > 0 ? '+' : ''}{prop.underOdds}
                      </span>
                    </div>
                  </Button>
                </div>

                {/* Stats */}
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-3">
                    {prop.projection && (
                      <div className="flex items-center gap-1">
                        <Target className="w-4 h-4 text-cyan-400" />
                        <span className="text-slate-300">Proj: {prop.projection.toFixed(1)}</span>
                      </div>
                    )}
                    {prop.edge !== undefined && (
                      <Badge className={
                        prop.edge > 7 ? 'bg-green-500/20 border-green-500' :
                        prop.edge > 4 ? 'bg-cyan-500/20 border-cyan-500' :
                        'bg-slate-500/20 border-slate-500'
                      }>
                        {prop.edge > 0 ? '+' : ''}{prop.edge.toFixed(1)}% Edge
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {prop.confidence && (
                      <span className="text-xs text-slate-200">
                        {prop.confidence}% Confidence
                      </span>
                    )}
                    {prop.hitRate && (
                      <span className="text-xs text-cyan-400">
                        {prop.hitRate}% Hit Rate
                      </span>
                    )}
                  </div>
                </div>
              </Card>
            ))}

            {filteredProps.length === 0 && (
              <Card className="p-12 bg-black/60 border-cyan-500/20 text-center">
                <Target className="w-16 h-16 text-slate-300 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">No Props Found</h3>
                <p className="text-white">Try adjusting your filters</p>
              </Card>
            )}
          </div>

          {/* Betting Slip - 1/3 width */}
          <div className="lg:col-span-1">
            <BettingSlip
              selections={selections}
              onRemove={removeFromBetSlip}
              onClear={() => setSelections([])}
            />
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

