import { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { X, DollarSign, TrendingUp, AlertTriangle, Calculator, Copy, Trash2 } from 'lucide-react';
import { bettingApi, type BetSelection } from '../services/betting-api';
import { toast } from 'react-hot-toast';

interface BettingSlipProps {
  selections: BetSelection[];
  onRemove: (id: string) => void;
  onClear: () => void;
}

export default function BettingSlip({ selections, onRemove, onClear }: BettingSlipProps) {
  const [stake, setStake] = useState('10');
  const [betType, setBetType] = useState<'straight' | 'parlay'>(selections.length > 1 ? 'parlay' : 'straight');
  const [payout, setPayout] = useState<any>(null);
  const [calculating, setCalculating] = useState(false);

  useEffect(() => {
    if (selections.length > 1) {
      setBetType('parlay');
    } else if (selections.length === 1) {
      setBetType('straight');
    }
  }, [selections.length]);

  useEffect(() => {
    if (selections.length > 0 && Number(stake) > 0) {
      calculatePayout();
    }
  }, [selections, stake, betType]);

  const calculatePayout = async () => {
    setCalculating(true);
    try {
      const result = await bettingApi.calculatePayout(selections, betType, Number(stake));
      setPayout(result);
    } catch (error) {
      console.error('Error calculating payout:', error);
    } finally {
      setCalculating(false);
    }
  };

  const handleCopyParlay = () => {
    const parlayText = selections.map(s => 
      `${s.player} (${s.team}) - ${s.prop} ${s.type.toUpperCase()} ${s.line} (${s.odds > 0 ? '+' : ''}${s.odds})`
    ).join('\n');
    
    navigator.clipboard.writeText(parlayText);
    toast.success('Parlay copied to clipboard!');
  };

  if (selections.length === 0) {
    return (
      <Card className="p-6 bg-black/60 border-cyan-500/20 sticky top-4">
        <div className="text-center py-8">
          <DollarSign className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">No selections yet</p>
          <p className="text-slate-500 text-sm mt-1">Add props to build your bet slip</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6 bg-black/60 border-cyan-500/20 sticky top-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Betting Slip</h3>
          <p className="text-sm text-slate-400">{selections.length} {betType === 'parlay' ? 'leg' : 'bet'} parlay</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={handleCopyParlay}>
            <Copy className="w-4 h-4" />
          </Button>
          <Button size="sm" variant="ghost" onClick={onClear}>
            <Trash2 className="w-4 h-4 text-red-400" />
          </Button>
        </div>
      </div>

      {/* Bet Type Toggle */}
      {selections.length > 1 && (
        <div className="flex gap-2 mb-4">
          <Button
            size="sm"
            variant={betType === 'straight' ? 'default' : 'outline'}
            onClick={() => setBetType('straight')}
            className={betType === 'straight' ? 'bg-cyan-500/20 border-cyan-500' : 'border-cyan-500/30'}
          >
            Straight
          </Button>
          <Button
            size="sm"
            variant={betType === 'parlay' ? 'default' : 'outline'}
            onClick={() => setBetType('parlay')}
            className={betType === 'parlay' ? 'bg-cyan-500/20 border-cyan-500' : 'border-cyan-500/30'}
          >
            Parlay
          </Button>
        </div>
      )}

      {/* Selections */}
      <div className="space-y-2 mb-4 max-h-[300px] overflow-y-auto">
        {selections.map((selection) => (
          <div 
            key={selection.id}
            className="p-3 bg-black/40 border border-cyan-500/10 rounded-lg"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <p className="text-white font-medium text-sm">{selection.player}</p>
                <p className="text-slate-400 text-xs">{selection.team}</p>
              </div>
              <Button 
                size="sm" 
                variant="ghost" 
                onClick={() => onRemove(selection.id)}
                className="h-6 w-6 p-0"
              >
                <X className="w-4 h-4 text-slate-400" />
              </Button>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-300">
                {selection.prop} {selection.type.toUpperCase()} {selection.line}
              </span>
              <Badge variant="outline" className="border-cyan-500/30">
                {selection.odds > 0 ? '+' : ''}{selection.odds}
              </Badge>
            </div>
          </div>
        ))}
      </div>

      <Separator className="my-4 bg-cyan-500/20" />

      {/* Stake Input */}
      <div className="mb-4">
        <label className="text-sm text-slate-400 mb-2 block">Stake Amount</label>
        <div className="relative">
          <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <Input
            type="number"
            value={stake}
            onChange={(e) => setStake(e.target.value)}
            min="0"
            step="5"
            className="pl-9 bg-black/60 border-cyan-500/20 text-white"
            placeholder="0.00"
          />
        </div>
      </div>

      {/* Payout Display */}
      {payout && (
        <div className="space-y-3 mb-4">
          <div className="p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-300">Combined Odds</span>
              <span className="text-lg font-bold text-cyan-400">{payout.totalOdds}</span>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-300">Potential Payout</span>
              <span className="text-xl font-bold text-white">${payout.potentialPayout.toFixed(2)}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">Profit</span>
              <span className="text-green-400 font-semibold">+${payout.profit.toFixed(2)}</span>
            </div>
          </div>

          {/* Probability */}
          <div className="flex items-center gap-2 text-sm">
            <Calculator className="w-4 h-4 text-slate-400" />
            <span className="text-slate-400">Implied Probability:</span>
            <span className="text-white font-medium">{payout.probability.toFixed(1)}%</span>
          </div>

          {/* Kelly Criterion */}
          {payout.kellyRecommendation && (
            <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-semibold text-blue-300">Kelly Criterion</span>
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Recommended Stake:</span>
                  <span className="text-white font-semibold">
                    ${payout.kellyRecommendation.recommendedStake.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Kelly %:</span>
                  <span className="text-blue-400">
                    {payout.kellyRecommendation.kellyPercentage.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Warning for large parlays */}
          {selections.length > 5 && (
            <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
              <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-yellow-200">
                Large parlays have lower hit rates. Consider splitting into smaller parlays for better results.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Place Bet Button */}
      <Button 
        className="w-full h-12 text-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
        disabled={!payout || calculating}
      >
        {calculating ? 'Calculating...' : `Place ${betType === 'parlay' ? 'Parlay' : 'Bet'}`}
      </Button>

      {/* Info */}
      <p className="text-xs text-slate-500 text-center mt-3">
        Responsible gambling. Never bet more than you can afford to lose.
      </p>
    </Card>
  );
}

