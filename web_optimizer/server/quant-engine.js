/**
 * Quantitative DFS Engine
 * 
 * Implements institutional-grade quantitative methods for DFS optimization:
 * - Monte Carlo Simulation for lineup outcome distributions
 * - Kelly Criterion for optimal player exposure sizing
 * - Volatility-adjusted player scoring (Sharpe-like metrics)
 * - Ownership leverage scoring for GPP tournaments
 * - Mean-Variance, Risk Parity, and Kelly optimization strategies
 * - VaR/CVaR risk metrics for lineups and portfolios
 */

class QuantEngine {
  constructor(settings = {}) {
    this.enabled = settings.enabled || false;
    this.strategy = settings.strategy || 'combined';
    this.riskTolerance = settings.riskTolerance || 1.0;
    this.varConfidence = settings.varConfidence || 0.95;
    this.targetVolatility = settings.targetVolatility || 0.20;
    this.monteCarloSims = settings.monteCarloSims || 10000;
    this.maxKellyFraction = settings.maxKellyFraction || 0.25;
    this.expectedWinRate = settings.expectedWinRate || 0.20;
    this.dependencyThreshold = settings.dependencyThreshold || 0.3;
  }

  // ─────────────────────────────────────────────────────
  // Player-Level Quantitative Scoring
  // ─────────────────────────────────────────────────────

  /**
   * Compute a quantitative score for each player based on the selected strategy.
   * Returns a Map of playerId → { quantScore, sharpe, leverage, kellyExposure, ... }
   */
  scorePlayersQuant(players, contestMode = 'gpp') {
    const scores = new Map();
    
    for (const player of players) {
      const projection = player.projection || player.projectedPoints || 0;
      const salary = player.salary || 1;
      const ceiling = player.ceiling || projection * 1.4;
      const floor = player.floor || projection * 0.5;
      const stdDev = player.stdDev || (ceiling - floor) / 4;
      const ownership = player.ownership || 0;

      // 1. Sharpe-like ratio: risk-adjusted value per dollar
      const playerSharpe = stdDev > 0 ? (projection / salary * 1000) / stdDev : projection / salary * 1000;

      // 2. Ownership leverage: high-ceiling, low-owned players score higher in GPPs
      const ownershipFraction = Math.min(ownership / 100, 0.99);
      const leverage = salary > 0 
        ? (ceiling * (1 - ownershipFraction)) / salary * 1000
        : 0;

      // 3. Kelly-optimal exposure: fraction of bankroll to allocate
      //    Kelly = (p * b - q) / b where p = win prob, b = payout odds, q = 1 - p
      //    For DFS: edge = (projection - baseline) / baseline, variance from stdDev
      const baseline = salary / 1000 * 2; // rough baseline: $1k salary → 2 pts
      const edge = baseline > 0 ? Math.max(0, (projection - baseline) / baseline) : 0;
      const variance = stdDev > 0 ? stdDev * stdDev : 1;
      const rawKelly = variance > 0 ? edge / variance : 0;
      const kellyExposure = Math.min(rawKelly, this.maxKellyFraction) * 100; // as percentage

      // 4. Ceiling probability: P(player exceeds 1.5x projection) using normal approximation
      const ceilingTarget = projection * 1.5;
      const ceilingProb = stdDev > 0 
        ? 1 - this._normalCDF((ceilingTarget - projection) / stdDev) 
        : 0;

      // 5. Composite quant score based on strategy
      let quantScore;
      switch (this.strategy) {
        case 'kelly':
          // Kelly-weighted: emphasize edge/variance ratio, salary-adjusted
          quantScore = (playerSharpe * 0.4) + (kellyExposure * 3) + (projection / salary * 1000 * 0.3);
          break;
        case 'mean_variance':
          // Mean-variance: high value, low variance per dollar, scaled by risk tolerance
          quantScore = (projection / salary * 1000) - (this.riskTolerance * stdDev * 0.5) + (playerSharpe * 0.5);
          break;
        case 'risk_parity':
          // Risk parity: normalize by volatility so each player contributes equal risk
          quantScore = stdDev > 0 ? projection / stdDev : projection;
          break;
        case 'equal_weight':
          // Simple projection/salary value
          quantScore = projection / salary * 1000;
          break;
        case 'combined':
        default:
          // Combined: blend all signals with salary-awareness
          if (contestMode === 'cash') {
            // Cash: prioritize floor, consistency, Sharpe (per-dollar value)
            quantScore = (playerSharpe * 0.4) + (floor / salary * 1000 * 0.3) + (projection / salary * 1000 * 0.2) - (stdDev * 0.1 * this.riskTolerance);
          } else {
            // GPP: prioritize ceiling leverage, upside per dollar, ownership fade
            quantScore = (leverage * 0.35) + (playerSharpe * 0.25) + (ceilingProb * 100 * 0.2) + (ceiling / salary * 1000 * 0.2);
          }
          break;
      }

      scores.set(player.id, {
        quantScore,
        sharpe: playerSharpe,
        leverage,
        kellyExposure,
        ceilingProb,
        stdDev,
        ceiling,
        floor,
        edge
      });
    }

    return scores;
  }

  // ─────────────────────────────────────────────────────
  // Monte Carlo Simulation
  // ─────────────────────────────────────────────────────

  /**
   * Run Monte Carlo simulation for a lineup.
   * Samples from each player's distribution, sums to lineup score.
   * Returns distribution statistics.
   */
  monteCarloLineup(lineupPlayers, numSims = null) {
    const sims = numSims || Math.min(this.monteCarloSims, 10000); // cap for performance in JS
    const scores = new Float64Array(sims);

    for (let i = 0; i < sims; i++) {
      let lineupScore = 0;
      for (const player of lineupPlayers) {
        const projection = player.projection || player.projectedPoints || 0;
        const ceiling = player.ceiling || projection * 1.4;
        const floor = player.floor || projection * 0.5;
        const stdDev = player.stdDev || (ceiling - floor) / 4;

        // Sample from truncated normal distribution (bounded by floor/ceiling)
        // Retry up to 10 times, then accept whatever sample we got
        let sample = projection;
        for (let attempt = 0; attempt < 10; attempt++) {
          sample = projection + stdDev * this._boxMullerRandom();
          if (sample >= floor * 0.5 && sample <= ceiling * 1.3) break;
        }
        
        // Clamp to reasonable bounds
        sample = Math.max(0, sample);
        lineupScore += sample;
      }
      scores[i] = lineupScore;
    }

    // Sort for percentile calculations
    scores.sort();

    const mean = scores.reduce((a, b) => a + b, 0) / sims;
    const variance = scores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / sims;
    const stdDev = Math.sqrt(variance);

    // VaR: what's the worst-case at the confidence level?
    const varIndex = Math.floor(sims * (1 - this.varConfidence));
    const valueAtRisk = scores[varIndex] || scores[0];

    // CVaR (Expected Shortfall): average of outcomes below VaR
    let cvarSum = 0;
    let cvarCount = 0;
    for (let i = 0; i <= varIndex; i++) {
      cvarSum += scores[i];
      cvarCount++;
    }
    const conditionalVaR = cvarCount > 0 ? cvarSum / cvarCount : valueAtRisk;

    // Percentiles
    const p10 = scores[Math.floor(sims * 0.10)];
    const p25 = scores[Math.floor(sims * 0.25)];
    const p50 = scores[Math.floor(sims * 0.50)];
    const p75 = scores[Math.floor(sims * 0.75)];
    const p90 = scores[Math.floor(sims * 0.90)];

    // Sharpe ratio (using mean and stdDev, baseline 0 for simplicity)
    const sharpeRatio = stdDev > 0 ? mean / stdDev : 0;

    // Ceiling probability: P(lineup > 1.3x mean projection)
    const totalProjection = lineupPlayers.reduce((s, p) => s + (p.projection || p.projectedPoints || 0), 0);
    const ceilingTarget = totalProjection * 1.3;
    let ceilingHits = 0;
    for (let i = 0; i < sims; i++) {
      if (scores[i] >= ceilingTarget) ceilingHits++;
    }
    const ceilingProbability = ceilingHits / sims;

    return {
      mean: Math.round(mean * 100) / 100,
      stdDev: Math.round(stdDev * 100) / 100,
      sharpeRatio: Math.round(sharpeRatio * 1000) / 1000,
      valueAtRisk: Math.round(valueAtRisk * 100) / 100,
      conditionalVaR: Math.round(conditionalVaR * 100) / 100,
      ceilingProbability: Math.round(ceilingProbability * 10000) / 10000,
      percentiles: {
        p10: Math.round(p10 * 100) / 100,
        p25: Math.round(p25 * 100) / 100,
        p50: Math.round(p50 * 100) / 100,
        p75: Math.round(p75 * 100) / 100,
        p90: Math.round(p90 * 100) / 100,
      },
      simulations: sims
    };
  }

  // ─────────────────────────────────────────────────────
  // Portfolio-Level Analysis
  // ─────────────────────────────────────────────────────

  /**
   * Compute portfolio-level metrics across all lineups.
   * Treats each lineup as an asset in a portfolio.
   */
  analyzePortfolio(lineups) {
    if (!lineups || lineups.length === 0) return null;

    const projections = lineups.map(l => l.totalProjection);
    const mean = projections.reduce((a, b) => a + b, 0) / projections.length;
    const variance = projections.reduce((s, p) => s + (p - mean) ** 2, 0) / projections.length;
    const stdDev = Math.sqrt(variance);
    const portfolioSharpe = stdDev > 0 ? mean / stdDev : 0;

    // Lineup diversity: average pairwise uniqueness
    let totalUniqueness = 0;
    let pairCount = 0;
    for (let i = 0; i < lineups.length; i++) {
      for (let j = i + 1; j < lineups.length; j++) {
        const playersI = new Set(lineups[i].players.map(p => p.id));
        const playersJ = new Set(lineups[j].players.map(p => p.id));
        let overlap = 0;
        for (const id of playersI) {
          if (playersJ.has(id)) overlap++;
        }
        const uniqueness = 1 - (overlap / Math.max(playersI.size, playersJ.size));
        totalUniqueness += uniqueness;
        pairCount++;
      }
    }
    const avgUniqueness = pairCount > 0 ? totalUniqueness / pairCount : 1;

    // Player exposure analysis
    const exposureMap = new Map();
    for (const lineup of lineups) {
      for (const player of lineup.players) {
        const count = exposureMap.get(player.id) || 0;
        exposureMap.set(player.id, count + 1);
      }
    }
    
    const exposures = Array.from(exposureMap.values()).map(c => c / lineups.length * 100);
    const maxExposure = Math.max(...exposures, 0);
    const avgExposure = exposures.length > 0 
      ? exposures.reduce((a, b) => a + b, 0) / exposures.length 
      : 0;

    // Exposure concentration (Herfindahl-like index)
    const exposureConcentration = exposures.length > 0
      ? exposures.reduce((s, e) => s + (e / 100) ** 2, 0)
      : 0;

    return {
      mean: Math.round(mean * 100) / 100,
      stdDev: Math.round(stdDev * 100) / 100,
      sharpeRatio: Math.round(portfolioSharpe * 1000) / 1000,
      avgUniqueness: Math.round(avgUniqueness * 1000) / 1000,
      maxExposure: Math.round(maxExposure * 10) / 10,
      avgExposure: Math.round(avgExposure * 10) / 10,
      exposureConcentration: Math.round(exposureConcentration * 10000) / 10000,
      lineupCount: lineups.length
    };
  }

  /**
   * Select the best player using quant-scored ranking.
   * Replaces simple greedy/balanced/value/projection strategies with quant-informed selection.
   */
  selectPlayerQuant(availablePlayers, quantScores, strategy, contestMode = 'gpp') {
    if (!availablePlayers || availablePlayers.length === 0) return null;

    // Score and sort players by their quant score
    const scored = availablePlayers
      .map(player => ({
        player,
        score: quantScores.get(player.id)?.quantScore || 0
      }))
      .sort((a, b) => b.score - a.score);

    switch (strategy) {
      case 'greedy':
        // Top quant-scored player
        return scored[0].player;
      
      case 'balanced': {
        // Random from top third, weighted by quant score
        const poolSize = Math.max(1, Math.floor(scored.length / 3));
        const pool = scored.slice(0, poolSize);
        // Weighted random: higher scores more likely
        const totalScore = pool.reduce((s, p) => s + Math.max(p.score, 0.01), 0);
        let rand = Math.random() * totalScore;
        for (const entry of pool) {
          rand -= Math.max(entry.score, 0.01);
          if (rand <= 0) return entry.player;
        }
        return pool[0].player;
      }
      
      case 'value':
        // Best quant score (already sorted)
        return scored[0].player;
      
      case 'projection':
        // Highest raw projection (fallback to traditional)
        return availablePlayers
          .sort((a, b) => (b.projection || b.projectedPoints || 0) - (a.projection || a.projectedPoints || 0))[0];
      
      default: {
        // Stochastic selection from top 5, weighted by quant score
        const topN = scored.slice(0, Math.min(5, scored.length));
        const total = topN.reduce((s, p) => s + Math.max(p.score, 0.01), 0);
        let r = Math.random() * total;
        for (const entry of topN) {
          r -= Math.max(entry.score, 0.01);
          if (r <= 0) return entry.player;
        }
        return topN[0].player;
      }
    }
  }

  /**
   * Compute Kelly-optimal exposure limits for players.
   * Returns a Map of playerId → recommended max exposure %.
   */
  kellyExposureLimits(players, defaultMaxExposure = 40) {
    const limits = new Map();
    
    for (const player of players) {
      const projection = player.projection || player.projectedPoints || 0;
      const salary = player.salary || 1;
      const ceiling = player.ceiling || projection * 1.4;
      const floor = player.floor || projection * 0.5;
      const stdDev = player.stdDev || (ceiling - floor) / 4;

      // Kelly fraction: edge / variance
      const baseline = salary / 1000 * 2;
      const edge = baseline > 0 ? Math.max(0, (projection - baseline) / baseline) : 0;
      const variance = stdDev > 0 ? stdDev * stdDev : 1;
      const kellyFraction = Math.min(edge / variance, this.maxKellyFraction);

      // Scale to exposure percentage, bounded by default max
      const kellyExposure = Math.min(kellyFraction * 100 * 2, defaultMaxExposure);
      
      // Minimum 5% exposure for any selected player
      limits.set(player.id, Math.max(5, Math.round(kellyExposure)));
    }

    return limits;
  }

  // ─────────────────────────────────────────────────────
  // Utility Functions
  // ─────────────────────────────────────────────────────

  /**
   * Box-Muller transform: generate standard normal random variable
   */
  _boxMullerRandom() {
    let u1 = 0, u2 = 0;
    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  /**
   * Standard normal CDF approximation (Abramowitz & Stegun)
   */
  _normalCDF(x) {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    const absX = Math.abs(x);
    const t = 1.0 / (1.0 + p * absX);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX / 2);

    return 0.5 * (1.0 + sign * y);
  }
}

module.exports = QuantEngine;
