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
    this.sport = (settings.sport || '').toUpperCase() || null;
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

    // Regime-aware risk adjustment (Step 7)
    const slateRegime = this.analyzeSlateRegime(players);
    this._lastSlateRegime = slateRegime;
    const effectiveRiskTolerance = this.riskTolerance * slateRegime.riskMultiplier;

    for (const player of players) {
      const projection = player.projection || player.projectedPoints || 0;
      const salary = player.salary || 1;
      const ceiling = player.ceiling || projection * 1.4;
      const floor = player.floor || projection * 0.5;
      const stdDev = this._getPlayerStdDev(player);
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

      // 4. Ceiling probability: use training pipeline's quantile regression if available,
      //    otherwise estimate from normal approximation
      let ceilingProb;
      if (player.probOver20 !== undefined) {
        // Training pipeline provided calibrated probabilities from quantile regression
        ceilingProb = player.probOver20;
      } else {
        const ceilingTarget = projection * 1.5;
        ceilingProb = stdDev > 0
          ? 1 - this._normalCDF((ceilingTarget - projection) / stdDev)
          : 0;
      }

      // Boom probability: use training prob_over thresholds if available
      const boomProb = player.probOver15 !== undefined
        ? player.probOver15
        : (stdDev > 0 ? 1 - this._normalCDF((projection * 1.5 - projection) / stdDev) : 0);

      // Bust probability: P(points < floor)
      const bustProb = stdDev > 0
        ? this._normalCDF((floor * 1.2 - projection) / stdDev)
        : 0;

      // 5. Composite quant score based on strategy
      //    Regime bonuses apply across all strategies when GARCH data is present:
      //    Cash: consistency_regime=1 → more reliable floor; bull_regime=1 → positive trend
      //    GPP: volatility_regime=1 → wider outcome range; high hurst → trending performance
      let quantScore;
      const regimeBonus = this._computeRegimeBonus(player, contestMode);
      switch (this.strategy) {
        case 'kelly':
          // Kelly-weighted: emphasize edge/variance ratio, salary-adjusted
          quantScore = (playerSharpe * 0.4) + (kellyExposure * 3) + (projection / salary * 1000 * 0.3);
          quantScore += regimeBonus;
          break;
        case 'mean_variance':
          // Mean-variance: high value, low variance per dollar, scaled by effective risk tolerance
          quantScore = (projection / salary * 1000) - (effectiveRiskTolerance * stdDev * 0.5) + (playerSharpe * 0.5);
          quantScore += regimeBonus;
          break;
        case 'risk_parity':
          // Risk parity: normalize by volatility so each player contributes equal risk
          quantScore = stdDev > 0 ? projection / stdDev : projection;
          quantScore += regimeBonus;
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
            quantScore = (playerSharpe * 0.4) + (floor / salary * 1000 * 0.3) + (projection / salary * 1000 * 0.2) - (stdDev * 0.1 * effectiveRiskTolerance);
          } else {
            // GPP: prioritize ceiling leverage, upside per dollar, ownership fade
            quantScore = (leverage * 0.35) + (playerSharpe * 0.25) + (ceilingProb * 100 * 0.2) + (ceiling / salary * 1000 * 0.2);
          }
          quantScore += regimeBonus;
          break;
      }

      scores.set(player.id, {
        quantScore,
        sharpe: playerSharpe,
        leverage,
        kellyExposure,
        ceilingProb,
        boomProb,
        bustProb,
        stdDev,
        ceiling,
        floor,
        edge,
        volatilitySource: this._getVolatilitySource(player),
        regimeBonus,
        // Pass through training pipeline probabilities if available
        probOver5: player.probOver5,
        probOver10: player.probOver10,
        probOver15: player.probOver15,
        probOver20: player.probOver20,
        probOver25: player.probOver25,
        probOver30: player.probOver30,
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
    const n = lineupPlayers.length;
    const scores = new Float64Array(sims);

    // Pre-extract player parameters into typed arrays for performance
    const projections = new Float64Array(n);
    const stdDevs = new Float64Array(n);
    const ceilings = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const player = lineupPlayers[k];
      projections[k] = player.projection || player.projectedPoints || 0;
      stdDevs[k] = this._getPlayerStdDev(player);
      ceilings[k] = player.ceiling || projections[k] * 1.4;
    }

    // Build correlation matrix and Cholesky decompose for correlated sampling
    const corrResult = this._buildCorrelationMatrix(lineupPlayers);
    let L = null;
    if (corrResult) {
      L = this._choleskyDecompose(corrResult.matrix, corrResult.n);
    }

    // Independent Z buffer and correlated Z output
    const independentZ = new Float64Array(n);
    const correlatedZ = new Float64Array(n);

    for (let i = 0; i < sims; i++) {
      // Generate N independent standard normal Z values
      for (let k = 0; k < n; k++) {
        independentZ[k] = this._boxMullerRandom();
      }

      // If Cholesky L exists, multiply L * Z for correlated samples
      if (L) {
        for (let row = 0; row < n; row++) {
          let sum = 0;
          for (let col = 0; col <= row; col++) {
            sum += L[row * n + col] * independentZ[col];
          }
          correlatedZ[row] = sum;
        }
      } else {
        // No correlation data — use independent Z directly
        for (let k = 0; k < n; k++) {
          correlatedZ[k] = independentZ[k];
        }
      }

      // Scale and sum: sample = projection + stdDev * z, clamped to [0, ceiling * 1.5]
      let lineupScore = 0;
      for (let k = 0; k < n; k++) {
        let sample = projections[k] + stdDevs[k] * correlatedZ[k];
        sample = Math.max(0, Math.min(sample, ceilings[k] * 1.5));
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

    // Track volatility sources used in this lineup
    let garchCount = 0;
    for (let k = 0; k < n; k++) {
      const src = this._getVolatilitySource(lineupPlayers[k]);
      if (src === 'garch_conditional' || src === 'garch_historical') garchCount++;
    }

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
      simulations: sims,
      garchPlayersUsed: garchCount,
      correlatedSampling: L !== null,
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
      const stdDev = this._getPlayerStdDev(player);

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
  // GARCH Volatility Integration
  // ─────────────────────────────────────────────────────

  /**
   * Get player standard deviation using GARCH volatility priority chain:
   * 1. garchConditionalVolatility * projection (forward-looking, most responsive)
   * 2. garchVolatility * projection (historical GARCH average)
   * 3. stdDev (from quantile regression)
   * 4. (ceiling - floor) / 4 (fallback)
   */
  _getPlayerStdDev(player) {
    const projection = player.projection || player.projectedPoints || 0;
    const ceiling = player.ceiling || projection * 1.4;
    const floor = player.floor || projection * 0.5;

    if (player.garchConditionalVolatility != null && player.garchConditionalVolatility > 0 && projection > 0) {
      return player.garchConditionalVolatility * projection;
    }
    if (player.garchVolatility != null && player.garchVolatility > 0 && projection > 0) {
      return player.garchVolatility * projection;
    }
    if (player.stdDev != null && player.stdDev > 0) {
      return player.stdDev;
    }
    return (ceiling - floor) / 4;
  }

  /**
   * Compute per-player regime bonus based on GARCH regime indicators and contest mode.
   * Cash mode rewards consistency and bull trends; GPP rewards volatility and momentum.
   */
  _computeRegimeBonus(player, contestMode) {
    let bonus = 0;
    if (contestMode === 'cash') {
      if (player.consistencyRegime === 1) bonus += 0.5;
      if (player.bullRegime === 1) bonus += 0.3;
      // Penalize high-volatility players in cash (they're risky for floor games)
      if (player.volatilityRegime === 1) bonus -= 0.2;
    } else {
      // GPP: reward volatility (wider distributions = higher ceilings)
      if (player.volatilityRegime === 1) bonus += 0.5;
      if (player.hurstExponent != null && player.hurstExponent > 0.5) bonus += 0.3;
      // Momentum regime = trending upward, good for GPP upside
      if (player.momentumRegime === 1) bonus += 0.2;
    }
    return bonus;
  }

  /**
   * Identify which volatility source was used for a player.
   * Returns 'garch_conditional' | 'garch_historical' | 'quantile_regression' | 'fallback'.
   */
  _getVolatilitySource(player) {
    const projection = player.projection || player.projectedPoints || 0;
    if (player.garchConditionalVolatility != null && player.garchConditionalVolatility > 0 && projection > 0) {
      return 'garch_conditional';
    }
    if (player.garchVolatility != null && player.garchVolatility > 0 && projection > 0) {
      return 'garch_historical';
    }
    if (player.stdDev != null && player.stdDev > 0) {
      return 'quantile_regression';
    }
    return 'fallback';
  }

  /**
   * Compute volatility source coverage for a set of players.
   * Returns { garch_conditional, garch_historical, quantile_regression, fallback, total }.
   */
  getVolatilityCoverage(players) {
    const counts = { garch_conditional: 0, garch_historical: 0, quantile_regression: 0, fallback: 0, total: players.length };
    for (const player of players) {
      counts[this._getVolatilitySource(player)]++;
    }
    return counts;
  }

  // ─────────────────────────────────────────────────────
  // Correlated Monte Carlo (Cholesky)
  // ─────────────────────────────────────────────────────

  /**
   * Build a correlation matrix for lineup players using sport-specific
   * base correlations and same-game opponent detection.
   *
   * Correlation structure per sport (from CLAUDE.md research):
   *   Same-team:   NFL +0.40, MLB +0.30, NBA +0.15
   *   Same-game (different teams / opponents): +0.05
   *   Different games: 0.0
   *
   * If a player has avgPlayerCorrelation from the training pipeline,
   * that value overrides the sport default for same-team pairs.
   *
   * Returns { matrix: Float64Array (row-major NxN), n } or null if
   * no team information is available on any player pair.
   */
  _buildCorrelationMatrix(players) {
    const n = players.length;
    const matrix = new Float64Array(n * n);

    // Sport-specific base correlations
    const SPORT_SAME_TEAM = { NFL: 0.40, MLB: 0.30, NBA: 0.15 };
    const SAME_GAME_DIFF_TEAM = 0.05;

    // Determine sport: use constructor value, or infer from position names
    let sport = this.sport;
    if (!sport) {
      for (const p of players) {
        const pos = (p.position || '').toUpperCase();
        if (pos === 'QB' || pos === 'WR' || pos === 'TE' || pos === 'DST' || pos === 'K') { sport = 'NFL'; break; }
        if (pos === 'PG' || pos === 'SG' || pos === 'SF' || pos === 'PF' || pos === 'C') { sport = 'NBA'; break; }
        if (pos === 'SP' || pos === 'RP' || pos === '1B' || pos === '2B' || pos === '3B' || pos === 'SS' || pos === 'OF' || pos === 'CF' || pos === 'LF' || pos === 'RF') { sport = 'MLB'; break; }
      }
    }

    const baseSameTeam = SPORT_SAME_TEAM[sport] || 0.20;

    // Build a mapping from team → opponent team using the opponent field
    // This lets us detect if two players on different teams share the same game.
    // player.opponent is typically "NYY" or "vs NYY" or "@NYY"
    const teamToOpponent = new Map();
    for (const p of players) {
      const team = p.team || p.teamAbbrev || '';
      if (!team) continue;
      let opp = p.opponent || '';
      // Strip common prefixes: "vs ", "@ ", "v "
      opp = opp.replace(/^(vs\.?\s*|@\s*|v\s+)/i, '').trim().toUpperCase();
      if (opp) {
        teamToOpponent.set(team.toUpperCase(), opp);
      }
    }

    let hasCorrelation = false;

    for (let i = 0; i < n; i++) {
      matrix[i * n + i] = 1.0; // diagonal
      for (let j = i + 1; j < n; j++) {
        let rho = 0.0;
        const pi = players[i];
        const pj = players[j];
        const teamI = (pi.team || pi.teamAbbrev || '').toUpperCase();
        const teamJ = (pj.team || pj.teamAbbrev || '').toUpperCase();

        if (teamI && teamJ) {
          if (teamI === teamJ) {
            // Same team: use training pipeline value if available, else sport default
            const corrI = pi.avgPlayerCorrelation || 0;
            const corrJ = pj.avgPlayerCorrelation || 0;
            const avgCorr = (corrI + corrJ) / 2;
            rho = avgCorr > 0
              ? Math.min(Math.max(avgCorr, baseSameTeam * 0.5), 0.60)
              : baseSameTeam;
            hasCorrelation = true;
          } else {
            // Different teams — check if they're in the same game
            const oppOfI = teamToOpponent.get(teamI) || '';
            const oppOfJ = teamToOpponent.get(teamJ) || '';
            const sameGame = (oppOfI === teamJ) || (oppOfJ === teamI);
            if (sameGame) {
              rho = SAME_GAME_DIFF_TEAM;
              hasCorrelation = true;
            }
            // else rho stays 0.0 (different games)
          }
        }

        matrix[i * n + j] = rho;
        matrix[j * n + i] = rho;
      }
    }

    if (!hasCorrelation) return null;
    return { matrix, n };
  }

  /**
   * Standard Cholesky decomposition on a row-major Float64Array.
   * Returns lower-triangular L such that L * L^T = A, or null if not positive definite.
   */
  _choleskyDecompose(matrix, n) {
    const L = new Float64Array(n * n);

    // Add small diagonal jitter for numerical stability
    for (let i = 0; i < n; i++) {
      matrix[i * n + i] += 1e-6;
    }

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += L[i * n + k] * L[j * n + k];
        }
        if (i === j) {
          const diag = matrix[i * n + i] - sum;
          if (diag <= 0) return null; // not positive definite
          L[i * n + j] = Math.sqrt(diag);
        } else {
          L[i * n + j] = (matrix[i * n + j] - sum) / L[j * n + j];
        }
      }
    }

    return L;
  }

  // ─────────────────────────────────────────────────────
  // Regime-Aware Strategy Tuning
  // ─────────────────────────────────────────────────────

  /**
   * Analyze slate-level regime indicators across all players.
   * Returns { riskMultiplier, stackBias, regimeLabel, metrics }.
   */
  analyzeSlateRegime(players) {
    let volatilityCount = 0;
    let bullCount = 0;
    let consistencyCount = 0;
    let regimePlayerCount = 0;

    for (const player of players) {
      const hasRegime = player.volatilityRegime != null || player.bullRegime != null || player.consistencyRegime != null;
      if (!hasRegime) continue;
      regimePlayerCount++;
      if (player.volatilityRegime === 1) volatilityCount++;
      if (player.bullRegime === 1) bullCount++;
      if (player.consistencyRegime === 1) consistencyCount++;
    }

    if (regimePlayerCount === 0) {
      return { riskMultiplier: 1.0, stackBias: 1.0, regimeLabel: 'neutral', metrics: { volatilityFraction: 0, bullFraction: 0, consistencyFraction: 0, regimePlayerCount: 0 } };
    }

    const volatilityFraction = volatilityCount / regimePlayerCount;
    const bullFraction = bullCount / regimePlayerCount;
    const consistencyFraction = consistencyCount / regimePlayerCount;

    let riskMultiplier = 1.0;
    let stackBias = 1.0;
    let regimeLabel = 'neutral';

    if (volatilityFraction > 0.5) {
      riskMultiplier = 1.3;
      regimeLabel = 'high_volatility';
    } else if (bullFraction > 0.6) {
      riskMultiplier = 0.8;
      regimeLabel = 'bull';
    }

    if (consistencyFraction > 0.5) {
      stackBias = 1.2;
      if (regimeLabel === 'neutral') regimeLabel = 'consistent';
    }

    return {
      riskMultiplier,
      stackBias,
      regimeLabel,
      metrics: { volatilityFraction, bullFraction, consistencyFraction, regimePlayerCount }
    };
  }

  /**
   * Returns the last computed slate regime, or null if not yet computed.
   */
  getLastSlateRegime() {
    return this._lastSlateRegime || null;
  }

  // ─────────────────────────────────────────────────────
  // Active Portfolio Construction
  // ─────────────────────────────────────────────────────

  /**
   * Select optimal lineup subset from candidates using greedy forward selection
   * (Haugh & Singal 2021). Maximizes portfolio Sharpe while maintaining diversity.
   *
   * Portfolio Sharpe = sum(individual Sharpes) / sqrt(sum of all pairwise correlations)
   * where pairwise correlation ~ sharedPlayers / rosterSize.
   * Overlap constraint: no pair of selected lineups shares more than maxOverlap players.
   */
  selectOptimalPortfolio(candidates, targetCount, contestMode = 'gpp', maxOverlap = 4) {
    if (!candidates || candidates.length === 0) return [];
    if (candidates.length <= targetCount) return candidates;

    const n = candidates.length;

    // Pre-compute player ID sets and roster sizes
    const playerSets = candidates.map(c => new Set((c.players || []).map(p => p.id)));
    const rosterSizes = playerSets.map(s => s.size);

    // Pre-compute pairwise overlap counts and correlations (symmetric, row-major)
    const overlapMatrix = new Int32Array(n * n);
    const correlationMatrix = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      overlapMatrix[i * n + i] = rosterSizes[i];
      correlationMatrix[i * n + i] = 1.0;
      for (let j = i + 1; j < n; j++) {
        let shared = 0;
        for (const id of playerSets[i]) {
          if (playerSets[j].has(id)) shared++;
        }
        overlapMatrix[i * n + j] = shared;
        overlapMatrix[j * n + i] = shared;
        const avgRoster = (rosterSizes[i] + rosterSizes[j]) / 2;
        const corr = avgRoster > 0 ? shared / avgRoster : 0;
        correlationMatrix[i * n + j] = corr;
        correlationMatrix[j * n + i] = corr;
      }
    }

    // Per-lineup quality score (Sharpe for cash, blended for GPP)
    const lineupScores = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const lineup = candidates[i];
      const qm = lineup.quantMetrics;
      if (contestMode === 'cash') {
        lineupScores[i] = qm && qm.sharpeRatio != null ? qm.sharpeRatio : (lineup.totalProjection || 0) / 100;
      } else {
        if (qm) {
          const p90 = (qm.percentiles && qm.percentiles.p90) || qm.mean || 0;
          const proj = lineup.totalProjection || qm.mean || 0;
          const ceilProb = qm.ceilingProbability || 0;
          lineupScores[i] = 0.5 * p90 + 0.3 * proj + 0.2 * ceilProb * 100;
        } else {
          lineupScores[i] = lineup.totalProjection || 0;
        }
      }
    }

    // Portfolio Sharpe: sum(individual scores) / sqrt(sum of all pairwise correlations)
    const computePortfolioSharpe = (indices) => {
      let sumScore = 0;
      for (const idx of indices) {
        sumScore += lineupScores[idx];
      }
      let sumCorr = 0;
      const arr = Array.from(indices);
      for (let a = 0; a < arr.length; a++) {
        for (let b = 0; b < arr.length; b++) {
          sumCorr += correlationMatrix[arr[a] * n + arr[b]];
        }
      }
      return sumScore / Math.sqrt(Math.max(sumCorr, 1e-10));
    };

    // Step 1: Seed with the highest individual score lineup
    const selectedIndices = new Set();
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < n; i++) {
      if (lineupScores[i] > bestVal) {
        bestVal = lineupScores[i];
        bestIdx = i;
      }
    }
    selectedIndices.add(bestIdx);

    // Step 2: Greedy forward selection — add lineup that maximally improves portfolio Sharpe
    while (selectedIndices.size < targetCount) {
      let bestCandidate = -1;
      let bestPortfolioSharpe = -Infinity;

      for (let i = 0; i < n; i++) {
        if (selectedIndices.has(i)) continue;

        // Overlap constraint: no pair exceeds maxOverlap shared players
        let passesOverlap = true;
        for (const si of selectedIndices) {
          if (overlapMatrix[i * n + si] > maxOverlap) {
            passesOverlap = false;
            break;
          }
        }
        if (!passesOverlap) continue;

        const candidateSet = new Set(selectedIndices);
        candidateSet.add(i);
        const portfolioSharpe = computePortfolioSharpe(candidateSet);

        if (portfolioSharpe > bestPortfolioSharpe) {
          bestPortfolioSharpe = portfolioSharpe;
          bestCandidate = i;
        }
      }

      if (bestCandidate === -1) {
        // No candidate passes overlap constraint — relax by picking best remaining
        let fallbackBest = -1;
        let fallbackScore = -Infinity;
        for (let i = 0; i < n; i++) {
          if (selectedIndices.has(i)) continue;
          if (lineupScores[i] > fallbackScore) {
            fallbackScore = lineupScores[i];
            fallbackBest = i;
          }
        }
        if (fallbackBest === -1) break;
        selectedIndices.add(fallbackBest);
      } else {
        selectedIndices.add(bestCandidate);
      }
    }

    return Array.from(selectedIndices).map(i => candidates[i]);
  }

  /**
   * Run Monte Carlo on each lineup, then select the optimal portfolio subset.
   * Intended for sport optimizers: generate 3x candidates, call this, get best N.
   * Returns { selected: Lineup[], portfolioMetrics, volCoverage }.
   */
  portfolioOptimize(candidates, targetCount, contestMode = 'gpp', allPlayers = []) {
    // Run MC on any candidate that doesn't already have quantMetrics
    for (const lineup of candidates) {
      if (lineup.players && lineup.players.length > 0 && !lineup.quantMetrics) {
        const mcResult = this.monteCarloLineup(lineup.players);
        lineup.quantMetrics = {
          simMean: mcResult.mean,
          simStdDev: mcResult.stdDev,
          sharpeRatio: mcResult.sharpeRatio,
          valueAtRisk: mcResult.valueAtRisk,
          conditionalVaR: mcResult.conditionalVaR,
          ceilingProbability: mcResult.ceilingProbability,
          percentiles: mcResult.percentiles,
          simulations: mcResult.simulations,
          garchPlayersUsed: mcResult.garchPlayersUsed,
        };
      }
    }

    // Pre-sort by contest mode
    if (contestMode === 'cash') {
      candidates.sort((a, b) => {
        const aS = a.quantMetrics ? a.quantMetrics.sharpeRatio : 0;
        const bS = b.quantMetrics ? b.quantMetrics.sharpeRatio : 0;
        return bS - aS;
      });
    } else {
      candidates.sort((a, b) => {
        const aP90 = a.quantMetrics && a.quantMetrics.percentiles ? a.quantMetrics.percentiles.p90 : 0;
        const bP90 = b.quantMetrics && b.quantMetrics.percentiles ? b.quantMetrics.percentiles.p90 : 0;
        if (bP90 !== aP90) return bP90 - aP90;
        return (b.totalProjection || 0) - (a.totalProjection || 0);
      });
    }

    // Select optimal subset
    const selected = candidates.length > targetCount
      ? this.selectOptimalPortfolio(candidates, targetCount, contestMode)
      : candidates;

    // Compute portfolio-level metrics on the selected set
    const portfolioMetrics = this.analyzePortfolio(selected);

    // Volatility coverage
    if (allPlayers.length > 0) {
      portfolioMetrics.volatilityCoverage = this.getVolatilityCoverage(allPlayers);
    }

    const slateRegime = this.getLastSlateRegime();
    if (slateRegime) {
      portfolioMetrics.slateRegime = slateRegime;
    }

    return { selected, portfolioMetrics };
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
