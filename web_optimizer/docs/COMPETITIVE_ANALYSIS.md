# DFS Optimizer Competitive Analysis & Feature Roadmap

## Research Date: 2026-02-10

## Market Leaders Analyzed
- SaberSim, Stokastic, FantasyCruncher, LineupHQ (RotoGrinders), FantasyLabs, DraftDime

---

## TIER 1: Table Stakes (Must-Have)

| Feature | SaberSim | Stokastic | FantasyCruncher | Us |
|---------|----------|-----------|-----------------|-----|
| Ownership Projections | Yes | Yes | Yes | Partial |
| FanDuel Support | Yes | Yes | Yes | No |
| DK + FD CSV Export | Yes | Yes | Yes | DK only |
| Player Lock/Exclude | Yes | Yes | Yes | Yes |
| Exposure Controls (per-player) | Yes | Yes | Yes | Yes |
| Lineup Uniqueness | Yes | Yes | Yes | Yes |
| Showdown/Captain Mode | Yes | Yes | Yes | Partial |

### Priority:
1. Ownership projections (uploadable + leverage score)
2. FanDuel support + dual-site CSV export
3. Showdown/Captain mode

---

## TIER 2: Competitive Differentiators

| Feature | SaberSim | Stokastic | LineupHQ | Us |
|---------|----------|-----------|----------|-----|
| Contest/Slate Simulation | Yes (core) | Yes (core) | No | Framework only |
| Boom/Bust Tool | Via sims | Yes (standalone) | No | Fields exist, no UI |
| Player Groups / Conditional Rules | Yes | Yes | Yes (premium) | No |
| Bring-Back / Opponent Correlation | Yes | Yes | Yes | Partial (NFL) |
| Portfolio Diversifier | Yes (major) | Yes | No | Basic |
| Leverage Score (proj vs ownership) | Yes | Yes | No | No |
| Projection Blending (merge sources) | No | No | Yes | No |
| Late Swap | Yes (premium) | No | No | No |

### Priority:
1. Player Groups & Conditional Rules
2. Leverage Score
3. Boom/Bust UI
4. Contest Simulation
5. Projection Blending
6. Portfolio Diversifier

---

## TIER 3: Premium Features ($30-100/mo)

| Feature | Who Has It | Us |
|---------|-----------|-----|
| Game-Level Simulation (play-by-play) | SaberSim | No |
| Contest Field Simulation | SaberSim, Stokastic | No |
| Real-Time Alerts (injuries, news) | SaberSim | WebSocket infra only |
| ROI / Results Tracking | RotoTracker | No |
| Bankroll Tracker + Kelly Sizing | The Solver | Designed, not functional |
| Contest Selection Advisor | None | No |
| Payout Structure Analysis | SaberSim | No |
| Swap Pools (auto late-swap) | SaberSim | No |

---

## TIER 4: Emerging Opportunities

| Feature | Market Status |
|---------|--------------|
| AI-powered projection adjustments | Emerging |
| Historical slate analysis | Minimal |
| Contest Selection Matrix | Very basic |
| Multi-entry lineup assignment | Only DraftDime |
| Collaborative/social features | None |
| Mobile-first optimizer | Most desktop-focused |

---

## Pricing Context

| Platform | Price | Key Differentiator |
|----------|-------|--------------------|
| SaberSim | $7-50/mo | Game sims, portfolio diversifier, late swap |
| Stokastic | $30-100/mo/sport | Sims, ownership, boom/bust, projections |
| FantasyCruncher | $29-99/mo | Optimizer, groups, multi-source blending |
| LineupHQ | $25-50/mo | Groups, stacking, correlation tools |

---

## Implementation Roadmap

### Phase 1: Table Stakes (Weeks 1-3)
1. Ownership projection upload + display + leverage
2. FanDuel roster support
3. DK + FD CSV export
4. Showdown/Captain mode

### Phase 2: Core Differentiators (Weeks 4-8)
5. Player Groups & Conditional Rules
6. Boom/Bust UI (ceiling/floor/std-dev display)
7. Leverage Score
8. Projection Blending
9. Portfolio Diversifier

### Phase 3: Premium Features (Weeks 9-16)
10. Monte Carlo Contest Simulator
11. Results Tracking (import DK/FD results)
12. Real-time player alerts
13. Late Swap Engine
14. Bankroll Management Dashboard

---

## Current Strengths
- Solid core optimization (PuLP-based, multiple algorithms)
- Excellent stacking (complex patterns with exposure control)
- Professional UI (React/Vite, dark theme)
- Multi-sport (MLB, NFL, NBA)
- Advanced design docs (Kelly Criterion, GARCH volatility)

## Biggest Gaps
1. No ownership projections / leverage scoring
2. No player groups / conditional rules
3. No FanDuel support
4. No boom/bust visualization
5. No contest simulation (framework only)

## Sources
- https://www.sabersim.com/how-it-works
- https://support.sabersim.com/en/articles/12079514-using-the-portfolio-diversifier-in-sabersim
- https://www.stokastic.com/news/nba-dfs-tools-2025-26-get-stokastic-sims-data-ac14
- https://www.stokastic.com/nfl/nfl-daily-fantasy-boom-bust-tool-draftkings-fanduel-projections
- https://windailysports.com/reviews/fantasycruncher/
- https://rotogrinders.com/lineuphq
- https://rototracker.com/
- https://dailyroto.com/using-kelly-criterion-for-dfs-bankroll-management/
