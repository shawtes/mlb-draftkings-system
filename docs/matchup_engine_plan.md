# Pitcher-Batter Matchup Engine — Implementation Plan

## Executive Summary

Add opposing pitcher context to the batter DK fantasy point prediction model. The opposing starter is the single largest external factor in single-game hitter outcomes — more than park or weather in most cases. This is the highest-ROI feature category we can add to the current pregame model.

**Current baseline (pregame model, no matchup context):**
- OOS MAE: ~5.75 DK points
- OOS R²: ~0.017

**Realistic target after matchup features:**
- OOS MAE: ~4.8–5.3 DK points (0.5–1.2 point reduction)
- OOS R²: ~0.04–0.08 (2x–5x lift)

---

## Research Foundation

### Academic & Industry Evidence

| Source | Key Finding |
|--------|-------------|
| **Bayesian Hierarchical Log5** (PMC 6192592, PLOS One) | Bayesian matchup model reduced prediction MSE by 40–45% vs standard log5 by combining pitcher+batter OBP with hierarchical priors |
| **Hierarchical PBR Model** (arXiv 2511.17733, 2025) | 4-level model (Pitcher→+Batter→+Recency→+BaseRunning) predicting PA outcomes; "an MLB team could gain roughly one additional win simply by improving matchup projections by a fraction of a percent" |
| **FanGraphs Matchup K%** (blogs.fangraphs.com) | Formula: `Expected K% = B×P / (0.84×B×P + 0.16)` where B=batter K%, P=pitcher K%. Pitcher and batter "contribute equally to the outcome." Validated on 1.5M PA (2002–2012) |
| **The Outcome Machine** (FanGraphs Community) | Log-linear regression combining 8 batter + 8 pitcher rate stats by handedness: K%, BB%, 1B%, 2B%, 3B%, HR%, HBP%, BABIP. R²=0.989 for K%, R²=0.848 for HR%, R²=0.966 for BABIP |
| **Pitcher Clustering** (FanGraphs Community) | K-means on pitch mix + velocity yields 8 pitcher archetypes. Batter wOBA varies meaningfully across clusters — some hitters crush cutters but struggle vs sinkers |
| **FanGraphs Stabilization Rates** (library.fangraphs.com) | Batter K%: 60 PA. Pitcher K%: 70 BF. Batter ISO: 160 AB. Pitcher HR rate: 1,320 BF. BABIP: 820 BIP (batter), 2,000 BIP (pitcher) |
| **Stuff+ / Location+ / Pitching+** (FanGraphs) | Pitch-level quality metrics; Stuff+ stabilizes in ~100 pitches; models trained on pitch characteristics (velocity, movement, spin) to predict run value |
| **Sloan Sports Conference** (2024) | "Leveraging Batter-Pitcher Matchups for Optimal Game Strategy" — matchup-aware lineup optimization outperforms aggregate-stat approaches |

### Why Matchup Features Help

1. **Suppression signal**: A contact-oriented hitter (low K%, high BABIP) facing a high-K pitcher gets appropriately suppressed — the model learns to lower the projection
2. **Ceiling amplification**: Power hitter (high ISO) vs fly-ball pitcher with high HR/FB → model learns to raise ceiling
3. **Platoon advantage**: L/R splits produce ~8–12% wOBA differential on average — a binary feature that meaningfully stratifies outcomes
4. **Pitch mix interaction**: A fastball-heavy pitcher vs a batter who crushes fastballs (high wFB/C) → specific vulnerability signal beyond aggregate stats

### Why NOT Direct BvP (Batter vs Pitcher History)

Small-sample BvP stats (e.g., "3-for-8 with 2 HR against this pitcher") are **noise, not signal**. Per FanGraphs stabilization research:
- Batting average requires 910 AB to stabilize
- ISO requires 160 AB
- Most BvP matchups have <20 PA lifetime

**Our approach: Use pitcher PROFILE stats (K%, GB%, SwStr%, pitch mix, velocity) against batter TENDENCY stats (K%, ISO, Contact%, wOBA vs handedness) — not direct historical BvP.**

---

## Data Architecture

### Available Data Sources

| Source | What It Provides | Coverage |
|--------|-----------------|----------|
| **Existing batter CSV** (`merged_fangraphs_data.csv`) | 201K rows, 202 cols, per-game FanGraphs stats for batters | 2005–2025, ~1038 dates |
| **Existing pitcher CSV** (`merged_fangraphs_data_pitchers.csv`) | 2,458 rows, 202 cols, per-game FanGraphs stats for pitchers | 2025 only (31 dates, 486 pitchers) |
| **pybaseball `batting_stats_range()`** | Per-batter per-game stats with `Opp` team + `mlbID` (31 cols) | Any date range, Baseball Reference |
| **pybaseball `pitching_stats_range()`** | Per-pitcher per-game stats with `GS` flag + `Opp` + `mlbID` (45 cols) | Any date range, Baseball Reference |
| **pybaseball `schedule_and_record()`** | Full schedule: Date + Tm + Opp + Home/Away + Win/Loss pitcher | Any year, per team |
| **FanGraphs scraper** (`scripts/fangraphs_pitchers.py`) | Our existing FanGraphs scraper for pitcher data | Customizable |

### Critical Gap

| Dataset | Date Range | Problem |
|---------|-----------|---------|
| Batter FanGraphs CSV | 2005–2025 (201K rows) | No `Opp` column, no opposing pitcher info |
| Pitcher FanGraphs CSV | 2025 only (2,458 rows, 31 dates) | Too small for historical training |
| pybaseball game logs | Any range | Has `Opp` column but only Baseball Reference stats (no FanGraphs advanced) |

### Resolution Strategy

**Phase 1 (Immediate — pybaseball bridge):**
Build a date+team→opponent lookup from `pitching_stats_range()` (GS=1 rows give us which starter faced which team on which date). Join this to our FanGraphs batter data to identify the opposing starter per batter-game row. Then join that starter's lagged FanGraphs stats from our pitcher CSV.

**Phase 2 (Scale — extend pitcher data):**
Extend `scripts/fangraphs_pitchers.py` to scrape historical pitcher data (2023–2025) to get FanGraphs-level pitcher stats for the full batter training window. This gives us rich pitcher features (Stuff+, pitch values, advanced plate discipline) rather than just Baseball Reference basics.

**Phase 3 (Enrichment — handedness + Statcast):**
Add pitcher handedness (L/R throws) and batter handedness (L/R bats) via `pybaseball.playerid_lookup()` or FanGraphs player pages. Optionally add Statcast expected stats (xwOBA, xERA) via Baseball Savant.

---

## Matchup Feature Engineering

### Tier 1: Opposing Pitcher Profile (Highest Impact)

These are the opposing starter's **lagged rolling averages** (shifted by 1 game to avoid leakage), merged onto each batter-game row.

| Feature | Source Column | Rolling Window | Why It Matters |
|---------|--------------|----------------|----------------|
| `opp_K_pct` | Pitcher K% | 7, 14, 28 games | Primary suppression signal; K% stabilizes fast (70 BF) |
| `opp_BB_pct` | Pitcher BB% | 7, 14, 28 games | Free baserunners → more RBI opportunities for batters |
| `opp_HR_per_9` | Pitcher HR/9 | 14, 28 games | Direct HR-allowed signal; slow to stabilize (1,320 BF) so use longer windows |
| `opp_FIP` | Pitcher FIP | 14, 28 games | Fielding-independent pitching; better true talent than ERA |
| `opp_xFIP` | Pitcher xFIP | 14, 28 games | Normalizes HR/FB to league average; most stable predictor |
| `opp_WHIP` | Pitcher WHIP | 7, 14 games | Baserunner density → scoring context |
| `opp_GB_pct` | Pitcher GB% | 14 games | Ground-ball pitchers suppress HR; stabilizes at 70 BIP |
| `opp_SwStr_pct` | Pitcher SwStr% | 7, 14 games | Swinging-strike rate; best proxy for "stuff quality" |
| `opp_Contact_pct` | Pitcher Contact% | 7, 14 games | Inverse of whiff tendency |
| `opp_ERA_minus` | Pitcher ERA- | 28 games | Park/league-adjusted ERA (100 = average) |

**Feature naming convention:** `opp_{stat}_{window}g` (e.g., `opp_K_pct_14g`)

### Tier 2: Matchup Interaction Features (Medium Impact)

Computed from the combination of batter tendencies and opposing pitcher profile.

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `matchup_K_rate` | `batter_K% × opp_K% / (0.84 × batter_K% × opp_K% + 0.16)` | FanGraphs log5 expected K rate for this specific matchup |
| `matchup_power_vs_flyball` | `batter_ISO × opp_FB%` | Power hitter vs fly-ball pitcher = HR upside |
| `matchup_contact_advantage` | `batter_Contact% - opp_Contact%` | Positive = batter makes more contact than pitcher allows |
| `matchup_discipline_edge` | `batter_BB% - opp_BB%` | Positive = batter walks more than pitcher walks people |
| `matchup_K_differential` | `opp_K% - batter_K%` | Positive = pitcher strikes out more than batter K's; suppression signal |
| `matchup_ground_ball_risk` | `batter_GB% × opp_GB%` | Both ground-ball-prone = low ceiling |
| `matchup_quality_score` | `(1 / opp_xFIP) × batter_wOBA` | Composite: good hitter vs bad pitcher = high score |

### Tier 3: Platoon & Handedness (Medium Impact, Requires Enrichment)

| Feature | Formula | Expected Effect |
|---------|---------|----------------|
| `platoon_advantage` | Binary: 1 if batter bats opposite of pitcher throws | +8–12% wOBA on average; +0.01–0.03 R² |
| `opp_K_pct_vs_hand` | Pitcher's K% against batter's specific handedness | Handedness-specific suppression (stronger signal than overall K%) |
| `opp_wOBA_allowed_vs_hand` | Pitcher's wOBA allowed vs L or R batters | Direct outcome predictor split by handedness |
| `batter_wOBA_vs_hand` | Batter's wOBA vs L or R pitchers | Batter's specific platoon split performance |

### Tier 4: Pitcher Arsenal Features (Lower Impact, Data Permitting)

| Feature | Source | Why |
|---------|--------|-----|
| `opp_FB_velocity` | Pitcher FBv | Velocity is the single best pitch-level predictor of K% |
| `opp_FB_pct` | Pitcher FB% | Fastball-heavy = different batter approach than breaking-ball-heavy |
| `opp_SL_pct` | Pitcher SL% | Slider usage; relevant for same-side matchups |
| `opp_CH_pct` | Pitcher CH% | Changeup usage; relevant for opposite-side matchups |
| `opp_wFB_per_C` | Pitcher wFB/C | Fastball run value per 100 pitches; quality metric |

### Feature Engineering Rules (Leakage Prevention)

1. **All pitcher features must be LAGGED by 1 game** — use the pitcher's stats from games BEFORE the current matchup, never including the current game
2. **Rolling windows use `.shift(1)` within pitcher groupby** — same pattern as `prepare_pregame_features()` in our existing pipeline
3. **For the first game of season**, pitcher features are NaN → let `SimpleImputer(strategy='median')` handle it
4. **Cross-validation must be temporal** — walk-forward splits ensure no future pitcher data leaks into past predictions
5. **Never use same-game pitcher stats as features** — e.g., don't use "pitcher threw 6 IP in this game" to predict batter outcome in the same game
6. **Pitcher rolling averages computed ONLY on games before the current date** — not a centered window

---

## Data Pipeline Architecture

### Step 1: Build Opponent Lookup Table

```
For each date in training data:
  1. Pull pitching_stats_range(date, date) from pybaseball
  2. Filter to GS=1 (starters only)
  3. Extract: (date, pitcher_team, pitcher_name, pitcher_mlbID, opponent_team)
  4. This gives us: "On 2025-04-01, SFG starter Logan Webb faced HOU"

Result: opponent_lookup.csv
  date | pitcher_name | pitcher_mlbID | pitcher_team | opponent_team
```

### Step 2: Merge Opposing Starter onto Batter Rows

```
For each batter-game row in merged_fangraphs_data.csv:
  1. Match by (date, batter_team == opponent_team in lookup)
     - Batter is on Team A → find starter from Team B who faced Team A on that date
  2. Attach opposing_pitcher_name and opposing_pitcher_mlbID

Result: batter_with_opponent.csv
  [all existing batter columns] + opposing_pitcher_name + opposing_pitcher_mlbID
```

### Step 3: Build Pitcher Profile Table

```
From merged_fangraphs_data_pitchers.csv (or extended historical scrape):
  1. For each pitcher, sort by date
  2. Compute lagged rolling averages (shifted by 1 game):
     - opp_K_pct_7g, opp_K_pct_14g, opp_K_pct_28g
     - opp_FIP_14g, opp_xFIP_28g
     - opp_SwStr_pct_7g, opp_GB_pct_14g
     - ... (all Tier 1 features)
  3. Key by (pitcher_name or mlbID, date)

Result: pitcher_profiles.csv
  pitcher_name | pitcher_mlbID | date | opp_K_pct_7g | opp_FIP_14g | ...
```

### Step 4: Final Merge

```
batter_with_opponent.csv
  LEFT JOIN pitcher_profiles.csv
  ON (opposing_pitcher_mlbID, date)

Result: full_matchup_dataset.csv
  [all batter features] + [all opp_ pitcher features] + [interaction features]
```

### Step 5: Compute Interaction Features

```
After merge, compute Tier 2 interaction features:
  matchup_K_rate = f(batter_K%, opp_K_pct_14g)
  matchup_power_vs_flyball = batter_ISO * opp_FB_pct_14g
  matchup_contact_advantage = batter_Contact% - opp_Contact_pct_14g
  ... etc.
```

---

## Implementation Plan

### Module: `1_CORE_TRAINING/matchup_engine.py`

New standalone module with the following functions:

```python
# --- Data Acquisition ---
def build_opponent_lookup(start_date, end_date, cache_dir):
    """Use pybaseball to build date+team→opposing_starter lookup table.
    Caches results to avoid repeated scraping."""

def extend_pitcher_history(start_date, end_date, cache_dir):
    """Scrape pitcher game logs from Baseball Reference via pybaseball.
    Returns DataFrame with per-game pitcher stats + opponent."""

# --- Feature Engineering ---
def compute_pitcher_rolling_profiles(pitcher_df, windows=[7, 14, 28]):
    """For each pitcher, compute lagged rolling averages of key stats.
    All features shifted by 1 game to prevent leakage.
    Returns pitcher_profiles keyed by (pitcher_id, date)."""

def merge_matchup_features(batter_df, pitcher_profiles, opponent_lookup):
    """Join opposing pitcher's lagged profile onto each batter-game row.
    Returns batter_df with opp_* columns added."""

def compute_interaction_features(df):
    """Compute Tier 2 matchup interaction features from
    batter stats + opposing pitcher profile.
    Returns df with matchup_* columns added."""

# --- Integration ---
def prepare_matchup_dataset(batter_csv, pitcher_csv, cache_dir,
                            start_date=None, end_date=None):
    """Full pipeline: build lookup → compute pitcher profiles →
    merge onto batters → compute interactions.
    Returns enriched DataFrame ready for training."""
```

### Integration with Existing Pipeline

**In `1_CORE_TRAINING/training.py`:**
- Add `--matchup` CLI flag
- When enabled, call `prepare_matchup_dataset()` before feature engineering
- Add `opp_*` and `matchup_*` columns to `PREGAME_NUMERIC_FEATURES` list in `config.py`
- These flow through the existing pipeline: imputer → scaler → SHAP selection → walk-forward CV

**In `1_CORE_TRAINING/config.py`:**
- Add `MATCHUP_PITCHER_FEATURES` list (Tier 1 feature names)
- Add `MATCHUP_INTERACTION_FEATURES` list (Tier 2 feature names)
- Add `MATCHUP_ROLLING_WINDOWS = [7, 14, 28]`
- Add to `PREGAME_NUMERIC_FEATURES` when matchup mode is enabled

**In `1_CORE_TRAINING/feature_engine.py`:**
- No changes needed — matchup features are computed in the new module before feature engineering runs

### File Structure After Implementation

```
1_CORE_TRAINING/
├── config.py              # + MATCHUP_* constants
├── feature_engine.py      # Unchanged
├── matchup_engine.py      # NEW — all matchup logic
├── model_builder.py       # Unchanged
├── validator.py           # Unchanged
├── training.py            # + --matchup flag, calls matchup_engine
└── run_pregame.py         # + --matchup flag support

data/
└── matchup_cache/         # NEW — cached pybaseball scrapes
    ├── opponent_lookup.csv
    ├── pitcher_game_logs/
    │   ├── 2023.csv
    │   ├── 2024.csv
    │   └── 2025.csv
    └── pitcher_profiles.csv
```

---

## Build Phases

### Phase 1: Core Matchup Engine (1–2 hours)

1. Create `matchup_engine.py` with opponent lookup builder
2. Build pitcher rolling profile computation (Tier 1 features)
3. Merge pipeline: batter → opponent lookup → pitcher profiles
4. Compute interaction features (Tier 2)
5. Add `--matchup` flag to `training.py` and `config.py`
6. Test on 2025 data (31 overlapping dates, ~6K batter rows)

### Phase 2: Historical Data Extension (30 min)

1. Extend pitcher scraping to 2023–2025 via pybaseball
2. Cache all pitcher game logs to `data/matchup_cache/`
3. Re-run matchup engine on full batter dataset (2023–2025 subset)
4. Validate: walk-forward CV with matchup features vs without

### Phase 3: Handedness / Platoon Splits (1 hour)

1. Scrape pitcher handedness (L/R throws) via pybaseball playerid tables
2. Scrape batter handedness (L/R bats) similarly
3. Add `platoon_advantage` binary feature
4. Add handedness-split pitcher features (K% vs LHB, wOBA allowed vs RHB)
5. Re-run validation

### Phase 4: Pitcher Model (Bonus — Parallel Track)

1. Apply same matchup logic in reverse: for pitcher DK point prediction, add opposing TEAM batting profile features (team wOBA, team K%, team ISO)
2. Compute team-level rolling averages from batter data
3. Train pitcher prediction model with lineup-quality context

---

## Validation Protocol

### A/B Comparison

Run walk-forward CV twice on the same data:
1. **Control**: Current pregame model (no matchup features)
2. **Treatment**: Pregame model + matchup features

Compare:
- OOS MAE per fold
- OOS R² per fold
- Per-player MAE breakdown (expect biggest improvement on boom/bust hitters)
- Feature importance (SHAP) — verify opp_* features get meaningful weight

### Leakage Guardrails

- If R² jumps above 0.15 → suspect leakage; audit pitcher feature lag
- Verify no `opp_*` feature has >0.3 correlation with target in-sample (would suggest same-game data leaked)
- Check that first-game-per-pitcher rows have NaN for rolling features (confirms shift)

### Expected Results by Phase

| Phase | Expected MAE | Expected R² | Key Signal |
|-------|-------------|-------------|------------|
| Baseline (no matchup) | ~5.75 | ~0.017 | Player history only |
| + Tier 1 (pitcher profile) | ~5.3–5.5 | ~0.03–0.05 | opp_K%, opp_FIP, opp_SwStr% dominate |
| + Tier 2 (interactions) | ~5.1–5.3 | ~0.04–0.06 | matchup_K_rate, matchup_power_vs_flyball |
| + Tier 3 (platoon) | ~4.9–5.2 | ~0.05–0.08 | platoon_advantage adds ~0.01–0.03 R² |
| + Tier 4 (arsenal) | ~4.8–5.1 | ~0.06–0.09 | Diminishing returns; opp_FB_velocity helps |

### Where the Improvement Shows Most

| Player Type | Expected MAE Improvement | Why |
|-------------|------------------------|-----|
| **Boom/bust sluggers** (Judge, Buxton, De La Cruz) | 1.0–1.5 DK | High variance is matchup-driven; K-prone hitters vs high-K pitchers get properly suppressed |
| **Contact-first hitters** (Arraez, McNeil) | 0.3–0.5 DK | Already consistent; less matchup-sensitive |
| **Platoon-dependent hitters** (vs specific handedness) | 0.8–1.2 DK | Platoon split captures large performance gap |
| **Value plays vs weak pitchers** | 0.5–0.8 DK | Cheap hitters facing high-ERA/FIP starters get boosted |

---

## Dependencies

```
pybaseball>=2.2        # Already installed — opponent lookup, game logs
pandas                 # Already installed
numpy                  # Already installed
```

No new dependencies required. Everything uses pybaseball (installed) + existing ML stack.

---

## References

1. Choi & Hamdan (2018). "Modeling the probability of a batter/pitcher matchup event: A Bayesian approach." PLOS One. PMC 6192592.
2. Brill et al. (2025). "Leveraging Hierarchical Bayesian Matchup Models." arXiv:2511.17733.
3. Sullivan (2013). "Batter-Pitcher Matchups: Expected Matchup K%." FanGraphs.
4. Appelman (2012). "Batter Performance vs. Pitcher Clusters." FanGraphs Community.
5. Carleton (2015). "The Outcome Machine: Predicting At-Bats Before They Happen." FanGraphs Community.
6. Russell (2016). "Adjusting Batter Performance by the Quality of the Opposing Pitcher." FanGraphs Community.
7. Turkenkopf. "Stuff+, Location+, and Pitching+ Primer." FanGraphs Sabermetrics Library.
8. FanGraphs. "Principles of Sample Size / Stabilization." library.fangraphs.com/principles/sample-size/
9. Lichtman (2013). "How Teams Value Platoon Splits." The Hardball Times.
