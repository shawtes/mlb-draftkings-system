# Matchup Engine — Status & Resume Guide

**Last updated**: 2026-02-18
**Status**: Phase 1 COMPLETE (code built + tested), not yet trained on full dataset

---

## What Was Built

### New File: `1_CORE_TRAINING/matchup_engine.py`

A pitcher-batter matchup feature engine that adds opposing pitcher context to batter DK fantasy point predictions. It produces **108 new features** per batter row:

- **100 `opp_*` features** — Opposing pitcher's lagged rolling stats (K/9, BB/9, ERA, FIP, xFIP, WHIP, K%, BB%, GB%, FB%, SwStr%, Contact%, etc.) across 7/14/28-game windows + season-to-date averages
- **8 `matchup_*` interaction features** — Log5 K rate, power vs flyball, contact advantage, discipline edge, K differential, ground ball risk, quality score, SwStr suppression

All pitcher stats are **shifted by 1 game** (leakage-free — only uses data before the current game).

### How It Works (Pipeline)

```
1. Load pitcher CSV (FanGraphs pitcher game logs)
2. Build opponent lookup: date + team → opposing starter
   - Primary: pybaseball day-by-day fetch (cached per date)
   - Fallback: pitcher CSV pairing for 2-team dates
3. Compute pitcher rolling profiles (lagged 7/14/28-game + season averages)
4. Merge opposing pitcher profile onto each batter row
5. Compute interaction features (batter stat × pitcher stat)
```

### Files Modified

| File | What Changed |
|------|-------------|
| `1_CORE_TRAINING/matchup_engine.py` | **NEW** — Full engine (~600 lines) |
| `1_CORE_TRAINING/config.py` | Added `--matchup`, `--pitcher-csv`, `--matchup-cache-dir` CLI flags + `MATCHUP_ALL_FEATURES` constant (128 feature names) |
| `1_CORE_TRAINING/training.py` | Calls `prepare_matchup_dataset()` after feature engineering when `--matchup` enabled; extends numeric features with opp_*/matchup_* |
| `1_CORE_TRAINING/run_pregame.py` | Same matchup integration for the fast pregame path |

### Test Results (Apr 1-2, 2025)

- **77% batter-to-starter match rate** (303/393 rows)
- **68% with full pitcher profile data** (pitcher had enough history for rolling averages)
- Matchups verified correct: Altuve vs Webb, Machado vs Allen, Santana vs King, etc.

---

## How to Run

### Full training pipeline with matchup features
```bash
python 1_CORE_TRAINING/training.py \
    --data-path /Users/sineshawmesfintesfaye/FangraphsData/merged_fangraphs_data.csv \
    --matchup \
    --pitcher-csv /Users/sineshawmesfintesfaye/FangraphsData/merged_fangraphs_data_pitchers.csv \
    --skip-hpo --n-splits 3
```

### Fast pregame path (skips 3-engine feature engineering)
```bash
python 1_CORE_TRAINING/run_pregame.py \
    --matchup \
    --pitcher-csv /Users/sineshawmesfintesfaye/FangraphsData/merged_fangraphs_data_pitchers.csv \
    --skip-hpo
```

### Standalone matchup engine test
```bash
python 1_CORE_TRAINING/matchup_engine.py \
    --batter-csv /Users/sineshawmesfintesfaye/FangraphsData/merged_fangraphs_data.csv \
    --pitcher-csv /Users/sineshawmesfintesfaye/FangraphsData/merged_fangraphs_data_pitchers.csv \
    --cache-dir ./data/matchup_cache \
    --start-date 2025-04-01 --end-date 2025-04-10
```

**Notes:**
- First run fetches pybaseball data (~10s per date, cached in `data/matchup_cache/`)
- `--pitcher-csv` auto-detects if file is next to the batter CSV
- `--matchup` flag defaults to OFF — existing pipeline behavior unchanged without it

---

## What's Left To Do (Next Session)

### Immediate: Run Full Training
1. Run the full pipeline with `--matchup` and compare OOS MAE/R² to the baseline (without matchup)
2. Expected improvement: MAE drop of 0.5-1.2 points, R² lift from ~0.017 to ~0.04-0.08

### Phase 2: Extend Historical Data
- Current pitcher CSV covers 2025 only (341 dates)
- Batter data goes back to 2005
- Use pybaseball to scrape 2023-2025 pitcher data: `pitching_stats_range()` day-by-day
- This would give matchup features for 3 seasons instead of 1

### Phase 3: Platoon Splits (Handedness)
- Neither CSV has L/R handedness columns
- pybaseball has `playerid_lookup()` with handedness
- Add: `is_platoon_advantage` (lefty batter vs righty pitcher = advantage)
- Expected R² lift: additional ~0.01-0.02

### Phase 4: Pitcher Arsenal Features
- Current engine already computes FBv, SL%, CH%, CB% rolling averages
- Next: add pitch-type-specific interaction features (e.g., fastball-heavy pitcher vs batter K% on fastballs)

---

## Key Design Decisions

1. **Day-by-day pybaseball fetch** — Multi-day `pitching_stats_range()` aggregates and drops Date/Opp columns. Single-day calls return per-game data with opponent info. Each day is cached individually.

2. **Team name mapping** — pybaseball uses full names ("San Diego"), FanGraphs uses abbreviations ("SDP"). `TEAM_NAME_TO_ABBREV` dict maps 30+ variants.

3. **Merge key** — Batter's Team matches to `opponent_team` in the lookup (the team the pitcher FACED), not `pitcher_team`. Deduped by both `(date, pitcher_team)` and `(date, opponent_team)` to prevent 1-to-many joins.

4. **Feature naming** — All pitcher features prefixed with `opp_` (e.g., `opp_K/9_14g`, `opp_ERA_season`). All interaction features prefixed with `matchup_` (e.g., `matchup_K_rate`). This makes them easy to filter and won't collide with batter features.

5. **Leakage prevention** — Every pitcher rolling stat uses `.shift(1)` within the pitcher groupby — the current game's stats are never included. Walk-forward CV in the training pipeline provides additional temporal separation.

---

## Other Recent Fixes (Same Session)

### Training Pipeline valid_mask Fix
- **File**: `training.py` line 885
- **Before**: `valid_mask = target.notna() & features.notna().all(axis=1)` — dropped any row where ANY feature was NaN (too strict, defeated SimpleImputer)
- **After**: `valid_mask = target.notna()` — only requires target; pregame mode additionally requires at least SOME pgm_/lag_ data via `.any(axis=1)`
- Added fit-state assertions for preprocessor and selector

### RL Hyperopt Import Fix
- **File**: `6_OPTIMIZATION/rl_hyperopt/run.py`
- Removed redundant relative imports inside functions that were already imported at the top level
- Q-Learning won the 3-agent comparison with mean reward 12.41
