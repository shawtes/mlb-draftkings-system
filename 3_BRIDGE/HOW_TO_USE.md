# Full Pipeline — Fetch → Train → Bridge → Optimize

## Quick Start

**One command does everything** (after data is fetched once):
```bash
python run_pipeline.py --skip-hpo
```

Or run each step separately:
```bash
# Step 1: Fetch historical data (one-time, interactive)
source scripts/.venv/bin/activate
python scripts/fangraphs_batters.py           # Pick option 2 for all data

# Step 2: Train model
python run_pipeline.py --train-only --skip-hpo

# Step 3: Drop DK file + bridge
#   Download DKEntries.csv from DraftKings → save to data/dk_drop/
python run_pipeline.py --bridge-only

# Step 4: Optimize
cd web_optimizer && npm run dev
#   Upload data/optimizer_ready/*.csv at http://localhost:3000
```

---

## Master Pipeline Commands

| Command | What It Does |
|---------|-------------|
| `python run_pipeline.py` | Train + bridge (uses existing data) |
| `python run_pipeline.py --fetch` | Fetch data first, then train + bridge |
| `python run_pipeline.py --train-only` | Just train, no bridge |
| `python run_pipeline.py --bridge-only` | Just bridge (already trained) |
| `python run_pipeline.py --skip-hpo` | Faster training (skip Optuna tuning) |
| `python run_pipeline.py --dk-file ~/Downloads/DKEntries.csv` | Use specific DK file |

---

## Step-by-Step Detail

### Step 1: Fetch Data from FanGraphs

Downloads historical MLB batter statistics (2005-2025) using Selenium.

```bash
source scripts/.venv/bin/activate
python scripts/fangraphs_batters.py
```

Menu options:
- **Option 2** — All data (2005-2025, ~3,800 days, ~8 hours)
- **Option 3** — Single season (e.g. 2024)
- **Option 4** — Year range (e.g. 2020-2025)

Output: `~/FangraphsData/merged_fangraphs_data.csv`

The pipeline auto-links this file into `data/` so training can find it.

**You only need to do this once.** Re-run periodically to add new games.

### Step 2: Train the ML Model

```bash
python run_pipeline.py --train-only --skip-hpo
```

This runs the full training pipeline:
1. Feature engineering (financial, probabilistic, copula — 100+ features)
2. Walk-forward cross-validation (5 folds, 7-day embargo)
3. SHAP feature selection (top 100 features)
4. Ensemble model (Ridge + Lasso + LightGBM + CatBoost → XGBoost meta)
5. Quantile regression (prediction intervals + ceiling/floor/stddev)

**Output directory:** `1_CORE_TRAINING/output/`

| File | Description |
|------|-------------|
| `final_predictions.csv` | Name, Date, Actual, Predicted |
| `final_predictions_with_probabilities.csv` | + quantile bounds, prob_over thresholds |
| `batters_final_ensemble_model_pipeline.pkl` | Serialized model pipeline |
| `quantile_models.pkl` | Quantile regression models + CQR adjustment |
| `oos_validation_results.csv` | Per-fold MAE, RMSE, R2 |
| `player_evaluation.csv` | Per-player accuracy breakdown |
| `feature_importances.csv` | SHAP-ranked feature importance |

Training flags:
```
--skip-hpo              Skip Optuna (faster, uses pre-tuned params)
--optuna-trials 50      Number of HPO trials (default: 50)
--n-splits 5            CV folds (default: 5)
--n-features 100        SHAP feature count (default: 100)
```

### Step 3: Download DK File + Run Bridge

1. Go to DraftKings, enter a contest
2. Click **Export to CSV** → downloads `DKEntries.csv`
3. Drop it into `data/dk_drop/`
4. Run:

```bash
python run_pipeline.py --bridge-only
```

The bridge script:
- Auto-finds the latest CSV in `data/dk_drop/` (or your Downloads folder)
- Parses the DK format (DKEntries or DKSalaries — auto-detected)
- Merges with training predictions on player name (fuzzy matching)
- Uses DK positions, salary, team, opponent directly from the DK file
- Falls back to DK AvgPointsPerGame for any unmatched players
- Outputs: `data/optimizer_ready/YYYY-MM-DD_SPORT_optimizer_ready.csv`

**Output columns** (all from DK except projections):

| Column | Source | Description |
|--------|--------|-------------|
| Name | DraftKings | Player name |
| Pos | DraftKings | DK roster positions (e.g. PG/G/UTIL) |
| Team | DraftKings | Team abbreviation |
| Salary | DraftKings | DK salary |
| Predicted_DK_Points | Training model | ML projection (or DK avg fallback) |
| Opponent | DraftKings | Opposing team |
| DK_ID | DraftKings | DraftKings player ID |
| Ceiling | Training model | Upper 80% prediction bound |
| Floor | Training model | Lower 80% prediction bound |
| StdDev | Training model | Prediction uncertainty |

### Step 4: Optimize

```bash
cd web_optimizer && npm run dev
```

Then at `http://localhost:3000`:
1. Select your sport (MLB/NBA/NFL)
2. Click Upload Players → select the `*_optimizer_ready.csv` file
3. Configure stacks, exposures, quant settings
4. Click **BUILD LINEUPS**

---

## Folder Structure

```
mlb-draftkings-system/
├── run_pipeline.py              ← Master pipeline (this ties everything together)
├── scripts/
│   ├── fangraphs_batters.py     ← Step 1: FanGraphs data fetcher
│   └── .venv/                   ← Python venv for fetcher (Selenium)
├── data/
│   ├── dk_drop/                 ← Drop DK exports here
│   ├── optimizer_ready/         ← Bridge output goes here
│   └── merged_fangraphs_data.csv ← Symlink to ~/FangraphsData/
├── 1_CORE_TRAINING/
│   ├── training.py              ← Step 2: ML training pipeline
│   ├── config.py                ← CLI args, constants, feature lists
│   ├── feature_engine.py        ← Feature engineering (3 engines)
│   ├── model_builder.py         ← Ensemble, SHAP, Optuna, quantile models
│   ├── validator.py             ← Walk-forward CV, CQR calibration
│   └── output/                  ← Training artifacts (predictions, models)
├── 3_BRIDGE/
│   ├── dk_to_optimizer.py       ← Step 3: DK + predictions → optimizer CSV
│   └── HOW_TO_USE.md            ← This file
├── 6_OPTIMIZATION/              ← Optimizer engines (PuLP, Markov, genetic)
└── web_optimizer/               ← Step 4: Web app (React + Node.js)
```

---

## Supported DK File Formats

| Format | How You Get It |
|--------|----------------|
| **DKEntries.csv** | Contest lobby > Export Entries (has your lineup + embedded player pool) |
| **DKSalaries.csv** | Contest lobby > Export Player List (clean player-only CSV) |

Both formats are parsed automatically. The script also checks `~/Downloads/` if nothing is in `data/dk_drop/`.

---

## Name Matching

When merging ML predictions with DK data, names are normalized:

| DK Name | Training Data Name | Matched? |
|---------|--------------------|----------|
| Michael Porter Jr. | Michael Porter Jr | Yes |
| CJ McCollum | C.J. McCollum | Yes |
| Nickeil Alexander-Walker | Nickeil Alexander Walker | Yes |

For unmatched players, fallback order:
1. DK's `AvgPointsPerGame`
2. Salary-based estimate (sport-specific pts per $1K)

---

## CLI Reference

### run_pipeline.py
```
python run_pipeline.py [OPTIONS]

Pipeline control:
  --fetch               Fetch data from FanGraphs (interactive)
  --train-only          Only train, skip bridge
  --bridge-only         Only bridge, skip train

Training:
  --data-path PATH      Path to training CSV (default: auto-detect)
  --skip-hpo            Skip Optuna tuning (use pre-tuned params)
  --optuna-trials N     HPO trials (default: 50)
  --n-splits N          CV folds (default: 5)
  --n-features N        SHAP feature count (default: 100)

Bridge:
  --dk-file PATH        Path to DK CSV (default: latest in data/dk_drop/)
  --sport {MLB,NBA,NFL} Force sport (default: auto-detect)
```

### 3_BRIDGE/dk_to_optimizer.py
```
python 3_BRIDGE/dk_to_optimizer.py [OPTIONS]

  --dk-file PATH         DK CSV path (default: latest in data/dk_drop/)
  --predictions-dir DIR  Training output dir (default: 1_CORE_TRAINING/output)
  --output-dir DIR       Output dir (default: data/optimizer_ready/)
  --sport {MLB,NBA,NFL}  Force sport
  --no-predictions       Skip ML predictions, use DK averages only
```
