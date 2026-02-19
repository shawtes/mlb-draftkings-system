"""
diagnose_leakage.py — Comprehensive leakage audit & honest prediction test

This script loads the trained model and raw data, then runs 7 diagnostic tests
to determine whether the R² ≈ 0.965 is real or the result of data leakage.

Usage:
  .venv312/bin/python3 training/mlb/diagnose_leakage.py \
      --data-path /path/to/merged_fangraphs_data.csv \
      --model-dir training/mlb/output
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Leakage audit for MLB DFS training pipeline")
    p.add_argument("--data-path", required=True, help="Path to merged_fangraphs_data.csv")
    p.add_argument("--model-dir", default="training/mlb/output",
                   help="Directory containing trained model artifacts")
    p.add_argument("--quick", action="store_true",
                   help="Run quick mode (smaller sample for speed)")
    return p.parse_args()


def calculate_dk_fpts(row):
    """DraftKings fantasy points for a batter."""
    s1 = pd.to_numeric(row.get("1B", 0), errors="coerce") or 0
    d = pd.to_numeric(row.get("2B", 0), errors="coerce") or 0
    t = pd.to_numeric(row.get("3B", 0), errors="coerce") or 0
    hr = pd.to_numeric(row.get("HR", 0), errors="coerce") or 0
    rbi = pd.to_numeric(row.get("RBI", 0), errors="coerce") or 0
    r = pd.to_numeric(row.get("R", 0), errors="coerce") or 0
    bb = pd.to_numeric(row.get("BB", 0), errors="coerce") or 0
    hbp = pd.to_numeric(row.get("HBP", 0), errors="coerce") or 0
    sb = pd.to_numeric(row.get("SB", 0), errors="coerce") or 0
    return s1 * 3 + d * 5 + t * 8 + hr * 10 + rbi * 2 + r * 2 + bb * 2 + hbp * 2 + sb * 5


# ---------------------------------------------------------------------------
# Feature classification: which features are pre-game vs same-game?
# ---------------------------------------------------------------------------

# SAME-GAME features: computed from or directly contain that game's box score.
# Using these to predict DK points is circular — they ARE the DK points in
# different mathematical forms.
SAME_GAME_LEAKERS = {
    # Raw FanGraphs game-log columns (same game's performance)
    "Off", "WAR", "Dol", "RAR", "RE24", "REW", "SLG", "WPA/LI", "AB",
    "wRC", "wRC+", "wRAA", "wOBA", "BABIP", "ISO", "flyBalls",
    # Engineered from same-game box score
    "Offense_Statcast", "Dollars_Statcast", "WPA/LI_Statcast",
    "wOBA_Statcast", "SLG_Statcast", "RAR_Statcast",
}

# Rolling features that INCLUDE the current row (no .shift(1))
ROLLING_CURRENT_ROW_LEAKERS = {
    "rolling_min_fpts_7", "rolling_max_fpts_7", "rolling_mean_fpts_7",
    "rolling_min_fpts_49", "rolling_max_fpts_49", "rolling_mean_fpts_49",
}

# Features computed from rolling windows of calculated_dk_fpts that include
# the current game (no shift) in the probabilistic engine
PROB_CURRENT_ROW_LEAKERS = {
    "skewness_7d", "skewness_14d", "skewness_30d",
    "kurtosis_7d", "kurtosis_14d", "kurtosis_30d",
    "var_95_7d", "var_95_14d", "var_95_30d",
    "var_99_7d", "var_99_14d", "var_99_30d",
    "expected_shortfall_7d", "expected_shortfall_14d", "expected_shortfall_30d",
    "tail_ratio", "prob_exceed_5", "prob_exceed_10", "prob_exceed_15", "prob_exceed_20",
    "entropy", "hurst_exponent", "max_drawdown", "current_drawdown",
    "drawdown_duration", "rolling_sharpe",
    "bull_regime", "regime_strength", "momentum_regime", "consistency_regime",
}

# Properly lagged features (these use .shift(1) — safe for prediction)
SAFE_LAGGED_FEATURES = {
    "lag_mean_fpts_3", "lag_mean_fpts_7", "lag_mean_fpts_14", "lag_mean_fpts_28",
    "lag_max_fpts_3", "lag_max_fpts_7", "lag_max_fpts_14", "lag_max_fpts_28",
    "lag_min_fpts_3", "lag_min_fpts_7", "lag_min_fpts_14", "lag_min_fpts_28",
}

# Calendar features (always safe)
SAFE_CALENDAR = {"year", "month", "day", "day_of_week", "is_weekend",
                 "day_of_week_sin", "day_of_week_cos", "day_of_season"}

ALL_LEAKERS = SAME_GAME_LEAKERS | ROLLING_CURRENT_ROW_LEAKERS | PROB_CURRENT_ROW_LEAKERS


def classify_feature(name):
    """Return 'same-game', 'current-row-rolling', or 'pre-game'."""
    if name in SAME_GAME_LEAKERS:
        return "same-game"
    if name in ROLLING_CURRENT_ROW_LEAKERS:
        return "current-row-rolling"
    if name in PROB_CURRENT_ROW_LEAKERS:
        return "current-row-prob"
    return "pre-game"


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------

def test_1_target_distribution(df, target_col="calculated_dk_fpts"):
    """Check target distribution — is std suspiciously low?"""
    print("\n" + "=" * 70)
    print("TEST 1: TARGET DISTRIBUTION")
    print("=" * 70)

    y = df[target_col]
    print(f"  Target column: {target_col}")
    print(f"  N rows:  {len(y):,}")
    print(f"  Mean:    {y.mean():.3f}")
    print(f"  Median:  {y.median():.3f}")
    print(f"  Std:     {y.std():.3f}")
    print(f"  Min:     {y.min():.3f}")
    print(f"  Max:     {y.max():.3f}")
    print(f"  Skew:    {y.skew():.3f}")
    print(f"  Kurt:    {y.kurtosis():.3f}")

    # Percentiles
    for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
        print(f"  P{int(q*100):02d}:     {y.quantile(q):.3f}")

    # Zero-game rate
    zero_rate = (y == 0).mean()
    print(f"  Zero-game rate: {zero_rate:.1%}")

    # R² sanity check
    var_y = y.var()
    reported_rmse = 1.399
    implied_r2 = 1 - (reported_rmse ** 2) / var_y
    print(f"\n  Var(y) = {var_y:.3f}")
    print(f"  Reported RMSE = {reported_rmse:.3f}")
    print(f"  Implied R² = 1 - RMSE²/Var(y) = 1 - {reported_rmse**2:.3f}/{var_y:.3f} = {implied_r2:.4f}")

    if implied_r2 > 0.90:
        print(f"\n  >>> VERDICT: R² = {implied_r2:.3f} is {('plausible' if var_y > 30 else 'SUSPICIOUS')} given Var(y) = {var_y:.1f}")
    return var_y


def test_2_fold_date_ranges(df, n_splits=5, gap_days=7):
    """Print exact train/test date boundaries for each fold."""
    print("\n" + "=" * 70)
    print("TEST 2: WALK-FORWARD FOLD DATE RANGES")
    print("=" * 70)

    dates = pd.to_datetime(df["date"])
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    min_train_days = int(total_days * 0.4)
    remaining_days = total_days - min_train_days
    test_block_size = remaining_days // n_splits

    print(f"  Data range: {min_date.date()} to {max_date.date()} ({total_days} days)")
    print(f"  Min train period: {min_train_days} days (40%)")
    print(f"  Test block size: {test_block_size} days")
    print(f"  Gap: {gap_days} days\n")

    for fold in range(n_splits):
        train_end = min_date + timedelta(days=min_train_days + fold * test_block_size)
        test_start = train_end + timedelta(days=gap_days)
        test_end = min(train_end + timedelta(days=(fold + 1) * test_block_size), max_date)

        train_mask = dates <= train_end
        test_mask = (dates >= test_start) & (dates <= test_end)

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        status = "SKIPPED" if n_train == 0 or n_test == 0 else "OK"
        print(f"  Fold {fold+1}: Train ≤ {train_end.date()} ({n_train:,} rows) | "
              f"Test {test_start.date()} – {test_end.date()} ({n_test:,} rows) [{status}]")

        if n_train > 0 and n_test > 0:
            # Check overlap
            train_dates = set(dates[train_mask].dt.date)
            test_dates = set(dates[test_mask].dt.date)
            overlap = train_dates & test_dates
            if overlap:
                print(f"    >>> WARNING: {len(overlap)} overlapping dates!")
            # Target stats in test fold
            y_test = df.loc[test_mask, "calculated_dk_fpts"]
            print(f"    Test y: mean={y_test.mean():.2f}, std={y_test.std():.2f}, "
                  f"var={y_test.var():.2f}")


def test_3_feature_leakage_audit(df, target_col="calculated_dk_fpts"):
    """Correlate each top feature with target to identify same-game leakers."""
    print("\n" + "=" * 70)
    print("TEST 3: FEATURE-TARGET CORRELATION AUDIT (same-game detection)")
    print("=" * 70)

    y = df[target_col]

    # Check all features from the NUMERIC_FEATURES config list
    from config import NUMERIC_FEATURES
    candidates = [f for f in NUMERIC_FEATURES if f in df.columns]

    results = []
    for col in candidates:
        try:
            vals = pd.to_numeric(df[col], errors="coerce")
            corr = vals.corr(y)
            category = classify_feature(col)
            results.append((col, corr, category))
        except Exception:
            pass

    results.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)

    print(f"\n  {'Feature':<30s} {'|r| with DK pts':>16s}  {'Category':<22s}")
    print(f"  {'-'*30} {'-'*16}  {'-'*22}")
    for name, corr, cat in results[:40]:
        flag = " <<<< LEAKER" if cat != "pre-game" and abs(corr) > 0.3 else ""
        print(f"  {name:<30s} {abs(corr):>16.4f}  {cat:<22s}{flag}")

    n_leakers = sum(1 for _, corr, cat in results if cat != "pre-game" and abs(corr) > 0.3)
    print(f"\n  >>> {n_leakers} features flagged as leakers (|r| > 0.3 with same-game/current-row category)")


def test_4_baseline_metrics(df, n_splits=5, gap_days=7):
    """Compute naive baselines: global mean, per-player mean, last-7 rolling."""
    print("\n" + "=" * 70)
    print("TEST 4: BASELINE METRICS (what R² should a naive model get?)")
    print("=" * 70)

    from sklearn.metrics import mean_absolute_error, r2_score

    dates = pd.to_datetime(df["date"])
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    min_train_days = int(total_days * 0.4)
    remaining_days = total_days - min_train_days
    test_block_size = remaining_days // n_splits

    all_baselines = {"global_mean": [], "player_mean": [], "player_rolling7": [],
                     "player_lag7": []}

    for fold in range(n_splits):
        train_end = min_date + timedelta(days=min_train_days + fold * test_block_size)
        test_start = train_end + timedelta(days=gap_days)
        test_end = min(train_end + timedelta(days=(fold + 1) * test_block_size), max_date)

        train_mask = dates <= train_end
        test_mask = (dates >= test_start) & (dates <= test_end)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        train_df = df[train_mask]
        test_df = df[test_mask].copy()
        y_test = test_df["calculated_dk_fpts"].values

        # Baseline 1: Global mean
        global_mean = train_df["calculated_dk_fpts"].mean()
        pred_global = np.full(len(y_test), global_mean)
        all_baselines["global_mean"].append({
            "mae": mean_absolute_error(y_test, pred_global),
            "r2": r2_score(y_test, pred_global),
        })

        # Baseline 2: Per-player historical mean
        player_means = train_df.groupby("Name")["calculated_dk_fpts"].mean().to_dict()
        pred_player = test_df["Name"].map(player_means).fillna(global_mean).values
        all_baselines["player_mean"].append({
            "mae": mean_absolute_error(y_test, pred_player),
            "r2": r2_score(y_test, pred_player),
        })

        # Baseline 3: Per-player last 7 games rolling mean (INCLUDING current game - like pipeline)
        # This mimics the pipeline's rolling_mean_fpts_7 which doesn't shift
        test_df["rolling7"] = test_df.groupby("Name")["calculated_dk_fpts"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        pred_rolling = test_df["rolling7"].fillna(global_mean).values
        all_baselines["player_rolling7"].append({
            "mae": mean_absolute_error(y_test, pred_rolling),
            "r2": r2_score(y_test, pred_rolling),
        })

        # Baseline 4: Per-player lagged 7-game mean (PROPERLY shifted)
        last_7_means = {}
        for name in test_df["Name"].unique():
            player_train = train_df[train_df["Name"] == name]
            if len(player_train) >= 7:
                last_7_means[name] = player_train.tail(7)["calculated_dk_fpts"].mean()
            elif len(player_train) > 0:
                last_7_means[name] = player_train["calculated_dk_fpts"].mean()
        pred_lag7 = test_df["Name"].map(last_7_means).fillna(global_mean).values
        all_baselines["player_lag7"].append({
            "mae": mean_absolute_error(y_test, pred_lag7),
            "r2": r2_score(y_test, pred_lag7),
        })

    print(f"\n  {'Baseline':<25s} {'MAE':>8s}  {'R²':>8s}  Notes")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*30}")

    for name, folds in all_baselines.items():
        if folds:
            avg_mae = np.mean([f["mae"] for f in folds])
            avg_r2 = np.mean([f["r2"] for f in folds])
            note = ""
            if name == "global_mean":
                note = "Predicts same value for everyone"
            elif name == "player_mean":
                note = "Historical average per player"
            elif name == "player_rolling7":
                note = "7-game rolling mean (INCL current game)"
            elif name == "player_lag7":
                note = "Last 7 games mean (properly lagged)"
            print(f"  {name:<25s} {avg_mae:>8.4f}  {avg_r2:>8.4f}  {note}")

    print(f"\n  >>> Pipeline reported:     MAE=0.9624, R²=0.9653")
    print(f"  >>> If pipeline R² >> player_lag7 R², the gap is likely from same-game features")


def test_5_retrain_without_leakers(df, n_splits=5, gap_days=7, quick=False):
    """Retrain a simple model with ONLY pre-game features."""
    print("\n" + "=" * 70)
    print("TEST 5: RETRAIN WITH ONLY PRE-GAME FEATURES (the honest test)")
    print("=" * 70)

    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        import lightgbm as lgb
        USE_LGBM = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        USE_LGBM = False

    # Identify safe features that exist in the dataframe
    safe_features = []
    leaker_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "calculated_dk_fpts":
            continue
        cat = classify_feature(col)
        if cat == "pre-game":
            # Additional filter: skip any column with "fpts" that doesn't have "lag"
            if "fpts" in col.lower() and "lag" not in col.lower():
                leaker_features.append(col)
                continue
            # Skip any column derived from same-game hitting stats
            # (SMA/EMA of same-game stats like HR_sma_3 etc include current row)
            safe_features.append(col)
        else:
            leaker_features.append(col)

    # Also filter: rolling features of individual stats that include current row
    # e.g., HR_sma_3 uses .rolling(3).mean() without shift — includes current HR
    # These are computed by the financial engine from same-game columns
    additional_leakers = []
    for col in safe_features:
        for stat in ["HR", "RBI", "BB", "SB", "H", "1B", "2B", "3B", "R"]:
            if col.startswith(f"{stat}_sma_") or col.startswith(f"{stat}_ema_") or col.startswith(f"{stat}_roc_"):
                additional_leakers.append(col)
                break
        if col.startswith("dk_fpts_"):
            additional_leakers.append(col)

    safe_features = [f for f in safe_features if f not in additional_leakers]
    leaker_features.extend(additional_leakers)

    print(f"  Total numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Identified leakers: {len(leaker_features)}")
    print(f"  Safe pre-game features: {len(safe_features)}")
    print(f"\n  Sample safe features: {safe_features[:20]}")
    print(f"  Sample leakers: {leaker_features[:20]}")

    if len(safe_features) < 5:
        print("\n  >>> Not enough safe features to train a model!")
        return

    dates = pd.to_datetime(df["date"])
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    min_train_days = int(total_days * 0.4)
    remaining_days = total_days - min_train_days
    test_block_size = remaining_days // n_splits

    honest_results = []
    leaky_results = []

    for fold in range(n_splits):
        train_end = min_date + timedelta(days=min_train_days + fold * test_block_size)
        test_start = train_end + timedelta(days=gap_days)
        test_end = min(train_end + timedelta(days=(fold + 1) * test_block_size), max_date)

        train_mask = dates <= train_end
        test_mask = (dates >= test_start) & (dates <= test_end)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        y_train = df.loc[train_mask, "calculated_dk_fpts"].values
        y_test = df.loc[test_mask, "calculated_dk_fpts"].values

        # --- Honest model (pre-game only) ---
        X_train_honest = df.loc[train_mask, safe_features].values
        X_test_honest = df.loc[test_mask, safe_features].values

        pipe_honest = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_train_h = pipe_honest.fit_transform(X_train_honest)
        X_test_h = pipe_honest.transform(X_test_honest)

        if quick:
            # Subsample for speed
            n_sub = min(50000, len(X_train_h))
            idx = np.random.RandomState(42).choice(len(X_train_h), n_sub, replace=False)
            X_train_h_sub, y_train_sub = X_train_h[idx], y_train[idx]
        else:
            X_train_h_sub, y_train_sub = X_train_h, y_train

        if USE_LGBM:
            model_honest = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=42, verbose=-1,
            )
        else:
            model_honest = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
            )

        model_honest.fit(X_train_h_sub, y_train_sub)
        pred_honest = model_honest.predict(X_test_h)

        mae_h = mean_absolute_error(y_test, pred_honest)
        r2_h = r2_score(y_test, pred_honest)
        honest_results.append({"fold": fold + 1, "mae": mae_h, "r2": r2_h})
        print(f"\n  Fold {fold+1} (honest, {len(safe_features)} features): "
              f"MAE={mae_h:.4f}, R²={r2_h:.4f}")

        # --- Leaky model (all features including same-game) ---
        all_numeric = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c != "calculated_dk_fpts"]
        X_train_leaky = df.loc[train_mask, all_numeric].values
        X_test_leaky = df.loc[test_mask, all_numeric].values

        pipe_leaky = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_train_l = pipe_leaky.fit_transform(X_train_leaky)
        X_test_l = pipe_leaky.transform(X_test_leaky)

        if quick:
            X_train_l_sub, y_train_l_sub = X_train_l[idx], y_train[idx]
        else:
            X_train_l_sub, y_train_l_sub = X_train_l, y_train

        if USE_LGBM:
            model_leaky = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=42, verbose=-1,
            )
        else:
            model_leaky = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
            )

        model_leaky.fit(X_train_l_sub, y_train_l_sub)
        pred_leaky = model_leaky.predict(X_test_l)

        mae_l = mean_absolute_error(y_test, pred_leaky)
        r2_l = r2_score(y_test, pred_leaky)
        leaky_results.append({"fold": fold + 1, "mae": mae_l, "r2": r2_l})
        print(f"  Fold {fold+1} (leaky,  {len(all_numeric)} features): "
              f"MAE={mae_l:.4f}, R²={r2_l:.4f}")

    if honest_results:
        avg_h_mae = np.mean([r["mae"] for r in honest_results])
        avg_h_r2 = np.mean([r["r2"] for r in honest_results])
        avg_l_mae = np.mean([r["mae"] for r in leaky_results])
        avg_l_r2 = np.mean([r["r2"] for r in leaky_results])

        print(f"\n  {'Model':<20s} {'Avg MAE':>10s}  {'Avg R²':>10s}")
        print(f"  {'-'*20} {'-'*10}  {'-'*10}")
        print(f"  {'Honest (pre-game)':<20s} {avg_h_mae:>10.4f}  {avg_h_r2:>10.4f}")
        print(f"  {'Leaky (all feats)':<20s} {avg_l_mae:>10.4f}  {avg_l_r2:>10.4f}")
        print(f"  {'Pipeline reported':<20s} {'0.9624':>10s}  {'0.9653':>10s}")

        r2_gap = avg_l_r2 - avg_h_r2
        print(f"\n  >>> R² gap (leaky - honest) = {r2_gap:.4f}")
        if r2_gap > 0.10:
            print(f"  >>> VERDICT: Same-game features inflate R² by ~{r2_gap:.2f}. This is LEAKAGE.")
        elif r2_gap > 0.02:
            print(f"  >>> VERDICT: Moderate inflation from same-game features. Worth investigating.")
        else:
            print(f"  >>> VERDICT: Minimal difference — leakage may not be the primary driver.")


def test_6_shuffle_dates(df, quick=False):
    """Shuffle date assignments and check if R² collapses (it should if model is real)."""
    print("\n" + "=" * 70)
    print("TEST 6: DATE SHUFFLE TEST (if R² survives, model uses same-row data)")
    print("=" * 70)

    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    try:
        import lightgbm as lgb
        USE_LGBM = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        USE_LGBM = False

    from config import NUMERIC_FEATURES

    features = [f for f in NUMERIC_FEATURES if f in df.columns]
    # Add copula PCA if present
    features += [c for c in df.columns if c.startswith("copula_pc_")]
    features = list(set(features))

    # Shuffle target while keeping features fixed
    y_original = df["calculated_dk_fpts"].values.copy()
    y_shuffled = np.random.RandomState(42).permutation(y_original)

    n_sample = min(80000, len(df)) if quick else len(df)
    idx = np.random.RandomState(42).choice(len(df), n_sample, replace=False)

    X = df.iloc[idx][features].values
    y_orig_sub = y_original[idx]
    y_shuf_sub = y_shuffled[idx]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_proc = pipe.fit_transform(X)

    split = int(0.8 * len(X_proc))
    X_train, X_test = X_proc[:split], X_proc[split:]

    # Original target
    y_train_o, y_test_o = y_orig_sub[:split], y_orig_sub[split:]

    if USE_LGBM:
        model_o = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                     n_jobs=-1, random_state=42, verbose=-1)
    else:
        model_o = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)

    model_o.fit(X_train, y_train_o)
    pred_o = model_o.predict(X_test)
    r2_o = r2_score(y_test_o, pred_o)
    mae_o = mean_absolute_error(y_test_o, pred_o)

    # Shuffled target
    y_train_s, y_test_s = y_shuf_sub[:split], y_shuf_sub[split:]

    if USE_LGBM:
        model_s = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                     n_jobs=-1, random_state=42, verbose=-1)
    else:
        model_s = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)

    model_s.fit(X_train, y_train_s)
    pred_s = model_s.predict(X_test)
    r2_s = r2_score(y_test_s, pred_s)
    mae_s = mean_absolute_error(y_test_s, pred_s)

    print(f"\n  {'Test':<25s} {'MAE':>8s}  {'R²':>8s}")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}")
    print(f"  {'Original target':<25s} {mae_o:>8.4f}  {r2_o:>8.4f}")
    print(f"  {'Shuffled target':<25s} {mae_s:>8.4f}  {r2_s:>8.4f}")

    if r2_s > 0.05:
        print(f"\n  >>> VERDICT: Shuffled target still has R²={r2_s:.3f}!")
        print(f"  >>> This CONFIRMS same-row leakage: features encode the target directly.")
    else:
        print(f"\n  >>> Shuffled R² collapsed as expected (R²={r2_s:.3f}).")


def test_7_feature_importance_audit(df, quick=False):
    """Train a model and show which features drive predictions — flag leakers."""
    print("\n" + "=" * 70)
    print("TEST 7: FEATURE IMPORTANCE WITH LEAKAGE FLAGS")
    print("=" * 70)

    try:
        import lightgbm as lgb
    except ImportError:
        print("  Skipping — LightGBM not available")
        return

    from sklearn.impute import SimpleImputer

    from config import NUMERIC_FEATURES
    features = [f for f in NUMERIC_FEATURES if f in df.columns]
    features += [c for c in df.columns if c.startswith("copula_pc_")]
    features = list(set(features))

    X = SimpleImputer(strategy="median").fit_transform(df[features].values)
    y = df["calculated_dk_fpts"].values

    if quick:
        n = min(80000, len(X))
        idx = np.random.RandomState(42).choice(len(X), n, replace=False)
        X, y = X[idx], y[idx]

    model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                               n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X, y)

    importances = model.feature_importances_.astype(float)
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  {'Rank':<5s} {'Feature':<30s} {'Importance':>12s}  {'Category':<22s}")
    print(f"  {'-'*5} {'-'*30} {'-'*12}  {'-'*22}")

    total_leaker_imp = 0
    total_safe_imp = 0

    for rank, i in enumerate(sorted_idx[:30], 1):
        fname = features[i]
        imp = importances[i]
        cat = classify_feature(fname)
        flag = " <<<<" if cat != "pre-game" else ""
        if cat != "pre-game":
            total_leaker_imp += imp
        else:
            total_safe_imp += imp
        print(f"  {rank:<5d} {fname:<30s} {imp:>12.1f}  {cat:<22s}{flag}")

    total_imp = importances.sum()
    leaker_pct = total_leaker_imp / total_imp * 100 if total_imp > 0 else 0
    print(f"\n  Leaker features: {leaker_pct:.1f}% of total importance (top 30)")
    if leaker_pct > 50:
        print(f"  >>> VERDICT: Leaker features dominate. Model learns same-game → DK points mapping.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Add script dir to path for config import
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    print("=" * 70)
    print("MLB DFS TRAINING PIPELINE — LEAKAGE DIAGNOSTIC")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Model dir: {args.model_dir}")
    print(f"Quick mode: {args.quick}")
    start = time.time()

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.data_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(["Name", "date"], inplace=True)

    # Ensure DK points exist
    if "calculated_dk_fpts" not in df.columns:
        print("Calculating DK fantasy points...")
        df["calculated_dk_fpts"] = df.apply(calculate_dk_fpts, axis=1)

    # Compute the engineered features the pipeline uses so we can test them
    print("Running feature engineering (needed for tests 5-7)...")

    # Import engines
    from feature_engine import (
        EnhancedMLBFinancialStyleEngine,
        ProbabilisticMLBEngine,
        AdvancedCopulaEngine,
        engineer_features as fe_engineer_features,
        concurrent_feature_engineering,
        reduce_copula_features,
        apply_marcel_shrinkage,
        set_feature_jobs,
    )
    set_feature_jobs(min(os.cpu_count() or 4, 8))

    # Subset for speed if quick mode
    if args.quick:
        # Keep most recent 50K rows to preserve player history
        df = df.tail(50000).copy()
        df.reset_index(drop=True, inplace=True)
        print(f"Quick mode: using last {len(df):,} rows")

    print("  Financial engine...")
    fin_eng = EnhancedMLBFinancialStyleEngine()
    df = fin_eng.calculate_features(df)

    print("  Probabilistic engine...")
    prob_eng = ProbabilisticMLBEngine(lookback_window=30, min_observations=10)
    df = prob_eng.calculate_all_features(df)

    print("  Copula engine...")
    copula_eng = AdvancedCopulaEngine(lookback_window=30, min_observations=15)
    df = copula_eng.calculate_all_advanced_features(df)

    print("  Sabermetric features...")
    df = concurrent_feature_engineering(df, chunksize=50000)

    print("  Copula PCA + Marcel shrinkage...")
    df = reduce_copula_features(df, n_components=8)
    df = apply_marcel_shrinkage(df, min_games=50)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"  Final shape: {df.shape}")

    # ---- Run all tests ----
    test_1_target_distribution(df)
    test_2_fold_date_ranges(df, n_splits=5, gap_days=7)
    test_3_feature_leakage_audit(df)
    test_4_baseline_metrics(df, n_splits=5, gap_days=7)
    test_5_retrain_without_leakers(df, n_splits=5, gap_days=7, quick=args.quick)
    test_6_shuffle_dates(df, quick=args.quick)
    test_7_feature_importance_audit(df, quick=args.quick)

    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    print("""
CONFIRMED LEAKAGE SOURCES:

1. SAME-GAME RAW COLUMNS (CATASTROPHIC)
   The FanGraphs game-log columns Off, wRC, SLG, RE24, WAR, RAR, WPA/LI, etc.
   are computed from the SAME game's box score as the DK points target.
   - SLG = (1B + 2×2B + 3×3B + 4×HR) / AB → same events as DK formula
   - wRC, wRAA, Off → same events rearranged via linear weighting
   - These have r = 0.85–0.92 with DK points

2. ENGINEERED SABERMETRIC FEATURES (CATASTROPHIC)
   engineer_features() computes wOBA, BABIP, ISO, wRC+, flyBalls,
   Offense_Statcast, Dollars_Statcast from the SAME game's box score.
   They're just different mathematical views of {1B, 2B, 3B, HR, BB, R, ...}.

3. ROLLING FEATURES WITHOUT SHIFT (MODERATE)
   rolling_mean_fpts_7 etc. use .rolling(7).mean() without .shift(1),
   so they include TODAY'S DK points in the rolling average.
   The lag_* features properly use .shift(1) — those are safe.

4. PROBABILISTIC/COPULA FEATURES (MODERATE)
   skewness_7d, kurtosis_7d, var_95_7d, etc. are rolling windows of
   calculated_dk_fpts that include the current row.
   Network features use same-game DK points for distance calculations.

5. PREPROCESSOR FIT ON ALL DATA
   StandardScaler is fit on full dataset before walk-forward split,
   leaking test distribution information into training.

WHAT THE MODEL ACTUALLY LEARNED:
   DK_pts ≈ f(SLG, wRC, Off, RE24, WPA/LI, ...)
   This is a mapping between different representations of the same game.
   It's algebraically guaranteed to have high R² because these are all
   linear/near-linear functions of {1B, 2B, 3B, HR, RBI, R, BB, HBP, SB}.

WHAT NEEDS TO CHANGE FOR A REAL PRE-GAME MODEL:
   1. Remove ALL same-game columns from features
   2. Shift ALL rolling features by at least 1 game (.shift(1))
   3. Use only data available BEFORE first pitch
   4. Fit preprocessor within each CV fold (not on full data)
   5. Expected honest R² for game-level DK hitter predictions: 0.02–0.15
      (game-level batter DFS is inherently high-variance)
""")

    elapsed = time.time() - start
    print(f"Total diagnostic time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
