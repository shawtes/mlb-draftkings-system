#!/usr/bin/env python3
"""
run_pipeline.py — Master Pipeline: Fetch → Train → Bridge → Optimize

Chains the entire MLB DFS workflow:
  1. FETCH   — Download historical batter data from FanGraphs (optional)
  2. TRAIN   — Run ML training pipeline on the data
  3. BRIDGE  — Merge training predictions with DraftKings CSV → optimizer-ready file

Usage:
  python run_pipeline.py                     # Full pipeline (skip fetch if data exists)
  python run_pipeline.py --fetch             # Force re-fetch data first
  python run_pipeline.py --train-only        # Just train (skip fetch + bridge)
  python run_pipeline.py --bridge-only       # Just bridge (skip fetch + train)
  python run_pipeline.py --skip-hpo          # Skip Optuna HPO during training (faster)
  python run_pipeline.py --season 2024       # Fetch only 2024 season data
  python run_pipeline.py --seasons 2020-2025 # Fetch a range of seasons
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FANGRAPHS_SCRIPT = os.path.join(REPO_ROOT, 'scripts', 'fangraphs_batters.py')
FANGRAPHS_VENV = os.path.join(REPO_ROOT, 'scripts', '.venv', 'bin', 'python3')
FANGRAPHS_DATA = os.path.join(os.path.expanduser('~'), 'FangraphsData', 'merged_fangraphs_data.csv')

TRAINING_SCRIPT = os.path.join(REPO_ROOT, '1_CORE_TRAINING', 'training.py')
TRAINING_OUTPUT = os.path.join(REPO_ROOT, '1_CORE_TRAINING', 'output')

DATA_DIR = os.path.join(REPO_ROOT, 'data')
DATA_LINK = os.path.join(DATA_DIR, 'merged_fangraphs_data.csv')

BRIDGE_SCRIPT = os.path.join(REPO_ROOT, '3_BRIDGE', 'dk_to_optimizer.py')
DK_DROP_DIR = os.path.join(DATA_DIR, 'dk_drop')
OPTIMIZER_READY_DIR = os.path.join(DATA_DIR, 'optimizer_ready')


def banner(text):
    width = 60
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")


def run_cmd(cmd, description, env=None, cwd=None):
    """Run a command, stream output, return success bool."""
    print(f"Running: {' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, cwd=cwd,
        )
        for line in proc.stdout:
            print(f"  {line}", end='')
        proc.wait()
        if proc.returncode != 0:
            print(f"\n[FAILED] {description} exited with code {proc.returncode}")
            return False
        print(f"\n[OK] {description} complete")
        return True
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return False


# ---------------------------------------------------------------------------
# Step 1: FETCH
# ---------------------------------------------------------------------------
def step_fetch(args):
    """Download historical MLB data from FanGraphs."""
    banner("STEP 1: FETCH DATA FROM FANGRAPHS")

    if not os.path.exists(FANGRAPHS_SCRIPT):
        print(f"Fangraphs script not found: {FANGRAPHS_SCRIPT}")
        return False

    # Use the scripts venv if available, else system python
    python = FANGRAPHS_VENV if os.path.exists(FANGRAPHS_VENV) else sys.executable

    print(f"Python: {python}")
    print(f"Script: {FANGRAPHS_SCRIPT}")
    print()
    print("This launches the interactive FanGraphs scraper.")
    print("You'll need to log in to FanGraphs in the browser window.")
    print()

    cmd = [python, FANGRAPHS_SCRIPT]
    return run_cmd(cmd, "FanGraphs data fetch", cwd=os.path.dirname(FANGRAPHS_SCRIPT))


# ---------------------------------------------------------------------------
# Step 1b: Link data into repo
# ---------------------------------------------------------------------------
def link_data():
    """Symlink or copy FanGraphs data into data/ so training can find it."""
    banner("LINKING DATA")

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(FANGRAPHS_DATA):
        print(f"FanGraphs data not found at: {FANGRAPHS_DATA}")
        print("Run with --fetch first, or download data manually.")
        return False

    file_size = os.path.getsize(FANGRAPHS_DATA) / (1024 * 1024)
    print(f"Source: {FANGRAPHS_DATA} ({file_size:.1f} MB)")

    # Remove old link/copy if it exists
    if os.path.islink(DATA_LINK):
        os.unlink(DATA_LINK)
    elif os.path.exists(DATA_LINK):
        os.remove(DATA_LINK)

    # Symlink (fast, no disk duplication)
    try:
        os.symlink(FANGRAPHS_DATA, DATA_LINK)
        print(f"Symlinked: {DATA_LINK} → {FANGRAPHS_DATA}")
    except OSError:
        # Fallback to copy on systems that don't support symlinks
        shutil.copy2(FANGRAPHS_DATA, DATA_LINK)
        print(f"Copied to: {DATA_LINK}")

    return True


# ---------------------------------------------------------------------------
# Step 2: TRAIN
# ---------------------------------------------------------------------------
def step_train(args):
    """Run the ML training pipeline."""
    banner("STEP 2: TRAIN ML MODEL")

    if not os.path.exists(TRAINING_SCRIPT):
        print(f"Training script not found: {TRAINING_SCRIPT}")
        return False

    # Make sure data is linked
    if not os.path.exists(DATA_LINK) and not args.data_path:
        if os.path.exists(FANGRAPHS_DATA):
            link_data()
        else:
            print("No training data available. Run --fetch first.")
            return False

    data_path = args.data_path or DATA_LINK
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False

    os.makedirs(TRAINING_OUTPUT, exist_ok=True)

    cmd = [
        sys.executable, TRAINING_SCRIPT,
        '--data-path', data_path,
        '--output-dir', TRAINING_OUTPUT,
        '--n-splits', str(args.n_splits),
        '--gap-days', str(args.gap_days),
    ]

    if args.skip_hpo:
        cmd.append('--skip-hpo')
    else:
        cmd.extend(['--optuna-trials', str(args.optuna_trials)])

    cmd.extend(['--n-features', str(args.n_features)])

    print(f"Data:   {data_path}")
    print(f"Output: {TRAINING_OUTPUT}")
    print(f"Folds:  {args.n_splits}")
    print(f"HPO:    {'skip' if args.skip_hpo else f'{args.optuna_trials} Optuna trials'}")
    print()

    start = time.time()
    ok = run_cmd(cmd, "ML training pipeline")
    elapsed = time.time() - start
    print(f"Training time: {elapsed/60:.1f} min")
    return ok


# ---------------------------------------------------------------------------
# Step 3: BRIDGE
# ---------------------------------------------------------------------------
def step_bridge(args):
    """Merge training predictions with DraftKings CSV."""
    banner("STEP 3: BRIDGE — DK + PREDICTIONS → OPTIMIZER")

    if not os.path.exists(BRIDGE_SCRIPT):
        print(f"Bridge script not found: {BRIDGE_SCRIPT}")
        return False

    # Check for DK file
    os.makedirs(DK_DROP_DIR, exist_ok=True)
    dk_file = args.dk_file
    if dk_file is None:
        # Auto-find latest CSV in drop folder
        import glob
        csvs = sorted(glob.glob(os.path.join(DK_DROP_DIR, '*.csv')),
                       key=os.path.getmtime, reverse=True)
        # Exclude README
        csvs = [f for f in csvs if not f.endswith('README.txt')]
        if csvs:
            dk_file = csvs[0]

    if not dk_file:
        # Check Downloads folder as fallback
        downloads = os.path.expanduser('~/Downloads')
        import glob
        dk_candidates = sorted(
            glob.glob(os.path.join(downloads, 'DK*.csv')) +
            glob.glob(os.path.join(downloads, 'dk*.csv')),
            key=os.path.getmtime, reverse=True,
        )
        if dk_candidates:
            dk_file = dk_candidates[0]
            print(f"Found DK file in Downloads: {os.path.basename(dk_file)}")
            # Copy to drop folder
            dest = os.path.join(DK_DROP_DIR, os.path.basename(dk_file))
            shutil.copy2(dk_file, dest)
            dk_file = dest
            print(f"Copied to: {dest}")

    if not dk_file:
        print("No DraftKings CSV found.")
        print(f"  Drop a DK export into: {DK_DROP_DIR}/")
        print(f"  Or specify: --dk-file /path/to/DKEntries.csv")
        print()
        print("Skipping bridge step. You can run it later:")
        print(f"  python {BRIDGE_SCRIPT}")
        return True  # Non-fatal — training still succeeded

    # Check for predictions
    predictions_dir = TRAINING_OUTPUT
    has_predictions = os.path.exists(os.path.join(predictions_dir, 'final_predictions.csv'))

    cmd = [
        sys.executable, BRIDGE_SCRIPT,
        '--dk-file', dk_file,
        '--predictions-dir', predictions_dir,
    ]
    if args.sport:
        cmd.extend(['--sport', args.sport])
    if not has_predictions:
        cmd.append('--no-predictions')
        print("No training predictions found — using DK averages")

    return run_cmd(cmd, "DK → Optimizer bridge")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='MLB DFS Master Pipeline: Fetch → Train → Bridge → Optimize',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                       # Train + bridge (uses existing data)
  python run_pipeline.py --fetch               # Fetch data first, then train + bridge
  python run_pipeline.py --train-only          # Just train, no bridge
  python run_pipeline.py --bridge-only         # Just bridge (already trained)
  python run_pipeline.py --skip-hpo            # Faster training (skip Optuna)
  python run_pipeline.py --dk-file ~/Downloads/DKEntries.csv  # Use specific DK file
        """,
    )

    # Pipeline control
    parser.add_argument('--fetch', action='store_true',
                        help='Run FanGraphs data fetch (interactive, needs browser)')
    parser.add_argument('--train-only', action='store_true',
                        help='Only run training (skip bridge)')
    parser.add_argument('--bridge-only', action='store_true',
                        help='Only run bridge (skip fetch + train)')

    # Training params
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to training data CSV (default: auto-detect)')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Walk-forward CV folds (default: 5)')
    parser.add_argument('--gap-days', type=int, default=7,
                        help='Embargo gap days (default: 7)')
    parser.add_argument('--skip-hpo', action='store_true',
                        help='Skip Optuna HPO (use hard-coded params)')
    parser.add_argument('--optuna-trials', type=int, default=50,
                        help='Optuna HPO trials (default: 50)')
    parser.add_argument('--n-features', type=int, default=100,
                        help='SHAP feature selection count (default: 100)')

    # Bridge params
    parser.add_argument('--dk-file', type=str, default=None,
                        help='Path to DraftKings CSV')
    parser.add_argument('--sport', type=str, choices=['MLB', 'NBA', 'NFL'], default=None,
                        help='Force sport (default: auto-detect)')

    args = parser.parse_args()

    start_time = time.time()

    banner("MLB DFS MASTER PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Repo:    {REPO_ROOT}")

    steps_run = []
    steps_ok = []

    # --- BRIDGE ONLY ---
    if args.bridge_only:
        ok = step_bridge(args)
        steps_run.append('bridge')
        steps_ok.append(ok)

    else:
        # --- FETCH ---
        if args.fetch:
            ok = step_fetch(args)
            steps_run.append('fetch')
            steps_ok.append(ok)
            if not ok:
                print("\nFetch failed. Continuing with existing data if available...")

        # --- LINK DATA ---
        if os.path.exists(FANGRAPHS_DATA) and not os.path.exists(DATA_LINK):
            link_data()

        # --- TRAIN ---
        if not args.bridge_only:
            ok = step_train(args)
            steps_run.append('train')
            steps_ok.append(ok)
            if not ok:
                print("\nTraining failed.")
                sys.exit(1)

        # --- BRIDGE ---
        if not args.train_only:
            ok = step_bridge(args)
            steps_run.append('bridge')
            steps_ok.append(ok)

    # --- SUMMARY ---
    elapsed = time.time() - start_time
    banner("PIPELINE COMPLETE")
    print(f"Time: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print()
    for step, ok in zip(steps_run, steps_ok):
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {step}")

    print()

    # Show output files
    if os.path.exists(TRAINING_OUTPUT):
        import glob
        outputs = glob.glob(os.path.join(TRAINING_OUTPUT, '*.csv')) + \
                  glob.glob(os.path.join(TRAINING_OUTPUT, '*.pkl'))
        if outputs:
            print("Training artifacts:")
            for f in sorted(outputs):
                size = os.path.getsize(f) / 1024
                print(f"  {os.path.basename(f):45s} {size:>8.0f} KB")

    if os.path.exists(OPTIMIZER_READY_DIR):
        import glob
        ready = glob.glob(os.path.join(OPTIMIZER_READY_DIR, '*_optimizer_ready.csv'))
        if ready:
            print()
            print("Optimizer-ready files:")
            for f in sorted(ready, key=os.path.getmtime, reverse=True)[:3]:
                print(f"  {f}")

    print()
    print("Next steps:")
    if 'bridge' in steps_run and steps_ok[steps_run.index('bridge')]:
        print("  1. Start web app:   cd web_optimizer && npm run dev")
        print("  2. Upload the optimizer-ready CSV at http://localhost:3000")
        print("  3. Configure settings and click BUILD LINEUPS")
    elif 'train' in steps_run:
        print("  1. Drop a DK CSV into data/dk_drop/")
        print("  2. Run:  python run_pipeline.py --bridge-only")
    else:
        print("  1. Run:  python run_pipeline.py --fetch")


if __name__ == '__main__':
    main()
