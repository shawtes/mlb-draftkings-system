"""
Enhanced training script runner with clean output and progress monitoring.
"""
import os
import sys
import time
import subprocess
import signal
from contextlib import contextmanager
from datetime import datetime

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print a clean header for the training script."""
    clear_screen()
    print("=" * 80)
    print("MLB DRAFTKINGS ENHANCED TRAINING SYSTEM")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Memory Optimized for 32GB RAM System")
    print("=" * 80)
    print()

def print_progress(message, step=None, total=None):
    """Print progress messages with timestamp."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    if step and total:
        progress = f"[{step}/{total}]"
        print(f"[{timestamp}] {progress} {message}")
    else:
        print(f"[{timestamp}] {message}")

@contextmanager
def suppress_warnings():
    """Context manager to suppress warnings during model training."""
    import warnings
    original_filters = warnings.filters[:]
    warnings.filterwarnings('ignore')
    try:
        yield
    finally:
        warnings.filters[:] = original_filters

def run_training():
    """Run the enhanced training script with clean output."""
    print_header()
    
    try:
        # Import and run the training script
        print_progress("Initializing training environment...")
        
        # Set environment variables for cleaner output
        os.environ['PYTHONWARNINGS'] = 'ignore'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        with suppress_warnings():
            import training.backup as training_module
            
            print_progress("Training script loaded successfully")
            print_progress("Starting data loading and processing...")
            
            # The training script should run its main logic here
            # Since it's designed to run on import, we just need to import it
            
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 80)
        return False
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    return True

def main():
    """Main entry point."""
    success = run_training()
    
    if success:
        print("\nTraining completed. Check the output files for results.")
    else:
        print("\nTraining failed or was interrupted.")
        sys.exit(1)

if __name__ == "__main__":
    main()
