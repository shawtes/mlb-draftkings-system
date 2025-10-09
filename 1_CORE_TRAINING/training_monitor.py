#!/usr/bin/env python3
"""
Training Progress Monitor
Monitor the progress of your ML training script
"""

import os
import glob
import pandas as pd
import time
from datetime import datetime

def monitor_training_progress():
    """Monitor training progress by checking output files and logs"""
    
    base_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/'
    
    print("🔍 Training Progress Monitor")
    print("=" * 50)
    
    start_time = time.time()
    
    while True:
        try:
            # Calculate timing information
            elapsed_time = time.time() - start_time
            elapsed_mins = elapsed_time / 60
            
            # Check for main output files from single-model training
            final_files = [
                "final_predictions.csv",
                "batters_final_ensemble_model_pipeline.pkl",
                "battersfinal_dataset_with_features.csv",
                "feature_importances.csv"
            ]
            
            completed_final = []
            for file in final_files:
                if os.path.exists(f"{base_path}{file}"):
                    completed_final.append(file)
            
            if completed_final:
                print(f"\n📊 Progress Update - {datetime.now().strftime('%H:%M:%S')}")
                print(f"Training outputs: {len(completed_final)}/{len(final_files)}")
                print(f"Elapsed time: {elapsed_mins:.1f} minutes")
                
                print(f"\n📁 Completed outputs:")
                for file in completed_final:
                    file_size = os.path.getsize(f"{base_path}{file}") / 1024  # KB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(f"{base_path}{file}"))
                    print(f"  ✅ {file} ({file_size:.1f} KB) - {mod_time.strftime('%H:%M:%S')}")
                
                if len(completed_final) == len(final_files):
                    total_time = elapsed_time / 60
                    print(f"\n🚀 Training completely finished in {total_time:.1f} minutes!")
                    break
                else:
                    print(f"  ⏳ Still generating remaining outputs...")
                        
            else:
                print(f"\n⏳ Training in progress... - {datetime.now().strftime('%H:%M:%S')}")
                print(f"  Runtime: {elapsed_mins:.1f} minutes")
                print("  Currently processing single-model training...")
                
                # Check if we can detect current activity from console output
                print("  💡 Tip: Check the console where you started traning.py for detailed progress")
            
            # Enhanced log file checking
            log_files_to_check = [
                "training.log",
                "gpu_training.log", 
                "error.log"
            ]
            
            for log_file in log_files_to_check:
                if os.path.exists(f"{base_path}{log_file}"):
                    print(f"  📝 {log_file} detected")
            
            # Check for any error indicators
            if os.path.exists(f"{base_path}error.log"):
                print("  ⚠️ Error log detected - check for issues")
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped by user after {elapsed_mins:.1f} minutes")
            break
        except Exception as e:
            print(f"\n❌ Error monitoring: {e}")
            time.sleep(30)

def show_system_info():
    """Show current system status"""
    import psutil
    import torch
    
    print("\n💻 System Status")
    print("-" * 30)
    print(f"CPU Usage: {psutil.cpu_percent():.1f}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent:.1f}%")
    
    if torch.cuda.is_available():
        print(f"GPU Available: ✅ {torch.cuda.get_device_name(0)}")
        try:
            # Try to get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory:.1f} GB")
        except:
            print("GPU Memory: Unable to query")
    else:
        print("GPU Available: ❌")

def quick_status_check():
    """Quick check of current training status"""
    base_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/'
    
    print("🔍 Quick Training Status Check")
    print("=" * 40)
    
    # Check for training output files
    final_files = [
        "final_predictions.csv",
        "batters_final_ensemble_model_pipeline.pkl", 
        "battersfinal_dataset_with_features.csv",
        "feature_importances.csv"
    ]
    
    completed_final = []
    for file in final_files:
        file_path = f"{base_path}{file}"
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            completed_final.append((file, file_size, mod_time))
    
    if completed_final:
        print(f"\n✅ Training Progress: {len(completed_final)}/{len(final_files)} outputs complete")
        for file, size, mod_time in completed_final:
            print(f"  📁 {file} ({size:.1f} KB) - Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if len(completed_final) == len(final_files):
            print("\n🚀 Training appears to be COMPLETE!")
            
            # Show model performance if predictions file exists
            try:
                pred_file = f"{base_path}final_predictions.csv"
                if os.path.exists(pred_file):
                    df = pd.read_csv(pred_file)
                    if 'accuracy' in df.columns:
                        avg_accuracy = df['accuracy'].mean()
                        print(f"  📈 Average model accuracy: {avg_accuracy:.3f}")
                    print(f"  � Predictions generated: {len(df)} rows")
            except Exception as e:
                print(f"  ⚠️ Could not read prediction metrics: {e}")
        else:
            print(f"\n⏳ Training still in progress ({len(completed_final)}/{len(final_files)} complete)")
    else:
        print("\n❌ No training outputs found - training may not have started or failed")
        
        # Check if training files exist
        training_files = ["traning.py", "config.py"]
        for file in training_files:
            if os.path.exists(f"{base_path}{file}"):
                print(f"  ✅ {file} exists")
            else:
                print(f"  ❌ {file} missing")
    
    # Training status
    if len(completed_final) == 0:
        print(f"\n🚀 Status: Training is starting or not yet begun")
    elif len(completed_final) < len(final_files):
        print(f"\n🔄 Status: Single-model training in progress")
    else:
        print(f"\n🎯 Status: Training complete - all outputs generated")
    
    print(f"\n💡 Use 'python training_monitor.py' for continuous monitoring")
    print(f"💡 Check console output of traning.py for detailed progress")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_status_check()
    else:
        print("🤖 ML Training Monitor")
        print("Press Ctrl+C to stop monitoring")
        print("💡 For a quick status check, run: python training_monitor.py quick\n")
        
        show_system_info()
        monitor_training_progress()
