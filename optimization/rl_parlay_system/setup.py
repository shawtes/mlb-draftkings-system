#!/usr/bin/env python3
"""
Setup script for RL Parlay System
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing RL Parlay System requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_rl.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "rl_training_data",
        "rl_models", 
        "rl_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ Directories created successfully!")

def main():
    """Main setup function"""
    print("🚀 RL Parlay System Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run demo: python main.py demo")
    print("2. Collect data: python main.py collect --api-key YOUR_KEY")
    print("3. Train model: python main.py train --api-key YOUR_KEY")
    print("4. Use GUI: python main.py gui")
    
    return True

if __name__ == "__main__":
    main()










