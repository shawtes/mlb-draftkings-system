#!/usr/bin/env python3
"""
DFS Optimizer Launcher
======================

Simple Python script to launch the DFS Optimizer web application.
Double-click this file to start the optimizer.

This script will:
1. Start the backend server (Node.js on port 5000)
2. Start the frontend server (Vite on port 5173)  
3. Open the web browser to the optimizer interface
4. Handle cleanup when you close the script
"""

import subprocess
import sys
import os
import time
import webbrowser
import signal
import atexit
from pathlib import Path

# Color codes for console output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color=Colors.WHITE):
    """Print colored message to console"""
    print(f"{color}{message}{Colors.END}")

def print_header():
    """Print the application header"""
    print_colored("=" * 60, Colors.CYAN)
    print_colored("🚀 DFS OPTIMIZER LAUNCHER", Colors.BOLD + Colors.GREEN)
    print_colored("=" * 60, Colors.CYAN)
    print_colored("Starting your MLB DFS Optimization tool...", Colors.WHITE)
    print()

def check_requirements():
    """Check if Node.js is installed"""
    print_colored("🔍 Checking requirements...", Colors.YELLOW)
    
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            node_version = result.stdout.strip()
            print_colored(f"✅ Node.js found: {node_version}", Colors.GREEN)
            return True
        else:
            print_colored("❌ Node.js not found", Colors.RED)
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_colored("❌ Node.js not found or not in PATH", Colors.RED)
        return False

def find_project_root():
    """Find the project root directory"""
    current_dir = Path(__file__).parent
    
    # Check if we're in the right directory
    server_dir = current_dir / "server"
    client_dir = current_dir / "client"
    
    if server_dir.exists() and client_dir.exists():
        return current_dir
    else:
        print_colored("❌ Could not find server and client directories", Colors.RED)
        print_colored(f"Looking in: {current_dir}", Colors.YELLOW)
        return None

def install_dependencies(project_root):
    """Install npm dependencies if needed"""
    server_dir = project_root / "server"
    client_dir = project_root / "client"
    
    # Check server dependencies
    if not (server_dir / "node_modules").exists():
        print_colored("📦 Installing server dependencies...", Colors.YELLOW)
        try:
            subprocess.run(['npm', 'install'], 
                         cwd=server_dir, 
                         check=True, 
                         timeout=120)
            print_colored("✅ Server dependencies installed", Colors.GREEN)
        except subprocess.TimeoutExpired:
            print_colored("⚠️  Server dependency installation timed out", Colors.YELLOW)
        except subprocess.CalledProcessError:
            print_colored("❌ Failed to install server dependencies", Colors.RED)
            return False
    
    # Check client dependencies
    if not (client_dir / "node_modules").exists():
        print_colored("📦 Installing client dependencies...", Colors.YELLOW)
        try:
            subprocess.run(['npm', 'install'], 
                         cwd=client_dir, 
                         check=True, 
                         timeout=180)
            print_colored("✅ Client dependencies installed", Colors.GREEN)
        except subprocess.TimeoutExpired:
            print_colored("⚠️  Client dependency installation timed out", Colors.YELLOW)
        except subprocess.CalledProcessError:
            print_colored("❌ Failed to install client dependencies", Colors.RED)
            return False
    
    return True

def kill_existing_processes():
    """Kill any existing Node.js processes on our ports"""
    print_colored("🔄 Checking for existing processes...", Colors.YELLOW)
    
    try:
        # Kill processes on port 5000 and 3000
        if sys.platform == "win32":
            subprocess.run(['taskkill', '/F', '/IM', 'node.exe'], 
                         capture_output=True, timeout=10)
        else:
            subprocess.run(['pkill', '-f', 'node'], 
                         capture_output=True, timeout=10)
        print_colored("✅ Cleared existing processes", Colors.GREEN)
    except:
        pass  # It's okay if this fails

def start_servers(project_root):
    """Start the backend and frontend servers"""
    server_dir = project_root / "server"
    client_dir = project_root / "client"
    
    print_colored("🚀 Starting backend server...", Colors.BLUE)
    
    # Start backend server
    try:
        backend_process = subprocess.Popen(
            ['node', 'index.js'],
            cwd=server_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        
        # Give backend time to start
        time.sleep(3)
        
        # Check if backend is still running
        if backend_process.poll() is None:
            print_colored("✅ Backend server started on port 5000", Colors.GREEN)
        else:
            print_colored("❌ Backend server failed to start", Colors.RED)
            return None, None
            
    except Exception as e:
        print_colored(f"❌ Failed to start backend: {e}", Colors.RED)
        return None, None
    
    print_colored("🚀 Starting frontend server...", Colors.BLUE)
    
    # Start frontend server
    try:
        frontend_process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=client_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        
        # Give frontend time to start
        print_colored("⏳ Starting Vite development server...", Colors.YELLOW)
        time.sleep(10)
        
        # Check if frontend is still running
        if frontend_process.poll() is None:
            print_colored("✅ Frontend server started on port 5173", Colors.GREEN)
        else:
            print_colored("❌ Frontend server failed to start", Colors.RED)
            return backend_process, None
            
    except Exception as e:
        print_colored(f"❌ Failed to start frontend: {e}", Colors.RED)
        return backend_process, None
    
    return backend_process, frontend_process

def open_browser():
    """Open the web browser to the optimizer"""
    print_colored("🌐 Opening web browser...", Colors.CYAN)
    time.sleep(2)
    
    try:
        webbrowser.open('http://localhost:5173')
        print_colored("✅ Web browser opened to http://localhost:5173", Colors.GREEN)
    except Exception as e:
        print_colored(f"⚠️  Could not open browser automatically: {e}", Colors.YELLOW)
        print_colored("Please manually open: http://localhost:5173", Colors.CYAN)

def cleanup_processes(backend_process, frontend_process):
    """Clean up server processes"""
    print_colored("\n🛑 Shutting down servers...", Colors.YELLOW)
    
    if frontend_process and frontend_process.poll() is None:
        try:
            if sys.platform == "win32":
                frontend_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                frontend_process.terminate()
            frontend_process.wait(timeout=5)
            print_colored("✅ Frontend server stopped", Colors.GREEN)
        except:
            try:
                frontend_process.kill()
            except:
                pass
    
    if backend_process and backend_process.poll() is None:
        try:
            if sys.platform == "win32":
                backend_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                backend_process.terminate()
            backend_process.wait(timeout=5)
            print_colored("✅ Backend server stopped", Colors.GREEN)
        except:
            try:
                backend_process.kill()
            except:
                pass

def main():
    """Main function"""
    backend_process = None
    frontend_process = None
    
    try:
        print_header()
        
        # Check requirements
        if not check_requirements():
            print_colored("\n❌ Node.js is required but not found.", Colors.RED)
            print_colored("Please install Node.js from: https://nodejs.org/", Colors.CYAN)
            input("\nPress Enter to exit...")
            return False
        
        # Find project root
        project_root = find_project_root()
        if not project_root:
            input("\nPress Enter to exit...")
            return False
        
        print_colored(f"📁 Project found at: {project_root}", Colors.CYAN)
        
        # Kill existing processes
        kill_existing_processes()
        
        # Install dependencies
        if not install_dependencies(project_root):
            input("\nPress Enter to exit...")
            return False
        
        # Start servers
        backend_process, frontend_process = start_servers(project_root)
        
        if not backend_process or not frontend_process:
            print_colored("\n❌ Failed to start servers", Colors.RED)
            cleanup_processes(backend_process, frontend_process)
            input("\nPress Enter to exit...")
            return False
        
        # Open browser
        open_browser()
        
        # Success message
        print()
        print_colored("🎉 DFS OPTIMIZER IS READY!", Colors.BOLD + Colors.GREEN)
        print_colored("=" * 60, Colors.CYAN)
        print_colored("📊 Backend API: http://localhost:5000", Colors.WHITE)
        print_colored("🌐 Web Interface: http://localhost:5173", Colors.WHITE)
        print_colored("=" * 60, Colors.CYAN)
        print()
        print_colored("💡 TIP: The optimizer will open in your web browser", Colors.YELLOW)
        print_colored("🔄 Both servers are running in the background", Colors.WHITE)
        print_colored("❌ Press Ctrl+C or close this window to stop", Colors.RED)
        print()
        
        # Register cleanup function
        atexit.register(cleanup_processes, backend_process, frontend_process)
        
        # Keep script running
        try:
            while True:
                # Check if processes are still alive
                if backend_process.poll() is not None:
                    print_colored("❌ Backend server stopped unexpectedly", Colors.RED)
                    break
                if frontend_process.poll() is not None:
                    print_colored("❌ Frontend server stopped unexpectedly", Colors.RED)
                    break
                
                time.sleep(5)
        except KeyboardInterrupt:
            print_colored("\n\n👋 Shutting down DFS Optimizer...", Colors.YELLOW)
        
        return True
        
    except Exception as e:
        print_colored(f"\n❌ Unexpected error: {e}", Colors.RED)
        return False
    
    finally:
        cleanup_processes(backend_process, frontend_process)
        print_colored("\n✅ Cleanup complete. Thanks for using DFS Optimizer!", Colors.GREEN)
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\n👋 Goodbye!", Colors.YELLOW)
        sys.exit(0)
    except Exception as e:
        print_colored(f"\n💥 Fatal error: {e}", Colors.RED)
        input("\nPress Enter to exit...")
        sys.exit(1)
