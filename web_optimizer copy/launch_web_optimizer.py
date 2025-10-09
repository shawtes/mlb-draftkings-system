#!/usr/bin/env python3
"""
DFS Web Optimizer Launcher
==========================

Double-click this file to automatically:
1. Check for Node.js and dependencies
2. Kill any existing processes on ports 3000/5000
3. Start backend server (port 5000)
4. Start frontend React app (port 3000)
5. Open the web UI in your default browser
6. Handle cleanup when you close the script

No command line needed - just double-click and go!
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import queue
import psutil
import socket

class WebOptimizerLauncher:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        
        # Smart path detection - check if we're in the web_optimizer directory or parent
        if (self.base_dir / "client").exists() and (self.base_dir / "server").exists():
            # We're in the web_optimizer directory
            self.web_optimizer_dir = self.base_dir
        else:
            # We're in the parent directory
            self.web_optimizer_dir = self.base_dir / "web_optimizer"
        
        self.client_dir = self.web_optimizer_dir / "client"
        self.server_dir = self.web_optimizer_dir / "server"
        
        self.backend_process = None
        self.frontend_process = None
        self.log_queue = queue.Queue()
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Create the launcher GUI"""
        self.root = tk.Tk()
        self.root.title("DFS Web Optimizer Launcher")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1f3a")
        
        # Header
        header_frame = tk.Frame(self.root, bg="#667eea", height=80)
        header_frame.pack(fill="x", pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🏈 DFS Web Optimizer Launcher", 
            font=("Arial", 16, "bold"),
            bg="#667eea",
            fg="white"
        )
        title_label.pack(expand=True)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg="#1a1f3a")
        status_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready to launch...",
            font=("Arial", 12),
            bg="#1a1f3a",
            fg="#b0bec5"
        )
        self.status_label.pack()
        
        # Buttons frame
        button_frame = tk.Frame(self.root, bg="#1a1f3a")
        button_frame.pack(fill="x", padx=20, pady=10)
        
        self.launch_btn = tk.Button(
            button_frame,
            text="🚀 Launch Web Optimizer",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            command=self.launch_optimizer
        )
        self.launch_btn.pack(side="left", padx=(0, 10))
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️ Stop All Servers",
            font=("Arial", 12),
            bg="#F44336",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            command=self.stop_servers,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(0, 10))
        
        self.browser_btn = tk.Button(
            button_frame,
            text="🌐 Open in Browser",
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            command=lambda: webbrowser.open("http://localhost:3000"),
            state="disabled"
        )
        self.browser_btn.pack(side="left", padx=(0, 10))
        
        self.install_btn = tk.Button(
            button_frame,
            text="📥 Install Node.js",
            font=("Arial", 12),
            bg="#FF9800",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            command=self.open_nodejs_download
        )
        self.install_btn.pack(side="left")
        
        # Log area
        log_frame = tk.Frame(self.root, bg="#1a1f3a")
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        log_label = tk.Label(
            log_frame,
            text="📋 Launch Log:",
            font=("Arial", 10, "bold"),
            bg="#1a1f3a",
            fg="#b0bec5",
            anchor="w"
        )
        log_label.pack(fill="x")
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=("Consolas", 9),
            bg="#0a0e27",
            fg="#b0bec5",
            insertbackground="white",
            relief="flat",
            wrap="word"
        )
        self.log_text.pack(fill="both", expand=True, pady=(5, 0))
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start log processor
        self.process_log_queue()
        
    def log(self, message, color="#b0bec5"):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put((f"[{timestamp}] {message}\n", color))
        
    def process_log_queue(self):
        """Process log messages from queue"""
        try:
            while True:
                message, color = self.log_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert("end", message)
                self.log_text.config(state="disabled")
                self.log_text.see("end")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)
    
    def check_dependencies(self):
        """Check if Node.js and npm are available"""
        self.log("🔍 Checking dependencies...")
        
        # Common Node.js installation paths on Windows
        node_paths = [
            "node",  # If in PATH
            r"C:\Program Files\nodejs\node.exe",
            r"C:\Program Files (x86)\nodejs\node.exe",
            os.path.expanduser(r"~\AppData\Roaming\npm\node.exe"),
            os.path.expanduser(r"~\AppData\Local\Programs\nodejs\node.exe"),
        ]
        
        node_found = False
        node_version = ""
        node_exe_path = ""
        
        # Check Node.js
        for node_path in node_paths:
            try:
                result = subprocess.run([node_path, "--version"], capture_output=True, text=True, shell=True, timeout=10)
                if result.returncode == 0:
                    node_version = result.stdout.strip()
                    node_exe_path = node_path
                    self.log(f"✅ Node.js found at {node_path}: {node_version}", "#4CAF50")
                    node_found = True
                    
                    # If not the first path (which is just "node"), add to PATH
                    if node_path != "node":
                        node_dir = os.path.dirname(node_path)
                        current_path = os.environ.get("PATH", "")
                        if node_dir not in current_path:
                            os.environ["PATH"] = node_dir + os.pathsep + current_path
                            self.log(f"✅ Added {node_dir} to PATH for this session", "#4CAF50")
                            
                            # Also update npm path
                            npm_dir = node_dir  # npm is usually in the same directory
                            if npm_dir not in current_path:
                                os.environ["PATH"] = npm_dir + os.pathsep + os.environ["PATH"]
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                continue
        
        if not node_found:
            self.log("❌ Node.js not found! Checking common installation locations...", "#F44336")
            self.log("   Searched paths:", "#F44336")
            for path in node_paths:
                self.log(f"   - {path}", "#F44336")
            self.log("", "#F44336")
            self.log("📥 Please install Node.js:", "#FF9800")
            self.log("   1. Go to https://nodejs.org", "#FF9800")
            self.log("   2. Download the latest LTS version for Windows", "#FF9800")
            self.log("   3. Run the installer and restart this launcher", "#FF9800")
            self.log("   4. Make sure to check 'Add to PATH' during installation", "#FF9800")
            
            # Offer to open the download page
            result = messagebox.askyesno(
                "Node.js Required",
                "Node.js is required to run the DFS Web Optimizer.\n\n"
                "Would you like to open the Node.js download page now?\n\n"
                "After installing Node.js, restart this launcher."
            )
            if result:
                webbrowser.open("https://nodejs.org/en/download/")
            
            return False
            
        # Check npm (usually comes with Node.js)
        npm_paths = [
            "npm",  # If in PATH
            r"C:\Program Files\nodejs\npm.cmd",
            r"C:\Program Files (x86)\nodejs\npm.cmd",
            os.path.expanduser(r"~\AppData\Roaming\npm\npm.cmd"),
        ]
        
        # If we found node.exe, try npm in the same directory
        if node_exe_path and node_exe_path != "node":
            node_dir = os.path.dirname(node_exe_path)
            npm_paths.insert(1, os.path.join(node_dir, "npm.cmd"))
        
        npm_found = False
        for npm_path in npm_paths:
            try:
                result = subprocess.run([npm_path, "--version"], capture_output=True, text=True, shell=True, timeout=10)
                if result.returncode == 0:
                    self.log(f"✅ npm found: {result.stdout.strip()}", "#4CAF50")
                    npm_found = True
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                continue
        
        if not npm_found:
            self.log("❌ npm not found! This should come with Node.js installation.", "#F44336")
            self.log("   Try reinstalling Node.js from https://nodejs.org", "#F44336")
            return False
            
        return True
    
    def open_nodejs_download(self):
        """Open Node.js download page and show installation instructions"""
        self.log("📥 Opening Node.js download page...", "#FF9800")
        webbrowser.open("https://nodejs.org/en/download/")
        
        messagebox.showinfo(
            "Node.js Installation Instructions",
            "Node.js download page opened in your browser.\n\n"
            "Installation Steps:\n"
            "1. Download the Windows Installer (.msi) - LTS version recommended\n"
            "2. Run the installer as Administrator\n"
            "3. Follow the installation wizard\n"
            "4. IMPORTANT: Make sure 'Add to PATH' is checked\n"
            "5. Restart this launcher after installation\n\n"
            "The installer will include both Node.js and npm."
        )
    
    def check_project_structure(self):
        """Check if the project directories exist"""
        self.log("📁 Checking project structure...")
        
        # List what we actually have
        if self.web_optimizer_dir.exists():
            contents = list(self.web_optimizer_dir.iterdir())
            self.log(f"📂 Web optimizer directory contents: {[item.name for item in contents]}")
        else:
            self.log(f"❌ Web optimizer directory not found: {self.web_optimizer_dir}", "#F44336")
            self.log("🔧 Creating basic project structure...", "#FF9800")
            self.create_simple_project()
            return True
            
        # Check for React/Node.js structure
        if (self.client_dir.exists() and (self.client_dir / "package.json").exists() and 
            self.server_dir.exists() and (self.server_dir / "index.js").exists()):
            self.log("✅ React/Node.js project structure found", "#4CAF50")
            return True
        
        # If no proper structure, create a simple Flask alternative
        self.log("⚠️ React/Node.js structure not found, creating simple Flask alternative...", "#FF9800")
        self.create_simple_project()
        return True
    
    def kill_existing_processes(self):
        """Kill any existing processes on ports 3000 and 5000"""
        self.log("🔪 Killing existing processes on ports 3000 and 5000...")
        
        ports_to_check = [3000, 5000]
        killed_any = False
        
        for port in ports_to_check:
            try:
                # Get processes using the port
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        # Get connections for this process
                        connections = proc.connections()
                        for conn in connections:
                            if conn.laddr.port == port:
                                self.log(f"🔪 Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                                proc.terminate()
                                killed_any = True
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
                        continue
            except Exception as e:
                # If psutil method fails, try alternative approach using netstat
                self.log(f"⚠️ psutil method failed, trying alternative approach...")
                try:
                    if os.name == 'nt':  # Windows
                        # Use netstat to find processes on the port
                        result = subprocess.run(
                            ['netstat', '-ano'], 
                            capture_output=True, 
                            text=True, 
                            shell=True
                        )
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if f':{port}' in line and 'LISTENING' in line:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    pid = parts[-1]
                                    try:
                                        subprocess.run(['taskkill', '/F', '/PID', pid], 
                                                     capture_output=True, shell=True)
                                        self.log(f"🔪 Killed process PID {pid} on port {port}")
                                        killed_any = True
                                    except Exception:
                                        pass
                except Exception:
                    self.log(f"⚠️ Could not kill processes on port {port}")
        
        if killed_any:
            time.sleep(2)  # Give processes time to terminate
            self.log("✅ Existing processes killed", "#4CAF50")
        else:
            self.log("✅ No existing processes found on target ports", "#4CAF50")
    
    def install_dependencies(self):
        """Install npm dependencies if needed"""
        self.log("📦 Checking and installing dependencies...")
        
        # Check if node_modules exists
        if not (self.client_dir / "node_modules").exists():
            self.log("📦 Installing client dependencies...")
            self.status_label.config(text="Installing dependencies...")
            
            try:
                process = subprocess.Popen(
                    ["npm", "install"],
                    cwd=self.client_dir,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output
                for line in process.stdout:
                    self.log(line.strip())
                    self.root.update()
                
                process.wait()
                
                if process.returncode == 0:
                    self.log("✅ Dependencies installed successfully", "#4CAF50")
                else:
                    self.log("❌ Failed to install dependencies", "#F44336")
                    return False
                    
            except Exception as e:
                self.log(f"❌ Error installing dependencies: {str(e)}", "#F44336")
                return False
        else:
            self.log("✅ Dependencies already installed", "#4CAF50")
        
        return True
    
    def start_backend_server(self):
        """Start the backend server"""
        self.log("🔧 Starting backend server...")
        
        try:
            self.backend_process = subprocess.Popen(
                ["node", "index.js"],
                cwd=self.server_dir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            if self.backend_process.poll() is None:
                self.log("✅ Backend server started on port 5000", "#4CAF50")
                return True
            else:
                self.log("❌ Backend server failed to start", "#F44336")
                return False
                
        except Exception as e:
            self.log(f"❌ Error starting backend server: {str(e)}", "#F44336")
            return False
    
    def start_frontend_server(self):
        """Start the frontend React development server"""
        self.log("⚛️ Starting frontend React server...")
        
        try:
            env = os.environ.copy()
            env["BROWSER"] = "none"  # Prevent automatic browser opening
            
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=self.client_dir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            # Wait for React to start (look for "webpack compiled" or similar)
            self.log("⏳ Waiting for React development server to start...")
            timeout = 60  # 60 seconds timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.frontend_process.poll() is not None:
                    self.log("❌ Frontend server process terminated", "#F44336")
                    return False
                
                # Check if port 3000 is open
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', 3000))
                    sock.close()
                    
                    if result == 0:
                        self.log("✅ Frontend server started on port 3000", "#4CAF50")
                        return True
                        
                except Exception:
                    pass
                
                time.sleep(2)
                self.log("⏳ Still waiting for React server...")
                self.root.update()
            
            self.log("❌ Frontend server startup timeout", "#F44336")
            return False
            
        except Exception as e:
            self.log(f"❌ Error starting frontend server: {str(e)}", "#F44336")
            return False
    
    def open_browser(self):
        """Open the web UI in the default browser"""
        self.log("🌐 Opening web UI in browser...")
        time.sleep(2)  # Give React a moment to fully load
        webbrowser.open("http://localhost:3000")
        self.log("✅ Web UI opened in browser", "#4CAF50")
    
    def launch_optimizer(self):
        """Main launch sequence"""
        self.launch_btn.config(state="disabled")
        self.status_label.config(text="Launching...")
        
        def launch_thread():
            try:
                # Check dependencies
                if not self.check_dependencies():
                    self.log("❌ Launch failed: Missing dependencies", "#F44336")
                    return
                
                # Check project structure
                if not self.check_project_structure():
                    self.log("❌ Launch failed: Invalid project structure", "#F44336")
                    return
                
                # Kill existing processes
                self.kill_existing_processes()
                
                # Install dependencies
                if not self.install_dependencies():
                    self.log("❌ Launch failed: Could not install dependencies", "#F44336")
                    return
                
                # Start backend server
                if not self.start_backend_server():
                    self.log("❌ Launch failed: Could not start backend server", "#F44336")
                    return
                
                # Start frontend server
                if not self.start_frontend_server():
                    self.log("❌ Launch failed: Could not start frontend server", "#F44336")
                    return
                
                # Open browser
                self.open_browser()
                
                # Update UI
                self.status_label.config(text="🚀 Web Optimizer is running!")
                self.stop_btn.config(state="normal")
                self.browser_btn.config(state="normal")
                self.log("🎉 Launch completed successfully!", "#4CAF50")
                self.log("📝 You can now use the DFS Web Optimizer in your browser")
                self.log("⏹️ Click 'Stop All Servers' to shut down when finished")
                
            except Exception as e:
                self.log(f"❌ Unexpected error during launch: {str(e)}", "#F44336")
            finally:
                self.launch_btn.config(state="normal")
        
        # Run in separate thread to prevent GUI freezing
        threading.Thread(target=launch_thread, daemon=True).start()
    
    def stop_servers(self):
        """Stop all running servers"""
        self.log("⏹️ Stopping all servers...")
        
        if self.frontend_process and self.frontend_process.poll() is None:
            self.log("⏹️ Stopping frontend server...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
                self.log("✅ Frontend server stopped", "#4CAF50")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                self.log("🔪 Frontend server killed (forced)", "#FF9800")
        
        if self.backend_process and self.backend_process.poll() is None:
            self.log("⏹️ Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
                self.log("✅ Backend server stopped", "#4CAF50")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                self.log("🔪 Backend server killed (forced)", "#FF9800")
        
        # Kill any remaining processes on ports
        self.kill_existing_processes()
        
        # Update UI
        self.status_label.config(text="⏹️ All servers stopped")
        self.stop_btn.config(state="disabled")
        self.browser_btn.config(state="disabled")
        self.log("✅ All servers stopped successfully", "#4CAF50")
    
    def on_closing(self):
        """Handle window close event"""
        if self.frontend_process or self.backend_process:
            result = messagebox.askyesno(
                "Confirm Exit", 
                "Servers are still running. Stop them before closing?"
            )
            if result:
                self.stop_servers()
                time.sleep(1)
        
        self.root.destroy()
    
    def run(self):
        """Start the launcher GUI"""
        self.log("🚀 DFS Web Optimizer Launcher ready!")
        self.log("📝 Click 'Launch Web Optimizer' to start both servers and open the UI")
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        launcher = WebOptimizerLauncher()
        launcher.run()
    except KeyboardInterrupt:
        print("\nLauncher interrupted by user")
    except Exception as e:
        # Fallback error handling if GUI fails
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
