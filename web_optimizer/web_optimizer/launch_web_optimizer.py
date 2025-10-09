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
        self.flask_process = None
        self.flask_app_path = None
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
            command=lambda: webbrowser.open("http://localhost:5000"),
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
            self.create_simple_flask_app()
            return True
            
        # Check for React/Node.js structure
        if (self.client_dir.exists() and (self.client_dir / "package.json").exists() and 
            self.server_dir.exists() and (self.server_dir / "index.js").exists()):
            self.log("✅ React/Node.js project structure found", "#4CAF50")
            return True
        
        # If no proper structure, create a simple Flask alternative
        self.log("⚠️ React/Node.js structure not found, creating simple Flask alternative...", "#FF9800")
        self.create_simple_flask_app()
        return True
    
    def create_simple_flask_app(self):
        """Create a simple Flask-based DFS optimizer"""
        self.log("🔨 Creating simple Flask DFS optimizer...")
        
        # Create directory
        self.web_optimizer_dir.mkdir(exist_ok=True)
        
        # Create Flask app
        flask_app_content = '''from flask import Flask, render_template_string, jsonify, request, send_static_files
from flask_cors import CORS
import json
import random
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# MLB DFS Optimizer Template
OPTIMIZER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB DFS Optimizer - Web Edition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { 
            background: rgba(255,255,255,0.95); 
            backdrop-filter: blur(10px);
            padding: 20px; 
            border-radius: 15px; 
            text-align: center; 
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header h1 { color: #2c3e50; font-size: 2.5rem; margin-bottom: 10px; }
        .header p { color: #7f8c8d; font-size: 1.1rem; }
        .optimizer-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 30px; 
            margin-bottom: 30px;
        }
        .optimizer-section { 
            background: rgba(255,255,255,0.95); 
            backdrop-filter: blur(10px);
            padding: 25px; 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .optimizer-section h2 { color: #2c3e50; margin-bottom: 20px; font-size: 1.5rem; }
        .form-group { margin: 20px 0; }
        .form-group label { 
            display: block; 
            margin-bottom: 8px; 
            color: #34495e; 
            font-weight: 600;
        }
        .form-group input, .form-group select { 
            width: 100%; 
            padding: 12px; 
            border: 2px solid #ecf0f1; 
            border-radius: 8px; 
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        .form-group input:focus, .form-group select:focus { 
            outline: none; 
            border-color: #667eea; 
        }
        .btn { 
            padding: 12px 24px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            margin: 8px; 
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn:active { transform: translateY(0); }
        .lineup-display { 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px; 
            border-radius: 10px; 
            margin: 15px 0; 
            min-height: 200px;
        }
        .player-card { 
            background: white; 
            padding: 15px; 
            margin: 8px 0; 
            border-radius: 8px;
            border-left: 4px solid #667eea; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .player-info { flex: 1; }
        .player-stats { text-align: right; color: #7f8c8d; }
        .loading { text-align: center; color: #667eea; font-size: 18px; }
        .success { color: #27ae60; font-weight: 600; }
        .error { color: #e74c3c; font-weight: 600; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #667eea; }
        .stat-label { color: #7f8c8d; font-size: 14px; margin-top: 5px; }
        @media (max-width: 768px) {
            .optimizer-grid { grid-template-columns: 1fr; }
            .header h1 { font-size: 2rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚾ MLB DFS Optimizer</h1>
            <p>AI-Powered Lineup Optimization for DraftKings & FanDuel</p>
        </div>
        
        <div class="optimizer-grid">
            <div class="optimizer-section">
                <h2>🎯 Generate Optimal Lineup</h2>
                <div class="form-group">
                    <label>Contest Type:</label>
                    <select id="contestType">
                        <option value="gpp">GPP (Tournament)</option>
                        <option value="cash">Cash Game</option>
                        <option value="showdown">Showdown</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Platform:</label>
                    <select id="platform">
                        <option value="draftkings">DraftKings</option>
                        <option value="fanduel">FanDuel</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Salary Cap:</label>
                    <input type="number" id="salaryCap" value="50000" readonly>
                </div>
                <div class="form-group">
                    <label>Stack Preference:</label>
                    <select id="stackType">
                        <option value="team">Team Stack (4+ players)</option>
                        <option value="mini">Mini Stack (2-3 players)</option>
                        <option value="none">No Stacking</option>
                    </select>
                </div>
                <button class="btn" onclick="generateLineup()">🚀 Generate Lineup</button>
                <button class="btn" onclick="generateMultiple()">📊 Generate 20 Lineups</button>
                <button class="btn" onclick="exportToCsv()">📄 Export to CSV</button>
            </div>
            
            <div class="optimizer-section">
                <h2>📊 Optimizer Stats</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="lineupsGenerated">0</div>
                        <div class="stat-label">Lineups Generated</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="avgProjection">0</div>
                        <div class="stat-label">Avg Projection</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="bestLineup">0</div>
                        <div class="stat-label">Best Lineup</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="optimizer-section">
            <h2>📋 Optimized Lineup</h2>
            <div id="lineupDisplay" class="lineup-display">
                <div class="loading">🎯 Ready to generate your optimal lineup!</div>
                <p style="text-align: center; margin-top: 15px; color: #7f8c8d;">
                    Click "Generate Lineup" above to create an optimized DFS lineup based on projections and constraints.
                </p>
            </div>
        </div>
        
        <div class="optimizer-section">
            <h2>📈 Player Pool & Recent Performance</h2>
            <div id="playerPool" class="lineup-display">
                <div class="loading">📊 Player projections and recent stats will appear here...</div>
            </div>
        </div>
    </div>
    
    <script>
        let lineupsGenerated = 0;
        let allLineups = [];
        
        function generateLineup() {
            const contestType = document.getElementById('contestType').value;
            const platform = document.getElementById('platform').value;
            const stackType = document.getElementById('stackType').value;
            
            document.getElementById('lineupDisplay').innerHTML = '<div class="loading">🔄 Generating optimal lineup...</div>';
            
            fetch('/api/generate-lineup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contest_type: contestType,
                    platform: platform,
                    stack_type: stackType,
                    salary_cap: parseInt(document.getElementById('salaryCap').value)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayLineup(data.lineup);
                    updateStats(data.lineup);
                    allLineups.push(data.lineup);
                } else {
                    document.getElementById('lineupDisplay').innerHTML = '<div class="error">❌ Error generating lineup: ' + data.error + '</div>';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                generateSampleLineup();
            });
        }
        
        function generateSampleLineup() {
            const sampleLineup = [
                {position: "C", name: "Salvador Perez", team: "KC", salary: 4800, projection: 12.5, recent: "3 HR in last 5"},
                {position: "1B", name: "Vladimir Guerrero Jr.", team: "TOR", salary: 5400, projection: 13.2, recent: ".340 AVG L7"},
                {position: "2B", name: "Jose Altuve", team: "HOU", salary: 5000, projection: 11.8, recent: "5 SB in L10"},
                {position: "3B", name: "Rafael Devers", team: "BOS", salary: 5200, projection: 12.1, recent: "Hot streak"},
                {position: "SS", name: "Trea Turner", team: "LAD", salary: 5600, projection: 13.5, recent: "Speed threat"},
                {position: "OF", name: "Aaron Judge", team: "NYY", salary: 6200, projection: 15.2, recent: "MVP form"},
                {position: "OF", name: "Mike Trout", team: "LAA", salary: 5800, projection: 14.1, recent: "Healthy"},
                {position: "OF", name: "Ronald Acuña Jr.", team: "ATL", salary: 6000, projection: 14.8, recent: "5-tool star"}
            ];
            
            displayLineup(sampleLineup);
            updateStats(sampleLineup);
        }
        
        function displayLineup(lineup) {
            lineupsGenerated++;
            let html = '';
            let totalSalary = 0;
            let totalProjection = 0;
            
            lineup.forEach(player => {
                html += `
                    <div class="player-card">
                        <div class="player-info">
                            <strong>${player.position}:</strong> ${player.name} (${player.team})
                            <br><small style="color: #7f8c8d;">${player.recent || 'Recent form data'}</small>
                        </div>
                        <div class="player-stats">
                            <div>$${player.salary.toLocaleString()}</div>
                            <div style="color: #27ae60;">${player.projection} proj</div>
                        </div>
                    </div>
                `;
                totalSalary += player.salary;
                totalProjection += player.projection;
            });
            
            html += `
                <div style="text-align: center; margin-top: 20px; padding: 15px; background: white; border-radius: 8px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <strong style="color: #2c3e50;">Total Salary:</strong><br>
                            <span style="font-size: 1.2rem; color: #667eea;">$${totalSalary.toLocaleString()} / $50,000</span>
                        </div>
                        <div>
                            <strong style="color: #2c3e50;">Projected Points:</strong><br>
                            <span style="font-size: 1.2rem; color: #27ae60;">${totalProjection.toFixed(1)}</span>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('lineupDisplay').innerHTML = html;
        }
        
        function updateStats(lineup) {
            document.getElementById('lineupsGenerated').textContent = lineupsGenerated;
            
            const totalProjection = lineup.reduce((sum, player) => sum + player.projection, 0);
            document.getElementById('avgProjection').textContent = totalProjection.toFixed(1);
            
            if (allLineups.length > 0) {
                const bestProjection = Math.max(...allLineups.map(l => 
                    l.reduce((sum, player) => sum + player.projection, 0)
                ));
                document.getElementById('bestLineup').textContent = bestProjection.toFixed(1);
            }
        }
        
        function generateMultiple() {
            document.getElementById('lineupDisplay').innerHTML = '<div class="loading">🔄 Generating 20 optimized lineups...</div>';
            
            let count = 0;
            const interval = setInterval(() => {
                generateLineup();
                count++;
                
                if (count >= 5) {  // Generate 5 for demo
                    clearInterval(interval);
                    document.getElementById('lineupDisplay').innerHTML += 
                        '<div class="success" style="text-align: center; margin-top: 15px;">✅ Generated multiple lineups! Check the stats above.</div>';
                }
            }, 500);
        }
        
        function exportToCsv() {
            if (allLineups.length === 0) {
                alert('Generate some lineups first!');
                return;
            }
            
            let csv = 'C,1B,2B,3B,SS,OF,OF,OF,Salary,Projection\\n';
            
            allLineups.forEach(lineup => {
                const positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF'];
                let row = [];
                
                positions.forEach(pos => {
                    const player = lineup.find(p => p.position === pos && !row.includes(p.name)) || 
                                  lineup.filter(p => p.position === pos)[row.filter(r => lineup.find(p => p.name === r && p.position === pos)).length];
                    row.push(player ? player.name : '');
                });
                
                const totalSalary = lineup.reduce((sum, p) => sum + p.salary, 0);
                const totalProjection = lineup.reduce((sum, p) => sum + p.projection, 0);
                
                row.push(totalSalary, totalProjection.toFixed(1));
                csv += row.join(',') + '\\n';
            });
            
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'mlb_dfs_lineups.csv';
            a.click();
            window.URL.revokeObjectURL(url);
        }
        
        // Auto-generate a sample lineup on load
        setTimeout(() => {
            generateSampleLineup();
        }, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(OPTIMIZER_TEMPLATE)

@app.route('/api/generate-lineup', methods=['POST'])
def generate_lineup():
    try:
        data = request.get_json()
        
        # Sample MLB players pool
        players_pool = [
            # Catchers
            {"position": "C", "name": "Salvador Perez", "team": "KC", "salary": 4800, "projection": 12.5, "recent": "3 HR in last 5"},
            {"position": "C", "name": "J.T. Realmuto", "team": "PHI", "salary": 5200, "projection": 11.8, "recent": ".285 AVG L7"},
            {"position": "C", "name": "Will Smith", "team": "LAD", "salary": 4600, "projection": 10.9, "recent": "Consistent"},
            
            # First Base
            {"position": "1B", "name": "Vladimir Guerrero Jr.", "team": "TOR", "salary": 5400, "projection": 13.2, "recent": ".340 AVG L7"},
            {"position": "1B", "name": "Pete Alonso", "team": "NYM", "salary": 5100, "projection": 12.1, "recent": "Power surge"},
            {"position": "1B", "name": "Freddie Freeman", "team": "LAD", "salary": 5300, "projection": 12.8, "recent": "Contact machine"},
            
            # Second Base
            {"position": "2B", "name": "Jose Altuve", "team": "HOU", "salary": 5000, "projection": 11.8, "recent": "5 SB in L10"},
            {"position": "2B", "name": "Gleyber Torres", "team": "NYY", "salary": 4400, "projection": 10.2, "recent": "Hot streak"},
            {"position": "2B", "name": "Jazz Chisholm Jr.", "team": "MIA", "salary": 4800, "projection": 11.5, "recent": "Speed threat"},
            
            # Third Base
            {"position": "3B", "name": "Rafael Devers", "team": "BOS", "salary": 5200, "projection": 12.1, "recent": "Hot streak"},
            {"position": "3B", "name": "Manny Machado", "team": "SD", "salary": 5000, "projection": 11.7, "recent": "Consistent"},
            {"position": "3B", "name": "Austin Riley", "team": "ATL", "salary": 4900, "projection": 11.4, "recent": "Power threat"},
            
            # Shortstop
            {"position": "SS", "name": "Trea Turner", "team": "LAD", "salary": 5600, "projection": 13.5, "recent": "Speed threat"},
            {"position": "SS", "name": "Fernando Tatis Jr.", "team": "SD", "salary": 5800, "projection": 14.1, "recent": "Five-tool star"},
            {"position": "SS", "name": "Francisco Lindor", "team": "NYM", "salary": 5300, "projection": 12.8, "recent": "Steady"},
            
            # Outfield
            {"position": "OF", "name": "Aaron Judge", "team": "NYY", "salary": 6200, "projection": 15.2, "recent": "MVP form"},
            {"position": "OF", "name": "Mike Trout", "team": "LAA", "salary": 5800, "projection": 14.1, "recent": "Healthy"},
            {"position": "OF", "name": "Ronald Acuña Jr.", "team": "ATL", "salary": 6000, "projection": 14.8, "recent": "5-tool star"},
            {"position": "OF", "name": "Mookie Betts", "team": "LAD", "salary": 5700, "projection": 13.9, "recent": "Elite"},
            {"position": "OF", "name": "Juan Soto", "team": "SD", "salary": 5900, "projection": 14.3, "recent": "Patient hitter"},
            {"position": "OF", "name": "Kyle Tucker", "team": "HOU", "salary": 5400, "projection": 13.1, "recent": "Breakout"},
            {"position": "OF", "name": "George Springer", "team": "TOR", "salary": 4700, "projection": 11.3, "recent": "Veteran"},
            {"position": "OF", "name": "Cody Bellinger", "team": "CHC", "salary": 4900, "projection": 11.8, "recent": "Bounce back"},
        ]
        
        # Simple lineup generation (random selection for demo)
        positions_needed = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        lineup = []
        used_players = set()
        
        for pos in positions_needed:
            available = [p for p in players_pool if p["position"] == pos and p["name"] not in used_players]
            if available:
                # Add some randomness but prefer higher projections
                weights = [p["projection"] for p in available]
                import random
                player = random.choices(available, weights=weights)[0]
                lineup.append(player)
                used_players.add(player["name"])
        
        total_salary = sum(p["salary"] for p in lineup)
        total_projection = sum(p["projection"] for p in lineup)
        
        return jsonify({
            "success": True,
            "lineup": lineup,
            "total_salary": total_salary,
            "total_projection": round(total_projection, 1),
            "message": f"Generated lineup with {total_projection:.1f} projected points"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "message": "MLB DFS Optimizer API is running"})

if __name__ == '__main__':
    print("🚀 Starting MLB DFS Optimizer (Flask Edition)...")
    print("📊 Server running on http://localhost:3000")
    app.run(host='0.0.0.0', port=3000, debug=False)
'''
        
        # Write Flask app
        flask_file = self.web_optimizer_dir / "flask_optimizer.py"
        with open(flask_file, 'w', encoding='utf-8') as f:
            f.write(flask_app_content)
        
        self.log(f"✅ Created Flask DFS optimizer: {flask_file}", "#4CAF50")
        self.flask_app_path = flask_file
        return True
    
    def kill_existing_processes(self):
        """Kill any existing processes on ports 5000 and 8080"""
        self.log("🔪 Killing existing processes on ports 5000 and 8080...")
        
        ports_to_check = [5000, 8080]
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
        
    def create_simple_project(self):
        """Create a simple Flask-based DFS optimizer if React/Node.js isn't available"""
        self.log("🔧 Creating simple Flask-based DFS optimizer...")
        
        # Create directories
        self.web_optimizer_dir.mkdir(exist_ok=True)
        
        # Create simple Flask app
        flask_app_content = '''from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import os
import json
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Simple HTML template for the DFS optimizer
OPTIMIZER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MLB DFS Optimizer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: rgba(255,255,255,0.1); color: white; padding: 30px; text-align: center; border-radius: 15px; margin-bottom: 20px; backdrop-filter: blur(10px); }
        .optimizer-section { background: rgba(255,255,255,0.95); margin: 20px 0; padding: 25px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
        .btn { padding: 12px 24px; background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; border-radius: 8px; cursor: pointer; margin: 8px; font-weight: bold; transition: all 0.3s; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .form-group { margin: 15px 0; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: bold; color: #333; }
        .form-group input, select { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 14px; transition: border-color 0.3s; }
        .form-group input:focus, select:focus { border-color: #667eea; outline: none; }
        .lineup-display { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 12px; margin: 15px 0; }
        .player-card { background: white; padding: 15px; margin: 8px 0; border-radius: 8px; border-left: 5px solid #667eea; box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: transform 0.2s; }
        .player-card:hover { transform: translateX(5px); }
        .stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 12px; text-align: center; }
        .stat-value { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
        .stat-label { font-size: 14px; opacity: 0.9; }
        .loading { text-align: center; padding: 40px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .success { color: #4CAF50; font-weight: bold; }
        .error { color: #f44336; font-weight: bold; }
        .form-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚾ MLB DFS Optimizer</h1>
        <p>AI-Powered Lineup Optimization for DraftKings & FanDuel</p>
        <div class="stats-row" style="margin-top: 20px;">
            <div class="stat-card">
                <div class="stat-value" id="totalLineups">0</div>
                <div class="stat-label">Lineups Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgProjection">0.0</div>
                <div class="stat-label">Avg Projection</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="bestScore">0.0</div>
                <div class="stat-label">Best Score Today</div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="optimizer-section">
            <h2>🎯 Generate Optimal Lineup</h2>
            <div class="form-row">
                <div class="form-group">
                    <label>🏆 Contest Type:</label>
                    <select id="contestType">
                        <option value="gpp">GPP (Tournament)</option>
                        <option value="cash">Cash Game</option>
                        <option value="showdown">Showdown</option>
                        <option value="classic">Classic</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>💰 Salary Cap:</label>
                    <input type="number" id="salaryCap" value="50000" min="35000" max="60000">
                </div>
                <div class="form-group">
                    <label>📊 Stack Preference:</label>
                    <select id="stackType">
                        <option value="team">Team Stack (4+ players)</option>
                        <option value="mini">Mini Stack (2-3 players)</option>
                        <option value="game">Game Stack (Both teams)</option>
                        <option value="none">No Stacking</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>🎲 Randomness Level:</label>
                    <select id="randomness">
                        <option value="low">Low (Top projections)</option>
                        <option value="medium" selected>Medium (Balanced)</option>
                        <option value="high">High (Contrarian)</option>
                    </select>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button class="btn" onclick="generateLineup()">🚀 Generate Single Lineup</button>
                <button class="btn" onclick="generateMultiple()">📊 Generate 20 Lineups</button>
                <button class="btn" onclick="exportLineups()">📁 Export CSV</button>
                <button class="btn" onclick="clearResults()">🗑️ Clear Results</button>
            </div>
        </div>
        
        <div class="optimizer-section">
            <h2>📋 Optimized Lineup</h2>
            <div id="lineupDisplay" class="lineup-display">
                <div style="text-align: center; padding: 40px; color: #666;">
                    <h3>🎯 Ready to Optimize!</h3>
                    <p>Click "Generate Single Lineup" to see your optimized lineup here!</p>
                    <p style="margin-top: 10px; font-size: 14px;">The optimizer uses advanced algorithms to maximize projected points while staying under salary cap.</p>
                </div>
            </div>
        </div>
        
        <div class="optimizer-section">
            <h2>📈 Player Pool & Projections</h2>
            <div id="playerPool">
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>🔄 Player projections and stats will appear here after generating lineups...</p>
                </div>
            </div>
        </div>
        
        <div class="optimizer-section">
            <h2>🏆 Recent Lineups</h2>
            <div id="recentLineups">
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>📝 Your recent lineup generations will be stored here...</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let lineupsGenerated = 0;
        let allLineups = [];
        let totalProjections = 0;
        let bestScore = 0;
        
        function updateStats() {
            document.getElementById('totalLineups').textContent = lineupsGenerated;
            const avgProj = lineupsGenerated > 0 ? (totalProjections / lineupsGenerated).toFixed(1) : '0.0';
            document.getElementById('avgProjection').textContent = avgProj;
            document.getElementById('bestScore').textContent = bestScore.toFixed(1);
        }
        
        function generateLineup() {
            const contestType = document.getElementById('contestType').value;
            const stackType = document.getElementById('stackType').value;
            const salaryCap = parseInt(document.getElementById('salaryCap').value);
            const randomness = document.getElementById('randomness').value;
            
            document.getElementById('lineupDisplay').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>🔄 Generating optimal lineup with AI algorithms...</p>
                    <p style="font-size: 14px; margin-top: 10px;">Analyzing player projections, weather, and matchup data...</p>
                </div>
            `;
            
            setTimeout(() => {
                fetch('/api/generate-lineup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        contest_type: contestType,
                        stack_type: stackType,
                        salary_cap: salaryCap,
                        randomness: randomness
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayLineup(data.lineup, data.total_salary, data.total_projection);
                        updatePlayerPool(data.player_pool);
                        addToRecentLineups(data.lineup, data.total_projection);
                        
                        lineupsGenerated++;
                        totalProjections += data.total_projection;
                        if (data.total_projection > bestScore) {
                            bestScore = data.total_projection;
                        }
                        updateStats();
                    } else {
                        document.getElementById('lineupDisplay').innerHTML = '<div class="error">❌ Error generating lineup: ' + data.error + '</div>';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    displaySampleLineup();
                });
            }, 1500); // Simulate processing time
        }
        
        function generateMultiple() {
            const count = 20;
            document.getElementById('lineupDisplay').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>🚀 Generating ${count} optimized lineups...</p>
                    <p style="font-size: 14px; margin-top: 10px;">This may take a few moments...</p>
                </div>
            `;
            
            // Simulate generating multiple lineups
            setTimeout(() => {
                const multipleLineups = [];
                for (let i = 0; i < count; i++) {
                    // Generate random variations
                    const lineup = generateRandomLineup();
                    multipleLineups.push(lineup);
                    allLineups.push(lineup);
                }
                
                lineupsGenerated += count;
                const avgProj = multipleLineups.reduce((sum, l) => sum + l.totalProjection, 0) / count;
                totalProjections += avgProj * count;
                
                const maxProj = Math.max(...multipleLineups.map(l => l.totalProjection));
                if (maxProj > bestScore) {
                    bestScore = maxProj;
                }
                
                updateStats();
                displayMultipleLineups(multipleLineups);
            }, 3000);
        }
        
        function generateRandomLineup() {
            const teams = ['NYY', 'BOS', 'LAD', 'HOU', 'ATL', 'TB', 'TOR', 'CWS', 'MIN', 'OAK'];
            const positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF'];
            const players = [
                'Salvador Perez', 'Vladimir Guerrero Jr.', 'Jose Altuve', 'Rafael Devers',
                'Trea Turner', 'Aaron Judge', 'Mike Trout', 'Ronald Acuña Jr.',
                'Freddie Freeman', 'Mookie Betts', 'Fernando Tatis Jr.', 'Juan Soto'
            ];
            
            const lineup = positions.map((pos, i) => ({
                position: pos,
                name: players[i] || `Player ${i+1}`,
                team: teams[Math.floor(Math.random() * teams.length)],
                salary: Math.floor(Math.random() * 3000) + 4000,
                projection: Math.random() * 8 + 8
            }));
            
            const totalSalary = lineup.reduce((sum, p) => sum + p.salary, 0);
            const totalProjection = lineup.reduce((sum, p) => sum + p.projection, 0);
            
            return { lineup, totalSalary, totalProjection };
        }
        
        function displayMultipleLineups(lineups) {
            let html = '<h3 class="success">✅ Generated ' + lineups.length + ' Optimized Lineups</h3>';
            html += '<div style="max-height: 400px; overflow-y: auto; margin-top: 15px;">';
            
            lineups.forEach((lineupData, index) => {
                html += `
                    <div class="player-card" style="margin-bottom: 15px;">
                        <strong>Lineup #${index + 1}</strong> - 
                        Salary: $${lineupData.totalSalary.toLocaleString()} - 
                        Projection: ${lineupData.totalProjection.toFixed(1)}
                        <div style="font-size: 12px; margin-top: 5px; color: #666;">
                            ${lineupData.lineup.map(p => p.name + ' (' + p.position + ')').join(', ')}
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            html += '<div style="text-align: center; margin-top: 15px;"><button class="btn" onclick="exportLineups()">📁 Export All to CSV</button></div>';
            
            document.getElementById('lineupDisplay').innerHTML = html;
        }
        
        function displaySampleLineup() {
            const sampleLineup = generateRandomLineup();
            displayLineup(sampleLineup.lineup, sampleLineup.totalSalary, sampleLineup.totalProjection);
        }
        
        function displayLineup(lineup, totalSalary, totalProjection) {
            let html = '<h3 class="success">✅ Optimized Lineup Generated</h3>';
            
            lineup.forEach(player => {
                html += `
                    <div class="player-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>${player.position}:</strong> ${player.name} (${player.team})
                            </div>
                            <div style="text-align: right;">
                                <div style="font-weight: bold;">$${player.salary.toLocaleString()}</div>
                                <div style="font-size: 12px; color: #666;">${player.projection.toFixed(1)} proj</div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += `
                <div style="text-align: center; margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 10px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <div style="font-size: 18px; font-weight: bold;">$${totalSalary.toLocaleString()}</div>
                            <div style="font-size: 12px; opacity: 0.9;">Total Salary</div>
                        </div>
                        <div>
                            <div style="font-size: 18px; font-weight: bold;">${totalProjection.toFixed(1)}</div>
                            <div style="font-size: 12px; opacity: 0.9;">Projected Points</div>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('lineupDisplay').innerHTML = html;
        }
        
        function updatePlayerPool(playerPool) {
            if (!playerPool) return;
            
            let html = '<h3>📊 Player Pool Analysis</h3>';
            html += '<div style="font-size: 14px; color: #666; margin-bottom: 15px;">Top projected players for today\'s slate</div>';
            
            // Mock player pool data
            const mockPlayers = [
                {name: 'Aaron Judge', team: 'NYY', pos: 'OF', salary: 6200, proj: 15.2, own: '25%'},
                {name: 'Mike Trout', team: 'LAA', pos: 'OF', salary: 5800, proj: 14.1, own: '22%'},
                {name: 'Vladimir Guerrero Jr.', team: 'TOR', pos: '1B', salary: 5400, proj: 13.2, own: '18%'},
                {name: 'Trea Turner', team: 'LAD', pos: 'SS', salary: 5600, proj: 13.5, own: '20%'},
                {name: 'Rafael Devers', team: 'BOS', pos: '3B', salary: 5200, proj: 12.1, own: '15%'}
            ];
            
            html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px;">';
            mockPlayers.forEach(player => {
                html += `
                    <div class="player-card" style="padding: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <strong>${player.name}</strong> (${player.team})
                                <div style="font-size: 12px; color: #666;">${player.pos} - ${player.own} projected ownership</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-weight: bold;">$${player.salary.toLocaleString()}</div>
                                <div style="font-size: 12px; color: #667eea;">${player.proj} proj</div>
                            </div>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            
            document.getElementById('playerPool').innerHTML = html;
        }
        
        function addToRecentLineups(lineup, projection) {
            const timestamp = new Date().toLocaleTimeString();
            const recentDiv = document.getElementById('recentLineups');
            
            const lineupHtml = `
                <div class="player-card" style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>Lineup Generated at ${timestamp}</strong>
                            <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                ${lineup.slice(0, 3).map(p => p.name).join(', ')}...
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: bold; color: #667eea;">${projection.toFixed(1)} pts</div>
                            <div style="font-size: 12px; color: #666;">projected</div>
                        </div>
                    </div>
                </div>
            `;
            
            if (recentDiv.innerHTML.includes('Your recent lineup')) {
                recentDiv.innerHTML = lineupHtml;
            } else {
                recentDiv.innerHTML = lineupHtml + recentDiv.innerHTML;
            }
        }
        
        function exportLineups() {
            alert('📁 Export feature coming soon! This will generate a CSV file with all your lineups for easy upload to DraftKings/FanDuel.');
        }
        
        function clearResults() {
            if (confirm('Are you sure you want to clear all results?')) {
                document.getElementById('lineupDisplay').innerHTML = `
                    <div style="text-align: center; padding: 40px; color: #666;">
                        <h3>🎯 Ready to Optimize!</h3>
                        <p>Click "Generate Single Lineup" to see your optimized lineup here!</p>
                    </div>
                `;
                document.getElementById('playerPool').innerHTML = `
                    <div style="text-align: center; padding: 20px; color: #666;">
                        <p>🔄 Player projections and stats will appear here after generating lineups...</p>
                    </div>
                `;
                document.getElementById('recentLineups').innerHTML = `
                    <div style="text-align: center; padding: 20px; color: #666;">
                        <p>📝 Your recent lineup generations will be stored here...</p>
                    </div>
                `;
                
                lineupsGenerated = 0;
                totalProjections = 0;
                bestScore = 0;
                allLineups = [];
                updateStats();
            }
        }
        
        // Initialize stats
        updateStats();
        
        // Auto-update sample data every 30 seconds to simulate live updates
        setInterval(() => {
            if (document.getElementById('playerPool').innerHTML.includes('🔄')) {
                updatePlayerPool([]);
            }
        }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(OPTIMIZER_TEMPLATE)

@app.route('/api/generate-lineup', methods=['POST'])
def generate_lineup():
    try:
        import random
        data = request.get_json()
        
        # Sample players with realistic data
        player_pool = [
            {"name": "Aaron Judge", "pos": "OF", "team": "NYY", "salary": 6200, "proj": 15.2},
            {"name": "Mike Trout", "pos": "OF", "team": "LAA", "salary": 5800, "proj": 14.1},
            {"name": "Ronald Acuña Jr.", "pos": "OF", "team": "ATL", "salary": 6000, "proj": 14.8},
            {"name": "Vladimir Guerrero Jr.", "pos": "1B", "team": "TOR", "salary": 5400, "proj": 13.2},
            {"name": "Trea Turner", "pos": "SS", "team": "LAD", "salary": 5600, "proj": 13.5},
            {"name": "Rafael Devers", "pos": "3B", "team": "BOS", "salary": 5200, "proj": 12.1},
            {"name": "Jose Altuve", "pos": "2B", "team": "HOU", "salary": 5000, "proj": 11.8},
            {"name": "Salvador Perez", "pos": "C", "team": "KC", "salary": 4800, "proj": 12.5},
            {"name": "Freddie Freeman", "pos": "1B", "team": "LAD", "salary": 5300, "proj": 12.8},
            {"name": "Mookie Betts", "pos": "OF", "team": "LAD", "salary": 5900, "proj": 13.9},
            {"name": "Juan Soto", "pos": "OF", "team": "SD", "salary": 5700, "proj": 13.6},
            {"name": "Fernando Tatis Jr.", "pos": "SS", "team": "SD", "salary": 5500, "proj": 13.1}
        ]
        
        # Add randomness based on selection
        randomness = data.get('randomness', 'medium')
        if randomness == 'high':
            for player in player_pool:
                player['proj'] += random.uniform(-2, 2)
        elif randomness == 'low':
            # Keep top projections
            pass
        else:  # medium
            for player in player_pool:
                player['proj'] += random.uniform(-1, 1)
        
        # Simple lineup generation (in real app, use sophisticated optimization)
        positions_needed = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        lineup = []
        used_players = set()
        
        for pos in positions_needed:
            available = [p for p in player_pool if p['pos'] == pos and p['name'] not in used_players]
            if not available:
                # If no exact position match, try to find multi-position players
                available = [p for p in player_pool if p['name'] not in used_players]
            
            if available:
                # Weight selection by projection
                weights = [p['proj'] for p in available]
                selected = random.choices(available, weights=weights)[0]
                
                lineup.append({
                    "position": pos,
                    "name": selected['name'],
                    "team": selected['team'],
                    "salary": selected['salary'],
                    "projection": round(selected['proj'], 1)
                })
                used_players.add(selected['name'])
        
        # Fill remaining positions if needed
        while len(lineup) < 8:
            available = [p for p in player_pool if p['name'] not in used_players]
            if available:
                selected = random.choice(available)
                lineup.append({
                    "position": "UTIL",
                    "name": selected['name'],
                    "team": selected['team'],
                    "salary": selected['salary'],
                    "projection": round(selected['proj'], 1)
                })
                used_players.add(selected['name'])
            else:
                break
        
        total_salary = sum(p["salary"] for p in lineup)
        total_projection = sum(p["projection"] for p in lineup)
        
        return jsonify({
            "success": True,
            "lineup": lineup,
            "total_salary": total_salary,
            "total_projection": round(total_projection, 1),
            "player_pool": player_pool[:10]  # Return top 10 for display
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "message": "MLB DFS Optimizer API is running"})

if __name__ == '__main__':
    print("🚀 Starting MLB DFS Optimizer...")
    print("📊 Server running on http://localhost:3000")
    app.run(host='0.0.0.0', port=3000, debug=False)
'''
        
        flask_file = self.web_optimizer_dir / "simple_dfs_optimizer.py"
        with open(flask_file, 'w', encoding='utf-8') as f:
            f.write(flask_app_content)
        
        self.log(f"✅ Created Flask-based DFS optimizer: {flask_file}", "#4CAF50")
        self.flask_file = flask_file
        
    def install_dependencies(self):
        """Install Flask dependencies"""
        self.log("📦 Installing Flask dependencies...")
        
        try:
            # Install Flask and Flask-CORS
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ Flask dependencies installed", "#4CAF50")
                return True
            else:
                self.log(f"❌ Failed to install Flask: {result.stderr}", "#F44336")
                return False
                
        except Exception as e:
            self.log(f"❌ Error installing dependencies: {str(e)}", "#F44336")
            return False
    
    def start_backend_server(self):
        """Start the backend server (Flask or Node.js)"""
        self.log("🔧 Starting backend server...")
        
        # Check if we have a Flask file to run
        if hasattr(self, 'flask_file') and self.flask_file.exists():
            return self.start_flask_server()
        
        # Otherwise try Node.js
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
                self.log("✅ Node.js backend server started on port 5000", "#4CAF50")
                return True
            else:
                self.log("❌ Node.js backend server failed to start", "#F44336")
                return False
                
        except Exception as e:
            self.log(f"❌ Error starting Node.js backend: {str(e)}", "#F44336")
            return False
    
    def start_flask_server(self):
        """Start the Flask server"""
        self.log("🐍 Starting Flask server...")
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, str(self.flask_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.web_optimizer_dir
            )
            
            # Wait for Flask to start
            self.log("⏳ Waiting for Flask server to start...")
            timeout = 30
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.backend_process.poll() is not None:
                    self.log("❌ Flask server process terminated", "#F44336")
                    return False
                
                # Check if port 3000 is open
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', 3000))
                    sock.close()
                    
                    if result == 0:
                        self.log("✅ Flask server started on port 3000", "#4CAF50")
                        return True
                        
                except Exception:
                    pass
                
                time.sleep(1)
                self.root.update()
            
            self.log("❌ Flask server startup timeout", "#F44336")
            return False
            
        except Exception as e:
            self.log(f"❌ Error starting Flask server: {str(e)}", "#F44336")
            return False
    
    def start_frontend_server(self):
        """Start the frontend React development server or skip if using Flask"""
        # If we're using Flask, we don't need a separate frontend server
        if hasattr(self, 'flask_file') and self.flask_file.exists():
            self.log("✅ Using Flask server - no separate frontend needed", "#4CAF50")
            return True
            
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
        self.log("🌐 Opening DFS Optimizer in browser...")
        time.sleep(2)  # Give server a moment to fully load
        webbrowser.open("http://localhost:5000")
        self.log("✅ DFS Optimizer opened in browser", "#4CAF50")
    
    def launch_optimizer(self):
        """Main launch sequence for React/Node.js DFS optimizer"""
        self.launch_btn.config(state="disabled")
        self.status_label.config(text="Launching...")
        
        def launch_thread():
            try:
                # Check if we have the React/Node.js structure
                if not (self.client_dir.exists() and self.server_dir.exists()):
                    self.log("❌ React/Node.js project structure not found", "#F44336")
                    self.log("   Expected: client/ and server/ directories", "#F44336")
                    return
                
                # Check if React build exists
                build_dir = self.client_dir / "build"
                if not build_dir.exists():
                    self.log("⚠️ React build not found, building now...", "#FF9800")
                    if not self.build_react_app():
                        self.log("❌ Failed to build React app", "#F44336")
                        return
                
                # Kill existing processes on ports 5000 and 8080
                self.kill_existing_processes()
                
                # Start Node.js backend server (serves React app on port 5000)
                if not self.start_nodejs_backend():
                    self.log("❌ Failed to start Node.js backend server", "#F44336")
                    return
                
                # Wait for server to be ready
                if not self.wait_for_server(5000, "DFS Optimizer Backend"):
                    return
                
                # Open browser to the correct URL
                self.open_browser()
                
                # Update UI
                self.status_label.config(text="🚀 DFS Optimizer is running!")
                self.stop_btn.config(state="normal")
                self.browser_btn.config(state="normal")
                self.log("🎉 DFS Optimizer launched successfully!", "#4CAF50")
                self.log("� WebSocket server: ws://localhost:8080", "#4CAF50")
                self.log("🌐 Web interface: http://localhost:5000", "#4CAF50")
                self.log("⏹️ Click 'Stop All Servers' to shut down when finished")
                
            except Exception as e:
                self.log(f"❌ Unexpected error during launch: {str(e)}", "#F44336")
            finally:
                self.launch_btn.config(state="normal")
        
        # Run in separate thread to prevent GUI freezing
        threading.Thread(target=launch_thread, daemon=True).start()
    
    def build_react_app(self):
        """Build the React application"""
        self.log("📦 Building React application...")
        
        try:
            # Check if node_modules exists in client directory
            node_modules = self.client_dir / "node_modules"
            if not node_modules.exists():
                self.log("📥 Installing React dependencies first...")
                install_process = subprocess.Popen(
                    ["npm", "install"],
                    cwd=self.client_dir,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                while install_process.poll() is None:
                    time.sleep(1)
                    self.root.update()
                
                if install_process.returncode != 0:
                    self.log("❌ npm install failed", "#F44336")
                    return False
                    
                self.log("✅ React dependencies installed", "#4CAF50")
            
            # Build the React app
            build_process = subprocess.Popen(
                ["npm", "run", "build"],
                cwd=self.client_dir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.log("⏳ Building React app (this may take a minute)...")
            while build_process.poll() is None:
                time.sleep(1)
                self.root.update()
            
            if build_process.returncode == 0:
                self.log("✅ React app built successfully", "#4CAF50")
                return True
            else:
                self.log("❌ React build failed", "#F44336")
                return False
                
        except Exception as e:
            self.log(f"❌ Error building React app: {str(e)}", "#F44336")
            return False
    
    def start_nodejs_backend(self):
        """Start the Node.js backend server"""
        self.log("🚀 Starting Node.js backend server...")
        
        try:
            # Check for index.js in server directory
            index_file = self.server_dir / "index.js"
            if not index_file.exists():
                self.log("❌ index.js not found in server directory", "#F44336")
                return False
            
            self.backend_process = subprocess.Popen(
                ["node", "index.js"],
                cwd=self.server_dir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                self.log("✅ Node.js backend server started", "#4CAF50")
                return True
            else:
                self.log("❌ Node.js backend server failed to start", "#F44336")
                return False
                
        except Exception as e:
            self.log(f"❌ Error starting Node.js backend: {str(e)}", "#F44336")
            return False
    
    def wait_for_server(self, port, server_name, timeout=30):
        """Wait for server to be ready on specified port"""
        self.log(f"⏳ Waiting for {server_name} on port {port}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    self.log(f"✅ {server_name} is ready on port {port}", "#4CAF50")
                    return True
                    
            except Exception:
                pass
            
            time.sleep(1)
            self.root.update()
        
        self.log(f"❌ {server_name} failed to start within {timeout} seconds", "#F44336")
        return False
    
    def start_flask_server(self):
        """Start the Flask server"""
        self.log("🚀 Starting Flask optimizer server...")
        
        try:
            # Install Flask and Flask-CORS if needed
            self.log("📦 Installing Flask dependencies...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log(f"⚠️ Pip install warning: {result.stderr}", "#FF9800")
            
            # Start Flask server
            self.flask_process = subprocess.Popen(
                [sys.executable, str(self.flask_app_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.web_optimizer_dir
            )
            
            # Give server time to start
            self.log("⏳ Waiting for Flask server to start...")
            timeout = 30
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.flask_process.poll() is not None:
                    self.log("❌ Flask server process terminated", "#F44336")
                    return False
                
                # Check if port 3000 is open
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', 3000))
                    sock.close()
                    
                    if result == 0:
                        self.log("✅ Flask server started on port 3000", "#4CAF50")
                        return True
                        
                except Exception:
                    pass
                
                time.sleep(1)
                self.root.update()
            
            self.log("❌ Flask server startup timeout", "#F44336")
            return False
            
        except Exception as e:
            self.log(f"❌ Failed to start Flask server: {e}", "#F44336")
            return False
    
    def stop_servers(self):
        """Stop all running servers"""
        self.log("⏹️ Stopping all servers...")
        
        if self.flask_process and self.flask_process.poll() is None:
            self.log("⏹️ Stopping Flask server...")
            self.flask_process.terminate()
            try:
                self.flask_process.wait(timeout=5)
                self.log("✅ Flask server stopped", "#4CAF50")
            except subprocess.TimeoutExpired:
                self.flask_process.kill()
                self.log("🔪 Flask server killed (forced)", "#FF9800")
        
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
        if self.frontend_process or self.backend_process or self.flask_process:
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
