# Frontend-Backend Integration Guide
## Connecting Your Custom Frontend to Existing Backend Algorithms

---

## Purpose

This guide explains how to integrate your **custom frontend design** with the **existing backend optimization algorithms** in `optimizer.genetic.algo.py`.

**Your Situation:**
- ✅ Backend algorithms exist and work (optimizer.genetic.algo.py - 6,318 lines)
- ✅ You have a frontend design you want to implement
- 🎯 Need to connect them properly

---

## Backend API Surface

### What the Backend Provides

The existing `optimizer.genetic.algo.py` has these key components you can use:

#### 1. Core Classes

```python
# Main application class
class FantasyBaseballApp(QMainWindow)
  ├─ __init__()                    # Initialize app
  ├─ load_players(file_path)       # Load CSV data
  ├─ run_optimization()            # Start optimization
  ├─ display_results(results)      # Handle results
  ├─ save_csv()                    # Export lineups
  └─ collect_team_selections()     # Get user selections

# Background worker
class OptimizationWorker(QThread)
  ├─ __init__(df_players, settings...)  # Configure worker
  ├─ run()                              # Execute optimization
  ├─ optimize_lineups()                 # Main optimization logic
  └─ optimization_done (signal)         # Results signal

# Genetic diversity engine
class GeneticDiversityEngine
  ├─ create_diverse_lineups()     # Generate diverse lineups
  ├─ evolve_population()          # Genetic algorithm
  └─ select_diverse_subset()      # Pick best diverse set
```

#### 2. Key Functions

```python
# Single lineup optimization (core algorithm)
def optimize_single_lineup(args):
    """
    Args: (df, stack_type, team_projected_runs, team_selections, min_salary)
    Returns: (lineup_df, stack_type)
    
    Uses PuLP to solve linear programming problem
    """
    
# Simulation for diversity
def simulate_iteration(df):
    """Adds random variation to projections"""
```

---

## Integration Architecture

### Option 1: Keep PyQt5, Restyle UI

**Your frontend:** Modified PyQt5 with custom styling

```python
# Your custom frontend
class YourCustomApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_your_custom_ui()
        
        # Import and use existing backend
        from optimizer.genetic.algo import OptimizationWorker
        
    def run_optimization_clicked(self):
        # Collect settings from YOUR UI
        settings = self.get_settings_from_your_ui()
        
        # Use existing worker
        self.worker = OptimizationWorker(
            df_players=self.df_players,
            num_lineups=settings['num_lineups'],
            team_selections=settings['team_selections'],
            # ... other settings
        )
        
        # Connect to YOUR result handler
        self.worker.optimization_done.connect(self.handle_results)
        self.worker.start()
    
    def handle_results(self, results, team_exp, stack_exp):
        # Display in YOUR custom UI
        self.display_in_your_results_panel(results)
```

### Option 2: Web Frontend (React/Vue/etc)

**Your frontend:** Web-based UI

```python
# Backend API server (Flask/FastAPI)
from flask import Flask, request, jsonify
from optimizer.genetic.algo import OptimizationWorker
import pandas as pd

app = Flask(__name__)

@app.route('/api/optimize', methods=['POST'])
def optimize():
    # Receive settings from web frontend
    data = request.json
    
    # Convert to DataFrame
    df_players = pd.DataFrame(data['players'])
    
    # Run optimization (synchronously for API)
    worker = OptimizationWorker(
        df_players=df_players,
        num_lineups=data['num_lineups'],
        team_selections=data['team_selections'],
        stack_settings=data['stack_settings'],
        min_salary=data['min_salary'],
        # ... other settings
    )
    
    # Run synchronously
    results, team_exp, stack_exp = worker.optimize_lineups()
    
    # Convert to JSON-serializable format
    lineups_json = [
        {
            'players': lineup.to_dict('records'),
            'total_points': lineup['Predicted_DK_Points'].sum(),
            'total_salary': lineup['Salary'].sum()
        }
        for lineup in results.values()
    ]
    
    return jsonify({
        'lineups': lineups_json,
        'team_exposure': dict(team_exp),
        'stack_exposure': dict(stack_exp)
    })

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    # Handle CSV upload from web frontend
    file = request.files['file']
    df = pd.read_csv(file)
    # Validate and return
    return jsonify({'players': df.to_dict('records')})
```

**Your web frontend (React example):**
```javascript
// Call backend API
async function runOptimization(settings) {
  const response = await fetch('/api/optimize', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      players: playerData,
      num_lineups: 100,
      team_selections: {4: ['NYY', 'LAD']},
      stack_settings: ['4', '3'],
      min_salary: 45000,
      // ... other settings from YOUR UI
    })
  });
  
  const results = await response.json();
  displayResults(results.lineups);
}
```

### Option 3: Desktop App (Electron/Tauri)

**Your frontend:** Electron with HTML/CSS/JS

```python
# Python backend runs as subprocess
# Electron frontend communicates via stdin/stdout or HTTP

# Python script: api_server.py
import sys
import json
from optimizer.genetic.algo import OptimizationWorker

def main():
    while True:
        # Read from stdin
        line = sys.stdin.readline()
        if not line:
            break
            
        command = json.loads(line)
        
        if command['action'] == 'optimize':
            # Run optimization
            results = run_optimization(command['settings'])
            
            # Write to stdout
            print(json.dumps(results))
            sys.stdout.flush()

if __name__ == '__main__':
    main()
```

```javascript
// Electron main process
const { spawn } = require('child_process');

const pythonProcess = spawn('python', ['api_server.py']);

function runOptimization(settings) {
  return new Promise((resolve) => {
    pythonProcess.stdout.once('data', (data) => {
      resolve(JSON.parse(data.toString()));
    });
    
    pythonProcess.stdin.write(JSON.stringify({
      action: 'optimize',
      settings: settings
    }) + '\n');
  });
}
```

---

## Required Data Mappings

### Frontend → Backend

**What your frontend needs to send:**

```javascript
// Settings object your UI collects
const optimizationSettings = {
  // From Players tab
  included_players: ['Shohei Ohtani', 'Aaron Judge', ...],
  
  // From Team Stacks tab
  team_selections: {
    "all": ['NYY', 'LAD', 'ATL', 'SF'],
    4: ['NYY', 'LAD'],
    3: ['ATL', 'SF']
  },
  
  // From Stack Exposure tab
  stack_settings: ['4', '3', '4|2'],
  
  // From Control Panel
  num_lineups: 100,
  min_unique: 3,
  min_salary: 45000,
  disable_kelly: false,
  
  // From Risk Management section
  bankroll: 1000,
  risk_tolerance: 'medium',
  enable_risk_mgmt: true,
  
  // From Advanced Quant tab
  use_advanced_quant: false,
  advanced_quant_params: {
    optimization_strategy: 'combined',
    risk_tolerance: 1.0,
    // ... other params
  }
}
```

### Backend → Frontend

**What the backend returns:**

```python
# Results structure
{
  'lineups': [
    {
      'lineup': [  # Array of 10 players
        {
          'Name': 'Shohei Ohtani',
          'Team': 'LAA',
          'Position': 'P',
          'Salary': 11000,
          'Predicted_DK_Points': 25.3
        },
        // ... 9 more players
      ],
      'total_points': 125.3,
      'total_salary': 49800,
      'stack_type': '4',
      'risk_info': {  # Optional
        'sharpe_ratio': 1.45,
        'volatility': 0.128,
        'kelly_fraction': 0.185
      }
    },
    // ... 99 more lineups
  ],
  'team_exposure': {
    'NYY': 45,  # Used in 45% of lineups
    'LAD': 38,
    'ATL': 22,
    // ...
  },
  'stack_exposure': {
    '4': 55,   # 55% of lineups are 4-stacks
    '3': 45    # 45% are 3-stacks
  }
}
```

---

## Critical Integration Points

### 1. CSV Upload → Player Data

**Your Frontend:**
```javascript
// Upload CSV file
const fileInput = document.getElementById('csv-upload');
const file = fileInput.files[0];

// Send to backend
const formData = new FormData();
formData.append('file', file);

fetch('/api/upload', {method: 'POST', body: formData})
  .then(res => res.json())
  .then(players => {
    // Display in YOUR Players table
    renderPlayersTable(players);
  });
```

**Backend Endpoint:**
```python
@app.route('/api/upload', methods=['POST'])
def upload_csv():
    file = request.files['file']
    
    # Use existing load logic
    df = pd.read_csv(file)
    
    # Validate (existing function)
    df = validate_and_clean(df)
    
    # Return as JSON
    return jsonify(df.to_dict('records'))
```

### 2. Team Selection → Checkbox State

**Your Frontend:**
```javascript
// User clicks team checkbox
function onTeamCheckboxChange(stackSize, team, checked) {
  // Update your state
  if (!teamSelections[stackSize]) {
    teamSelections[stackSize] = [];
  }
  
  if (checked) {
    teamSelections[stackSize].push(team);
  } else {
    teamSelections[stackSize] = teamSelections[stackSize].filter(t => t !== team);
  }
  
  // teamSelections now matches backend format!
}
```

**Backend Receives:**
```python
team_selections = {
    4: ['NYY', 'LAD'],
    3: ['ATL', 'SF']
}
# Exactly what OptimizationWorker expects!
```

### 3. Run Optimization → Progress Updates

**Your Frontend:**
```javascript
// Start optimization
async function runOptimization() {
  showLoadingState();
  
  const results = await fetch('/api/optimize', {
    method: 'POST',
    body: JSON.stringify(optimizationSettings)
  }).then(res => res.json());
  
  hideLoadingState();
  displayResults(results);
}
```

**Backend Process:**
```python
def optimize():
    data = request.json
    
    # Use existing worker
    worker = OptimizationWorker(
        df_players=pd.DataFrame(data['players']),
        num_lineups=data['num_lineups'],
        team_selections=data['team_selections'],
        # ... etc
    )
    
    # Run synchronously for API
    results, team_exp, stack_exp = worker.optimize_lineups()
    
    # Convert to JSON
    return jsonify(serialize_results(results))
```

---

## Data Format Conversions

### DataFrame ↔ JSON

**Backend DataFrame to JSON:**
```python
# In backend
lineup_df = pd.DataFrame({...})  # Optimization result

# Convert for frontend
lineup_json = lineup_df.to_dict('records')
# Result: [{Name: 'Player', Team: 'NYY', ...}, ...]
```

**Frontend JSON to Backend DataFrame:**
```python
# Received from frontend
players_json = [
    {'Name': 'Shohei Ohtani', 'Salary': 11000, ...},
    ...
]

# Convert for backend
df_players = pd.DataFrame(players_json)
# Now can use with existing algorithms!
```

---

## Minimal Integration Example

### Simple API Wrapper

Create a thin API layer over your existing backend:

```python
# File: api_wrapper.py
"""
Thin API wrapper around existing optimizer.genetic.algo.py
Exposes backend functionality to your custom frontend
"""

from optimizer_genetic_algo import OptimizationWorker, optimize_single_lineup
import pandas as pd
import json

class OptimizerAPI:
    """API interface for custom frontend"""
    
    def __init__(self):
        self.df_players = None
        self.optimized_lineups = None
    
    def load_players_from_json(self, players_json):
        """
        Load player data from your frontend
        
        Args:
            players_json: List of player dictionaries
        Returns:
            Success status
        """
        try:
            self.df_players = pd.DataFrame(players_json)
            return {'success': True, 'count': len(self.df_players)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_optimization(self, settings):
        """
        Run optimization with settings from your frontend
        
        Args:
            settings: Dictionary with all configuration
        Returns:
            Results dictionary
        """
        # Create worker with existing class
        worker = OptimizationWorker(
            df_players=self.df_players,
            salary_cap=50000,
            position_limits={'P':2, 'C':1, '1B':1, '2B':1, '3B':1, 'SS':1, 'OF':3},
            included_players=settings.get('included_players', []),
            stack_settings=settings.get('stack_settings', ['No Stacks']),
            min_exposure={},
            max_exposure={},
            min_points=1,
            monte_carlo_iterations=100,
            num_lineups=settings.get('num_lineups', 100),
            team_selections=settings.get('team_selections', {}),
            min_unique=settings.get('min_unique', 0),
            bankroll=settings.get('bankroll', 1000),
            risk_tolerance=settings.get('risk_tolerance', 'medium'),
            disable_kelly=settings.get('disable_kelly', False),
            min_salary=settings.get('min_salary', 45000),
            use_advanced_quant=settings.get('use_advanced_quant', False),
            advanced_quant_params=settings.get('advanced_quant_params', {})
        )
        
        # Run optimization (existing method)
        results, team_exp, stack_exp = worker.optimize_lineups()
        
        # Convert to JSON-serializable format
        return self.serialize_results(results, team_exp, stack_exp)
    
    def serialize_results(self, results, team_exp, stack_exp):
        """Convert backend results to frontend format"""
        lineups_json = []
        
        for idx, result in results.items():
            lineup_df = result['lineup']
            
            lineup_json = {
                'id': idx,
                'players': lineup_df.to_dict('records'),
                'total_points': float(result['total_points']),
                'total_salary': int(lineup_df['Salary'].sum()),
                'stack_type': result.get('stack_type', 'Unknown')
            }
            
            # Add risk info if available
            if 'risk_info' in result:
                lineup_json['risk_info'] = result['risk_info']
            
            lineups_json.append(lineup_json)
        
        return {
            'lineups': lineups_json,
            'team_exposure': dict(team_exp),
            'stack_exposure': dict(stack_exp),
            'total_count': len(lineups_json)
        }
    
    def export_to_dk_format(self, lineups, output_path):
        """
        Export lineups to DraftKings CSV format
        
        Args:
            lineups: List of lineup DataFrames
            output_path: Where to save
        """
        # Use existing export logic
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF'])
            
            for lineup_df in lineups:
                dk_lineup = self.format_lineup_for_dk(lineup_df)
                writer.writerow(dk_lineup)
        
        return {'success': True, 'path': output_path}
```

---

## Frontend Requirements

### What Your Frontend Needs to Implement

Based on the design docs, your frontend should have:

#### 1. Data Input Components
```
✓ CSV file upload
✓ Player data display (table/grid)
✓ Player selection (checkboxes)
✓ Team selection per stack size
✓ Stack type selection
```

#### 2. Configuration Components
```
✓ Number of lineups input
✓ Min unique input
✓ Min/Max salary inputs
✓ Disable Kelly checkbox
✓ Bankroll input
✓ Risk profile selector
```

#### 3. Output Components
```
✓ Results table/grid
✓ Exposure displays
✓ Status/progress indicators
✓ Export buttons
```

#### 4. Data Collection Methods
```javascript
// Your frontend needs these functions
function getIncludedPlayers() {
  // Return array of selected player names
  return ['Shohei Ohtani', 'Aaron Judge', ...];
}

function getTeamSelections() {
  // Return object mapping stack sizes to teams
  return {
    4: ['NYY', 'LAD'],
    3: ['ATL', 'SF']
  };
}

function getStackSettings() {
  // Return array of enabled stack types
  return ['4', '3'];
}

function getOptimizationSettings() {
  // Collect all settings
  return {
    num_lineups: parseInt(numLineupsInput.value),
    min_unique: parseInt(minUniqueInput.value),
    min_salary: parseInt(minSalaryInput.value),
    // ... etc
  };
}
```

---

## Key Backend Functions to Use

### Functions You Can Call Directly

```python
# 1. Load and validate player data
from optimizer.genetic.algo import FantasyBaseballApp
app = FantasyBaseballApp()
df_players = app.load_players('path/to/csv')

# 2. Run single lineup optimization
from optimizer.genetic.algo import optimize_single_lineup
lineup, stack_type = optimize_single_lineup((
    df_players,
    stack_type='4',
    team_projected_runs={},
    team_selections={4: ['NYY', 'LAD']},
    min_salary=45000
))

# 3. Use genetic diversity engine
from optimizer.genetic.algo import GeneticDiversityEngine
engine = GeneticDiversityEngine(
    df_players=df_players,
    position_limits={'P':2, 'C':1, '1B':1, '2B':1, '3B':1, 'SS':1, 'OF':3},
    salary_cap=50000,
    team_selections={4: ['NYY']},
    min_salary=45000
)
diverse_lineups = engine.create_diverse_lineups(num_lineups=20, stack_type='4')

# 4. Full optimization workflow
from optimizer.genetic.algo import OptimizationWorker
worker = OptimizationWorker(
    df_players=df_players,
    # ... all settings
)
results, team_exp, stack_exp = worker.optimize_lineups()
```

---

## Configuration Mapping Reference

### UI Input → Backend Parameter

| Your UI Element | Backend Parameter | Type | Example |
|----------------|------------------|------|---------|
| Player checkboxes | `included_players` | List[str] | `['Player A', 'Player B']` |
| Team checkboxes | `team_selections` | Dict[int, List[str]] | `{4: ['NYY', 'LAD']}` |
| Stack type checkboxes | `stack_settings` | List[str] | `['4', '3', '4\|2']` |
| Number of lineups | `num_lineups` | int | `100` |
| Min unique | `min_unique` | int | `3` |
| Min salary | `min_salary` | int | `45000` |
| Disable Kelly | `disable_kelly` | bool | `False` |
| Bankroll | `bankroll` | float | `1000.0` |
| Risk profile | `risk_tolerance` | str | `'medium'` |
| Advanced quant toggle | `use_advanced_quant` | bool | `False` |

---

## Example: Complete Integration Flow

```python
# File: backend_adapter.py
"""
Adapter between your frontend and existing backend
"""

class OptimizerAdapter:
    def __init__(self):
        self.backend = None
        self.df_players = None
    
    def initialize(self):
        """Setup backend"""
        from optimizer.genetic.algo import FantasyBaseballApp
        self.backend = FantasyBaseballApp()
    
    def load_players(self, file_path_or_data):
        """
        Load player data
        
        Args:
            file_path_or_data: CSV path or DataFrame/dict
        """
        if isinstance(file_path_or_data, str):
            # File path
            self.df_players = self.backend.load_players(file_path_or_data)
        elif isinstance(file_path_or_data, dict):
            # JSON data from frontend
            self.df_players = pd.DataFrame(file_path_or_data)
        elif isinstance(file_path_or_data, pd.DataFrame):
            # Already a DataFrame
            self.df_players = file_path_or_data
        
        return {
            'success': True,
            'players': self.df_players.to_dict('records'),
            'count': len(self.df_players)
        }
    
    def optimize(self, frontend_settings):
        """
        Run optimization with frontend settings
        
        Args:
            frontend_settings: Dict from your UI
        Returns:
            Serialized results for your UI
        """
        # Map frontend settings to backend format
        backend_settings = self.map_settings(frontend_settings)
        
        # Create worker (existing class)
        from optimizer.genetic.algo import OptimizationWorker
        worker = OptimizationWorker(**backend_settings)
        
        # Run optimization (existing method)
        results, team_exp, stack_exp = worker.optimize_lineups()
        
        # Convert to frontend format
        return self.format_for_frontend(results, team_exp, stack_exp)
    
    def map_settings(self, frontend):
        """Convert frontend settings to backend format"""
        return {
            'df_players': self.df_players,
            'salary_cap': 50000,
            'position_limits': {'P':2, 'C':1, '1B':1, '2B':1, '3B':1, 'SS':1, 'OF':3},
            'included_players': frontend.get('included_players', []),
            'stack_settings': frontend.get('stack_settings', ['No Stacks']),
            'min_exposure': {},
            'max_exposure': {},
            'min_points': 1,
            'monte_carlo_iterations': 100,
            'num_lineups': frontend.get('num_lineups', 100),
            'team_selections': frontend.get('team_selections', {}),
            'min_unique': frontend.get('min_unique', 0),
            'bankroll': frontend.get('bankroll', 1000),
            'risk_tolerance': frontend.get('risk_tolerance', 'medium'),
            'disable_kelly': frontend.get('disable_kelly', False),
            'min_salary': frontend.get('min_salary', 45000),
            'use_advanced_quant': frontend.get('use_advanced_quant', False),
            'advanced_quant_params': frontend.get('advanced_quant_params', {})
        }
    
    def format_for_frontend(self, results, team_exp, stack_exp):
        """Convert backend results to frontend format"""
        lineups = []
        
        for idx, result in results.items():
            lineup_df = result['lineup']
            
            lineups.append({
                'id': idx,
                'players': lineup_df.to_dict('records'),
                'total_points': float(result.get('total_points', 0)),
                'total_salary': int(lineup_df['Salary'].sum()),
                'stack_type': result.get('stack_type', 'Unknown'),
                'risk_info': result.get('risk_info', None)
            })
        
        return {
            'success': True,
            'lineups': lineups,
            'team_exposure': {k: int(v) for k, v in team_exp.items()},
            'stack_exposure': {k: int(v) for k, v in stack_exp.items()},
            'count': len(lineups)
        }

# Usage in your app
adapter = OptimizerAdapter()
adapter.initialize()

# Load data
adapter.load_players(csv_file)

# Run optimization
results = adapter.optimize(your_frontend_settings)

# Display in your UI
display_results(results['lineups'])
```

---

## Testing Your Integration

### Test Checklist

```
□ Can upload CSV and see players in your UI
□ Can select players and selection persists
□ Can select teams for stacking
□ Can configure stack types
□ Can set number of lineups
□ Can run optimization and see progress
□ Can display results in your results view
□ Can see exposure percentages
□ Can export to CSV
□ Can load DK entries file
□ Can fill entries with lineups
□ Can add to favorites
□ Can export favorites
```

### Test Scenarios

**Scenario 1: Basic Flow**
```
1. Load CSV with 187 players
2. Select 50 players
3. Select NYY, LAD for 4-stack
4. Enable 4-Stack in Stack Exposure
5. Set 20 lineups
6. Run optimization
7. Verify: 20 lineups generated
8. Verify: Each has 4+ NYY or LAD batters
9. Export to CSV
10. Success: File created ✓
```

**Scenario 2: Advanced Features**
```
1. Load CSV with probability columns
2. Select 100 players
3. Configure multi-stacks (4|2|2)
4. Enable advanced quant
5. Set risk parameters
6. Run optimization with 100 lineups
7. Verify: Risk metrics displayed
8. Verify: Proper stack distribution
9. Add top 50 to favorites
10. Export favorites
11. Success: DK-format file created ✓
```

---

## Recommendation: Architecture Pattern

### Suggested Approach

```
Your Frontend (UI Layer)
        ↕
    API Adapter
        ↕
Existing Backend (Algorithm Layer)
```

**Benefits:**
- ✅ Clean separation of concerns
- ✅ Existing algorithms unchanged
- ✅ Easy to test independently
- ✅ Can swap frontend without touching backend
- ✅ Can update backend without breaking frontend

**Implementation:**
1. Keep `optimizer.genetic.algo.py` as-is (working code!)
2. Create thin adapter layer (`api_wrapper.py` or REST API)
3. Build your frontend to communicate through adapter
4. Adapter handles all data format conversions

---

## Summary

Your design docs now provide:

✅ **What data flows where** - Data Flow document  
✅ **What each UI component does** - Individual tab documents  
✅ **What the backend expects** - This integration guide  
✅ **How to connect them** - API adapter examples  

**Next Steps:**
1. Build your frontend UI based on tab design docs
2. Create adapter layer using examples above
3. Connect frontend to adapter
4. Test with example data
5. Refine and polish

The existing backend algorithms are production-ready - you just need to wire your frontend to them! 🚀

