# DFS Optimizer - Quick Start Guide

## Overview

The DFS Optimizer is now fully integrated with the backend. When you click "Optimize", it actually calls the backend API and generates real lineups!

## Changes Made

### Backend Integration
✅ **File Upload** - Now sends CSV to backend API (`/api/upload-players`)
✅ **Optimization** - Calls backend optimization API (`/api/optimize`)  
✅ **Export** - Downloads actual CSV from backend (`/api/export/draftkings`)
✅ **Results Display** - Shows real optimization results, not mock data

### What Works Now

1. **Load CSV** → Uploads to backend and fetches player data
2. **Select Players** → Marks players as selected in backend
3. **Click Optimize** → Runs actual optimization (NFL or MLB)
4. **View Results** → Displays real generated lineups
5. **Save CSV** → Exports DraftKings-ready file

## Step-by-Step Usage

### 1. Start the Application

```bash
# Terminal 1: Start backend
cd web_optimizer
npm run server

# Terminal 2: Start frontend  
cd web_optimizer/client
npm run dev
```

Backend should be running on `http://localhost:5001`
Frontend should be running on `http://localhost:5173`

### 2. Select Sport

1. Open browser to `http://localhost:5173`
2. In the Control Panel (right side), find "Sport" section at top
3. Select **🏈 NFL** or **⚾ MLB**

### 3. Load Player Data

1. Click "Load CSV" button
2. Select your CSV file (must have columns: Name, Position, Team, Salary, Predicted_DK_Points)
3. Wait for upload to complete
4. You should see: "✅ Loaded X players successfully!"
5. Players will appear in the "Players" tab

**Example CSV Format (NFL):**
```csv
Name,Position,Team,Salary,Predicted_DK_Points,Ownership
Patrick Mahomes,QB,KC,8800,26.5,35
Travis Kelce,TE,KC,7000,15.8,25
Tyreek Hill,WR,MIA,8000,20.1,30
...
```

### 4. Select Players

1. Go to "Players" tab
2. Use position filters (QB, RB, WR, TE, DST for NFL)
3. Check boxes next to players you want to include
4. OR click "Select All" to select all visible players

**Minimum Required:**
- MLB: 10+ players
- NFL: 9+ players

### 5. Configure Settings

**In Control Panel:**

- **Lineups**: How many to generate (e.g., 20)
- **Min Salary**: Defaults to $48k (NFL) or $45k (MLB)
- **Min Unique**: Players that must differ between lineups (e.g., 3)

**Optional - Stack Exposure Tab:**
1. Go to "Stack Exposure" tab
2. Enable stack types (e.g., "QB + 2 WR + TE" for NFL)
3. Set min/max exposure percentages

### 6. Run Optimization

1. Click the blue **"Optimize"** button in Control Panel
2. Wait while optimization runs (10-30 seconds for 20 lineups)
3. Progress shown in button text: "Optimizing..."
4. When complete: "✅ Generated X optimal lineups!"
5. Auto-switches to "My Entries" tab

### 7. View Results

**My Entries Tab** shows:
- List of generated lineups
- Player names, positions, teams
- Total projected points
- Total salary used
- Salary remaining

**Features:**
- Sort by points/salary
- Filter by run number
- Add to favorites
- Select multiple lineups

### 8. Export to DraftKings

1. Click **"Save CSV"** button in Control Panel
2. File downloads as `nfl_lineups_2025-10-22.csv` (or mlb)
3. Open file to verify format
4. Upload directly to DraftKings contest

**DraftKings CSV Format (NFL):**
```csv
QB,RB,RB,WR,WR,WR,TE,FLEX,DST
Patrick Mahomes,Christian McCaffrey,Saquon Barkley,Tyreek Hill,Cooper Kupp,Stefon Diggs,Travis Kelce,Derrick Henry,49ers
```

## Troubleshooting

### "❌ Please load player data first"
**Solution:** Click "Load CSV" and upload a valid player file

### "❌ Please select at least X players"
**Solution:** Go to Players tab and select more players (10 for MLB, 9 for NFL)

### "❌ Upload failed"
**Causes:**
- CSV format incorrect (missing required columns)
- File not CSV format
- Backend not running

**Solution:** 
1. Check CSV has: Name, Position, Team, Salary, Predicted_DK_Points
2. Verify backend running on port 5001
3. Check browser console for errors

### "❌ Optimization failed"
**Causes:**
- Not enough players selected
- Invalid stack configuration
- Backend error

**Solution:**
1. Select more players
2. Disable stack types temporarily
3. Check backend logs in terminal

### Results show 0 lineups
**Solution:** 
- Check "My Entries" tab (auto-switches after optimization)
- Verify optimization completed successfully  
- Check browser console for errors

### Export downloads empty/wrong file
**Causes:**
- No lineups generated yet
- Backend export error

**Solution:**
1. Run optimization first
2. Verify results visible in "My Entries" tab
3. Check backend running

## Backend API Reference

### Upload Players
```http
POST /api/upload-players
Content-Type: multipart/form-data

FormData:
  playersFile: File
```

### Get Players
```http
GET /api/players
```

### Update Player Selection
```http
PUT /api/players/:id
Content-Type: application/json

Body:
{
  "selected": true
}
```

### Run Optimization
```http
POST /api/optimize
Content-Type: application/json

Body:
{
  "sport": "NFL",
  "numLineups": 20,
  "minSalary": 48000,
  "maxSalary": 50000,
  "stackSettings": {
    "enabled": true,
    "types": ["QB + 2 WR + TE"],
    "teams": ["KC", "BUF", "SF"]
  },
  "uniquePlayers": 3,
  "maxExposure": 40
}
```

### Export Lineups
```http
GET /api/export/draftkings?sport=NFL
```

## Example Workflow

**Complete optimization in 2 minutes:**

1. **Start servers** (if not running)
   ```bash
   cd web_optimizer && npm run server
   cd client && npm run dev
   ```

2. **Select NFL** in Sport dropdown

3. **Load CSV**
   - Click "Load CSV"
   - Select `nfl_week7_projections.csv`
   - Wait for "✅ Loaded 150 players"

4. **Select players**
   - Click "Players" tab
   - Click "Select All" button
   - Verify 150 players selected

5. **Configure stacks** (optional)
   - Click "Stack Exposure" tab
   - Enable "QB + 2 WR + TE"
   - Set min 20%, max 50%

6. **Optimize**
   - Set Lineups: 20
   - Click blue "Optimize" button
   - Wait 10-15 seconds

7. **Export**
   - Verify results in "My Entries" tab
   - Click "Save CSV"
   - Upload to DraftKings

Done! 🎉

## Tips & Best Practices

### For Better Results

1. **Load Quality Projections**
   - Use reliable projection sources
   - Update ownership estimates
   - Include ceiling/floor data

2. **Select Diverse Player Pool**
   - Don't over-constrain (select 50+ players)
   - Include value plays and chalk
   - Multiple teams represented

3. **Configure Stacks Wisely**
   - GPP: Aggressive stacks (QB+2WR+TE)
   - Cash: Safe stacks (QB+WR)
   - Limit exposure (20-30% per stack type)

4. **Generate More Lineups**
   - 20-50 for small contests
   - 100-150 for large GPPs
   - More = better diversification

5. **Review Before Exporting**
   - Check salary usage (aim for $49k+)
   - Verify stack patterns make sense
   - Ensure no duplicates

### Contest-Specific Strategies

**Cash Games (50/50, Double-Ups):**
- Min Salary: $49,000+
- Stacks: QB + WR only
- Lineups: 5-10
- Focus: High floor players

**GPP Tournaments:**
- Min Salary: $48,000
- Stacks: QB + 2 WR + TE
- Lineups: 20-150
- Focus: High ceiling, contrarian

**Showdown:**
- Different format (Captain + 5 FLEX)
- Use Showdown-specific CSV
- More correlation important

## Advanced Features

### Team Combinations (Coming Soon)
- Generate specific team stacks
- Double stacks (Team A + Team B)
- Custom correlation matrices

### Advanced Quant (Coming Soon)
- Monte Carlo simulation
- Kelly criterion sizing
- Sharpe ratio optimization

### Risk Management (Coming Soon)
- Bankroll management
- Position limits
- Exposure caps

## Support

**Need Help?**
- Check backend logs in terminal
- Open browser console (F12)
- Review error messages
- Check API endpoints with curl/Postman

**Found a Bug?**
- Note steps to reproduce
- Check browser console for errors
- Review backend terminal output
- Create detailed issue report

---

**Last Updated:** October 22, 2025
**Version:** 2.0 with Backend Integration
**Status:** ✅ Fully Functional


