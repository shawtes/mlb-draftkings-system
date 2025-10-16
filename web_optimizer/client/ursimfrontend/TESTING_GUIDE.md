# UrSim UX Optimization - Testing Guide

## Quick Start Testing

### 1. Start the Application
```bash
# From: mlb-draftkings-system/web_optimizer/
simple_start.bat
```

You'll see 3 windows:
- 🚀 Launcher (can close)
- 🔧 Backend (port 5000)
- 🌐 Frontend (port 3000)

### 2. Login/Register
- Create account or login
- You'll land on the dashboard

---

## Test Plan by Section

### ✅ Test 1: Games Hub

**Steps**:
1. Click **"Games Hub"** in sidebar (under RESEARCH)
2. Verify you see game cards with:
   - Team names
   - Spread and total
   - Time
   - Weather icons
   - Injury alerts
3. Click any game card
4. Verify right panel shows:
   - **Matchup Analysis** tab (default)
   - Offense/Defense rankings
   - Weather details
   - Key matchups
   - Injuries
5. Click **"Top Props"** tab
6. Verify props display for that game

**Expected Result**:
- ✅ All games display in grid
- ✅ Selected game highlights
- ✅ Matchup analysis shows
- ✅ Props tab works
- ✅ No errors in console

---

### ✅ Test 2: Prop Betting Center

**Steps**:
1. Click **"Prop Betting Center"** in sidebar (under BETTING)
2. Verify layout: Props list (left) + Betting slip (right)
3. Enter "Mahomes" in search
4. Verify props filter
5. Click **"Over"** button on a prop
6. Verify betting slip shows:
   - Added prop
   - Stake input ($10 default)
   - Combined odds
   - Potential payout
7. Add 2-3 more props
8. Verify:
   - Parlay odds accumulate
   - Payout updates
   - Kelly recommendation appears
9. Click **"Copy"** button (clipboard icon)
10. Verify toast message
11. Click **"Clear"** (trash icon)
12. Verify slip empties

**Expected Result**:
- ✅ Props browse and filter
- ✅ Add to slip works
- ✅ Odds calculate correctly
- ✅ Kelly shows (if >0 edge)
- ✅ Copy/clear works
- ✅ Warning shows for >5 leg parlays

---

### ✅ Test 3: DFS Lineup Optimizer

#### Tab 1: Players & Projections

**Steps**:
1. Click **"DFS Lineup Optimizer"** in sidebar (under DFS)
2. Verify you're on "Players & Projections" tab
3. Click **"Upload CSV"** button
4. Select a CSV file (or skip if none)
5. If players load, verify table shows:
   - Player names
   - Positions
   - Teams
   - Salaries
   - Projections
   - Value calculation
6. Click **Lock** icon (first player)
7. Verify it turns green
8. Click **Exclude** icon (second player)
9. Verify it turns red
10. Click **Star** icon (third player)
11. Verify it turns yellow

**Expected Result**:
- ✅ Upload works or shows empty state
- ✅ Player table displays correctly
- ✅ Lock/Exclude/Favorite toggle works
- ✅ Icons change color appropriately

#### Tab 2: Strategy & Settings

**Steps**:
1. Click **"Strategy & Settings"** tab
2. Under "2-Stack Teams", click 2-3 team badges
3. Verify they highlight (cyan background)
4. Under "3-Stack Teams", click 1-2 teams
5. Verify they highlight (blue background)
6. Scroll to "Exposure Limits"
7. Enter min/max values for a team
8. Verify inputs accept numbers

**Expected Result**:
- ✅ Team badges toggle on/off
- ✅ Visual feedback (color change)
- ✅ Exposure inputs work
- ✅ Layout is clean

#### Tab 3: Optimization Control

**Steps**:
1. Click **"Optimization Control"** tab
2. Change "Number of Lineups" to 5
3. Select "GPP (Tournament)" from Contest Type
4. Move "Min Salary" slider
5. Move "Max Salary" slider
6. Move "Uniqueness" slider
7. Verify value updates show
8. Scroll to "Advanced Settings"
9. Move "Monte Carlo Iterations" slider
10. Change "Risk Tolerance" dropdown
11. Toggle "Kelly Criterion" switch
12. Click **"Start Optimization"** button
13. Verify:
    - Button changes to "Optimizing..."
    - Progress bar appears
    - Progress updates
14. Wait for completion
15. Verify auto-navigation to Results tab

**Expected Result**:
- ✅ All inputs responsive
- ✅ Sliders move smoothly
- ✅ Advanced settings accessible
- ✅ Optimization runs
- ✅ Progress shows
- ✅ Auto-navigates to results

#### Tab 4: Results & Export

**Steps**:
1. Verify you're on "Results & Export" tab (after optimization)
2. Check lineup table shows:
   - Rank badges (#1, #2, etc.)
   - Projected points
   - Total salary
   - Player chips
3. Click **"DraftKings"** export button
4. Verify download starts (or mock action)
5. Click **"FanDuel"** export button
6. Click **"CSV"** export button

**Expected Result**:
- ✅ Results table displays
- ✅ All lineup data visible
- ✅ Export buttons work
- ✅ Toast messages appear

---

### ✅ Test 4: How to Use UrSim

**Steps**:
1. Click **"How to Use UrSim"** in sidebar (under HELP)
2. Verify 6 tabs show:
   - Getting Started
   - Building DFS Lineups
   - Prop Betting & Parlays
   - Using Games Hub
   - Advanced Strategies
   - FAQ
   - Glossary
3. Click each tab
4. Expand/collapse accordion items
5. Type "Kelly" in search box
6. Verify content filters
7. Clear search
8. Verify all content returns
9. Scroll to bottom
10. Verify 3 tip cards display

**Expected Result**:
- ✅ All tabs accessible
- ✅ Accordions expand/collapse
- ✅ Search filters correctly
- ✅ Content readable
- ✅ Tips cards display

---

## Browser Console Check

### Should See (No Errors)
```
✅ React components mounted
✅ API calls (may show 404 if backend not running - OK)
✅ WebSocket connection attempt
✅ No red errors
```

### May See (OK)
```
⚠️ API connection failed (if backend not running)
⚠️ Using fallback calculations
⚠️ Mock data loaded
```

### Should NOT See
```
❌ Component failed to render
❌ Cannot read property of undefined
❌ Module not found
❌ Uncaught TypeError
```

---

## Visual Testing

### Check on Different Screens

**Desktop (1920x1080)**:
- ✅ Sidebar fully visible
- ✅ Betting slip side-by-side with props
- ✅ Tables display properly
- ✅ No horizontal scroll

**Laptop (1366x768)**:
- ✅ Content fits
- ✅ Scrollbars where needed
- ✅ Readable fonts

**Tablet (768px)**:
- ✅ Sidebar collapses
- ✅ Betting slip stacks below props
- ✅ Touch-friendly buttons

**Mobile (375px)**:
- ✅ Full collapse
- ✅ Vertical layout
- ✅ Readable content

---

## Integration Testing

### With Backend Running

**Test**:
1. Start backend: `cd server && node index.js`
2. Upload CSV with real players
3. Run optimization
4. Verify lineups generate
5. Export to DraftKings format

**Expected**:
- ✅ Real data loads
- ✅ Optimization completes
- ✅ Export downloads file
- ✅ WebSocket updates in real-time

### Without Backend (Fallback Mode)

**Test**:
1. Don't start backend
2. Click through all sections
3. Try all features

**Expected**:
- ✅ Mock data displays
- ✅ UI fully functional
- ✅ Calculations work client-side
- ✅ Graceful error messages
- ✅ No crashes

---

## Performance Testing

### Load Times (Target)
- Initial page load: <2s
- Tab switch: <200ms
- API call: <500ms
- Betting slip calculation: <50ms
- Optimization: <10s for 20 lineups

### Memory Usage
- Initial: ~50MB
- After navigation: <100MB
- No memory leaks (check DevTools)

---

## Accessibility Testing

### Keyboard Navigation
- Tab through buttons: Works
- Enter to select: Works
- ESC to close: Works (modals)

### Screen Reader
- Proper labels: ✅
- Alt text: ✅
- ARIA labels: ✅

### Color Contrast
- Text readable: ✅
- Buttons clear: ✅
- WCAG AA compliant: ✅

---

## Regression Testing

### Make Sure Old Features Still Work

**Account**:
- ✅ Logout button works
- ✅ Settings page loads
- ✅ Profile dropdown works

**Navigation**:
- ✅ Sidebar collapse/expand
- ✅ Active state highlighting
- ✅ Mobile menu works

**Styling**:
- ✅ Consistent theme
- ✅ No broken layouts
- ✅ Icons display correctly

---

## Bug Tracking

### Found Issues? Document:
1. **What**: Describe the issue
2. **Where**: Section → Tab → Action
3. **Expected**: What should happen
4. **Actual**: What actually happened
5. **Console**: Any error messages
6. **Reproduce**: Steps to recreate

### Report To:
- GitHub issues
- Development team
- User feedback form

---

## Success Checklist

### Before Marking Complete
- [ ] All 5 sections accessible
- [ ] No console errors
- [ ] All tabs functional
- [ ] Betting slip calculates
- [ ] DFS optimization runs
- [ ] Export works
- [ ] Search works
- [ ] Mobile responsive
- [ ] No duplicate features
- [ ] Professional appearance
- [ ] Tutorials complete
- [ ] API integrations ready

---

## What to Test First

### Priority 1 (Core Functionality)
1. ✅ Games Hub loads
2. ✅ DFS Optimizer all 4 tabs
3. ✅ Prop Betting Center + Slip

### Priority 2 (Features)
1. ✅ Search and filters
2. ✅ Lock/Exclude controls
3. ✅ Parlay calculator
4. ✅ Export functions

### Priority 3 (Polish)
1. ✅ How to Use guide
2. ✅ Animations smooth
3. ✅ Tooltips helpful
4. ✅ Colors consistent

---

## Demo Walkthrough Script

### For DFS Players (2 min demo)
```
"Welcome to UrSim's new unified experience!

1. [Games Hub] - See all games in one place, with matchups
2. [DFS Optimizer] - Streamlined to 4 tabs
3. [Upload players] - One click
4. [Configure stacks] - Visual team selection
5. [Advanced settings] - Monte Carlo, Kelly, risk management
6. [Start optimization] - Real-time progress
7. [Export] - Direct to DraftKings

Everything you need, nothing you don't."
```

### For Prop Bettors (2 min demo)
```
"New prop betting experience!

1. [Games Hub] - Research games and matchups
2. [Prop Betting Center] - Browse props with edge filters
3. [Add to slip] - One-click adds
4. [Betting slip] - See payout live
5. [Kelly recommendation] - Optimal stake sizing
6. [Build parlay] - 2-4 legs for best odds
7. [Place or copy] - Easy submission

Find edges, build smart parlays, bet with confidence."
```

---

**Testing Estimated Time**: 30 minutes full test  
**Quick Test**: 10 minutes (one path through each section)

**Status**: Ready for testing! 🚀

