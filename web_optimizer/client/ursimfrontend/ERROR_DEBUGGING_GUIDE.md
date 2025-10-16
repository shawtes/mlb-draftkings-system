# Error Debugging Guide

## Fixes Applied

### 1. ✅ Fixed ChevronDown import error
- Added `ChevronDown` to Dashboard.tsx imports

### 2. ✅ Fixed HowToUse.tsx syntax error
- Removed extra closing brace in tutorials array

### 3. ✅ Added Toaster at app level
- Moved Toaster to App.tsx
- Removed duplicate Toasters from components

### 4. ✅ Added error catching to lazy imports
- All components now show specific error messages if they fail to load

### 5. ✅ Fixed backend API endpoint
- Changed `/api/players/upload` → `/api/upload-players`
- Changed form field `file` → `playersFile`

---

## How to Debug Current Issues

### Step 1: Open Browser Console
Press `F12` in your browser and click the "Console" tab

### Step 2: Look for Error Messages
You should now see specific errors like:
```
"Failed to load GamesHub: [error details]"
"Failed to load PropBettingCenter: [error details]"
etc.
```

### Step 3: Common Errors and Fixes

#### Error: "Cannot find module"
**Cause**: Missing dependency
**Fix**:
```bash
cd C:\Users\Carte\Desktop\UrSim\mlb-draftkings-system\web_optimizer\client\ursimfrontend
npm install
```

#### Error: "Unexpected token" or "Syntax Error"
**Cause**: Syntax error in component
**Fix**: Check the specific file mentioned in error

#### Error: "X is not defined"
**Cause**: Missing import
**Fix**: Add the missing import

---

## Quick Fix: Use Fallback Components

If new components still don't load, temporarily use old ones:

### Edit Dashboard.tsx imports:
```typescript
// Fallback to old components temporarily
const GamesHub = lazy(() => import('./DashboardOverview')); // Use old one
const PropBettingCenter = lazy(() => import('./PropBetFinder')); // Use old one  
const DFSOptimizerUnified = lazy(() => import('./DFSOptimizer')); // Use old one
```

This will restore old functionality while we debug.

---

## Backend Connection Status

### Current Backend Endpoints
```
✅ /api/health
✅ /api/upload-players
✅ /api/players
✅ /api/players/:id
✅ /api/players/bulk
✅ /api/optimize
✅ /api/results  
✅ /api/export/:format
✅ /api/export-advanced
✅ /api/teams
✅ /api/favorites
✅ /api/contest-formats
✅ /api/stack-analysis
```

### Missing Endpoints (Using Fallbacks)
```
⚠️ /api/odds/props - Using mock data
⚠️ /api/odds/games - Using mock data
⚠️ /api/bets/calculate-payout - Client-side calculation
⚠️ /api/bets/place - Not implemented
```

These are OK - components work with mock data!

---

## What to Share for Help

If components still won't load, copy/paste from browser console:

1. **Red error messages** (full text)
2. **Component name** that failed
3. **Line numbers** if shown
4. **Stack trace** (the gray text below error)

Example:
```
Failed to load GamesHub: Error: Cannot resolve './ui/something'
  at loadComponent (chunk-xyz.js:123)
  ...
```

---

## Quick Test Script

Run this in browser console to test components:
```javascript
// Test if components exist
console.log('Testing imports...');

import('./components/GamesHub').then(() => console.log('✅ GamesHub OK'))
  .catch(err => console.error('❌ GamesHub failed:', err));

import('./components/PropBettingCenter').then(() => console.log('✅ PropBettingCenter OK'))
  .catch(err => console.error('❌ PropBettingCenter failed:', err));

import('./components/DFSOptimizerUnified').then(() => console.log('✅ DFSOptimizerUnified OK'))
  .catch(err => console.error('❌ DFSOptimizerUnified failed:', err));

import('./components/HowToUse').then(() => console.log('✅ HowToUse OK'))
  .catch(err => console.error('❌ HowToUse failed:', err));
```

---

## Next Steps

1. **Refresh browser** (Ctrl + F5)
2. **Check console** for new error messages  
3. **Copy errors** and share them
4. **Try clicking each section** to see which ones show errors

The error boundaries should now show you exactly what's failing!

