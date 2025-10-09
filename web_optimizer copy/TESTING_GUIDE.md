# Testing Guide for MLB DFS Optimizer

This guide will help you test all features of the MLB DFS Optimizer web application.

## Prerequisites Test

1. **Check Node.js Version**
   ```bash
   node --version
   npm --version
   ```
   Required: Node.js v16+ and npm v7+

## Installation Test

1. **Install Dependencies**
   ```bash
   # Server dependencies
   cd server
   npm install
   
   # Client dependencies  
   cd ../client
   npm install
   ```

2. **Verify Installation**
   - Check that `node_modules` folders exist in both `/server` and `/client` directories
   - No error messages during installation

## Backend Server Test

1. **Start Backend Server**
   ```bash
   cd server
   npm start
   ```

2. **Test API Endpoints**
   
   **Health Check:**
   ```bash
   curl http://localhost:5000/api/health
   ```
   Expected: `{"status":"ok","timestamp":"..."}`

   **Get Players (empty initially):**
   ```bash
   curl http://localhost:5000/api/players
   ```
   Expected: `{"players":[],"count":0}`

## Frontend Application Test

1. **Start Frontend**
   ```bash
   cd client
   npm start
   ```
   
2. **Open Browser**
   - Navigate to http://localhost:3000
   - Application should load with dark theme
   - All tabs should be visible: Players, Team Stacks, Stack Exposure, Team Combos, Control Panel, Favorites

## Feature Testing

### 1. File Upload Test

1. **Upload Sample CSV**
   - Use the included `sample_players.csv` file
   - Click "Upload CSV" button in Players tab
   - Select the sample file
   - Verify players appear in the table

2. **Validation Tests**
   - Try uploading a non-CSV file (should fail)
   - Try uploading invalid CSV format (should show error)

### 2. Player Management Test

1. **Bulk Selection**
   - Click "Select All" button
   - Verify all players are selected (checkboxes checked)
   - Click "Deselect All" 
   - Verify all players are deselected

2. **Individual Player Controls**
   - Select individual players using checkboxes
   - Modify exposure settings for specific players
   - Verify changes are saved

### 3. Control Panel Test

1. **Configuration**
   - Set number of lineups (try 1, 5, 10)
   - Adjust salary range (min: 45000, max: 50000)
   - Set uniqueness and exposure settings

2. **Validation**
   - Try invalid settings (should show errors)
   - Verify settings are preserved when switching tabs

### 4. Team Stacking Test

1. **Enable Stacking**
   - Toggle "Enable Team Stacking"
   - Select teams from dropdown
   - Set minimum players per team (2-4)

2. **Advanced Settings**
   - Configure multiple team stacks
   - Test different stack sizes

### 5. Optimization Test

1. **Prerequisites**
   - Ensure players are uploaded and selected
   - Set basic configuration in Control Panel

2. **Run Optimization**
   - Click "Run Optimization" button
   - Watch status bar for progress updates
   - Verify lineups are generated

3. **Results Validation**
   - Check that lineups meet position requirements:
     - 2 Pitchers (P)
     - 1 Catcher (C)  
     - 1 First Baseman (1B)
     - 1 Second Baseman (2B)
     - 1 Third Baseman (3B)
     - 1 Shortstop (SS)
     - 3 Outfielders (OF)
   - Verify salary constraints are met
   - Check for duplicate lineups (based on uniqueness settings)

### 6. WebSocket Test

1. **Real-time Updates**
   - Start optimization and watch status bar
   - Should see: "Started", "Progress %", "Completed"
   - Messages should appear in real-time

2. **Error Handling**
   - Try optimization with insufficient players
   - Verify error messages appear

### 7. Export Test

1. **CSV Export**
   - Generate some lineups
   - Click "Export CSV" button
   - Verify download starts
   - Open CSV and check format

2. **DraftKings Format**
   - Export in DraftKings format
   - Verify column headers match DK requirements

## Performance Testing

### 1. Large Player Pool
- Upload CSV with 500+ players
- Test filtering and selection performance
- Verify UI remains responsive

### 2. Multiple Lineups
- Generate 50-100 lineups
- Monitor optimization time
- Check memory usage

### 3. Concurrent Users (if applicable)
- Open multiple browser tabs
- Test WebSocket connections
- Verify data isolation

## Error Testing

### 1. Invalid Data
- Upload CSV with missing required columns
- Try optimization with no players selected
- Test with invalid salary constraints

### 2. Network Issues
- Disconnect network during optimization
- Verify error handling and reconnection

### 3. Edge Cases
- Upload CSV with special characters
- Test with duplicate player names
- Try extreme salary ranges

## Browser Compatibility

Test in multiple browsers:
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Mobile Testing

1. **Responsive Design**
   - Test on mobile devices or use browser dev tools
   - Verify UI scales properly
   - Check touch interactions

2. **Performance**
   - Monitor loading times on mobile
   - Test file upload on mobile browsers

## Troubleshooting Common Issues

### Backend Won't Start
```bash
# Check if port is in use
netstat -ano | findstr :5000

# Kill existing process if needed
taskkill /PID <process_id> /F
```

### Frontend Won't Start  
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

### Optimization Fails
- Verify enough players selected (minimum 10)
- Check position requirements can be met  
- Ensure salary constraints are reasonable

### File Upload Issues
- Check file size (max 10MB)
- Verify CSV format with proper headers
- Test with sample_players.csv first

## Test Results Documentation

Create a test log with:
- Date and time of testing
- Features tested
- Issues found
- Performance observations
- Browser/OS information

## Automated Testing (Future)

Consider adding:
- Unit tests for optimization algorithms
- Integration tests for API endpoints
- E2E tests with Playwright or Cypress
- Performance benchmarks
