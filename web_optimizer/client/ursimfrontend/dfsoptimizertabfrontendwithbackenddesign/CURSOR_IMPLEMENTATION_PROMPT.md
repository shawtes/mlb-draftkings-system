# Cursor Implementation Prompt
## Instructions for Implementing DFS Optimizer in Your Full Stack Application

---

## Copy This Prompt to Cursor

```
I need to integrate the DFS MLB Optimizer into my full stack application.

EXISTING BACKEND:
- Location: optimizer.genetic.algo.py (6,318 lines - FULLY WORKING)
- Technology: Python, PuLP, Pandas, NumPy
- Features: All optimization algorithms are complete and functional
- Classes: OptimizationWorker, GeneticDiversityEngine, FantasyBaseballApp
- DO NOT modify the existing backend algorithms - they work perfectly

MY FRONTEND STACK:
[Tell Cursor about YOUR technology stack, for example:]
- Framework: React / Vue / Angular / Next.js / [YOUR FRAMEWORK]
- Styling: Tailwind / Bootstrap / Material UI / [YOUR DESIGN SYSTEM]
- State Management: Redux / Zustand / Context / [YOUR STATE MGMT]
- API Layer: REST / GraphQL / tRPC / [YOUR API TYPE]

DESIGN DOCUMENTATION:
I have complete design specifications in /DESIGN_DOCS/:
- 00_OVERVIEW.md - System overview and purpose
- 01_ARCHITECTURE.md - Application architecture
- 02_PLAYERS_TAB.md - Player selection interface (553 lines)
- 03_TEAM_STACKS_TAB.md - Team stacking configuration (637 lines)
- 04_STACK_EXPOSURE_TAB.md - Stack exposure management (495 lines)
- 05_TEAM_COMBINATIONS_TAB.md - Combination generator (533 lines)
- 06_ADVANCED_QUANT_TAB.md - Advanced optimization (501 lines)
- 07_MY_ENTRIES_TAB.md - Favorites management (472 lines)
- 08_CONTROL_PANEL.md - Control panel (523 lines)
- 09_DATA_FLOW.md - Complete data flow (650+ lines)
- 10_FRONTEND_BACKEND_INTEGRATION.md - Integration guide (944 lines)

WHAT I NEED YOU TO DO:

1. CREATE API ADAPTER LAYER
   - Create a Flask/FastAPI REST API that wraps the existing backend
   - Use the code examples in 10_FRONTEND_BACKEND_INTEGRATION.md
   - Endpoints needed:
     • POST /api/upload-csv - Upload and validate player CSV
     • POST /api/optimize - Run optimization with settings
     • POST /api/export - Export lineups to DraftKings format
     • GET /api/status - Check optimization status
   - The adapter should:
     • Accept JSON from my frontend
     • Call existing OptimizationWorker class
     • Return JSON results to frontend
     • Handle all data format conversions

2. CREATE FRONTEND COMPONENTS
   Using MY existing design system and component library:
   
   a) LAYOUT (see 01_ARCHITECTURE.md):
      - Main container with left panel (70%) + right panel (30%)
      - Tab navigation for main content area
      - Control panel always visible on right
      - Responsive splitter between panels
      - STYLE: Use my existing layout components and CSS
   
   b) PLAYERS TAB (see 02_PLAYERS_TAB.md):
      - Player data table with sorting and filtering
      - Position sub-tabs (All Batters, C, 1B, 2B, 3B, SS, OF, P)
      - Checkbox selection for each player
      - Columns: Select, Name, Team, Position, Salary, Projected Points, Value
      - Select All / Deselect All buttons
      - STYLE: Use my existing table components
   
   c) TEAM STACKS TAB (see 03_TEAM_STACKS_TAB.md):
      - Stack size sub-tabs (All Stacks, 2 Stack, 3 Stack, 4 Stack, 5 Stack)
      - Team checkboxes for each stack size
      - Team statistics display (Proj Runs, Time, Status)
      - Test Detection button for debugging
      - STYLE: Use my existing card/table components
   
   d) STACK EXPOSURE TAB (see 04_STACK_EXPOSURE_TAB.md):
      - Stack type selection table
      - Stack types: 5, 4, 3, No Stacks, 4|2|2, 4|2, 3|3|2, etc.
      - Min/Max exposure spinboxes
      - Actual exposure display after optimization
      - STYLE: Use my existing form controls
   
   e) TEAM COMBINATIONS TAB (see 05_TEAM_COMBINATIONS_TAB.md):
      - Team selection panel
      - Stack pattern dropdown
      - Auto-generate all combinations button
      - Combinations table with lineups per combo
      - Generate All button
      - STYLE: Use my existing grid/table components
   
   f) ADVANCED QUANT TAB (see 06_ADVANCED_QUANT_TAB.md):
      - Enable/disable master toggle
      - Optimization strategy dropdown
      - Risk parameters (sliders/spinboxes)
      - Monte Carlo settings
      - GARCH parameters
      - Copula settings
      - Kelly Criterion settings
      - Library status display
      - STYLE: Use my existing form components and accordions
   
   g) MY ENTRIES TAB (see 07_MY_ENTRIES_TAB.md):
      - Favorites management table
      - Add to Favorites button
      - Export Favorites button
      - Clear All button
      - Multi-run tracking with run numbers
      - STYLE: Use my existing table and action buttons
   
   h) CONTROL PANEL (see 08_CONTROL_PANEL.md):
      - File upload buttons (Load CSV, Load DK Entries)
      - Optimization settings inputs (Number of Lineups, Min Unique, Min Salary)
      - Risk management section
      - Action buttons (Run Optimization, Save CSV, Fill Entries)
      - Results display table
      - Status bar
      - STYLE: Use my existing sidebar/panel components

3. IMPLEMENT DATA FLOW
   Follow the complete data flow in 09_DATA_FLOW.md:
   
   Frontend State Management:
   ```javascript
   const [playerData, setPlayerData] = useState([]);
   const [selectedPlayers, setSelectedPlayers] = useState([]);
   const [teamSelections, setTeamSelections] = useState({});
   const [stackSettings, setStackSettings] = useState([]);
   const [optimizationSettings, setOptimizationSettings] = useState({});
   const [results, setResults] = useState(null);
   const [isOptimizing, setIsOptimizing] = useState(false);
   ```
   
   API Calls:
   ```javascript
   // Upload CSV
   const uploadCSV = async (file) => {
     const formData = new FormData();
     formData.append('file', file);
     const res = await fetch('/api/upload-csv', {method: 'POST', body: formData});
     const data = await res.json();
     setPlayerData(data.players);
   };
   
   // Run optimization
   const runOptimization = async () => {
     setIsOptimizing(true);
     const res = await fetch('/api/optimize', {
       method: 'POST',
       headers: {'Content-Type': 'application/json'},
       body: JSON.stringify({
         players: playerData,
         included_players: selectedPlayers,
         team_selections: teamSelections,
         stack_settings: stackSettings,
         ...optimizationSettings
       })
     });
     const results = await res.json();
     setResults(results);
     setIsOptimizing(false);
   };
   ```

4. STYLING REQUIREMENTS
   Match my existing design system:
   - Use my color palette: [TELL CURSOR YOUR COLORS]
   - Use my typography: [TELL CURSOR YOUR FONTS]
   - Use my component library: [TELL CURSOR YOUR COMPONENTS]
   - Use my spacing/sizing: [TELL CURSOR YOUR DESIGN TOKENS]
   - Match my button styles
   - Match my form input styles
   - Match my table styles
   - Maintain visual consistency with rest of my app

5. INTEGRATION CHECKLIST
   Ensure these work correctly:
   □ CSV file upload shows players in table
   □ Player selection persists across tab changes
   □ Team selection works for each stack size
   □ Stack type enables/disables properly
   □ Optimization runs and shows progress
   □ Results display with exposure percentages
   □ Export to CSV works
   □ Favorites can be saved and exported
   □ All validation errors display properly
   □ Loading states show during processing
   □ Error messages are user-friendly

6. KEY CONSTRAINTS FROM BACKEND
   Respect these DraftKings rules (hardcoded in backend):
   - Salary cap: $50,000 (fixed)
   - Team size: 10 players (fixed)
   - Positions: 2 P, 1 C, 1 1B, 1 2B, 1 3B, 1 SS, 3 OF
   - Stack sizes: 2-5 players from same team
   - Min salary: Default $45,000 (adjustable)

IMPORTANT NOTES:
- The backend algorithms are production-ready - DO NOT change them
- Focus on building the frontend UI and API adapter
- All algorithm logic is already implemented
- Just need to wire my frontend to call the existing backend
- Use the integration examples in 10_FRONTEND_BACKEND_INTEGRATION.md
- Reference the design docs for what each UI component should do

Please create:
1. API adapter layer (Flask or FastAPI based on my stack)
2. Frontend components matching my design system
3. Integration between frontend and backend
4. Follow the exact data structures documented in the design docs
```

---

## Customization Template

**Add this section with YOUR specific details:**

```
MY SPECIFIC STACK DETAILS:

Frontend Framework: [e.g., Next.js 14 with App Router]
Styling System: [e.g., Tailwind CSS with custom config]
Component Library: [e.g., shadcn/ui components]
State Management: [e.g., Zustand or React Context]
API Type: [e.g., Next.js API routes or separate FastAPI server]

My Design Tokens:
Colors:
  - Primary: #[YOUR_COLOR]
  - Secondary: #[YOUR_COLOR]
  - Success: #[YOUR_COLOR]
  - Warning: #[YOUR_COLOR]
  - Error: #[YOUR_COLOR]

Typography:
  - Font Family: [YOUR_FONT]
  - Heading Sizes: [YOUR_SIZES]
  - Body Size: [YOUR_SIZE]

Component Styles:
  - Buttons: [Describe your button style]
  - Inputs: [Describe your input style]
  - Tables: [Describe your table style]
  - Cards: [Describe your card style]

Layout:
  - Max Width: [e.g., 1920px]
  - Sidebar Width: [e.g., 280px]
  - Spacing Unit: [e.g., 4px base]
  - Border Radius: [e.g., 8px]
```

---

## Example: Complete Cursor Prompt

**Copy and customize this for your use:**

```
I need to integrate a DFS MLB Lineup Optimizer into my existing full stack application.

CONTEXT:
- I have a WORKING Python backend (optimizer.genetic.algo.py) with all algorithms implemented
- I have complete design documentation in /DESIGN_DOCS/ explaining every feature
- I want to build a custom frontend that calls this backend
- The frontend should match MY existing design system

MY STACK:
- Frontend: Next.js 14 with TypeScript
- Styling: Tailwind CSS with my custom design tokens
- Components: Radix UI primitives + custom components
- Backend: Next.js API routes calling Python subprocess
- Database: PostgreSQL (for saving favorites)

TASK 1: Create Python API Adapter
File: /api/optimizer_adapter.py

Based on examples in DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md, create:
- OptimizerAdapter class that wraps existing backend
- Methods: load_players(), run_optimization(), export_lineups()
- All data conversions (DataFrame ↔ JSON)
- Error handling with proper status codes

TASK 2: Create Next.js API Routes
Files: /app/api/optimizer/*

Create these endpoints:
- POST /api/optimizer/upload - Upload CSV, return player data
- POST /api/optimizer/optimize - Run optimization, return results
- POST /api/optimizer/export - Export to DK format
- GET /api/optimizer/status - Check if running

Each endpoint should:
- Call Python adapter via subprocess
- Handle async properly
- Return JSON matching design doc structures
- Include error handling

TASK 3: Create Frontend Components
Location: /components/optimizer/*

Using MY component library and Tailwind classes, create:

A. Layout Component (see DESIGN_DOCS/01_ARCHITECTURE.md)
   - Resizable split panel (70/30)
   - Left: Tab navigation
   - Right: Control panel (fixed)
   - Use my existing layout components

B. Players Tab Component (see DESIGN_DOCS/02_PLAYERS_TAB.md)
   - Data table with sorting
   - Position sub-tabs
   - Checkbox column for selection
   - Use my Table component with Tailwind styling
   - Match my existing table designs

C. Team Stacks Component (see DESIGN_DOCS/03_TEAM_STACKS_TAB.md)
   - Stack size tabs
   - Team checkboxes
   - Use my Tabs and Checkbox components
   - Match my card/panel styling

D. Stack Exposure Component (see DESIGN_DOCS/04_STACK_EXPOSURE_TAB.md)
   - Stack type selection table
   - Min/max exposure inputs
   - Use my Form components
   - Match my input field styles

E. Team Combinations Component (see DESIGN_DOCS/05_TEAM_COMBINATIONS_TAB.md)
   - Team selection grid
   - Combinations generator
   - Results table
   - Use my Grid and Card components

F. Advanced Quant Component (see DESIGN_DOCS/06_ADVANCED_QUANT_TAB.md)
   - Collapsible sections for each parameter group
   - Sliders and number inputs
   - Use my Accordion and Slider components
   - Match my form styling

G. My Entries Component (see DESIGN_DOCS/07_MY_ENTRIES_TAB.md)
   - Favorites table
   - Action buttons
   - Use my Table and Button components

H. Control Panel Component (see DESIGN_DOCS/08_CONTROL_PANEL.md)
   - Vertical layout of controls
   - File upload buttons
   - Settings inputs
   - Action buttons
   - Results display
   - Use my Sidebar component
   - Match my button and input styles

TASK 4: Implement State Management

Create state store for:
```typescript
interface OptimizerState {
  // Data
  playerData: Player[];
  optimizedLineups: Lineup[];
  favorites: Lineup[];
  
  // Selections
  selectedPlayers: string[];
  teamSelections: Record<number, string[]>;
  stackSettings: string[];
  
  // Configuration
  numLineups: number;
  minUnique: number;
  minSalary: number;
  disableKelly: boolean;
  bankroll: number;
  riskTolerance: 'conservative' | 'medium' | 'aggressive';
  
  // UI State
  isOptimizing: boolean;
  currentTab: string;
  
  // Actions
  loadPlayers: (file: File) => Promise<void>;
  runOptimization: () => Promise<void>;
  exportLineups: () => Promise<void>;
  addToFavorites: (count: number) => void;
}
```

TASK 5: Follow Design Specifications

Reference these documents for exact requirements:
- Data structures: DESIGN_DOCS/09_DATA_FLOW.md
- API format: DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md
- UI behavior: Individual tab documents (02-08)
- Validation rules: Each tab document has validation section
- Error handling: Each tab document has error handling section

TASK 6: Match My Design System

Apply my existing styles:
- Color scheme: [YOUR COLORS]
- Typography: [YOUR FONTS]
- Spacing: [YOUR SPACING SYSTEM]
- Border radius: [YOUR RADIUS]
- Shadows: [YOUR SHADOWS]
- Animations: [YOUR TRANSITIONS]

Use my existing components where possible:
- Button component: [YOUR BUTTON COMPONENT]
- Input component: [YOUR INPUT COMPONENT]
- Table component: [YOUR TABLE COMPONENT]
- Card component: [YOUR CARD COMPONENT]
- Tab component: [YOUR TAB COMPONENT]

CRITICAL REQUIREMENTS:
1. DO NOT modify optimizer.genetic.algo.py - it works perfectly
2. Create adapter layer to bridge frontend ↔ backend
3. All data formats must match DESIGN_DOCS specifications
4. Visual design must match MY existing application
5. Performance: Optimization should run in background (async)
6. Error handling: Show user-friendly messages from design docs

DELIVERABLES:
1. Python API adapter (optimizer_adapter.py or Flask/FastAPI app)
2. API routes/endpoints
3. Frontend components for all 7 tabs + control panel
4. State management implementation
5. Integration working end-to-end
6. Styled to match my design system

START WITH:
1. Read DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md for integration patterns
2. Create the API adapter using the OptimizerAdapter class example
3. Build one component at a time (start with Players Tab)
4. Test data flow: Upload CSV → Display → Select → Optimize → Results
5. Add remaining tabs once basic flow works
```

---

## Additional Context You Can Provide

### If Using React/Next.js:

```
ADDITIONAL CONTEXT FOR REACT:

Project Structure:
/app
  /api
    /optimizer
      route.ts  ← API endpoints
  /optimizer
    page.tsx  ← Main optimizer page
/components
  /optimizer
    PlayersTab.tsx
    TeamStacksTab.tsx
    StackExposureTab.tsx
    TeamCombinationsTab.tsx
    AdvancedQuantTab.tsx
    MyEntriesTab.tsx
    ControlPanel.tsx
    OptimizerLayout.tsx

State Management:
- Use Zustand store (see store pattern in my other components)
- Or use React Context if that's what rest of app uses

Component Library:
- Using shadcn/ui components
- Extend Button, Input, Table, Card, Tabs from @/components/ui
- Apply Tailwind classes matching my design tokens

API Communication:
- Use fetch or React Query
- Error handling with toast notifications (using my existing toast system)
- Loading states with my Spinner component
```

### If Using Vue/Nuxt:

```
ADDITIONAL CONTEXT FOR VUE:

Project Structure:
/pages
  /optimizer
    index.vue  ← Main page with tabs
/components
  /optimizer
    PlayersTab.vue
    TeamStacksTab.vue
    StackExposureTab.vue
    [etc...]
/composables
  useOptimizer.ts  ← State and API logic
/server
  /api
    optimizer.ts  ← Nuxt server API

State Management:
- Use Pinia store (like my other features)
- Or use Vue composables pattern

Component Library:
- Using [YOUR VUE COMPONENT LIBRARY]
- Apply my existing component patterns

API Communication:
- Use $fetch (Nuxt) or axios
- Error handling with my notification system
```

### If Building Desktop App (Electron):

```
ADDITIONAL CONTEXT FOR ELECTRON:

Project Structure:
/src
  /main
    main.ts  ← Electron main process
    python_bridge.ts  ← Spawn Python process
  /renderer
    /components
      /optimizer
        [All UI components]

Python Communication:
- Spawn optimizer.genetic.algo.py as subprocess
- Communicate via stdin/stdout with JSON
- Or run Flask server and use HTTP locally

Packaging:
- Include Python in package (PyInstaller or similar)
- Bundle all dependencies
- Ensure backend works in packaged app
```

---

## What Cursor Will Need from You

### 1. Your Design Tokens
```
Provide a file like design-tokens.ts or tailwind.config.js with:
- Color palette
- Typography scale
- Spacing system
- Border radius values
- Shadow definitions
```

### 2. Your Component Examples
```
Show Cursor examples of:
- How you build tables
- How you style buttons
- How you handle forms
- How you do modals/dialogs
- How you show loading states
```

### 3. Your API Pattern
```
Show Cursor:
- How you structure API routes
- How you handle errors
- How you do async operations
- How you manage authentication (if needed)
```

---

## Suggested Cursor Conversation Flow

```
1. YOU: "I need to integrate the DFS optimizer. Here's my stack: [details]
        Read DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md first."

2. CURSOR: [Reads integration doc, asks clarifying questions]

3. YOU: "Start with the API adapter using the OptimizerAdapter example.
        My backend API framework is [Flask/FastAPI/Next.js API]."

4. CURSOR: [Creates adapter, shows you the code]

5. YOU: "Good! Now create the Players Tab component.
        Use my existing Table component from @/components/ui/table.
        Reference DESIGN_DOCS/02_PLAYERS_TAB.md for requirements."

6. CURSOR: [Creates PlayersTab component]

7. YOU: "Now connect it to the API. When user uploads CSV,
        call /api/optimizer/upload and populate the table."

8. CURSOR: [Implements data flow]

9. Continue this pattern for each tab...
```

---

## Testing Your Integration

### Verification Steps

```
After implementation, test this flow:

1. Upload CSV
   ✓ File uploads successfully
   ✓ Players appear in table
   ✓ Count shows correctly

2. Select Players
   ✓ Checkboxes work
   ✓ Select All works
   ✓ Selection persists across tabs
   ✓ Count updates

3. Configure Teams
   ✓ Team checkboxes appear
   ✓ Can select teams per stack size
   ✓ Test Detection shows selections

4. Configure Stacks
   ✓ Can enable stack types
   ✓ Can set exposures

5. Run Optimization
   ✓ Button triggers API call
   ✓ Loading state shows
   ✓ Results return
   ✓ Results display correctly

6. Export
   ✓ Can save CSV
   ✓ File format matches DraftKings
   ✓ Player IDs included (if available)

7. Favorites
   ✓ Can add to favorites
   ✓ Can export favorites
   ✓ Persistence works across sessions
```

---

## Summary

**You have:**
- ✅ Working backend (optimizer.genetic.algo.py)
- ✅ Complete design documentation
- ✅ Integration guide and examples

**Cursor needs to build:**
- API adapter layer (thin wrapper)
- Frontend components (your design system)
- State management (your framework)
- API communication (your patterns)

**The design docs provide:**
- What each component does
- What data flows where
- How components interact
- Validation rules
- Error handling
- Best practices

**You tell Cursor:**
- Your technology stack
- Your design system
- Your component library
- Where to put the code
- Point to the design docs

**Result:**
Your custom-styled DFS optimizer integrated into your full stack app! 🎯

---

**Next Step:** Copy the prompt above, customize with YOUR stack details, and paste into Cursor!

