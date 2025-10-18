# START HERE - Integration Instructions
## How to Use These Design Docs with Cursor

---

## What You Have

✅ **Working Backend:** `optimizer.genetic.algo.py` (6,318 lines of production-ready Python code)
- All optimization algorithms implemented
- Genetic diversity engine
- Risk management
- Monte Carlo simulations
- PuLP linear programming solver
- **DO NOT MODIFY - IT WORKS!**

✅ **Complete Design Docs:** 13 comprehensive specification documents (~7,500 lines)
- Every UI component specified
- All data structures documented
- Integration patterns provided
- Code examples included

---

## What You're Building

**Goal:** Integrate the DFS optimizer into your full stack web application with YOUR custom design

**Approach:** Build a thin API layer + custom frontend that calls the existing backend

---

## Step-by-Step: What to Tell Cursor

### Step 1: Provide Your Tech Stack Info

Tell Cursor:
```
My full stack application uses:

FRONTEND:
- Framework: [React / Vue / Angular / Next.js / etc.]
- Styling: [Tailwind / CSS Modules / Styled Components / etc.]
- Component Library: [shadcn/ui / Material UI / Chakra / Custom / etc.]
- State Management: [Redux / Zustand / Context / Pinia / etc.]

BACKEND:
- Framework: [Next.js API / Flask / FastAPI / Django / Express / etc.]
- Database: [PostgreSQL / MongoDB / MySQL / etc.] (for favorites)
- Deployment: [Vercel / AWS / Docker / etc.]

DESIGN SYSTEM:
- Colors: Primary #[HEX], Secondary #[HEX], etc.
- Fonts: [Font names and sizes]
- Component patterns: [Link to your component library or examples]
```

### Step 2: Give Cursor the Implementation Prompt

**Open:** `CURSOR_IMPLEMENTATION_PROMPT.md`

**Customize the template** with your specific details:
- Replace `[YOUR FRAMEWORK]` with your actual framework
- Replace `[YOUR COLORS]` with your color palette
- Replace `[YOUR COMPONENTS]` with your component library
- Add your design system specifics

**Copy the customized prompt** and paste into Cursor

### Step 3: Tell Cursor the Implementation Order

```
Please implement in this order:

PHASE 1 - API Layer (Start Here):
1. Read DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md
2. Create API adapter that wraps optimizer.genetic.algo.py
3. Implement these endpoints:
   - POST /api/optimizer/upload-csv
   - POST /api/optimizer/optimize  
   - POST /api/optimizer/export
4. Test: Can call backend and get results

PHASE 2 - Basic UI:
1. Read DESIGN_DOCS/01_ARCHITECTURE.md for layout structure
2. Create main layout (split panel: tabs left, control panel right)
3. Create tab navigation component
4. Create control panel component skeleton
5. Test: Layout renders correctly

PHASE 3 - Players Tab (First Feature):
1. Read DESIGN_DOCS/02_PLAYERS_TAB.md
2. Create PlayersTab component using MY table component
3. Implement CSV upload → display players
4. Implement checkbox selection
5. Test: Can load CSV, select players, see selections

PHASE 4 - Team Configuration:
1. Read DESIGN_DOCS/03_TEAM_STACKS_TAB.md
2. Create TeamStacksTab component
3. Implement team checkboxes per stack size
4. Read DESIGN_DOCS/04_STACK_EXPOSURE_TAB.md
5. Create StackExposureTab component
6. Test: Can configure teams and stacks

PHASE 5 - Optimization Flow:
1. Read DESIGN_DOCS/09_DATA_FLOW.md
2. Implement state management for all settings
3. Wire "Run Optimization" button to API
4. Display results in control panel
5. Test: Full flow works (Upload → Configure → Optimize → Results)

PHASE 6 - Advanced Features:
1. Team Combinations tab (DESIGN_DOCS/05_TEAM_COMBINATIONS_TAB.md)
2. Advanced Quant tab (DESIGN_DOCS/06_ADVANCED_QUANT_TAB.md)
3. My Entries tab (DESIGN_DOCS/07_MY_ENTRIES_TAB.md)
4. Export functionality

PHASE 7 - Polish:
1. Error handling (each doc has error handling section)
2. Loading states
3. Validation messages
4. Visual polish
```

### Step 4: Reference Documents During Build

**When Cursor asks "How should X work?"**

Point to the relevant document:
- "Check DESIGN_DOCS/02_PLAYERS_TAB.md section 3 for table columns"
- "See DESIGN_DOCS/09_DATA_FLOW.md for the complete optimization flow"
- "Reference DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md for data format"

---

## Example: Complete Cursor Conversation

### Your First Message to Cursor:

```
I need to integrate a DFS MLB lineup optimizer into my full stack application.

EXISTING BACKEND:
- File: optimizer.genetic.algo.py (6,318 lines)
- Status: FULLY WORKING - all algorithms implemented
- DO NOT modify this file - just wrap it with an API

MY TECH STACK:
- Frontend: Next.js 14 with TypeScript and Tailwind CSS
- Components: shadcn/ui component library
- State: Zustand for state management
- API: Next.js API routes
- Backend: Python subprocess calls
- Database: PostgreSQL (for favorites persistence)

DESIGN DOCUMENTATION:
Complete specifications in /DESIGN_DOCS/:
- Start by reading: 10_FRONTEND_BACKEND_INTEGRATION.md
- Then reference other docs as needed

FIRST TASK:
Create a Python API adapter (optimizer_adapter.py) that:
1. Wraps the existing OptimizationWorker class
2. Accepts JSON from my Next.js frontend
3. Calls the existing backend methods
4. Returns JSON results

Use the OptimizerAPI class example from:
DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md (starting around line 100)

After you create the adapter, I'll tell you what to build next.
```

### Cursor Will Respond:

```
I'll create the API adapter. First, let me read the integration guide...

[Cursor reads 10_FRONTEND_BACKEND_INTEGRATION.md]

Here's the optimizer_adapter.py based on the design docs:
[Shows code]

This adapter provides:
- load_players_from_json()
- run_optimization()
- serialize_results()
- export_to_dk_format()

Should I proceed with the Next.js API routes next?
```

### You Continue:

```
Yes, create Next.js API routes in /app/api/optimizer/:
- route.ts for main endpoints
- Use the adapter we just created
- Reference DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md for endpoint structure
```

### Then Build UI:

```
Now create the Players Tab component:
- Read DESIGN_DOCS/02_PLAYERS_TAB.md
- Use shadcn/ui Table component
- Match my existing table styling (show example from your app)
- Implement checkbox selection
- Position sub-tabs
```

---

## What Each Document Tells Cursor

| Document | What Cursor Learns | When to Reference |
|----------|-------------------|-------------------|
| 00_OVERVIEW.md | System purpose, tech stack | Initial understanding |
| 01_ARCHITECTURE.md | Layout structure, component hierarchy | Building layout |
| 02_PLAYERS_TAB.md | Player table specs, columns, interactions | Building Players tab |
| 03_TEAM_STACKS_TAB.md | Team selection UI, stack logic | Building Team Stacks |
| 04_STACK_EXPOSURE_TAB.md | Stack type configuration | Building Stack Exposure |
| 05_TEAM_COMBINATIONS_TAB.md | Combination generator logic | Building Combinations |
| 06_ADVANCED_QUANT_TAB.md | Advanced settings UI | Building Advanced tab |
| 07_MY_ENTRIES_TAB.md | Favorites management | Building My Entries |
| 08_CONTROL_PANEL.md | Control panel layout, all settings | Building right panel |
| 09_DATA_FLOW.md | How data moves, state management | Connecting components |
| 10_FRONTEND_BACKEND_INTEGRATION.md | API patterns, data formats | Building API layer |
| CURSOR_IMPLEMENTATION_PROMPT.md | Complete implementation plan | Starting the project |
| QUICK_REFERENCE.md | Fast lookup for common tasks | During development |

---

## Common Cursor Questions & Answers

**Q: "Should I modify optimizer.genetic.algo.py?"**
A: No! It's working perfectly. Just wrap it with an API adapter.

**Q: "What data format should the API use?"**
A: See DESIGN_DOCS/10_FRONTEND_BACKEND_INTEGRATION.md "Data Format Conversions" section

**Q: "How should the Players table look?"**
A: See DESIGN_DOCS/02_PLAYERS_TAB.md, use your own styling

**Q: "What's the team_selections format?"**
A: See DESIGN_DOCS/QUICK_REFERENCE.md "Team Selection Format"

**Q: "How do I run the optimization?"**
A: See DESIGN_DOCS/QUICK_REFERENCE.md "Backend Function Calls"

**Q: "What validation rules should I implement?"**
A: Each tab document (02-08) has a "Validation" section

**Q: "How should errors be displayed?"**
A: Each tab document has an "Error Handling" section

---

## Success Criteria

Your integration is successful when:

✅ Can upload CSV and see players in your styled table  
✅ Can select players using your checkboxes  
✅ Can configure team stacks and see selections  
✅ Can click "Run Optimization" and see loading state  
✅ Results return and display in your results component  
✅ Can export to CSV in DraftKings format  
✅ Can add to favorites and persist across sessions  
✅ Everything matches YOUR design system  

---

## Support Documents

**If Cursor gets stuck, point it to:**

- **Data format issues?** → `QUICK_REFERENCE.md` "Data Structures"
- **API integration issues?** → `10_FRONTEND_BACKEND_INTEGRATION.md`
- **UI component questions?** → Specific tab document (02-08)
- **State management?** → `09_DATA_FLOW.md` "State Synchronization"
- **Validation logic?** → Tab documents have validation sections
- **Performance concerns?** → `01_ARCHITECTURE.md` "Performance Optimizations"

---

## Final Checklist

Before giving to Cursor:

- [ ] I've customized CURSOR_IMPLEMENTATION_PROMPT.md with my tech stack
- [ ] I've specified my frontend framework (React/Vue/etc.)
- [ ] I've provided my design system details (colors, fonts, components)
- [ ] I've specified my API approach (REST/GraphQL/etc.)
- [ ] I've indicated where code should go in my project structure
- [ ] I'm ready to iterate with Cursor as it builds each component

---

## You're Ready! 🚀

1. Open `CURSOR_IMPLEMENTATION_PROMPT.md`
2. Fill in YOUR tech stack details (marked with [YOUR_*])
3. Copy the customized prompt
4. Start new Cursor conversation
5. Paste the prompt
6. Let Cursor read the design docs and build your integration!

The design docs contain everything Cursor needs to know about:
- What to build (UI components)
- How it works (data flow)
- What format to use (data structures)
- How to connect (integration patterns)

**Your backend is done. Your design is documented. Time to build! 💪**

