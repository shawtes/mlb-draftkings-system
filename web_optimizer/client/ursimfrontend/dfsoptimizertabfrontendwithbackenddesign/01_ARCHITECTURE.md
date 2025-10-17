# Architecture & Layout Design
## Application Structure and Navigation

---

## Overall Layout

### Window Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ Advanced MLB DFS Optimizer                            [─][□][×] │
├─────────────────────────────────────┬───────────────────────────┤
│                                     │                           │
│  ┌───────────────────────────────┐ │   CONTROL PANEL          │
│  │ TAB NAVIGATION                │ │                           │
│  ├───────────────────────────────┤ │   • Load CSV              │
│  │ Players | Team Stacks |       │ │   • Number of Lineups     │
│  │ Stack Exposure |               │ │   • Min Unique            │
│  │ Team Combinations |            │ │   • Min Salary            │
│  │ Advanced Quant | My Entries   │ │   • Stack Settings        │
│  └───────────────────────────────┘ │   • Risk Management       │
│                                     │   • Run Optimization      │
│  ┌───────────────────────────────┐ │   • Save/Export           │
│  │                               │ │                           │
│  │  ACTIVE TAB CONTENT           │ │   ┌─────────────────────┐│
│  │  (Table or Form Interface)    │ │   │ RESULTS TABLE       ││
│  │                               │ │   │                     ││
│  │                               │ │   │ • Player            ││
│  │                               │ │   │ • Position          ││
│  │                               │ │   │ • Salary            ││
│  │                               │ │   │ • Projected Points  ││
│  │                               │ │   │ • Exposure %        ││
│  └───────────────────────────────┘ │   └─────────────────────┘│
│                                     │                           │
├─────────────────────────────────────┴───────────────────────────┤
│ Status: Ready | Players: 0 | Lineups: 0 | Last Run: Never       │
└─────────────────────────────────────────────────────────────────┘
```

### Dimensions
- **Window Size:** 1600x1000 pixels (default)
- **Minimum Size:** 1200x800 pixels
- **Splitter Ratio:** 70% (left tabs) / 30% (right control panel)
- **Splitter:** Draggable to adjust ratio

---

## Component Hierarchy

```
QMainWindow (FantasyBaseballApp)
│
├── Central Widget
│   └── Main Layout (Vertical)
│       ├── QSplitter (Horizontal)
│       │   │
│       │   ├── LEFT SECTION (70%)
│       │   │   └── QTabWidget (tabs)
│       │   │       ├── Tab 0: Players
│       │   │       ├── Tab 1: Team Stacks
│       │   │       ├── Tab 2: Stack Exposure
│       │   │       ├── Tab 3: Team Combinations
│       │   │       ├── Tab 4: Advanced Quant
│       │   │       └── Tab 5: My Entries
│       │   │
│       │   └── RIGHT SECTION (30%)
│       │       └── Control Panel (Frame)
│       │           ├── File Operations
│       │           ├── Optimization Settings
│       │           ├── Risk Management
│       │           ├── Action Buttons
│       │           └── Results Display
│       │
│       └── Status Bar
│
└── Background Worker Threads
    └── OptimizationWorker (QThread)
```

---

## Navigation Flow

### Primary Navigation
**Tab-Based Interface** - Users switch between major功能 using tabs

```
User Flow:
1. Load Data (Control Panel) → Players Tab populates
2. Select Players (Players Tab) → Mark desired players
3. Configure Stacks (Team Stacks Tab) → Select teams for stacking
4. Set Constraints (Stack Exposure Tab) → Define stack requirements
5. Advanced Options (Advanced Quant Tab) → Optional: fine-tune algorithms
6. Run Optimization (Control Panel) → Generate lineups
7. Review Results (Control Panel Results Table) → See generated lineups
8. Manage Favorites (My Entries Tab) → Optional: save best lineups
9. Export (Control Panel) → Save to CSV for DraftKings
```

### Secondary Navigation
- **Keyboard Shortcuts** (future implementation)
  - `Ctrl+O` - Open CSV
  - `Ctrl+R` - Run Optimization
  - `Ctrl+S` - Save Results
  - `Ctrl+Tab` - Next Tab
  - `Ctrl+Shift+Tab` - Previous Tab

---

## Data Flow Architecture

```
┌─────────────┐         ┌──────────────────┐
│  CSV File   │────────>│  Load Players    │
└─────────────┘         └──────────────────┘
                                │
                                ↓
                    ┌─────────────────────────┐
                    │  df_players (DataFrame) │
                    │  • Name                 │
                    │  • Position             │
                    │  • Salary               │
                    │  • Predicted_DK_Points  │
                    └─────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ↓               ↓               ↓
        ┌──────────────┐ ┌────────────┐ ┌─────────────┐
        │ Players Tab  │ │Team Stacks │ │ Combinations│
        │ (Display)    │ │   Tab      │ │     Tab     │
        └──────────────┘ └────────────┘ └─────────────┘
                                │
                                ↓
                    ┌────────────────────────┐
                    │  User Configuration    │
                    │  • selected_players    │
                    │  • team_selections     │
                    │  • stack_settings      │
                    │  • min_unique          │
                    │  • min_salary          │
                    └────────────────────────┘
                                │
                                ↓
                    ┌────────────────────────┐
                    │ OptimizationWorker     │
                    │  (Background Thread)   │
                    │                        │
                    │  1. preprocess_data()  │
                    │  2. optimize_lineups() │
                    │  3. apply_constraints()│
                    └────────────────────────┘
                                │
                                ↓
                    ┌────────────────────────┐
                    │  Results Dictionary    │
                    │  {                     │
                    │   0: {lineup: df,      │
                    │        total_points,   │
                    │        total_salary},  │
                    │   1: {...},            │
                    │   ...                  │
                    │  }                     │
                    └────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ↓               ↓               ↓
        ┌──────────────┐ ┌────────────┐ ┌─────────────┐
        │Results Table │ │ My Entries │ │  Export CSV │
        │  (Display)   │ │    Tab     │ │    File     │
        └──────────────┘ └────────────┘ └─────────────┘
```

---

## State Management

### Application States

```python
class AppState:
    INITIAL = "initial"           # No data loaded
    DATA_LOADED = "data_loaded"   # CSV imported, ready to configure
    CONFIGURING = "configuring"   # User adjusting settings
    OPTIMIZING = "optimizing"     # Background optimization running
    RESULTS_READY = "results"     # Lineups generated, ready to export
    ERROR = "error"               # Something went wrong
```

### State Transitions

```
INITIAL
  │
  ├─[Load CSV]──────────────────> DATA_LOADED
  │                                     │
  └─[Error]────────────> ERROR         │
                           │            │
                           └────────────┘
                                        │
                                        ├─[Adjust Settings]─> CONFIGURING
                                        │                          │
                                        ↓                          │
                                  [Run Optimization]               │
                                        │                          │
                                        ↓                          │
                                   OPTIMIZING                      │
                                        │                          │
                                        ├─[Success]─────────> RESULTS_READY
                                        │                          │
                                        ├─[Error]──────────> ERROR │
                                        │                          │
                                        └──────────────────────────┘
                                                                   │
                                                          [Export/Save New Run]
                                                                   │
                                                                   ↓
                                                             DATA_LOADED
```

### UI Element State

| State | Tabs | Load Button | Run Button | Export Button | Input Fields |
|-------|------|-------------|------------|---------------|--------------|
| INITIAL | Empty | Enabled | Disabled | Disabled | Disabled |
| DATA_LOADED | Populated | Enabled | Enabled | Disabled | Enabled |
| CONFIGURING | Populated | Enabled | Enabled | Disabled | Enabled |
| OPTIMIZING | Locked | Disabled | Disabled (Cancel shown) | Disabled | Locked |
| RESULTS_READY | Populated | Enabled | Enabled | Enabled | Enabled |
| ERROR | Varies | Enabled | Depends | Disabled | Enabled |

---

## Threading Architecture

### Main Thread (GUI Thread)
**Responsibilities:**
- Handle all UI updates
- Process user input events
- Update tables and displays
- Show dialogs and messages

### Worker Thread (OptimizationWorker)
**Responsibilities:**
- Run optimization algorithms
- Process large datasets
- Perform heavy calculations
- Emit progress signals

### Communication
```python
# Signal-Slot Architecture (PyQt5)

OptimizationWorker.optimization_done.connect(MainWindow.display_results)
#                  ↑                                    ↑
#              Signal emitted                    Slot method called
#           (from worker thread)              (in main GUI thread)
```

**Signals Emitted:**
- `optimization_done(results, team_exposure, stack_exposure)` - Optimization complete
- `progress_update(current, total)` - Progress tracking (future)
- `error_occurred(error_message)` - Error handling (future)

---

## Memory Management

### Data Retention

**Persistent Data** (kept in memory):
- `self.df_players` - Original player DataFrame
- `self.optimized_lineups` - Last optimization results
- `self.favorites_lineups` - Saved favorite lineups
- `self.player_exposure` - Exposure tracking dictionary

**Transient Data** (cleared after use):
- Worker thread instances
- Temporary CSV data during import
- Dialog form data

### Large Dataset Handling

For 200+ players with 500+ lineups:
```python
# Efficient storage
lineups = []  # List of DataFrames (lightweight)

# Instead of:
lineups = full_dataframe.copy()  # Heavy duplication

# Use views where possible
lineup_view = df_players[df_players['Position'] == 'P']
```

---

## Styling & Theming

### Color Scheme
```python
# Primary Colors
BACKGROUND = "#FFFFFF"
PANEL_BG = "#F5F5F5"
BORDER = "#DDDDDD"

# Accent Colors
PRIMARY = "#2196F3"     # Blue - Main actions
SUCCESS = "#4CAF50"     # Green - Success states
WARNING = "#FF9800"     # Orange - Warnings
ERROR = "#F44336"       # Red - Errors
INFO = "#9C27B0"        # Purple - Info

# Text Colors
TEXT_PRIMARY = "#212121"
TEXT_SECONDARY = "#757575"
TEXT_DISABLED = "#BDBDBD"
```

### Typography
```python
# Font Sizes
TITLE = "16px"
HEADER = "14px"
BODY = "12px"
SMALL = "10px"

# Font Weights
BOLD = "bold"
NORMAL = "normal"
```

### Component Styling

**Buttons:**
```css
QPushButton {
    background-color: #2196F3;
    color: white;
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 4px;
}

QPushButton:hover {
    background-color: #1976D2;
}

QPushButton:disabled {
    background-color: #BDBDBD;
}
```

**Tables:**
```css
QTableWidget {
    alternate-row-colors: true;
    gridline-color: #DDDDDD;
}

QTableWidget::item:selected {
    background-color: #2196F3;
    color: white;
}
```

---

## Responsive Behavior

### Window Resizing
- **Splitter adapts** - Ratio maintained but adjustable
- **Tables resize** - Columns auto-adjust to available width
- **Control panel** - Vertical scroll if content exceeds height
- **Minimum size enforced** - Window cannot shrink below 1200x800

### Table Column Management
- **Fixed Width Columns:** Checkboxes (50px), Actions (80px)
- **Auto-Resize Columns:** Name, Team (content-based)
- **Proportional Columns:** Others split remaining space

---

## Accessibility Considerations

### Keyboard Navigation
- **Tab Order:** Logical left-to-right, top-to-bottom
- **Enter Key:** Activates default button in context
- **Escape Key:** Closes dialogs
- **Arrow Keys:** Navigate table cells

### Visual Feedback
- **Focus Indicators:** Blue border on focused elements
- **Hover States:** Light background change
- **Active States:** Darker press effect
- **Disabled States:** Grayed out appearance

### Screen Reader Support (Future)
- **Labels:** All inputs have accessible labels
- **Alt Text:** Buttons have descriptive text
- **Announcements:** Status updates announced

---

## Error States & Edge Cases

### No Data Loaded
**Display:**
```
┌─────────────────────────────────┐
│  No data loaded                 │
│                                 │
│  Click "Load CSV" to begin     │
│            ↑                    │
└─────────────────────────────────┘
```

### Optimization Running
**Display:**
```
┌─────────────────────────────────┐
│  ⏳ Optimizing...               │
│  [████████░░░░░░] 60%          │
│  [Cancel Optimization]          │
└─────────────────────────────────┘
```

### No Results Found
**Display:**
```
┌─────────────────────────────────┐
│  ⚠ No valid lineups generated   │
│                                 │
│  Possible causes:               │
│  • Constraints too restrictive  │
│  • Not enough eligible players  │
│  • Salary cap impossible        │
│                                 │
│  Try:                           │
│  • Selecting more players       │
│  • Relaxing stack requirements  │
│  • Lowering min salary          │
└─────────────────────────────────┘
```

---

## Performance Optimizations

### Lazy Loading
- **Player tables:** Only populate visible position tabs
- **Results:** Load first 50 lineups, rest on scroll
- **Images:** No images currently (lightweight)

### Caching
- **Table sorting:** Cache sort results
- **Calculations:** Memoize expensive functions
- **Validation:** Cache validation results

### Background Processing
- **All optimization** - Worker thread
- **Large CSV parsing** - Chunked reading (future)
- **Export operations** - Async writing (future)

---

## Next: Individual Tab Documentation

Now proceed to detailed specifications for each tab:
- [Players Tab](02_PLAYERS_TAB.md)
- [Team Stacks Tab](03_TEAM_STACKS_TAB.md)
- [Stack Exposure Tab](04_STACK_EXPOSURE_TAB.md)
- [Team Combinations Tab](05_TEAM_COMBINATIONS_TAB.md)
- [Advanced Quant Tab](06_ADVANCED_QUANT_TAB.md)
- [My Entries Tab](07_MY_ENTRIES_TAB.md)
- [Control Panel](08_CONTROL_PANEL.md)

