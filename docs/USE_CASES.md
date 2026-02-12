# UrSim DFS Optimizer — Use Case Document

**Project:** UrSim DFS Optimization System
**Team:** SWE Group
**Date:** February 2026

---

## Use Case 1: Upload Player CSV

### Description
1. The user clicks the "CSV" button in the Build Control Bar.
2. The system opens a file picker dialog filtered to `.csv` files.
3. The user selects a DraftKings or custom player CSV file.
4. The system detects the sport (NFL, NBA, or MLB) from column headers.
5. The system displays a CSV Preview modal showing detected columns and sample rows.
6. The user reviews column mappings (Name, Team, Pos, Salary, Projection, Ownership, etc.) and adjusts if needed.
7. The user clicks "Confirm & Upload."
8. The system parses all players, populates the Players tab, auto-selects all players, and initializes stack settings for the detected sport.

### Exception Paths
1. If the CSV file is malformed or empty, the system displays an error toast and returns to step 1.
2. If the sport cannot be auto-detected, the system prompts the user to manually select NFL, NBA, or MLB before proceeding.
3. If required columns (Name, Salary) are missing, the system highlights unmapped fields in red and disables the Confirm button.

### Alternate Paths
1. The user can manually remap any column via the dropdown selectors in the CSV Preview modal (e.g., map "FPTS" to "Projection").
2. The user can cancel the upload at step 6 and return to the previous state.

### Prerequisites
1. The user has navigated to the DFS Optimizer page.
2. A valid DraftKings export CSV or custom projection CSV exists on the user's device.

### Postrequisites
1. The Players tab is populated with player data.
2. All players are selected by default.
3. The sport mode is locked to the detected sport for the current build.
4. Stack settings are initialized with sport-specific stack types.

---

## Use Case 2: Configure Stack Types

### Description
1. The user clicks the "Stack Types" tab in the main content area.
2. The system displays a table of sport-specific stack types with checkboxes, correlation badges, and min/max exposure inputs.
3. The user enables or disables individual stack types by clicking the checkbox or the row.
4. The user sets minimum and maximum exposure percentages for each enabled stack type.
5. The system validates that total minimum exposure does not exceed 100%.
6. The system displays a status bar showing active stack count, total min/max percentages, and remaining headroom.

### Exception Paths
1. If total minimum exposure exceeds 100%, the system displays a yellow warning banner and the optimizer will reject the configuration.
2. If no stack types are enabled, the system displays a red warning prompting the user to select at least one.
3. If the user sets min > max for a stack type, the system auto-adjusts the other value to match.

### Alternate Paths
1. The user can click "Enable All" to activate all stack types at once.
2. The user can click "Disable All" to deactivate all stack types.
3. The user can click "Reset Ranges" to restore all min/max values to 0/100 defaults.
4. The user can hover over the info icon next to each stack type to see its correlation description.

### Prerequisites
1. A CSV has been uploaded and players are loaded.
2. The sport mode is set (determines which stack types are shown).

### Postrequisites
1. Stack type settings are stored in the current build state.
2. The optimizer will respect the enabled stack types and exposure ranges during lineup generation.

---

## Use Case 3: Configure Team Stacks

### Description
1. The user clicks the "Stacks" tab in the main content area.
2. The system displays a table of all teams derived from loaded player data, with checkboxes, team stats, and per-team min/max exposure inputs.
3. The user selects a stack size sub-tab (All, 2-Stack, 3-Stack, 4-Stack, or 5-Stack).
4. The user checks teams they want included in stacks for the selected stack size.
5. The user optionally sets per-team minimum and maximum exposure percentages.
6. The system displays a status bar showing selected teams and their count for the current stack size.

### Exception Paths
1. If a team has fewer players than the selected stack size, the team row is grayed out and disabled with a tooltip explaining the insufficient player count.
2. If no teams are selected, the optimizer will not generate stacked lineups for that stack size.

### Alternate Paths
1. The user can click "Select All" to check all eligible teams for the current stack size.
2. The user can click "Deselect All" to uncheck all teams.
3. The user can configure different team selections for each stack size independently (e.g., different teams for 3-stack vs 5-stack).

### Prerequisites
1. A CSV has been uploaded and players are loaded.
2. Player data includes valid team abbreviations.

### Postrequisites
1. Team selections and per-team exposures are stored in the current build state.
2. The optimizer will generate lineups respecting team stack selections and exposure limits.

---

## Use Case 4: Lock and Exclude Players

### Description
1. The user navigates to the "Players" tab.
2. The user locates a player in the table using search, position filters, or scrolling.
3. To lock a player into all lineups, the user clicks the lock icon in the player's row.
4. The locked player's row highlights green and the lock icon fills in.
5. To exclude a player from all lineups, the user clicks the exclude (X) icon in the player's row.
6. The excluded player's row highlights red and is marked with a strikethrough.

### Exception Paths
1. If the user attempts to lock and exclude the same player, the system treats them as mutually exclusive — locking clears exclude status and vice versa.
2. If too many players are locked (exceeding lineup size), the optimizer will return an error indicating the constraint is infeasible.

### Alternate Paths
1. The user can click the lock/exclude icon again to toggle the status off, returning the player to normal selection.
2. The user can use the "Leverage Plays" filter to quickly find high-value players worth locking.
3. The user can bulk select/deselect players using the header checkbox.

### Prerequisites
1. A CSV has been uploaded and players are loaded in the Players tab.

### Postrequisites
1. Locked players will appear in every generated lineup.
2. Excluded players will not appear in any generated lineup.
3. The player's lock/exclude status is stored in the current build state.

---

## Use Case 5: Run Lineup Optimization

### Description
1. The user configures desired settings: number of lineups, minimum unique players, salary range, stack types, and team stacks.
2. The user clicks the "BUILD LINEUPS" button in the Build Control Bar.
3. The system sends player data, selections, stack settings, team exposures, and quant settings to the backend API.
4. The backend optimizer generates lineups using the selected strategy (greedy, projection, balanced, value, or quant-scored).
5. A progress indicator shows optimization status.
6. The system populates the Lineups tab with generated results and displays lineup cards in the Sidebar.
7. If quant engine is enabled, each lineup includes VaR, Sharpe ratio, and ceiling probability metrics.

### Exception Paths
1. If no players are selected, the system displays an error: "Select players before optimizing."
2. If a required position cannot be filled (not enough eligible players), the optimizer throws a position constraint error.
3. If the server is unreachable, the system displays a connection error toast.
4. If the optimizer generates fewer lineups than requested (due to diversity constraints), the system returns the partial set with a warning.

### Alternate Paths
1. The user can cancel optimization while it is running.
2. The user can adjust settings and re-run optimization, replacing previous results.
3. The user can switch to the "Quant" tab to enable the quantitative engine before running optimization.

### Prerequisites
1. A CSV has been uploaded and players are loaded.
2. At least enough players are selected to fill all required positions for the sport.
3. The backend server is running (port 5001).

### Postrequisites
1. Generated lineups are displayed in the Lineups tab and Sidebar.
2. If quant is enabled, portfolio metrics (Sharpe, uniqueness, concentration) are displayed above the lineups.
3. Results are stored in the current build state and persist when switching tabs.

---

## Use Case 6: Configure Quant Engine Settings

### Description
1. The user clicks the "Quant" tab in the main content area.
2. The system displays the Advanced Quant Settings panel with toggles and parameter inputs.
3. The user enables the quant engine by toggling the master enable switch.
4. The user selects an optimization strategy from the dropdown (Combined, Kelly, Mean-Variance, Risk Parity, or Equal Weight).
5. The user adjusts parameters: Monte Carlo simulations count, VaR confidence level, Kelly fraction, risk tolerance, and GARCH settings.
6. The user selects the contest mode (GPP or Cash) which changes how player scoring weights are applied.

### Exception Paths
1. If the quant engine is enabled but Monte Carlo simulation count is set below 100, the system warns that results may be unreliable.
2. If GARCH is enabled but the Python `arch` library is not installed, the system falls back to static standard deviation.

### Alternate Paths
1. The user can leave the quant engine disabled and the optimizer will use standard projection-based optimization.
2. The user can choose "Cash Mode" for floor-optimized, low-variance lineups or "GPP Mode" for ceiling-optimized, high-leverage lineups.

### Prerequisites
1. A CSV has been uploaded and players are loaded.
2. Player data ideally includes ceiling, floor, and standard deviation fields for full quant functionality.

### Postrequisites
1. Quant settings are stored in the current build state.
2. The next optimization run will use the quant engine for player scoring, Kelly exposure limits, Monte Carlo simulation, and portfolio analysis.
3. Lineup results will include per-lineup quantitative metrics (VaR, Sharpe, ceiling probability).

---

## Use Case 7: Export Lineups to DraftKings

### Description
1. The user generates lineups via the optimizer (Use Case 5).
2. The user clicks the "Export" button in the Build Control Bar or within the Lineups tab.
3. The system formats the lineups into DraftKings-compatible CSV format with correct position columns for the active sport.
4. The browser downloads the CSV file named `{sport}_lineups_{date}.csv`.
5. The user uploads the CSV to DraftKings contest entry page.

### Exception Paths
1. If no lineups have been generated, the system displays an alert: "No lineups to export."
2. If the export API fails, the system displays an error toast.

### Alternate Paths
1. The user can export in different formats if multiple export options are available.
2. The user can save individual lineups as favorites before exporting (Use Case 9).

### Prerequisites
1. At least one lineup has been generated.
2. The backend server is running.

### Postrequisites
1. A CSV file is downloaded to the user's device.
2. The CSV is formatted for direct upload to DraftKings.

---

## Use Case 8: Blend Multiple Projection Sources

### Description
1. The user clicks the "Blend" button in the Build Control Bar.
2. The system opens the Projection Blending modal.
3. The user clicks "Add Source" to upload a projection CSV file.
4. The system parses the CSV and adds it as a named projection source with a default weight.
5. The user repeats step 3 to add additional projection sources (e.g., FantasyLabs, NumberFire, custom model).
6. The user adjusts the weight percentage for each source (e.g., 50% Source A, 30% Source B, 20% Source C).
7. The user clicks "Blend & Apply."
8. The system computes weighted average projections for each player and updates the Players tab.

### Exception Paths
1. If a projection source CSV cannot be matched to existing players by name, those entries are skipped.
2. If all source weights are set to 0, the Blend & Apply button is disabled.
3. If no projection sources have been added, the modal shows "No projection sources added yet."

### Alternate Paths
1. The user can remove individual projection sources by clicking the X button next to each source.
2. The user can cancel the blending process and keep original projections.

### Prerequisites
1. A base CSV has been uploaded and players are loaded.
2. Additional projection source CSV files exist on the user's device with matching player names.

### Postrequisites
1. Player projections in the Players tab are updated to blended values.
2. Original projections are preserved and can be viewed via the projection sources tooltip.
3. Blended projections are used in subsequent optimization runs.

---

## Use Case 9: Save and Manage Favorite Lineups

### Description
1. The user generates lineups via the optimizer.
2. The user navigates to the Lineups tab and reviews generated lineups.
3. The user clicks the "Save" or star icon on a lineup they want to keep.
4. The system saves the lineup to the favorites store with a timestamp and the current sport.
5. The user navigates to the "Entries" tab to view all saved favorites.
6. The user can review, compare, or remove saved lineups from favorites.

### Exception Paths
1. If the favorites store is full or the save API fails, the system displays an error toast.
2. If the user tries to save a duplicate lineup (same players), the system warns that it already exists.

### Alternate Paths
1. The user can save multiple lineups at once by selecting them in the Lineups tab.
2. The user can organize favorites by run number or date.

### Prerequisites
1. At least one lineup has been generated.

### Postrequisites
1. The lineup is persisted in the favorites file on the server.
2. Saved lineups are available across sessions (44 favorites currently loaded on server start).

---

## Use Case 10: Multi-Build Workflow

### Description
1. The user clicks the "+" button in the Build Control Bar to create a new build tab.
2. The system creates a new build with default empty state and switches to it.
3. The user selects a sport for the new build (can be different from other builds).
4. The user uploads a CSV and configures settings independently for this build.
5. The user switches between builds by clicking on build tabs in the Build Control Bar.
6. Each build maintains its own isolated state: sport, players, selections, stack settings, quant settings, and results.

### Exception Paths
1. If the user tries to remove the last remaining build, the system prevents deletion (at least one build must exist).
2. If the user switches builds during an active optimization, the optimization continues in the background for the original build.

### Alternate Paths
1. The user can rename builds by editing the build tab label.
2. The user can remove builds by clicking the X on the build tab.
3. The user can create separate builds for different slates (e.g., main slate vs early slate) of the same sport.

### Prerequisites
1. The user has navigated to the DFS Optimizer page.

### Postrequisites
1. Multiple builds exist with fully isolated state.
2. The user can compare results across builds by switching tabs.
3. Each build can be independently optimized and exported.

---

## Use Case 11: View Portfolio Analytics

### Description
1. The user enables the quant engine in the "Quant" tab (Use Case 6).
2. The user runs lineup optimization (Use Case 5).
3. The system generates lineups with per-lineup quantitative metrics.
4. The user navigates to the Lineups tab.
5. The system displays a Portfolio Analysis bar at the top showing: Portfolio Sharpe Ratio, Average Uniqueness, Max Exposure, and Exposure Concentration (Herfindahl index).
6. Each lineup card in the Sidebar shows inline quant badges: VaR (blue), Sharpe (green/yellow/red), and Ceiling Probability (purple).
7. The user reviews the Players tab to see Boom% and Leverage columns for each player.

### Exception Paths
1. If the quant engine is disabled, no portfolio metrics or quant badges are displayed.
2. If player data lacks ceiling/floor/stdDev fields, the quant engine uses default estimates which may reduce accuracy.

### Alternate Paths
1. The user can click "Leverage Plays" filter in the Players tab to show only high-leverage, low-ownership players.
2. The user can sort lineups by Sharpe ratio instead of total projection.

### Prerequisites
1. The quant engine is enabled with valid settings.
2. Lineups have been generated.

### Postrequisites
1. Portfolio-level metrics are displayed for cross-lineup analysis.
2. Per-lineup risk metrics help the user identify which lineups are best for GPP vs cash contests.
3. Player-level Boom% and Leverage scores guide individual player selection decisions.
