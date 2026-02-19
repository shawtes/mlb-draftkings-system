"""
COMPREHENSIVE CHECKBOX FIX FOR OPTIMIZER

This module provides improved checkbox handling for team stack selections.
The key improvements are:

1. Enhanced checkbox state collection with multiple fallback methods
2. Checkbox state change tracking 
3. Better error handling and debugging
4. Direct checkbox access methods
5. State persistence verification
"""

import logging
from PyQt5.QtWidgets import QCheckBox, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt

# Import Windows-safe logging
try:
    from safe_logging import safe_log_info, safe_log_debug, safe_log_warning, safe_log_error
    SAFE_LOGGING_AVAILABLE = True
except ImportError:
    SAFE_LOGGING_AVAILABLE = False
    # Fallback to regular logging
    def safe_log_info(msg): logging.info(msg)
    def safe_log_debug(msg): logging.debug(msg)
    def safe_log_warning(msg): logging.warning(msg)
    def safe_log_error(msg): logging.error(msg)

class CheckboxManager:
    """Manages checkbox states and provides robust access methods"""
    
    def __init__(self):
        self.checkbox_states = {}  # Manual state tracking
        self.checkbox_widgets = {}  # Direct widget references
    
    def create_checkbox_widget(self, row_id, initial_state=False):
        """Create a checkbox widget with proper state tracking"""
        checkbox = QCheckBox()
        checkbox.setChecked(initial_state)
        
        # Create the container widget
        checkbox_widget = QWidget()
        layout = QHBoxLayout(checkbox_widget)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Store references for direct access
        self.checkbox_widgets[row_id] = checkbox
        self.checkbox_states[row_id] = initial_state
        
        # Connect state change signal
        checkbox.stateChanged.connect(lambda state, rid=row_id: self._on_checkbox_changed(rid, state))
        
        return checkbox_widget
    
    def _on_checkbox_changed(self, row_id, state):
        """Handle checkbox state changes"""
        is_checked = state == Qt.Checked
        self.checkbox_states[row_id] = is_checked
        safe_log_info(f"Checkbox {row_id} state changed to: {is_checked}")
    
    def get_checkbox_state(self, row_id):
        """Get checkbox state with multiple fallback methods"""
        # Method 1: Direct widget reference (most reliable)
        if row_id in self.checkbox_widgets:
            try:
                return self.checkbox_widgets[row_id].isChecked()
            except Exception as e:
                safe_log_warning(f"Direct widget access failed for {row_id}: {e}")
        
        # Method 2: Manual state tracking (fallback)
        if row_id in self.checkbox_states:
            return self.checkbox_states[row_id]
        
        # Method 3: Default
        return False
    
    def set_checkbox_state(self, row_id, checked):
        """Set checkbox state with verification"""
        if row_id in self.checkbox_widgets:
            try:
                self.checkbox_widgets[row_id].setChecked(checked)
                self.checkbox_states[row_id] = checked
                safe_log_info(f"Set checkbox {row_id} to {checked}")
                return True
            except Exception as e:
                safe_log_error(f"Failed to set checkbox {row_id}: {e}")
        return False
    
    def get_all_checked_states(self):
        """Get all checkbox states"""
        states = {}
        for row_id in self.checkbox_widgets:
            states[row_id] = self.get_checkbox_state(row_id)
        return states
    
    def get_checked_items(self):
        """Get list of checked row IDs"""
        return [row_id for row_id, checked in self.get_all_checked_states().items() if checked]


def collect_team_selections_enhanced(team_stack_tables, checkbox_managers=None):
    """Enhanced team selection collection with multiple methods"""
    team_selections = {}
    
    if not team_stack_tables:
        logging.debug("No team stack tables available")
        return team_selections
    
    safe_log_info("ENHANCED: Starting team selection collection...")
    
    for stack_size, table in team_stack_tables.items():
        safe_log_info(f"ENHANCED: Checking table for stack size: {stack_size}")
        selected_teams = []
        
        # Method 1: Use checkbox manager if available
        if checkbox_managers and stack_size in checkbox_managers:
            manager = checkbox_managers[stack_size]
            checked_rows = manager.get_checked_items()
            
            for row_id in checked_rows:
                # Extract team name from row ID or table
                try:
                    row_num = int(row_id.split('_')[-1])  # Assuming row_id format like "team_0"
                    team_item = table.item(row_num, 1)
                    if team_item:
                        team_name = team_item.text()
                        selected_teams.append(team_name)
                        safe_log_info(f"ENHANCED: Found selected team: {team_name} (from manager)")
                except Exception as e:
                    safe_log_warning(f"Error extracting team from row_id {row_id}: {e}")
        
        # Always run Method 2 as well to double-check or as fallback
        safe_log_info("Running widget-based collection for verification...")
        fallback_teams = []
        
        for row in range(table.rowCount()):
            try:
                checkbox_widget = table.cellWidget(row, 0)
                team_item = table.item(row, 1)
                
                if not team_item:
                    continue
                
                team_name = team_item.text()
                is_checked = False
                
                if checkbox_widget:
                    # Try multiple ways to get the checkbox
                    checkbox = None
                    
                    # Way 1: Through layout
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        widget = layout.itemAt(0).widget()
                        if isinstance(widget, QCheckBox):
                            checkbox = widget
                    
                    # Way 2: Direct child search
                    if not checkbox:
                        checkbox = checkbox_widget.findChild(QCheckBox)
                    
                    # Way 3: Check if widget itself is checkbox
                    if not checkbox and isinstance(checkbox_widget, QCheckBox):
                        checkbox = checkbox_widget
                    
                    if checkbox:
                        is_checked = checkbox.isChecked()
                        safe_log_info(f"ENHANCED: Row {row} ({team_name}): {is_checked}")
                    else:
                        safe_log_warning(f"ENHANCED: No checkbox found for row {row}")
                
                if is_checked:
                    fallback_teams.append(team_name)
                    if team_name not in selected_teams:  # Add if not already found by manager
                        selected_teams.append(team_name)
                        safe_log_info(f"ENHANCED: Found selected team: {team_name} (from widget fallback)")
            
            except Exception as e:
                safe_log_error(f"Error processing row {row}: {e}")
        
        # Store results
        if selected_teams:
            if stack_size == "All Stacks":
                team_selections["all"] = selected_teams
            else:
                try:
                    stack_num = int(stack_size.split('-')[0])
                    team_selections[stack_num] = selected_teams
                except (ValueError, IndexError):
                    team_selections[stack_size] = selected_teams
            
            safe_log_info(f"ENHANCED: Stack {stack_size}: {len(selected_teams)} teams selected")
        else:
            safe_log_info(f"ENHANCED: Stack {stack_size}: No teams selected")
    
    safe_log_info(f"ENHANCED RESULT: {team_selections}")
    return team_selections


def create_enhanced_checkbox_widget(team_name, row_idx, manager=None):
    """Create checkbox widget with enhanced tracking"""
    if manager:
        row_id = f"team_{row_idx}"
        return manager.create_checkbox_widget(row_id, False)
    else:
        # Fallback to standard creation
        checkbox = QCheckBox()
        checkbox_widget = QWidget()
        layout = QHBoxLayout(checkbox_widget)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return checkbox_widget
