#!/usr/bin/env python3
"""
DraftKings file format handler to handle the complex CSV format
"""

import pandas as pd
import logging
import os

def analyze_dk_file(file_path):
    """Analyze a DraftKings entries file to understand its structure"""
    try:
        # Read the first 20 lines to analyze structure
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines()[:20]]
        
        # Identify different sections
        header_line_idx = None
        instruction_line_idx = None
        player_data_idx = None
        
        for i, line in enumerate(lines):
            if "Entry ID,Contest Name,Contest ID,Entry Fee,P,P,C,1B,2B,3B,SS,OF,OF,OF" in line:
                header_line_idx = i
                logging.info(f"Found header line at row {i+1}")
            
            if "Instructions" in line and ",1. Column A lists" in line:
                instruction_line_idx = i
                logging.info(f"Found instruction line at row {i+1}")
            
            if "Position,Name + ID,Name,ID,Roster Position" in line:
                player_data_idx = i
                logging.info(f"Found player data section at row {i+1}")
        
        return {
            "header_line": header_line_idx,
            "instruction_line": instruction_line_idx,
            "player_data_line": player_data_idx
        }
    
    except Exception as e:
        logging.error(f"Error analyzing file: {str(e)}")
        return None

def load_dk_entries(file_path):
    """Smart loader for DraftKings entries that handles the complex format"""
    analysis = analyze_dk_file(file_path)
    
    if not analysis or analysis["header_line"] is None:
        # Fall back to standard pandas read
        logging.warning("Could not analyze DK file structure, using standard read")
        return pd.read_csv(file_path, on_bad_lines='skip')
    
    try:
        # Read only the actual entries section using the detected format
        header_line = analysis["header_line"]
        
        if analysis["instruction_line"]:
            # Read only up to the instructions line
            nrows = analysis["instruction_line"] - header_line - 1
            df = pd.read_csv(file_path, skiprows=header_line, nrows=nrows)
        else:
            # Read all rows after header, handle special formats
            df = pd.read_csv(file_path, skiprows=header_line)
            
            # Try to detect the end of entry rows
            for i, row in df.iterrows():
                # Look for rows that don't have proper entry data
                first_col = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                if not first_col.isdigit() or "Instructions" in first_col or "Position" in first_col:
                    df = df.iloc[:i]
                    break
        
        # Clean up the DataFrame
        cols_to_drop = []
        for col in df.columns:
            if 'Unnamed' in str(col) or str(col).strip() == '':
                if df[col].isna().all() or (df[col] == '').all():
                    cols_to_drop.append(col)
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logging.info(f"Removed {len(cols_to_drop)} empty columns")
        
        logging.info(f"Successfully extracted {len(df)} entries from DraftKings file.")
        return df
    
    except Exception as e:
        logging.error(f"Error using smart reader: {str(e)}")
        # Fall back to standard read
        return pd.read_csv(file_path, on_bad_lines='skip')

def extract_player_data(file_path):
    """Extract player reference data from DraftKings entries file"""
    analysis = analyze_dk_file(file_path)
    
    if not analysis or analysis["player_data_line"] is None:
        logging.warning("Could not find player data section in DK file")
        return None
    
    try:
        # Read starting from the player data header line
        player_data_line = analysis["player_data_line"]
        player_df = pd.read_csv(file_path, skiprows=player_data_line)
        
        # Clean up the DataFrame
        cols_to_drop = []
        for col in player_df.columns:
            if 'Unnamed' in str(col) or str(col).strip() == '':
                if player_df[col].isna().all() or (player_df[col] == '').all():
                    cols_to_drop.append(col)
        
        if cols_to_drop:
            player_df = player_df.drop(columns=cols_to_drop)
        
        logging.info(f"Successfully extracted {len(player_df)} player records from DK file.")
        return player_df
    
    except Exception as e:
        logging.error(f"Error extracting player data: {str(e)}")
        return None

def extract_player_mapping(file_path):
    """Extract player name to ID mapping from DraftKings entries file"""
    player_df = extract_player_data(file_path)
    
    if player_df is None or player_df.empty:
        logging.warning("No player data available for mapping")
        return {}
    
    try:
        # Identify the name and ID columns
        name_col = None
        id_col = None
        
        for col in player_df.columns:
            if 'Name' in str(col) and '+' not in str(col):
                name_col = col
            elif 'ID' == str(col) or 'Id' == str(col):
                id_col = col
        
        if name_col and id_col:
            # Create mapping
            player_map = {}
            for _, row in player_df.iterrows():
                if pd.notna(row[name_col]) and pd.notna(row[id_col]):
                    name = str(row[name_col]).strip()
                    player_id = str(row[id_col]).strip()
                    if name and player_id and player_id.isdigit():
                        player_map[name] = player_id
            
            logging.info(f"Created player mapping with {len(player_map)} players")
            return player_map
        else:
            logging.warning("Could not identify name and ID columns in player data")
            return {}
    
    except Exception as e:
        logging.error(f"Error creating player mapping: {str(e)}")
        return {}
