#!/usr/bin/env python3
"""
SportsData.io NFL API - Player Game Stats by Week
==================================================

API Documentation: https://sportsdata.io/developers/api-documentation/nfl
Endpoint: Player Game Stats by Week

This script fetches NFL player game statistics for a specific week and season.

API Endpoint Format:
https://api.sportsdata.io/api/nfl/fantasy/json/PlayerGameStatsByWeek/{season}/{week}

Parameters:
- season: Year and season type (e.g., "2024REG", "2024PRE", "2024POST")
- week: Week number (Preseason: 0-4, Regular: 1-18, Postseason: 1-4)
- API Key: Required for authentication (passed in header)
"""

import requests
import json
import os
from typing import Optional, Dict, List
import pandas as pd


class SportsDataNFLAPI:
    """
    Client for SportsData.io NFL API
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the API client
        
        Args:
            api_key: Your SportsData.io API key
        """
        self.api_key = api_key
        self.base_url = "https://api.sportsdata.io/api/nfl/fantasy/json"
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
    
    def get_injuries_by_week(
        self,
        season: str,
        week: int,
        save_to_file: bool = False,
        filename: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get injury reports for a specific week
        
        Args:
            season: Season (e.g., "2025REG", "2025POST")
            week: Week number (1-18 for regular season)
            save_to_file: Whether to save the response to a file
            filename: Custom filename for saving (optional)
            
        Returns:
            List of injury records or None if request fails
        """
        endpoint = f"{self.base_url}/Injuries/{season}/{week}"
        
        print(f"📡 Fetching NFL Injuries...")
        print(f"   Season: {season}")
        print(f"   Week: {week}")
        print(f"   Endpoint: {endpoint}")
        
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Retrieved {len(data)} injury records")
                
                if save_to_file:
                    if filename is None:
                        filename = f"nfl_injuries_{season}_week{week}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"💾 Saved to: {filename}")
                
                return data
            
            elif response.status_code == 401:
                print("❌ Error: Invalid API Key (401 Unauthorized)")
                return None
            
            elif response.status_code == 404:
                print("❌ Error: Data not found (404)")
                print(f"   No injury data available for {season} Week {week}")
                return None
            
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        
        except requests.exceptions.Timeout:
            print("❌ Error: Request timed out")
            return None
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
    
    def get_dfs_slates_by_date(
        self,
        date: str,
        save_to_file: bool = False,
        filename: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Fetch DFS slate information including DraftKings salaries for a specific date
        
        Args:
            date: Date in format YYYY-MMM-DD or YYYY-MM-DD (e.g., "2025-OCT-18" or "2025-10-18")
            save_to_file: Whether to save results to JSON file
            filename: Custom filename for saved data
        
        Returns:
            List of DFS slate dictionaries with salary information, or None if error
        
        Example:
            >>> api = SportsDataNFLAPI("your_api_key")
            >>> slates = api.get_dfs_slates_by_date("2025-10-20")
        """
        # Build the API endpoint for DFS slates
        endpoint = f"{self.base_url}/DfsSlatesByDate/{date}"
        
        print(f"📡 Fetching DFS Slates (DraftKings Salaries)...")
        print(f"   Date: {date}")
        print(f"   Endpoint: {endpoint}")
        
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Retrieved {len(data)} DFS slates")
                
                # Show slate information
                for slate in data:
                    if 'DfsSlateGames' in slate:
                        num_games = len(slate.get('DfsSlateGames', []))
                        num_players = len(slate.get('DfsSlatePlayers', []))
                        operator = slate.get('Operator', 'Unknown')
                        slate_id = slate.get('SlateID', 'N/A')
                        print(f"   📋 Slate {slate_id} ({operator}): {num_games} games, {num_players} players")
                
                if save_to_file:
                    if filename is None:
                        filename = f"nfl_dfs_slates_{date}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"💾 Saved to: {filename}")
                
                return data
            
            elif response.status_code == 401:
                print("❌ Error: Invalid API Key (401 Unauthorized)")
                return None
            
            elif response.status_code == 404:
                print("❌ Error: Data not found (404)")
                print(f"   No DFS slates available for {date}")
                return None
            
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        
        except requests.exceptions.Timeout:
            print("❌ Error: Request timed out")
            return None
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
    
    def get_player_projections_by_week(
        self,
        season: str,
        week: int,
        save_to_file: bool = False,
        filename: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Fetch player PROJECTIONS for a specific week
        
        Args:
            season: Season year and type (e.g., "2025REG")
            week: Week number
            save_to_file: Whether to save results to JSON file
            filename: Custom filename for saved data
        
        Returns:
            List of player projection dictionaries, or None if error
        """
        # Build the API endpoint for projections
        endpoint = f"{self.base_url}/PlayerGameProjectionStatsByWeek/{season}/{week}"
        
        print(f"📡 Fetching NFL Player PROJECTIONS...")
        print(f"   Season: {season}")
        print(f"   Week: {week}")
        print(f"   Endpoint: {endpoint}")
        
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Retrieved {len(data)} player projections")
                
                if save_to_file:
                    if filename is None:
                        filename = f"nfl_projections_{season}_week{week}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"💾 Saved to: {filename}")
                
                return data
            
            elif response.status_code == 401:
                print("❌ Error: Invalid API Key (401 Unauthorized)")
                return None
            
            elif response.status_code == 404:
                print("❌ Error: Data not found (404)")
                print(f"   No projections available for {season} Week {week}")
                return None
            
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        
        except requests.exceptions.Timeout:
            print("❌ Error: Request timed out")
            return None
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
    
    def get_player_stats_by_week(
        self, 
        season: str, 
        week: int,
        save_to_file: bool = False,
        filename: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Fetch player game stats for a specific week
        
        Args:
            season: Season year and type (e.g., "2024REG", "2024PRE", "2024POST")
                   - REG = Regular Season
                   - PRE = Preseason  
                   - POST = Postseason/Playoffs
            week: Week number
                  - Preseason: 0-4
                  - Regular Season: 1-18
                  - Postseason: 1-4
            save_to_file: Whether to save results to JSON file
            filename: Custom filename for saved data
        
        Returns:
            List of player stat dictionaries, or None if error
        
        Example:
            >>> api = SportsDataNFLAPI("1dd5e646265649af87e0d9cdb80d1c8c")
            >>> stats = api.get_player_stats_by_week("2025REG", 6)
        """
        # Build the API endpoint
        endpoint = f"{self.base_url}/PlayerGameStatsByWeek/{season}/{week}"
        
        print(f"📡 Fetching NFL Player Stats...")
        print(f"   Season: {season}")
        print(f"   Week: {week}")
        print(f"   Endpoint: {endpoint}")
        
        try:
            # Make the API request
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            
            # Check for successful response
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Retrieved {len(data)} player records")
                
                # Save to file if requested
                if save_to_file:
                    if filename is None:
                        filename = f"nfl_stats_{season}_week{week}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"💾 Saved to: {filename}")
                
                return data
            
            elif response.status_code == 401:
                print("❌ Error: Invalid API Key (401 Unauthorized)")
                print("   Please check your API key is correct")
                return None
            
            elif response.status_code == 404:
                print("❌ Error: Data not found (404)")
                print(f"   No data available for {season} Week {week}")
                return None
            
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        
        except requests.exceptions.Timeout:
            print("❌ Error: Request timed out")
            return None
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
    
    def get_top_scorers(
        self, 
        season: str, 
        week: int, 
        position: Optional[str] = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top scoring players for a week
        
        Args:
            season: Season (e.g., "2024REG")
            week: Week number
            position: Filter by position (QB, RB, WR, TE, K, DEF, etc.)
            top_n: Number of top players to return
        
        Returns:
            DataFrame with top scorers
        """
        stats = self.get_player_stats_by_week(season, week)
        
        if not stats:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(stats)
        
        # Filter by position if specified
        if position:
            df = df[df['Position'] == position]
        
        # Sort by fantasy points and get top N
        if 'FantasyPoints' in df.columns:
            df = df.sort_values('FantasyPoints', ascending=False).head(top_n)
            
            # Select key columns
            columns = ['Name', 'Position', 'Team', 'FantasyPoints', 
                      'PassingYards', 'RushingYards', 'ReceivingYards',
                      'Touchdowns']
            
            # Only include columns that exist
            columns = [col for col in columns if col in df.columns]
            df = df[columns]
        
        return df
    
    def save_to_csv(
        self, 
        season: str, 
        week: int,
        filename: Optional[str] = None
    ) -> bool:
        """
        Fetch stats and save directly to CSV
        
        Args:
            season: Season (e.g., "2024REG")
            week: Week number
            filename: Custom CSV filename
        
        Returns:
            True if successful, False otherwise
        """
        stats = self.get_player_stats_by_week(season, week)
        
        if not stats:
            return False
        
        if filename is None:
            filename = f"nfl_stats_{season}_week{week}.csv"
        
        df = pd.DataFrame(stats)
        df.to_csv(filename, index=False)
        print(f"💾 Saved to CSV: {filename}")
        
        return True
    
    def calculate_mae(
        self,
        season: str,
        week: int,
        stat_columns: Optional[List[str]] = None,
        min_fantasy_points: float = 0.0
    ) -> pd.DataFrame:
        """
        Calculate Mean Absolute Error (MAE) between projections and actual stats
        
        Args:
            season: Season (e.g., "2025REG")
            week: Week number
            stat_columns: List of stat columns to compare
                         If None, uses common fantasy stats
            min_fantasy_points: Only include players who scored this many points or more
                              Default: 0.0 (include all players)
        
        Returns:
            DataFrame with MAE for each stat column
        """
        print("\n" + "="*60)
        print("CALCULATING MAE: PROJECTIONS VS ACTUAL")
        print("="*60)
        
        # Fetch both projections and actual stats
        print("\n1️⃣ Fetching PROJECTIONS...")
        projections = self.get_player_projections_by_week(season, week)
        
        print("\n2️⃣ Fetching ACTUAL stats...")
        actuals = self.get_player_stats_by_week(season, week)
        
        if not projections or not actuals:
            print("❌ Could not fetch data for MAE calculation")
            return pd.DataFrame()
        
        # Convert to DataFrames
        df_proj = pd.DataFrame(projections)
        df_actual = pd.DataFrame(actuals)
        
        print(f"\n✅ Projections: {len(df_proj)} players")
        print(f"✅ Actual stats: {len(df_actual)} players")
        
        # Default stat columns if not provided
        if stat_columns is None:
            stat_columns = [
                'FantasyPoints',
                'PassingYards',
                'PassingTouchdowns',
                'RushingYards',
                'RushingTouchdowns',
                'ReceivingYards',
                'ReceivingTouchdowns',
                'Receptions'
            ]
        
        # Merge on PlayerID or Name
        merge_key = 'PlayerID' if 'PlayerID' in df_proj.columns else 'Name'
        
        merged = pd.merge(
            df_proj,
            df_actual,
            on=merge_key,
            suffixes=('_proj', '_actual'),
            how='inner'
        )
        
        print(f"✅ Matched {len(merged)} players with both projections and actuals")
        
        # Filter by minimum fantasy points if specified
        if min_fantasy_points > 0:
            if 'FantasyPoints_actual' in merged.columns:
                before_filter = len(merged)
                merged = merged[merged['FantasyPoints_actual'] >= min_fantasy_points]
                print(f"🎯 Filtered to {len(merged)} players with {min_fantasy_points}+ fantasy points (removed {before_filter - len(merged)} low scorers)")
        
        # Calculate MAE for each stat
        mae_results = {}
        
        for stat in stat_columns:
            proj_col = f"{stat}_proj"
            actual_col = f"{stat}_actual"
            
            # Check if columns exist
            if proj_col in merged.columns and actual_col in merged.columns:
                # Remove NaN values
                valid_data = merged[[proj_col, actual_col]].dropna()
                
                if len(valid_data) > 0:
                    # Calculate MAE
                    mae = abs(valid_data[proj_col] - valid_data[actual_col]).mean()
                    
                    # Calculate other metrics
                    mse = ((valid_data[proj_col] - valid_data[actual_col]) ** 2).mean()
                    rmse = mse ** 0.5
                    
                    # Mean values for context
                    mean_projected = valid_data[proj_col].mean()
                    mean_actual = valid_data[actual_col].mean()
                    
                    mae_results[stat] = {
                        'MAE': mae,
                        'RMSE': rmse,
                        'Mean_Projected': mean_projected,
                        'Mean_Actual': mean_actual,
                        'Sample_Size': len(valid_data)
                    }
        
        # Create results DataFrame
        results_df = pd.DataFrame(mae_results).T
        
        print("\n" + "="*60)
        print("MAE RESULTS")
        print("="*60)
        print(results_df.to_string())
        
        # Save to CSV
        filter_suffix = f"_min{int(min_fantasy_points)}pts" if min_fantasy_points > 0 else ""
        filename = f"mae_analysis_{season}_week{week}{filter_suffix}.csv"
        results_df.to_csv(filename)
        print(f"\n💾 Saved MAE analysis to: {filename}")
        
        # Save detailed comparison
        comparison_cols = [merge_key, 'Name_proj', 'Position_proj']
        for stat in stat_columns:
            proj_col = f"{stat}_proj"
            actual_col = f"{stat}_actual"
            if proj_col in merged.columns and actual_col in merged.columns:
                comparison_cols.extend([proj_col, actual_col])
                # Add error column
                merged[f"{stat}_error"] = abs(merged[proj_col] - merged[actual_col])
                comparison_cols.append(f"{stat}_error")
        
        # Keep only columns that exist
        comparison_cols = [col for col in comparison_cols if col in merged.columns]
        comparison_df = merged[comparison_cols]
        
        comparison_filename = f"detailed_comparison_{season}_week{week}{filter_suffix}.csv"
        comparison_df.to_csv(comparison_filename, index=False)
        print(f"💾 Saved detailed comparison to: {comparison_filename}")
        
        return results_df


def example_usage():
    """
    Example usage of the API
    """
    # ⚠️ IMPORTANT: Replace with your actual API key
    # Get your free API key at: https://sportsdata.io/
    API_KEY = "YOUR_API_KEY_HERE"
    
    # You can also use environment variable
    # API_KEY = os.getenv('SPORTSDATA_API_KEY')
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  Please set your API key first!")
        print("   Get a free key at: https://sportsdata.io/")
        print("   Then replace 'YOUR_API_KEY_HERE' in this script")
        return
    
    # Initialize API client
    api = SportsDataNFLAPI(API_KEY)
    
    # Example 1: Get all player stats for Week 7 of 2024 Regular Season
    print("\n" + "="*60)
    print("EXAMPLE 1: Get Player Stats by Week")
    print("="*60)
    stats = api.get_player_stats_by_week("2024REG", 7, save_to_file=True)
    
    if stats:
        print(f"\nFirst player stats:")
        print(json.dumps(stats[0], indent=2))
    
    # Example 2: Get top 10 quarterbacks
    print("\n" + "="*60)
    print("EXAMPLE 2: Top 10 Quarterbacks")
    print("="*60)
    top_qbs = api.get_top_scorers("2024REG", 7, position="QB", top_n=10)
    print(top_qbs)
    
    # Example 3: Get top 10 running backs
    print("\n" + "="*60)
    print("EXAMPLE 3: Top 10 Running Backs")
    print("="*60)
    top_rbs = api.get_top_scorers("2024REG", 7, position="RB", top_n=10)
    print(top_rbs)
    
    # Example 4: Save all stats to CSV
    print("\n" + "="*60)
    print("EXAMPLE 4: Save to CSV")
    print("="*60)
    api.save_to_csv("2024REG", 7)
    
    # Example 5: Calculate MAE between projections and actuals
    print("\n" + "="*60)
    print("EXAMPLE 5: Calculate MAE (Projections vs Actuals)")
    print("="*60)
    mae_results = api.calculate_mae("2024REG", 7)


def calculate_mae_for_week6():
    """
    Standalone function to calculate MAE for Week 6
    Only includes players who scored 10+ fantasy points
    """
    API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
    
    api = SportsDataNFLAPI(API_KEY)
    
    # Calculate MAE for Week 6 of 2025 Regular Season
    # Only for players who scored 10+ fantasy points
    mae_results = api.calculate_mae("2025REG", 6, min_fantasy_points=10.0)
    
    return mae_results


def interactive_mode():
    """
    Interactive mode to fetch data with user input
    """
    print("\n" + "="*60)
    print("NFL PLAYER STATS FETCHER - Interactive Mode")
    print("="*60)
    
    api_key = input("\nEnter your SportsData.io API Key: ").strip()
    
    if not api_key:
        print("❌ API key required!")
        return
    
    api = SportsDataNFLAPI(api_key)
    
    # Get season
    print("\nSeason Types:")
    print("  REG  - Regular Season")
    print("  PRE  - Preseason")
    print("  POST - Postseason/Playoffs")
    
    year = input("\nEnter year (e.g., 2024): ").strip()
    season_type = input("Enter season type (REG/PRE/POST): ").strip().upper()
    season = f"{year}{season_type}"
    
    # Get week
    week = int(input("Enter week number: ").strip())
    
    # Fetch data
    stats = api.get_player_stats_by_week(season, week, save_to_file=True)
    
    if stats:
        print(f"\n✅ Successfully retrieved {len(stats)} player records!")
        
        # Ask if user wants to see top scorers
        show_top = input("\nShow top scorers? (y/n): ").strip().lower()
        if show_top == 'y':
            position = input("Filter by position (or press Enter for all): ").strip().upper()
            position = position if position else None
            
            top_players = api.get_top_scorers(season, week, position=position, top_n=20)
            print("\n" + "="*60)
            print("TOP SCORERS")
            print("="*60)
            print(top_players.to_string())


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     SportsData.io NFL API - Player Stats by Week        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("Choose mode:")
    print("1. Run examples (requires API key in code)")
    print("2. Interactive mode (enter API key when prompted)")
    print("3. Calculate MAE for Week 6 (2025REG)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        example_usage()
    elif choice == "2":
        interactive_mode()
    elif choice == "3":
        print("\n🎯 Calculating MAE for Week 6...")
        calculate_mae_for_week6()
    else:
        print("Goodbye!")

