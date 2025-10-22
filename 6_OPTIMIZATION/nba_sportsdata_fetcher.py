"""
NBA SportsData.io Data Fetcher
================================
Retrieves projections and historical data from SportsData.io API
Integrates with research-based optimizer for opponent modeling and optimization

API Documentation: NBA Fantasy & Odds endpoints
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import time

logging.basicConfig(level=logging.INFO)

class NBADataFetcher:
    """
    Fetches NBA data from SportsData.io API
    Provides projections, historical stats, and game information
    """
    
    def __init__(self, api_key: str):
        """
        Initialize NBA Data Fetcher
        
        Args:
            api_key: Your SportsData.io API key
        """
        self.api_key = api_key
        self.base_url = "https://api.sportsdata.io/api/nba"
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
        
        # Cache for API calls (avoid rate limits)
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def _make_request(self, endpoint: str, cache_key: Optional[str] = None) -> Dict:
        """
        Make API request with caching
        
        Args:
            endpoint: API endpoint (e.g., '/fantasy/json/Players')
            cache_key: Unique key for caching
            
        Returns:
            JSON response as dictionary
        """
        # Check cache
        if cache_key and cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                logging.debug(f"Using cached data for {cache_key}")
                return cached_data
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache result
            if cache_key:
                self.cache[cache_key] = (time.time(), data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return {}
    
    # ========================================================================
    # PROJECTIONS (for current slate optimization)
    # ========================================================================
    
    def get_daily_projections(self, date: str = None) -> pd.DataFrame:
        """
        Get projected player stats for a specific date
        
        Args:
            date: Date string in format 'YYYY-MMM-DD' (e.g., '2024-FEB-15')
                 If None, uses today's date
        
        Returns:
            DataFrame with player projections including:
            - Name, Position, Team, Opponent
            - Projected fantasy points
            - Projected stats (points, rebounds, assists, etc.)
        
        API Endpoint: /fantasy/json/PlayerGameProjectionStatsByDate/{date}
        """
        if date is None:
            date = datetime.now().strftime('%Y-%b-%d').upper()
        
        logging.info(f"📥 Fetching NBA projections for {date}...")
        
        endpoint = f"/fantasy/json/PlayerGameProjectionStatsByDate/{date}"
        data = self._make_request(endpoint, cache_key=f"proj_{date}")
        
        if not data:
            logging.warning(f"No projection data found for {date}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Calculate DraftKings points if not provided
        if 'FantasyPointsDraftKings' not in df.columns:
            df = self._calculate_dk_points(df)
        
        # Rename columns for consistency with optimizer
        df = df.rename(columns={
            'PlayerID': 'ID',
            'FantasyPointsDraftKings': 'Predicted_DK_Points',
            'Points': 'ProjectedPoints',
            'Rebounds': 'ProjectedRebounds',
            'Assists': 'ProjectedAssists',
            'Steals': 'ProjectedSteals',
            'BlockedShots': 'ProjectedBlocks',
            'Turnovers': 'ProjectedTurnovers'
        })
        
        # Add required columns for optimizer
        if 'Salary' not in df.columns:
            df['Salary'] = 5000  # Default salary if not provided
        
        # Add game information
        if 'HomeOrAway' in df.columns and 'Opponent' in df.columns:
            df['Game'] = df.apply(
                lambda x: f"{x['Team']}@{x['Opponent']}" if x['HomeOrAway'] == 'AWAY' 
                else f"{x['Opponent']}@{x['Team']}", axis=1
            )
        
        logging.info(f"✅ Retrieved {len(df)} player projections for {date}")
        
        return df
    
    def get_dfs_slate_info(self, date: str = None) -> Dict:
        """
        Get DFS slate information (game totals, Vegas lines)
        
        Args:
            date: Date string in format 'YYYY-MMM-DD'
        
        Returns:
            Dictionary with slate info and games
            
        API Endpoint: /fantasy/json/DfsSlatesByDate/{date}
        """
        if date is None:
            date = datetime.now().strftime('%Y-%b-%d').upper()
        
        endpoint = f"/fantasy/json/DfsSlatesByDate/{date}"
        data = self._make_request(endpoint, cache_key=f"slate_{date}")
        
        return data
    
    def get_games_with_odds(self, date: str = None) -> pd.DataFrame:
        """
        Get games with Vegas odds (totals, spreads)
        Critical for game stacking and high-scoring game identification
        
        Args:
            date: Date string in format 'YYYY-MMM-DD'
            
        Returns:
            DataFrame with games and Vegas totals
            
        API Endpoint: /odds/json/GameOddsByDate/{date}
        """
        if date is None:
            date = datetime.now().strftime('%Y-%b-%d').upper()
        
        logging.info(f"📥 Fetching game odds for {date}...")
        
        endpoint = f"/odds/json/GameOddsByDate/{date}"
        data = self._make_request(endpoint, cache_key=f"odds_{date}")
        
        if not data:
            return pd.DataFrame()
        
        games = []
        for game in data:
            if 'PregameOdds' in game and game['PregameOdds']:
                # Get consensus total
                totals = [odd.get('OverUnder', 0) for odd in game['PregameOdds'] 
                         if odd.get('OverUnder')]
                avg_total = np.mean(totals) if totals else 0
                
                games.append({
                    'GameID': game.get('GameID'),
                    'HomeTeam': game.get('HomeTeam'),
                    'AwayTeam': game.get('AwayTeam'),
                    'GameTotal': avg_total,
                    'DateTime': game.get('DateTime'),
                    'Status': game.get('Status')
                })
        
        df = pd.DataFrame(games)
        logging.info(f"✅ Retrieved odds for {len(df)} games")
        
        return df
    
    # ========================================================================
    # HISTORICAL DATA (for opponent modeling)
    # ========================================================================
    
    def get_historical_stats(self, start_date: str, end_date: str = None, 
                            num_days: int = 30) -> pd.DataFrame:
        """
        Get historical player game stats for Dirichlet regression
        Used for opponent portfolio modeling (MIT paper Section 3)
        
        Args:
            start_date: Start date 'YYYY-MMM-DD'
            end_date: End date (optional)
            num_days: Number of days to fetch if end_date not specified
            
        Returns:
            DataFrame with historical player performances
            
        API Endpoint: /fantasy/json/PlayerGameStatsByDate/{date}
        """
        if end_date is None:
            start = datetime.strptime(start_date.replace('-', ' '), '%Y %b %d')
            dates = [(start - timedelta(days=i)).strftime('%Y-%b-%d').upper() 
                    for i in range(num_days)]
        else:
            start = datetime.strptime(start_date.replace('-', ' '), '%Y %b %d')
            end = datetime.strptime(end_date.replace('-', ' '), '%Y %b %d')
            dates = []
            current = start
            while current <= end:
                dates.append(current.strftime('%Y-%b-%d').upper())
                current += timedelta(days=1)
        
        logging.info(f"📥 Fetching historical data for {len(dates)} days...")
        
        all_stats = []
        for date in dates:
            endpoint = f"/fantasy/json/PlayerGameStatsByDate/{date}"
            data = self._make_request(endpoint, cache_key=f"hist_{date}")
            
            if data:
                df_day = pd.DataFrame(data)
                df_day['Date'] = date
                all_stats.append(df_day)
            
            time.sleep(0.1)  # Rate limiting
        
        if not all_stats:
            logging.warning("No historical data retrieved")
            return pd.DataFrame()
        
        df = pd.concat(all_stats, ignore_index=True)
        
        # Calculate DK points
        if 'FantasyPointsDraftKings' not in df.columns:
            df = self._calculate_dk_points(df)
        
        logging.info(f"✅ Retrieved {len(df)} historical player game stats")
        
        return df
    
    def get_season_stats(self, season: str = '2025') -> pd.DataFrame:
        """
        Get full season stats for all players
        Used for calculating player consistency, variance, etc.
        
        Args:
            season: Season year (e.g., '2025')
            
        Returns:
            DataFrame with season-long statistics
            
        API Endpoint: /fantasy/json/PlayerSeasonStats/{season}
        """
        logging.info(f"📥 Fetching season stats for {season}...")
        
        endpoint = f"/fantasy/json/PlayerSeasonStats/{season}"
        data = self._make_request(endpoint, cache_key=f"season_{season}")
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Add consistency metrics
        if 'Games' in df.columns and 'FantasyPointsDraftKings' in df.columns:
            df['AvgDKPoints'] = df['FantasyPointsDraftKings'] / df['Games']
            # Estimate variance (30% coefficient of variation)
            df['DKPoints_StdDev'] = df['AvgDKPoints'] * 0.3
        
        logging.info(f"✅ Retrieved season stats for {len(df)} players")
        
        return df
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _calculate_dk_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DraftKings fantasy points
        
        DK Scoring:
        - Points: 1.0
        - 3PT Made: 0.5
        - Rebounds: 1.25
        - Assists: 1.5
        - Steals: 2.0
        - Blocks: 2.0
        - Turnovers: -0.5
        - Double-Double: 1.5 bonus
        - Triple-Double: 3.0 bonus
        """
        df['FantasyPointsDraftKings'] = (
            df.get('Points', 0) * 1.0 +
            df.get('ThreePointersHitMade', 0) * 0.5 +
            df.get('Rebounds', 0) * 1.25 +
            df.get('Assists', 0) * 1.5 +
            df.get('Steals', 0) * 2.0 +
            df.get('BlockedShots', 0) * 2.0 +
            df.get('Turnovers', 0) * -0.5
        )
        
        # Add bonuses for double-doubles and triple-doubles
        if all(col in df.columns for col in ['Points', 'Rebounds', 'Assists', 'Steals', 'BlockedShots']):
            # Count categories >= 10
            stat_cols = ['Points', 'Rebounds', 'Assists', 'Steals', 'BlockedShots']
            double_digits = df[stat_cols].apply(lambda x: (x >= 10).sum(), axis=1)
            
            # Double-double bonus (2 categories >= 10)
            df.loc[double_digits >= 2, 'FantasyPointsDraftKings'] += 1.5
            
            # Triple-double bonus (3 categories >= 10)
            df.loc[double_digits >= 3, 'FantasyPointsDraftKings'] += 3.0
        
        return df
    
    def get_active_players(self) -> pd.DataFrame:
        """
        Get all active NBA players
        
        Returns:
            DataFrame with active player information
            
        API Endpoint: /fantasy/json/Players
        """
        logging.info("📥 Fetching active NBA players...")
        
        endpoint = "/fantasy/json/Players"
        data = self._make_request(endpoint, cache_key="active_players")
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        logging.info(f"✅ Retrieved {len(df)} active players")
        
        return df
    
    def prepare_optimizer_input(self, date: str = None, 
                               include_odds: bool = True,
                               min_minutes: float = 15.0) -> pd.DataFrame:
        """
        Prepare complete dataset for research-based optimizer
        
        Combines:
        - Daily projections
        - Vegas odds (for game stacking)
        - Filters low-minute players
        
        Args:
            date: Date for slate
            include_odds: Include Vegas totals
            min_minutes: Minimum projected minutes
            
        Returns:
            DataFrame ready for ResearchBasedNBAOptimizer
        """
        # Get projections
        df = self.get_daily_projections(date)
        
        if df.empty:
            return df
        
        # Filter by minutes
        if 'Minutes' in df.columns:
            df = df[df['Minutes'] >= min_minutes]
        
        # Add Vegas odds
        if include_odds:
            odds = self.get_games_with_odds(date)
            if not odds.empty:
                # Map game totals to players
                df = self._add_game_totals(df, odds)
        
        # Ensure all required columns exist
        required_cols = ['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            logging.warning(f"Missing columns: {missing}")
        
        logging.info(f"✅ Prepared {len(df)} players for optimization")
        
        return df
    
    def _add_game_totals(self, players_df: pd.DataFrame, 
                        odds_df: pd.DataFrame) -> pd.DataFrame:
        """Add game total (O/U) to each player"""
        if 'Team' not in players_df.columns:
            return players_df
        
        # Create team to total mapping
        team_totals = {}
        for _, game in odds_df.iterrows():
            team_totals[game['HomeTeam']] = game['GameTotal']
            team_totals[game['AwayTeam']] = game['GameTotal']
        
        # Add to players
        players_df['GameTotal'] = players_df['Team'].map(team_totals)
        
        return players_df


# ============================================================================
# INTEGRATION WITH RESEARCH OPTIMIZER
# ============================================================================

class NBAResearchPipeline:
    """
    Complete pipeline: Data Fetching → Opponent Modeling → Optimization
    Integrates SportsData.io API with MIT research-based optimizer
    """
    
    def __init__(self, api_key: str):
        """
        Initialize complete NBA DFS research pipeline
        
        Args:
            api_key: SportsData.io API key
        """
        self.fetcher = NBADataFetcher(api_key)
        
        # Import research optimizer
        try:
            from nba_research_optimizer_core import ResearchBasedNBAOptimizer
            self.optimizer = ResearchBasedNBAOptimizer()
            self.optimizer_available = True
        except ImportError:
            logging.warning("Research optimizer not available")
            self.optimizer_available = False
    
    def run_cash_optimization(self, date: str = None, 
                             num_lineups: int = 1) -> Tuple[List, pd.DataFrame]:
        """
        Complete cash game optimization pipeline
        
        Args:
            date: Slate date
            num_lineups: Number of lineups to generate
            
        Returns:
            (lineups, player_pool_df)
        """
        logging.info("🏀 Running CASH GAME optimization pipeline...")
        
        # 1. Fetch projections
        player_pool = self.fetcher.prepare_optimizer_input(date)
        
        if player_pool.empty:
            logging.error("No player data available")
            return [], player_pool
        
        # 2. Fetch historical data for opponent modeling
        end_date = date or datetime.now().strftime('%Y-%b-%d').upper()
        historical = self.fetcher.get_historical_stats(end_date, num_days=30)
        
        # 3. Optimize
        if self.optimizer_available:
            lineups = self.optimizer.optimize(
                player_pool=player_pool,
                contest_type='cash',
                position_limits={
                    'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
                    'G': 1, 'F': 1, 'UTIL': 1
                },
                salary_cap=50000,
                num_lineups=num_lineups,
                historical_data=historical
            )
        else:
            logging.warning("Using fallback optimization")
            lineups = self._fallback_optimize(player_pool, num_lineups)
        
        logging.info(f"✅ Generated {len(lineups)} cash game lineup(s)")
        
        return lineups, player_pool
    
    def run_gpp_optimization(self, date: str = None,
                            num_lineups: int = 20,
                            stack_type: str = 'game_stack') -> Tuple[List, pd.DataFrame]:
        """
        Complete GPP tournament optimization pipeline
        
        Args:
            date: Slate date
            num_lineups: Number of diverse lineups
            stack_type: 'pg_c_stack' or 'game_stack'
            
        Returns:
            (lineups, player_pool_df)
        """
        logging.info("🏆 Running GPP TOURNAMENT optimization pipeline...")
        
        # 1. Fetch projections with odds
        player_pool = self.fetcher.prepare_optimizer_input(date, include_odds=True)
        
        if player_pool.empty:
            logging.error("No player data available")
            return [], player_pool
        
        # 2. Add ownership projections (if available)
        # TODO: Integrate with ownership projection service
        player_pool['Ownership%'] = 10.0  # Placeholder
        
        # 3. Optimize for GPP
        if self.optimizer_available:
            lineups = self.optimizer.optimize(
                player_pool=player_pool,
                contest_type='gpp',
                position_limits={
                    'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
                    'G': 1, 'F': 1, 'UTIL': 1
                },
                salary_cap=50000,
                num_lineups=num_lineups,
                ownership_data=player_pool['Ownership%'],
                stack_config={'type': stack_type}
            )
        else:
            logging.warning("Using fallback optimization")
            lineups = self._fallback_optimize(player_pool, num_lineups)
        
        logging.info(f"✅ Generated {len(lineups)} GPP lineup(s)")
        
        return lineups, player_pool
    
    def _fallback_optimize(self, player_pool: pd.DataFrame, 
                          num_lineups: int) -> List:
        """Simple greedy optimization fallback"""
        player_pool = player_pool.copy()
        player_pool['value'] = player_pool['Predicted_DK_Points'] / player_pool['Salary']
        
        lineups = []
        for _ in range(num_lineups):
            lineup = []
            used_players = set()
            total_salary = 0
            
            for pos in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']:
                available = player_pool[
                    (~player_pool.index.isin(used_players)) &
                    (player_pool['Position'].isin([pos.replace('UTIL', '')] if pos != 'UTIL' else ['PG', 'SG', 'SF', 'PF', 'C']))
                ].sort_values('value', ascending=False)
                
                for idx, player in available.iterrows():
                    if total_salary + player['Salary'] <= 50000:
                        lineup.append(idx)
                        used_players.add(idx)
                        total_salary += player['Salary']
                        break
            
            if len(lineup) == 8:
                lineups.append(lineup)
        
        return lineups


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    API_KEY = "your_sportsdata_io_api_key_here"
    
    print("🏀 NBA SportsData.io Data Fetcher")
    print("=" * 60)
    
    # Initialize
    fetcher = NBADataFetcher(API_KEY)
    
    # Get today's projections
    print("\n1. Fetching Daily Projections...")
    projections = fetcher.get_daily_projections()
    print(f"   Retrieved {len(projections)} player projections")
    
    # Get game odds
    print("\n2. Fetching Game Odds...")
    odds = fetcher.get_games_with_odds()
    print(f"   Retrieved odds for {len(odds)} games")
    
    # Get historical data
    print("\n3. Fetching Historical Data (last 30 days)...")
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%b-%d').upper()
    historical = fetcher.get_historical_stats(start_date, num_days=7)  # Last 7 days
    print(f"   Retrieved {len(historical)} historical game stats")
    
    # Prepare for optimizer
    print("\n4. Preparing Optimizer Input...")
    optimizer_ready = fetcher.prepare_optimizer_input()
    print(f"   {len(optimizer_ready)} players ready for optimization")
    
    print("\n" + "=" * 60)
    print("✅ Data fetching complete!")
    print("\nNext: Use NBAResearchPipeline for complete optimization:")
    print("  pipeline = NBAResearchPipeline(API_KEY)")
    print("  cash_lineups, players = pipeline.run_cash_optimization()")
    print("  gpp_lineups, players = pipeline.run_gpp_optimization(num_lineups=20)")

