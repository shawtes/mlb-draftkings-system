import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import concurrent.futures
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor, RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
import warnings
import multiprocessing
import os

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Define paths for saving the label encoders and scaler
name_encoder_path = '/Users/sineshawmesfintesfaye/newenv/name_encoder_nfl3.pkl'
team_encoder_path = '/Users/sineshawmesfintesfaye/newenv/team_encoder_nfl3.pkl'
scaler_path = '/Users/sineshawmesfintesfaye/newenv/scaler_nfl3.pkl'

# NFL features for predictions
selected_features = ["FG%", "Name_encoded", "Team_encoded", "XP%",
     "FGA", "FGM", "Pnt", "Pnt.1", "XPM", "XPA", 
     "Pts", "Y/P", "Yds", "Cmp.1", "Cmp", "Inc", 
     "Y/C", "Att", "Rate", "Sk%", "TD", "AY/A", 
     "Yds.1", "ANY/A", "Sk", "Blck", "TD%", "Int",
     "Rec", "Int%", "Y/R", "Y/Tgt", "1D", "Y/A", 
     "Ctch%", "Tgt", "Tgt.1", 'Completion%',
     'YardsPerAttempt', 'TDPerAttempt', 'InterceptionsPerAttempt',
     'RushYardsPerAttempt', 'TotalOffYards',
     'rolling_avg_fpts_3', 'rolling_avg_fpts_4', 'calculated_dk_fpts',
     'fpts_std', 'fpts_volatility', '5_game_avg', 'rolling_min_fpts_3', 'rolling_max_fpts_3',
     'rolling_min_fpts_4', 'rolling_max_fpts_4', 'lag_mean_fpts_2', 'lag_max_fpts_2',
     'lag_min_fpts_2', 'lag_mean_fpts_4', 'lag_max_fpts_4', 'lag_min_fpts_4',
     'lag_mean_fpts_5', 'lag_max_fpts_5', 'lag_min_fpts_5', 'lag_mean_fpts_10',
     'lag_max_fpts_10', 'lag_min_fpts_10',
     'Player_Team_Interaction', 'Player_Game_Interaction',
     'day_of_season', 'week_of_season', 'day_of_year'
]

def load_or_create_label_encoders(df):
    le_name = LabelEncoder()
    le_team = LabelEncoder()
    
    # Fill NaN values for Name and Team
    df['Player'] = df['Player'].fillna('Unknown')
    df['Team'] = df['Team'].fillna('Unknown')

    le_name.fit(df['Player'].unique())
    le_team.fit(df['Team'].unique())
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    
    return le_name, le_team

def load_or_create_scaler(df, numeric_features):
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])
        joblib.dump(scaler, scaler_path)
    return scaler

def engineer_features(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isnull().any():
        print("Warning: Some date values could not be parsed. Please check the 'date' column in your CSV file.")

    # Date-related features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_season'] = (df['date'] - df['date'].min()).dt.days
    df['week_of_season'] = (df['date'] - df['date'].min()).dt.days // 7
    df['day_of_year'] = df['date'].dt.dayofyear

    # Efficiency metrics
    if 'Cmp' in df.columns and 'Att' in df.columns:
        df['Completion%'] = df['Cmp'] / df['Att']
    if 'Yds' in df.columns and 'Att' in df.columns:
        df['YardsPerAttempt'] = df['Yds'] / df['Att']
    if 'TD' in df.columns and 'Att' in df.columns:
        df['TDPerAttempt'] = df['TD'] / df['Att']
    if 'Int' in df.columns and 'Att' in df.columns:
        df['InterceptionsPerAttempt'] = df['Int'] / df['Att']
    
    # Rushing stats
    if 'RushYds' in df.columns and 'RushAtt' in df.columns:
        df['RushYardsPerAttempt'] = df['RushYds'] / df['RushAtt']
    
    # Total offense
    if 'Yds' in df.columns and 'RushYds' in df.columns:
        df['TotalOffYards'] = df['Yds'] + df['RushYds']

    # Interaction terms
    df['Player_Team_Interaction'] = df['Player'] + '_' + df['Team']
    df['Player_Game_Interaction'] = df['Player'] + '_' + df['date'].astype(str)

    # Rolling averages and other statistical features
    df = df.sort_values(['Player', 'date'])
    df['rolling_avg_fpts_3'] = df.groupby('Player')['calculated_dk_fpts'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_avg_fpts_4'] = df.groupby('Player')['calculated_dk_fpts'].rolling(window=4, min_periods=1).mean().reset_index(0, drop=True)
    df['fpts_std'] = df.groupby('Player')['calculated_dk_fpts'].rolling(window=5, min_periods=1).std().reset_index(0, drop=True)
    df['fpts_volatility'] = df['fpts_std'] / df['rolling_avg_fpts_4']
    df['5_game_avg'] = df.groupby('Player')['calculated_dk_fpts'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)

    # Add more rolling statistics
    for window in [3, 4]:
        df[f'rolling_min_fpts_{window}'] = df.groupby('Player')['calculated_dk_fpts'].rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
        df[f'rolling_max_fpts_{window}'] = df.groupby('Player')['calculated_dk_fpts'].rolling(window=window, min_periods=1).max().reset_index(0, drop=True)

    # Add lag features
    for lag in [2, 4, 5, 10]:
        df[f'lag_mean_fpts_{lag}'] = df.groupby('Player')['calculated_dk_fpts'].shift(lag)
        df[f'lag_max_fpts_{lag}'] = df.groupby('Player')['calculated_dk_fpts'].shift(lag).rolling(window=lag, min_periods=1).max().reset_index(0, drop=True)
        df[f'lag_min_fpts_{lag}'] = df.groupby('Player')['calculated_dk_fpts'].shift(lag).rolling(window=lag, min_periods=1).min().reset_index(0, drop=True)

    return df

def engineer_chunk(chunk):
    return engineer_features(chunk)

def concurrent_feature_engineering(df, chunksize):
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        processed_chunks = list(executor.map(engineer_chunk, chunks))
    return pd.concat(processed_chunks)

def create_synthetic_rows_for_all_players(df, all_players, prediction_date):
    print(f"Creating synthetic rows for all players for date: {prediction_date}...")
    synthetic_rows = []
    for player in all_players:
        player_df = df[df['Player'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            print(f"No historical data found for player {player}. Using randomized default values.")
            default_row = pd.DataFrame([{col: np.random.uniform(0, 1) for col in df.columns}])
            default_row['date'] = prediction_date
            default_row['Player'] = player
            default_row['calculated_dk_fpts'] = np.random.uniform(0, 5)
            default_row['has_historical_data'] = False
            synthetic_rows.append(default_row)
        else:
            print(f"Using {len(player_df)} rows of data for {player}. Date range: {player_df['date'].min()} to {player_df['date'].max()}")
            
            player_df = player_df.head(15)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Player'] = player
            synthetic_row['has_historical_data'] = True
            
            if 'calculated_dk_fpts' in player_df.columns:
                synthetic_row['calculated_dk_fpts'] = player_df['calculated_dk_fpts'].mean()
            else:
                synthetic_row['calculated_dk_fpts'] = np.nan
            
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Player']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    if synthetic_rows:
        synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
        print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
        print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
        return synthetic_df
    else:
        print("Warning: No synthetic rows created.")
        return pd.DataFrame()

def analyze_prediction_differences(df):
    df = df.sort_values(by=['Player', 'date'])
    df['last_5_games'] = df.groupby('Player')['date'].transform(lambda x: x >= (x.max() - pd.Timedelta(days=5)))
    last_5_games_df = df[df['last_5_games']]
    last_5_games_df['difference'] = last_5_games_df['predicted_dk_fpts'] - last_5_games_df['calculated_dk_fpts']
    
    player_adjustments = last_5_games_df.groupby('Player').agg({
        'difference': [
            ('avg_positive_diff', lambda x: x[x > 0].mean()),
            ('avg_negative_diff', lambda x: x[x < 0].mean())
        ]
    })
    
    player_adjustments.columns = player_adjustments.columns.droplevel()
    player_adjustments['avg_positive_diff'] = player_adjustments['avg_positive_diff'].fillna(0)
    player_adjustments['avg_negative_diff'] = player_adjustments['avg_negative_diff'].fillna(0)
    
    print("Player-specific adjustments calculated for the last 5 games.")
    return player_adjustments

def adjust_predictions(row, player_adjustments):
    prediction = row['predicted_dk_fpts']
    player = row['Player']
    
    if player in player_adjustments.index:
        if prediction > row['calculated_dk_fpts']:
            adjusted_prediction = prediction - (player_adjustments.loc[player, 'avg_positive_diff'])
        else:
            adjusted_prediction = prediction - (player_adjustments.loc[player, 'avg_negative_diff'])
    else:
        if prediction > row['calculated_dk_fpts']:
            adjusted_prediction = prediction - (player_adjustments['avg_positive_diff'].mean())
        else:
            adjusted_prediction = prediction - (player_adjustments['avg_negative_diff'].mean())
    
    return max(0, adjusted_prediction)

def process_predictions(chunk, pipeline, player_adjustments):
    features = chunk.drop(columns=['calculated_dk_fpts', 'Ind. Games Link'])
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    features_selected = pipeline.named_steps['selector'].transform(features_preprocessed)
    chunk['predicted_dk_fpts'] = pipeline.named_steps['model'].predict(features_selected)
    
    chunk['predicted_dk_fpts'] = chunk.apply(lambda row: adjust_predictions(row, player_adjustments), axis=1)
    
    return chunk

def rolling_predictions(train_data, model_pipeline, test_dates, chunksize, player_adjustments):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        train_data_filtered = train_data[train_data['date'] < current_date]
        synthetic_rows = create_synthetic_rows_for_all_players(train_data_filtered, train_data_filtered['Player'].unique(), current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, 
                                                 [model_pipeline]*len(chunks), 
                                                 [player_adjustments]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def evaluate_model(y_true, y_pred):
    print("Evaluating model...")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    epsilon = 1e-10
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted DK Points")
    plt.savefig('/Users/sineshawmesfintesfaye/newenv/actual_vs_predicted.png')
    plt.close()
    
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig('/Users/sineshawmesfintesfaye/newenv/residual_plot.png')
    plt.close()
    
    print("Model evaluation completed.")
    return mae, mse, r2, mape

base_models = [
    ('lr', LinearRegression()),
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('dt', DecisionTreeRegressor()),
    ('svr', SVR()),
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', XGBRegressor(objective='reg:squarederror', n_jobs=-1)),
    ('bagging', BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10)),
    ('poly', make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
]

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.7
    )
)

final_model = StackingRegressor(
    estimators=[('stacking', stacking_model)],
    final_estimator=XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.7
    )
)

def clean_infinite_values(df):
    print("Starting to clean infinite values...")
    df = df.replace([np.inf, -np.inf], np.nan)
    print("Replaced inf and -inf with NaN")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    total_columns = len(numeric_columns)
    for i, col in enumerate(numeric_columns, 1):
        if i % 10 == 0:
            print(f"Processing numeric column {i}/{total_columns}: {col}")
        if col != 'calculated_dk_fpts':
            df[col] = df[col].fillna(df[col].mean())
    
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        df[col] = df[col].fillna('Unknown')
    
    return df

def rolling_window_prediction(df, pipeline, window_size=30, step_size=1):
    results = []
    dates = df['date'].sort_values().unique()
    
    print(f"Total unique dates: {len(dates)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    if len(dates) <= window_size:
        print(f"Warning: Not enough dates for specified window size. Adjusting window size to {len(dates) - 1}")
        window_size = len(dates) - 1
    
    for i in range(window_size, len(dates)):
        train_dates = dates[max(0, i-window_size):i]
        test_date = dates[i]
        
        print(f"Processing window: Train dates from {train_dates[0]} to {train_dates[-1]}, Test date: {test_date}")
        
        train_data = df[df['date'].isin(train_dates)]
        test_data = df[df['date'] == test_date]
        
        if train_data.empty or test_data.empty:
            print(f"Warning: Empty train or test data for window ending on {test_date}")
            continue
        
        train_features = train_data[features_to_use]
        test_features = test_data[features_to_use]
        
        print(f"Train data shape: {train_features.shape}, Test data shape: {test_features.shape}")
        
        try:
            pipeline.fit(train_features, train_data['calculated_dk_fpts'])
            predictions = pipeline.predict(test_features)
            test_data['predicted_dk_fpts'] = predictions
            results.append(test_data)
            print(f"Processed window ending on {test_date}, predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"Error during fitting/prediction: {str(e)}")
            continue
    
    if not results:
        print("Warning: No predictions were generated.")
        return df
    
    return pd.concat(results)

def iterative_training_and_prediction(df, n_iterations=2):
    print(f"Starting iterative training and prediction with {n_iterations} iterations...")
    
    # Initial split of data
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        
        # Train the model
        pipeline = train_model(train_data)
        
        # Make predictions on test data
        test_predictions = rolling_window_prediction(test_data, pipeline)
        
        # Evaluate the model
        mae, mse, r2, mape = evaluate_model(test_predictions['calculated_dk_fpts'], test_predictions['predicted_dk_fpts'])
        print(f"Iteration {iteration + 1} Results:")
        print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}%")
        
        # Analyze prediction differences
        player_adjustments = analyze_prediction_differences(test_predictions)
        
        # Apply feedback: adjust the training data based on prediction errors
        train_data = apply_feedback(train_data, test_predictions, player_adjustments)
        
        # Update test data for next iteration (optional, depending on your strategy)
        test_data = update_test_data(test_data, test_predictions)
    
    return pipeline, player_adjustments

def train_model(train_data):
    # ... (keep existing model training code)
    return pipeline

def apply_feedback(train_data, test_predictions, player_adjustments):
    print("Applying feedback to training data...")
    
    # Calculate overall average adjustment
    avg_adjustment = player_adjustments['difference'].mean()
    
    # Apply player-specific or average adjustment to training data
    for player in train_data['Player'].unique():
        if player in player_adjustments.index:
            adjustment = player_adjustments.loc[player, 'difference']
        else:
            adjustment = avg_adjustment
        
        # Adjust the target variable (calculated_dk_fpts) in the training data
        mask = (train_data['Player'] == player)
        train_data.loc[mask, 'calculated_dk_fpts'] += adjustment
    
    # Ensure no negative values after adjustment
    train_data['calculated_dk_fpts'] = train_data['calculated_dk_fpts'].clip(lower=0)
    
    return train_data

def update_test_data(test_data, test_predictions):
    print("Updating test data for next iteration...")
    
    # Merge the original test data with the new predictions
    updated_test_data = test_data.merge(test_predictions[['Player', 'date', 'predicted_dk_fpts']], 
                                        on=['Player', 'date'], 
                                        how='left')
    
    # Use the predicted values as the new 'calculated_dk_fpts' for the next iteration
    updated_test_data['calculated_dk_fpts'] = updated_test_data['predicted_dk_fpts']
    
    # Drop the 'predicted_dk_fpts' column
    updated_test_data = updated_test_data.drop(columns=['predicted_dk_fpts'])
    
    return updated_test_data

def calculate_player_adjustments(df):
    df['prediction_error'] = df['predicted_dk_fpts'] - df['calculated_dk_fpts']
    player_adjustments = df.groupby('Player')['prediction_error'].mean().to_dict()
    return player_adjustments

if __name__ == "__main__":
    try:
        start_time = time.time()
        
        print("Loading dataset...")
        df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/nfl_fantasy_points.csv',
                         dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                         low_memory=False)
        
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by=['Player', 'date'], inplace=True)

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

        df.fillna(0, inplace=True)
        print("Dataset loaded and preprocessed.")

        # Load or create LabelEncoders
        le_name, le_team = load_or_create_label_encoders(df)

        # Ensure 'Name_encoded' and 'Team_encoded' columns are created
        df['Name_encoded'] = le_name.transform(df['Player'])
        df['Team_encoded'] = le_team.transform(df['Team'])
      
        chunksize = 20000
        df = concurrent_feature_engineering(df, chunksize)

        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str)

        # Define the list of all selected and engineered features
        features = selected_features + ['date']

        # Define numeric and categorical features
        numeric_features = ["FG%",'calculated_dk_fpts',"Name_encoded","Team_encoded","XP%",
         "FGA", "FGM", "Pnt", "Pnt.1", "XPM", "XPA", 
         "Pts", "Y/P", "Yds", "Cmp.1", "Cmp", "Inc", 
         "Y/C", "Att", "Rate", "Sk%", "TD", "AY/A", 
         "Yds.1", "ANY/A", "Sk", "Blck", "TD%", "Int",
         "Rec", "Int%", "Y/R", "Y/Tgt", "1D", "Y/A", 
         "Ctch%", "Tgt", "Tgt.1",'Completion%',
         'YardsPerAttempt','TDPerAttempt','InterceptionsPerAttempt',
         'RushYardsPerAttempt','TotalOffYards',
         'rolling_avg_fpts_3', 'rolling_avg_fpts_4',    
            'fpts_std', 'fpts_volatility', '5_game_avg','rolling_min_fpts_3', 'rolling_max_fpts_3',
            'rolling_min_fpts_4', 'rolling_max_fpts_4', 'lag_mean_fpts_2', 'lag_max_fpts_2',
            'lag_min_fpts_2', 'lag_mean_fpts_4', 'lag_max_fpts_4', 'lag_min_fpts_4',
            'lag_mean_fpts_5', 'lag_max_fpts_5', 'lag_min_fpts_5', 'lag_mean_fpts_10',
            'lag_max_fpts_10', 'lag_min_fpts_10', 'fpts_std', 'fpts_volatility',
            'rolling_avg_fpts_3', 
        'rolling_avg_fpts_4', 'fpts_std', 'fpts_volatility',
        '5_game_avg', 'rolling_min_fpts_3', 'rolling_max_fpts_3',
        'rolling_min_fpts_4', 'rolling_max_fpts_4', 
        'lag_mean_fpts_2', 'lag_max_fpts_2', 
        'lag_min_fpts_2', 'lag_mean_fpts_4', 
        'lag_max_fpts_4', 'lag_min_fpts_4',
        'lag_mean_fpts_5', 'lag_max_fpts_5',
        'lag_min_fpts_5', 'lag_mean_fpts_10', 
        'lag_max_fpts_10', 'lag_min_fpts_10',
         
        ]

        # Remove duplicates while preserving order
        numeric_features = list(dict.fromkeys(numeric_features))

        categorical_features = ['Player', 'Team'
       ,
        
    ]

        # Update the list of features to include the new interaction terms
        numeric_features += [
            
        ]
        categorical_features += []

        # Remove duplicates while preserving order
        numeric_features = list(dict.fromkeys(numeric_features))
        categorical_features = list(dict.fromkeys(categorical_features))

        all_features = numeric_features + categorical_features
        all_features = list(dict.fromkeys(all_features))

        print("All features after adding teammate interactions:")
        print(all_features)

        # Use all_features instead of numeric_features + categorical_features
        features = df[all_features]

        # Debug prints to check feature lists and data types
        print("Numeric features:", numeric_features)
        print("Categorical features:", categorical_features)
        print("Data types in DataFrame:")
        print(df.dtypes)

        # Clean infinite values in the dataset before fitting the scaler
        print("Cleaning infinite values in the dataset...")
        df = clean_infinite_values(df)

        # Load or create Scaler
        scaler = load_or_create_scaler(df, numeric_features)

        # Define transformers for preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', scaler)
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # After loading the dataset and before creating the preprocessor
        print("Columns in the original DataFrame:")
        print(df.columns.tolist())

        # Update all_features to only include columns that are in the dataset
        all_features = [feature for feature in selected_features if feature in df.columns]

        print("Updated all_features:")
        print(all_features)

        # Define features_to_use
        leaky_features = ['calculated_dk_fpts', '5_game_avg', 'rolling_mean_fpts']
        features_to_use = [f for f in all_features if f not in leaky_features]

        # Update numeric_features and categorical_features
        numeric_features = [f for f in features_to_use if df[f].dtype in ['int64', 'float64']]
        categorical_features = [f for f in features_to_use if df[f].dtype == 'object']

        print("Updated numeric features:")
        print(numeric_features)
        print("Updated categorical features:")
        print(categorical_features)

        # Create the preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Clean the data before fitting the preprocessor
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Before preparing features for preprocessing
        print("Preparing features for preprocessing...")
        features = df[numeric_features + categorical_features]

        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]

        # Debug print to check data types in features DataFrame
        print("Data types in features DataFrame before preprocessing:")
        print(features.dtypes)

        # Clean the data before fitting the preprocessor
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0, inplace=True)

        # Fit the preprocessor
        print("Fitting preprocessor...")
        preprocessed_features = preprocessor.fit_transform(features)
        n_features = preprocessed_features.shape[1]

        # Feature selection based on the actual number of features
        k = min(50, n_features)  # Increase from 35 to 50

        selector = SelectKBest(f_regression, k=k)

        # Create the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('selector', SelectKBest(f_regression, k=min(50, len(features_to_use)))),
            ('model', final_model)
        ])

        # Clean the data before fitting the preprocessor
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Sort the dataframe by date
        df = df.sort_values('date')

        # Split the data into train and test sets
        train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

        # Use features_to_use for training and testing
        train_features = train_data[features_to_use]
        test_features = test_data[features_to_use]

        print("Fitting the pipeline...")
        print("Shape of train_features:", train_features.shape)
        print("Shape of train_data['calculated_dk_fpts']:", train_data['calculated_dk_fpts'].shape)
        
        pipeline.fit(train_features, train_data['calculated_dk_fpts'])
        
        print("Pipeline fitted successfully.")

        rolling_preds = rolling_window_prediction(test_data, pipeline)
        
        if not rolling_preds.empty and 'predicted_dk_fpts' in rolling_preds.columns:
            # Evaluate rolling predictions
            y_true = rolling_preds['calculated_dk_fpts']
            y_pred = rolling_preds['predicted_dk_fpts']
            mae, mse, r2, mape = evaluate_model(y_true, y_pred)

            print(f'Rolling predictions MAE: {mae}')
            print(f'Rolling predictions MSE: {mse}')
            print(f'Rolling predictions R2: {r2}')
            print(f'Rolling predictions MAPE: {mape}')

            # Calculate player adjustments
            player_adjustments = calculate_player_adjustments(rolling_preds)

            # Save player adjustments
            pd.DataFrame.from_dict(player_adjustments, orient='index', columns=['adjustment']).to_csv('/Users/sineshawmesfintesfaye/newenv/player_adjustments.csv')
            print("Player-specific adjustments saved.")
        else:
            print("No predictions were generated or 'predicted_dk_fpts' column is missing.")

        # Save the final model
        joblib.dump(pipeline, '/Users/sineshawmesfintesfaye/newenv/ensemble_model_pipeline_1_2_sep5nfl.pkl')
        print("Final model pipeline saved.")

        # Save the final data to a CSV file
        df.to_csv('/Users/sineshawmesfintesfaye/newenv/nfl_fantasy_points.csv', index=False)
        print("Final dataset with all features saved.")

        # Save the LabelEncoders
        joblib.dump(le_name, name_encoder_path)
        joblib.dump(le_team, team_encoder_path)
      
        print("LabelEncoders saved.")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total script execution time: {total_time:.2f} seconds.")

        # After cleaning infinite values and before fitting the preprocessor
        print("Columns in DataFrame:")
        print(df.columns.tolist())

        print("Columns in train_features:")
        print(train_features.columns.tolist())

        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            print(f"Warning: Duplicate columns found in DataFrame: {duplicate_columns}")

        duplicate_features = train_features.columns[train_features.columns.duplicated()].tolist()
        if duplicate_features:
            print(f"Warning: Duplicate columns found in train_features: {duplicate_features}")

        # Check the new interaction terms
        print("Sample of new interaction terms:")
        print(df[['Player', 'Team', 'date', 'calculated_dk_fpts']].head())

        # After loading the dataset
        print("Columns in the DataFrame:")
        print(df.columns.tolist())

        # Update all_features to only include columns that are in the dataset
        all_features = [feature for feature in all_features if feature in df.columns]

        print("Updated all_features:")
        print(all_features)

        # Update numeric_features and categorical_features
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]

        print("Updated numeric features:")
        print(numeric_features)
        print("Updated categorical features:")
        print(categorical_features)

        # Remove 'calculated_dk_fpts' from features used for training
        features_to_use = [f for f in all_features if f != 'calculated_dk_fpts']

        # Now use the updated features_to_use
        train_features = train_data[features_to_use]
        test_features = test_data[features_to_use]

        # Continue with the rest of your code...

        # After loading the dataset
        print("Columns in the original DataFrame:")
        print(df.columns.tolist())

        if 'calculated_dk_fpts' not in df.columns:
            print("Warning: 'calculated_dk_fpts' is not in the DataFrame. Please check your data source.")
            # You might want to calculate it here if it's missing
            # df['calculated_dk_fpts'] = calculate_fantasy_points(df)
        else:
            print("'calculated_dk_fpts' is present in the DataFrame.")

    except Exception as e:
        print(f"An error occurred during script execution: {str(e)}")
        import traceback
        traceback.print_exc()