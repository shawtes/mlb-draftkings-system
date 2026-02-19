import pandas as pd
import os

def calculate_dk_fpts_hitters(row):
    # Use .get for fallback to 0 if column is missing
    singles = row.get('1B', 0)
    doubles = row.get('2B', 0)
    triples = row.get('3B', 0)
    home_runs = row.get('HR', 0)
    rbis = row.get('RBI', 0)
    runs = row.get('R', 0)
    walks = row.get('BB', 0)
    hbp = row.get('HBP', 0)
    stolen_bases = row.get('SB', 0)

    # Some columns may be floats, ensure they're numbers
    try:
        singles = float(singles)
        doubles = float(doubles)
        triples = float(triples)
        home_runs = float(home_runs)
        rbis = float(rbis)
        runs = float(runs)
        walks = float(walks)
        hbp = float(hbp)
        stolen_bases = float(stolen_bases)
    except Exception:
        singles = doubles = triples = home_runs = rbis = runs = walks = hbp = stolen_bases = 0

    dk_fpts = (
        singles * 3 +
        doubles * 5 +
        triples * 8 +
        home_runs * 10 +
        rbis * 2 +
        runs * 2 +
        walks * 2 +
        hbp * 2 +
        stolen_bases * 5
    )
   
    return dk_fpts

def main(): 
    input_csv = 'C:\\Users\\smtes\\FangraphsData\\merged_fangraphs_data.csv'
    output_csv = 'C:\\Users\\smtes\\FangraphsData\\merged_fangraphs_data_output.csv'

    df = pd.read_csv(input_csv)
    print("Initial DataFrame columns:", df.columns.tolist())

    # If any columns are missing, add them as 0
    for col in ['1B', '2B', '3B', 'HR', 'RBI', 'R', 'BB', 'HBP', 'SB']:
        if col not in df.columns:
            df[col] = 0
    print("DataFrame after adding missing columns:", df.head())

    df['dk_fpts'] = df.apply(calculate_dk_fpts_hitters, axis=1)
    print("DataFrame after calculating dk_fpts:", df[['dk_fpts']].head())

    # Ensure 'dk_fpts' column exists before saving
    if 'dk_fpts' in df.columns:
        print("'dk_fpts' column exists in DataFrame before saving.")
    else:
        print("Error: 'dk_fpts' column is missing in DataFrame before saving.")

    # Rename 'dk_fpts_x' to 'dk_fpts' and drop 'dk_fpts_y' if it exists
    if 'dk_fpts_x' in df.columns:
        df.rename(columns={'dk_fpts_x': 'dk_fpts'}, inplace=True)
        print("Renamed 'dk_fpts_x' to 'dk_fpts'.")
    if 'dk_fpts_y' in df.columns:
        df.drop(columns=['dk_fpts_y'], inplace=True)
        print("Dropped 'dk_fpts_y' column.")

    df.to_csv(output_csv, index=False)
    print(f"DraftKings fantasy points calculated and saved to {output_csv}")

    # Verify the saved CSV file
    saved_df = pd.read_csv(output_csv)
    print("Columns in the saved CSV file:", saved_df.columns.tolist())

if __name__ == "__main__":
    main()