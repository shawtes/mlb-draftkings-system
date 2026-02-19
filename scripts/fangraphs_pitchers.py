# Copyright (c) 2026 Shaw Technologies, LLC
# Licensed under the Research and Personal Use License (RPUL)
# See LICENSE.md for full terms

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import pandas as pd
import time
import os
import glob
import hashlib

# MLB regular season start/end dates (Opening Day through final day)
MLB_SEASONS = {
    2005: ("2005-04-03", "2005-10-02"),
    2006: ("2006-04-02", "2006-10-01"),
    2007: ("2007-04-01", "2007-09-30"),
    2008: ("2008-03-25", "2008-09-28"),
    2009: ("2009-04-05", "2009-10-04"),
    2010: ("2010-04-04", "2010-10-03"),
    2011: ("2011-03-31", "2011-09-28"),
    2012: ("2012-03-28", "2012-10-03"),
    2013: ("2013-03-31", "2013-09-29"),
    2014: ("2014-03-22", "2014-09-28"),
    2015: ("2015-04-05", "2015-10-04"),
    2016: ("2016-04-03", "2016-10-02"),
    2017: ("2017-04-02", "2017-10-01"),
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-20", "2019-09-29"),
    2020: ("2020-07-23", "2020-09-27"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-18", "2025-09-28"),
}

def init_browser(download_dir):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--disable-extensions")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-gpu-logging")
    options.add_argument("--silence-local-ntp")
    options.add_argument("--log-level=3")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Use proper Windows path format
    download_dir = os.path.normpath(download_dir)
    prefs = {"download.default_directory": download_dir}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=options)
    return driver

def clean_download_directory(download_dir):
    """Remove existing CSV files from download directory before starting"""
    csv_files = glob.glob(os.path.join(download_dir, "*.csv"))
    for file in csv_files:
        try:
            os.remove(file)
            print(f"Removed old file: {os.path.basename(file)}")
        except Exception as e:
            print(f"Warning: Could not remove {file}: {e}")

def wait_for_new_download(download_dir, timeout=30):
    """Wait for a new CSV file to appear in the download directory"""
    start_time = time.time()
    initial_files = set(glob.glob(os.path.join(download_dir, "*.csv")))
    
    while time.time() - start_time < timeout:
        current_files = set(glob.glob(os.path.join(download_dir, "*.csv")))
        new_files = current_files - initial_files
        if new_files:
            # Return the newest file
            return max(new_files, key=os.path.getctime)
        time.sleep(1)
    
    return None

def get_file_hash(file_path):
    """Get MD5 hash of a file to verify uniqueness"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None

def analyze_downloaded_data(file_path, date):
    """Analyze the downloaded data to understand what we got"""
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Data analysis for {date}:")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {len(df.columns)}")
        
        # Show first few column names
        cols_preview = list(df.columns[:5])
        print(f"   - First 5 columns: {cols_preview}")
        
        # Check if there's any date-related column
        date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'day', 'game'])]
        if date_cols:
            print(f"   - Date-related columns: {date_cols}")
            # Show unique values in date columns
            for col in date_cols[:2]:  # Check first 2 date columns
                unique_vals = df[col].unique()[:5]  # First 5 unique values
                print(f"     {col} sample values: {unique_vals}")
        
        # Check if data looks empty/minimal
        if len(df) == 0:
            print("   ⚠️  EMPTY DATASET - No data for this date")
        elif len(df) < 10:
            print("   ⚠️  MINIMAL DATA - Very few rows, might be off-season")
        
        return True
    except Exception as e:
        print(f"   ❌ Error analyzing data: {e}")
        return False

def verify_data_changed(current_hash, previous_hashes, date):
    """Verify that downloaded data is unique for this date"""
    if current_hash in previous_hashes:
        print(f"WARNING: Data for {date} appears to be identical to previous downloads!")
        print("This might indicate the date filter is not working correctly.")
        print(f"File hash: {current_hash}")
        return False
    else:
        print(f"✅ Data verified as unique for {date}")
        return True

def download_data_for_date(driver, date, download_dir):
    formatted_date = date.strftime('%Y-%m-%d')
    # Set both start and end date to the same value for single-day download
    custom_report_url = f"https://www.fangraphs.com/leaders/major-league?pos=all&stats=pit&lg=all&qual=y&type=c%2C4%2C6%2C11%2C12%2C13%2C21%2C-1%2C34%2C35%2C40%2C41%2C-1%2C23%2C37%2C38%2C50%2C317%2C61%2C-1%2C111%2C-1%2C203%2C199%2C58%2C-1%2C0%2C1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%2C11%2C12%2C13%2C14%2C15%2C16%2C17%2C18%2C19%2C20%2C21%2C22%2C23%2C24%2C25%2C26%2C27%2C28%2C29%2C30%2C31%2C32%2C33%2C34%2C35%2C36%2C37%2C38%2C39%2C40%2C41%2C42%2C43%2C44%2C45%2C46%2C47%2C48%2C49%2C50%2C51%2C52%2C53%2C54%2C55%2C56%2C57%2C58%2C59%2C60%2C61%2C62%2C63%2C64%2C65%2C66%2C67%2C68%2C69%2C70%2C71%2C72%2C73%2C74%2C75%2C76%2C77%2C78%2C79%2C80%2C81%2C82%2C83%2C84%2C85%2C86%2C87%2C88%2C89%2C90%2C91%2C92%2C93%2C94%2C95%2C96%2C97%2C98%2C99%2C100%2C101%2C102%2C103%2C104%2C105%2C106%2C107%2C108%2C109%2C110%2C111%2C112%2C113%2C114%2C115%2C116%2C117%2C118%2C119%2C120%2C121%2C122%2C123%2C124%2C125%2C126%2C127%2C128%2C129%2C130%2C131%2C132%2C133%2C134%2C135%2C136%2C137%2C138%2C139%2C140%2C141%2C142%2C143%2C144%2C145%2C146%2C147%2C148%2C149%2C150%2C151%2C152%2C153%2C154%2C155%2C156%2C157%2C158%2C159%2C160%2C161%2C162%2C163%2C164%2C165%2C166%2C167%2C168%2C169%2C170%2C171%2C172%2C173%2C174&season={date.year}&month=1000&season1={date.year}&ind=0&startdate={formatted_date}&enddate={formatted_date}&team=0&v_cr=202301"

    print(f"Downloading data for {formatted_date}...")
    print(f"URL: {custom_report_url[:100]}...")  # Show truncated URL for debugging
    driver.get(custom_report_url)
    time.sleep(5)

    try:
        # Click the export data button
        export_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.data-export'))
        )
        export_button.click()
        print(f"Export button clicked for {formatted_date}, waiting for download...")
        
        # Wait for the new file to be downloaded
        latest_file_path = wait_for_new_download(download_dir, timeout=30)
        
        if latest_file_path:
            print(f"Successfully downloaded file for {formatted_date}: {os.path.basename(latest_file_path)}")
            print(f"Full path: {latest_file_path}")
            return latest_file_path, formatted_date
        else:
            print(f"Error: No new file detected for {formatted_date}")
            return None
            
    except Exception as e:
        print(f"Error during data download for {formatted_date}: {e}")
        return None

def append_data_to_daily_file(file_path, game_date, save_dir):
    try:
        # Ensure the file exists and is readable
        if not os.path.exists(file_path):
            print(f"Error: Downloaded file {file_path} does not exist")
            return False
            
        if not os.path.isfile(file_path):
            print(f"Error: {file_path} is not a file")
            return False
            
        data = pd.read_csv(file_path)
        
        # Add the download date as a column
        data['download_date'] = game_date
        data['date'] = game_date  # Also add as 'date' for clarity
        
        print(f"   - Adding date column: {game_date}")
        print(f"   - Data shape: {data.shape}")
        
        # Define the master file path
        master_filename = os.path.join(save_dir, "merged_fangraphs_data_pitchers.csv")
        
        # Check if master file exists
        if os.path.exists(master_filename):
            print(f"   - Appending to existing master file")
            try:
                existing_data = pd.read_csv(master_filename)
                print(f"   - Existing data shape: {existing_data.shape}")
                
                # Merge the data
                updated_data = pd.concat([existing_data, data], ignore_index=True)
                print(f"   - Combined data shape: {updated_data.shape}")
            except Exception as e:
                print(f"   - Error reading existing file, creating new one: {e}")
                updated_data = data
        else:
            print(f"   - Creating new master file")
            updated_data = data

        # Save the merged data
        updated_data.to_csv(master_filename, index=False)
        print(f"✅ Data for {game_date} merged into {master_filename}")
        
        # Also save individual daily file as backup
        daily_filename = os.path.join(save_dir, f"daily_backup_{game_date}pitchers.csv")
        data.to_csv(daily_filename, index=False)
        print(f"   - Daily backup saved: {os.path.basename(daily_filename)}")
        
        # Clean up the original download file
        try:
            os.remove(file_path)
            print(f"   - Cleaned up temporary download file")
        except Exception as e:
            print(f"   - Warning: Could not remove temp file: {e}")
            
        return True
        
    except PermissionError as e:
        print(f"❌ Permission error reading file {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing file {file_path}: {e}")
        return False

def view_merged_data_stats():
    """View statistics of the merged data file"""
    save_dir = os.path.join(os.path.expanduser("~"), "FangraphsData")
    master_file = os.path.join(save_dir, "merged_fangraphs_datapitchers.csv")
    
    print("📊 MERGED DATA STATISTICS")
    print("=" * 50)
    
    if not os.path.exists(master_file):
        print("❌ No merged data file found!")
        print(f"   Expected location: {master_file}")
        return
    
    try:
        df = pd.read_csv(master_file)
        
        print(f"📈 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"💾 File size: {os.path.getsize(master_file) / (1024*1024):.2f} MB")
        
        # Date information
        if 'date' in df.columns:
            print(f"\n📅 DATE INFORMATION:")
            print(f"   Unique dates: {df['date'].nunique()}")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Records per date
            date_counts = df['date'].value_counts().sort_index()
            print(f"   Average records per date: {date_counts.mean():.1f}")
            print(f"   Min records per date: {date_counts.min()}")
            print(f"   Max records per date: {date_counts.max()}")
            
            print(f"\n📋 SAMPLE DATES AND RECORD COUNTS:")
            for date, count in date_counts.head(10).items():
                print(f"   {date}: {count} records")
            if len(date_counts) > 10:
                print(f"   ... and {len(date_counts) - 10} more dates")
        
        # Column information
        print(f"\n📊 COLUMN INFORMATION:")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Sample columns: {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more columns")
        
        # Data quality
        print(f"\n🔍 DATA QUALITY:")
        missing_data = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (missing_data / total_cells) * 100
        print(f"   Missing values: {missing_data:,} ({missing_pct:.2f}% of all cells)")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        print(f"   Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
        
        print(f"\n✅ Data file location: {master_file}")
        
    except Exception as e:
        print(f"❌ Error reading merged data: {e}")

def run_download(date_ranges, label="custom"):
    """Core download loop. date_ranges is a list of (start_date, end_date) tuples."""
    download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    save_dir = os.path.join(os.path.expanduser("~"), "FangraphsData")
    os.makedirs(save_dir, exist_ok=True)

    # Build the full list of dates across all ranges
    all_dates = []
    for start_date, end_date in date_ranges:
        days = (end_date - start_date).days + 1
        all_dates.extend([start_date + timedelta(days=i) for i in range(days)])

    total_days = len(all_dates)
    first_date = all_dates[0].strftime('%Y-%m-%d')
    last_date = all_dates[-1].strftime('%Y-%m-%d')

    print(f"\n📁 Download directory: {download_dir}")
    print(f"💾 Save directory: {save_dir}")
    print(f"📊 Master file: {os.path.join(save_dir, 'merged_fangraphs_data_pitchers.csv')}")
    print(f"📅 Total days to process: {total_days} ({first_date} to {last_date})")
    print(f"📦 Seasons: {len(date_ranges)}")

    print("\n🧹 Cleaning download directory...")
    clean_download_directory(download_dir)

    driver = init_browser(download_dir)
    driver.get("https://www.fangraphs.com/")
    input("🔐 Please log in to Fangraphs in the opened browser and press Enter to continue...")

    print("🚀 Proceeding with data download...")

    downloaded_dates = []
    successful_merges = []
    file_hashes = []
    skipped_empty = 0

    for i, date in enumerate(all_dates):
        print(f"\n📊 [{i+1}/{total_days}] {date.strftime('%Y-%m-%d')}")
        print("-" * 50)

        result = download_data_for_date(driver, date, download_dir)
        if result:
            file_path, game_date = result

            analyze_downloaded_data(file_path, game_date)

            file_hash = get_file_hash(file_path)
            if file_hash:
                if verify_data_changed(file_hash, file_hashes, game_date):
                    file_hashes.append(file_hash)

                    merge_success = append_data_to_daily_file(file_path, game_date, save_dir)
                    if merge_success:
                        downloaded_dates.append(game_date)
                        successful_merges.append(game_date)
                        print(f"✅ Successfully processed {game_date}")
                    else:
                        print(f"❌ Failed to merge data for {game_date}")
                else:
                    print(f"⚠️  Skipping duplicate data for {game_date}")
                    skipped_empty += 1
            else:
                print(f"⚠️  Warning: Could not verify data for {game_date}, proceeding anyway")
                merge_success = append_data_to_daily_file(file_path, game_date, save_dir)
                if merge_success:
                    downloaded_dates.append(game_date)
                    successful_merges.append(game_date)
        else:
            print(f"❌ No data for {date.strftime('%Y-%m-%d')} (off-day or no games)")
            skipped_empty += 1

    driver.quit()

    # Save download log
    download_log_file = os.path.join(save_dir, f"download_log_pitchers_{label}.txt")
    with open(download_log_file, "w") as file:
        file.write(f"Download completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Mode: {label}\n")
        file.write(f"Date range: {first_date} to {last_date}\n")
        file.write(f"Total days attempted: {total_days}\n")
        file.write(f"Successfully processed: {len(successful_merges)}\n")
        file.write(f"Skipped/failed: {skipped_empty}\n\n")
        file.write(f"Successfully processed dates:\n")
        for date in successful_merges:
            file.write(f"{date}\n")

    # Final summary
    print(f"\n🎉 DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"📊 Total days attempted: {total_days}")
    print(f"✅ Successfully downloaded: {len(downloaded_dates)}")
    print(f"💾 Successfully merged: {len(successful_merges)}")
    print(f"⏭️  Skipped (off-days/dupes): {skipped_empty}")

    master_file = os.path.join(save_dir, "merged_fangraphs_data_pitchers.csv")
    if os.path.exists(master_file):
        try:
            final_data = pd.read_csv(master_file)
            print(f"📈 Final merged dataset: {final_data.shape[0]} rows, {final_data.shape[1]} columns")

            if 'date' in final_data.columns:
                date_counts = final_data['date'].value_counts().sort_index()
                unique_years = sorted(set(d[:4] for d in final_data['date'].astype(str)))
                print(f"📅 Years covered: {', '.join(unique_years)}")
                print(f"📅 Unique game dates: {len(date_counts)}")
                print(f"📅 Date distribution:")
                for date, count in date_counts.head(10).items():
                    print(f"   {date}: {count} records")
                if len(date_counts) > 10:
                    print(f"   ... and {len(date_counts) - 10} more dates")

        except Exception as e:
            print(f"⚠️  Could not analyze final dataset: {e}")

    print(f"📁 All files saved to: {save_dir}")
    print(f"📋 Download log saved to: {download_log_file}")

    unique_files = len(set(file_hashes))
    if unique_files < len(file_hashes):
        print(f"⚠️  WARNING: Only {unique_files} unique files out of {len(file_hashes)} downloads")


def main():
    """Download data for a custom date range."""
    start_date = datetime.strptime(input("Enter the start date (YYYY-MM-DD): "), '%Y-%m-%d')
    end_date = datetime.strptime(input("Enter the end date (YYYY-MM-DD): "), '%Y-%m-%d')
    run_download([(start_date, end_date)], label="custom")


def download_all_seasons():
    """Download all MLB seasons from 2025 down to 2005 (newest first)."""
    print("\n⚾ ALL PITCHERS DATA — MLB Seasons 2025-2005 (newest first)")
    print("=" * 60)

    total_days = 0
    for year in sorted(MLB_SEASONS.keys(), reverse=True):
        start_str, end_str = MLB_SEASONS[year]
        start = datetime.strptime(start_str, '%Y-%m-%d')
        end = datetime.strptime(end_str, '%Y-%m-%d')
        days = (end - start).days + 1
        total_days += days
        note = ""
        if year == 2020:
            note = " (COVID 60-game season)"
        elif year == 2022:
            note = " (lockout delayed)"
        print(f"   {year}: {start_str} to {end_str}  ({days} days){note}")

    print(f"\n   TOTAL: {total_days} days across {len(MLB_SEASONS)} seasons")
    print(f"   Estimated time: ~{total_days * 8 // 3600}h {(total_days * 8 % 3600) // 60}m (at ~8s per day)")

    confirm = input(f"\nProceed with downloading all {total_days} days? (yes/no): ").strip().lower()
    if confirm not in ('yes', 'y'):
        print("Cancelled.")
        return

    date_ranges = []
    for year in sorted(MLB_SEASONS.keys(), reverse=True):
        start_str, end_str = MLB_SEASONS[year]
        start = datetime.strptime(start_str, '%Y-%m-%d')
        end = datetime.strptime(end_str, '%Y-%m-%d')
        date_ranges.append((start, end))

    run_download(date_ranges, label="all_seasons_2025_2005")


def download_single_season():
    """Download a single MLB season by year."""
    year_str = input("Enter the season year (2005-2025): ").strip()
    try:
        year = int(year_str)
    except ValueError:
        print(f"❌ Invalid year: {year_str}")
        return

    if year not in MLB_SEASONS:
        print(f"❌ No season data for {year}. Available: {min(MLB_SEASONS)}-{max(MLB_SEASONS)}")
        return

    start_str, end_str = MLB_SEASONS[year]
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end = datetime.strptime(end_str, '%Y-%m-%d')
    days = (end - start).days + 1

    print(f"\n⚾ {year} MLB Season: {start_str} to {end_str} ({days} days)")
    run_download([(start, end)], label=f"season_{year}")


def download_year_range():
    """Download a range of MLB seasons (e.g. 2018-2023), newest first."""
    start_year = int(input("Enter start year (2005-2025): ").strip())
    end_year = int(input("Enter end year (2005-2025): ").strip())

    if start_year > end_year:
        start_year, end_year = end_year, start_year

    date_ranges = []
    for year in range(end_year, start_year - 1, -1):  # newest first
        if year not in MLB_SEASONS:
            print(f"⚠️  Skipping {year} — no season data available")
            continue
        start_str, end_str = MLB_SEASONS[year]
        start = datetime.strptime(start_str, '%Y-%m-%d')
        end = datetime.strptime(end_str, '%Y-%m-%d')
        date_ranges.append((start, end))
        days = (end - start).days + 1
        print(f"   {year}: {start_str} to {end_str} ({days} days)")

    if not date_ranges:
        print("❌ No valid seasons in that range.")
        return

    total_days = sum((e - s).days + 1 for s, e in date_ranges)
    print(f"\n   TOTAL: {total_days} days across {len(date_ranges)} seasons")

    confirm = input(f"\nProceed? (yes/no): ").strip().lower()
    if confirm not in ('yes', 'y'):
        print("Cancelled.")
        return

    run_download(date_ranges, label=f"seasons_{end_year}_{start_year}")


if __name__ == "__main__":
    print("🔗 FANGRAPHS PITCHERS DATA DOWNLOADER")
    print("=" * 50)
    print("1. Download custom date range")
    print("2. Download ALL data (2025-2005 — every MLB season, newest first)")
    print("3. Download single season (pick a year)")
    print("4. Download year range (e.g. 2018-2023)")
    print("5. View merged data statistics")
    print("6. Exit")

    choice = input("\nSelect an option (1-6): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        download_all_seasons()
    elif choice == "3":
        download_single_season()
    elif choice == "4":
        download_year_range()
    elif choice == "5":
        view_merged_data_stats()
    elif choice == "6":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice.")