import pandas as pd
import json
import numpy as np
import os

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- CONFIGURATION --- (using new folder structure)
CSV_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'Processed_data15.csv')
LONG_TERM_SUMMARY_FILE = os.path.join(PROJECT_ROOT, 'data', 'long_term_summary.json')
AVAILABLE_ROUTES_FILE = os.path.join(PROJECT_ROOT, 'config', 'available_routes.json')
MAJOR_ROUTES_FILE = os.path.join(PROJECT_ROOT, 'config', 'major_routes.json')

TOP_N_ROUTES = 250
# ---

def calculate_flight_statistics(df):
    """
    Calculate delay rates per unique flight route (aggregated across all months)
    """
    print("📊 Calculating flight statistics from raw data...")
    
    DELAY_THRESHOLD = 15
    results = []
    
    # ✅ Group by route only (no month)
    grouped = df.groupby(['FL_NUMBER', 'ORIGIN', 'DEST'])
    
    for (flight_num, origin, dest), group in grouped:
        total_flights = len(group)
        
        if total_flights < 3:  # Skip routes with too few samples
            continue
        
        # Cancellation rate
        cancelled_flights = group['CANCELLED'].sum()
        cancel_rate = (cancelled_flights / total_flights) * 100 if total_flights > 0 else 0
        
        # Delay rate
        delayed_flights = len(group[group['DEP_DELAY'] > DELAY_THRESHOLD])
        delay_rate = (delayed_flights / total_flights) * 100 if total_flights > 0 else 0
        
        # Average delay
        delays = group[group['DEP_DELAY'] > 0]['DEP_DELAY']
        avg_delay = delays.mean() if len(delays) > 0 else 0
        if pd.isna(avg_delay):
            avg_delay = 0
        
        # Delay reasons
        delayed_group = group[group['DEP_DELAY'] > 0]
        total_delay_minutes = delayed_group['DEP_DELAY'].sum()
        
        delay_reasons = {}
        if total_delay_minutes > 0:
            reason_columns = {
                'reason_carrier_percent': 'DELAY_DUE_CARRIER',
                'reason_weather_percent': 'DELAY_DUE_WEATHER', 
                'reason_nas_percent': 'DELAY_DUE_NAS',
                'reason_security_percent': 'DELAY_DUE_SECURITY',
                'reason_late_aircraft_percent': 'DELAY_DUE_LATE_AIRCRAFT'
            }
            
            for reason_key, reason_col in reason_columns.items():
                if reason_col in delayed_group.columns:
                    reason_delay = pd.to_numeric(delayed_group[reason_col], errors='coerce').fillna(0).sum()
                    if total_delay_minutes > 0:
                        delay_reasons[reason_key] = round((reason_delay / total_delay_minutes) * 100, 2)
        
        stats = {
            'flight_number': flight_num,
            'origin': origin,
            'destination': dest,
            'total_flights': int(total_flights),
            'delay_rate': round(delay_rate, 2),
            'cancel_rate': round(cancel_rate, 2),
            'avg_delay_time': round(avg_delay, 2),
            **delay_reasons
        }
        
        results.append(stats)
    
    return pd.DataFrame(results)

def create_long_term_summary():
    """
    Creates long_term_summary.json with route-level aggregation (no month grouping)
    """
    print(f"Starting to process {CSV_FILE_PATH} for Layer 1 (Long-Term History)...")
    
    try:
        # Read the CSV file
        print("📖 Reading CSV file...")
        df = pd.read_csv(CSV_FILE_PATH)
        
        # Calculate statistics from raw data
        print("🧮 Calculating flight statistics...")
        stats_df = calculate_flight_statistics(df)
        
        # Create the JSON structure
        summary_stats = {}
        
        for _, row in stats_df.iterrows():
            # ✅ Simple key: flight_number_origin_destination (no month)
            key = f"{row['flight_number']}_{row['origin']}_{row['destination']}"
            
            stats = {
                'delay_rate': row['delay_rate'],
                'cancel_rate': row['cancel_rate'],
                'avg_delay_time': row['avg_delay_time'],
                'total_flights_analyzed': int(row['total_flights']),
                'origin': row['origin'],
                'destination': row['destination']
            }
            
            # Add delay reasons if available
            delay_reason_cols = ['reason_carrier_percent', 'reason_weather_percent', 'reason_nas_percent', 
                               'reason_security_percent', 'reason_late_aircraft_percent']
            for col in delay_reason_cols:
                if col in row and not pd.isna(row[col]):
                    stats[col] = row[col]
            
            summary_stats[key] = stats
        
        # Save the summary JSON
        with open(LONG_TERM_SUMMARY_FILE, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"✅ SUCCESS: Created {LONG_TERM_SUMMARY_FILE} with {len(summary_stats)} records.")
        print(f"📈 Sample statistics:")
        
        # Show some sample data
        sample_keys = list(summary_stats.keys())[:3]
        for key in sample_keys:
            print(f"   {key}: {summary_stats[key]}")

    except FileNotFoundError:
        print(f"❌ ERROR: '{CSV_FILE_PATH}' not found.")
    except Exception as e:
        print(f"❌ ERROR: Failed to process long-term summary: {e}")
        import traceback
        traceback.print_exc()

def create_route_lists():
    """
    Creates two files:
    1. available_routes.json: A full list of ALL routes for the UI.
    2. major_routes.json: The Top 250 routes we will track with our paid API.
    """
    print(f"Starting to process {CSV_FILE_PATH} for route lists...")
    
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        # --- 1. Create available_routes.json (for UI dropdowns) ---
        unique_routes_df = df[['ORIGIN', 'DEST']].drop_duplicates().dropna()
        available_routes_list = unique_routes_df.to_dict('records')
        
        with open(AVAILABLE_ROUTES_FILE, 'w') as f:
            json.dump(available_routes_list, f, indent=2)
        print(f"✅ SUCCESS: Created {AVAILABLE_ROUTES_FILE} with {len(available_routes_list)} total routes.")

        # --- 2. Create major_routes.json (for API tracking) ---
        route_counts = df.groupby(['ORIGIN', 'DEST']).size()
        top_routes_series = route_counts.sort_values(ascending=False).head(TOP_N_ROUTES)
        
        major_routes_list = []
        for (origin, dest) in top_routes_series.index:
            major_routes_list.append({
                "ORIGIN": origin,
                "DEST": dest
            })
            
        with open(MAJOR_ROUTES_FILE, 'w') as f:
            json.dump(major_routes_list, f, indent=2)
        print(f"✅ SUCCESS: Created {MAJOR_ROUTES_FILE} with {len(major_routes_list)} major routes to track.")
        
        # Show top 10 routes by frequency
        print("🏆 Top 10 routes by frequency:")
        for i, ((origin, dest), count) in enumerate(top_routes_series.head(10).items()):
            print(f"   {i+1}. {origin} → {dest}: {count} flights")
        
    except FileNotFoundError:
        print(f"❌ ERROR: '{CSV_FILE_PATH}' not found.")
    except Exception as e:
        print(f"❌ ERROR: Failed to create route lists: {e}")

if __name__ == "__main__":
    print("🚀 Processing flight data for AI prediction system...")
    print("=" * 60)
    
    create_long_term_summary()
    print("\n" + "=" * 60)
    create_route_lists()
    print("\n" + "=" * 60)
    print("🎯 Processing complete! Your system is ready for AI predictions.")