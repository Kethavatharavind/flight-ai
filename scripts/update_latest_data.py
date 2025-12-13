"""
CRITICAL FIXES:
1. Added pandas import at top
2. Changed days=4 to days=1 for correct "yesterday"
3. Added fresh signals to RL learning
4. Updated paths for new folder structure
"""

import os
import sys
import requests
import json
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import pandas as pd  # ✅ FIX 1: Added here instead of inside function

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add src folder to Python path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# --- CONFIGURATION --- (using new folder structure)
DB_NAME = os.path.join(PROJECT_ROOT, 'data', 'india_data.db')
ROUTES_FILE = os.path.join(PROJECT_ROOT, 'config', 'major_routes.json')
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, 'data')
ROUTES_TO_TRACK = 20
API_SLEEP = 15.0
# ---------------------


def iso_to_time(iso_str):
    """Convert ISO time → Normal 24-hour time"""
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace('Z', '+00:00')).strftime('%H:%M:%S')
    except:
        return None


def init_db(conn):
    """Initialize database tables and indexes"""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_date TEXT,
            flight_number TEXT,
            airline_code TEXT,
            airline_name TEXT,
            origin TEXT,
            destination TEXT,
            scheduled_departure TEXT,
            actual_departure TEXT,
            scheduled_arrival TEXT,
            actual_arrival TEXT,
            departure_delay INTEGER,
            arrival_delay INTEGER,
            status TEXT
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recent_flights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_number TEXT,
            origin TEXT,
            destination TEXT,
            flight_date TEXT,
            status TEXT,
            delay_minutes INTEGER
        );
    """)
    
    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_route ON flights(origin, destination);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON flights(flight_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_flight ON flights(flight_number);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recent_route ON recent_flights(origin, destination);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recent_date ON recent_flights(flight_date);")

    conn.commit()


def archive_old_data(conn, days_to_keep=180):
    """
    Archive old data to CSV before deleting from database.
    Keeps last 180 days in database, exports older to CSV.
    """
    # ✅ FIX: Removed 'import pandas as pd' from here
    
    cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM flights WHERE flight_date < ?", (cutoff_date,))
    old_count = cursor.fetchone()[0]
    
    if old_count == 0:
        print(f"📂 No data older than {days_to_keep} days to archive")
        return
    
    print(f"\n📦 Archiving {old_count} records older than {cutoff_date}...")
    
    archive_filename = os.path.join(ARCHIVE_DIR, f"archived_flights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    try:
        df = pd.read_sql_query(
            "SELECT * FROM flights WHERE flight_date < ?", 
            conn, 
            params=(cutoff_date,)
        )
        
        df.to_csv(archive_filename, index=False)
        print(f"✅ Exported to: {archive_filename}")
        
        cursor.execute("DELETE FROM flights WHERE flight_date < ?", (cutoff_date,))
        conn.commit()
        print(f"🗑️ Removed {old_count} old records from database")
        
    except Exception as e:
        print(f"❌ Archive failed: {e}")


def fetch_with_retry(origin, dest, date_str, retries=3):
    """Fetch flights with retry logic"""
    for attempt in range(retries):
        flights = fetch_historical_flights(origin, dest, date_str)
        if flights:
            return flights

        print(f"    ⚠ Retry {attempt + 1}/{retries} after 10 seconds...")
        time.sleep(10)

    print("    ❌ Failed after retry attempts.")
    return []


def fetch_historical_flights(origin_iata, dest_iata, date_str):
    """Fetch flights for a specific date from AviationStack API"""
    params = {
        'access_key': os.getenv("AVIATIONSTACK_API_KEY"),
        'dep_iata': origin_iata,
        'arr_iata': dest_iata,
        'flight_date': date_str,
        'limit': 100
    }

    try:
        print(f"  Fetching {origin_iata}->{dest_iata} for {date_str}...")
        response = requests.get(
            "http://api.aviationstack.com/v1/flights",
            params=params,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        if data.get('error'):
            error_info = data['error']
            print(f"    API Error: {error_info.get('info', str(error_info))}")
            return []

        flights = data.get("data", [])
        print(f"    Found {len(flights)} flights")
        return flights

    except Exception as e:
        print(f"    Error: {e}")
        return []


def run_update():
    """update function - fetches yesterday's flight data"""
    print("\n STARTING FLIGHT DATA COLLECTION")
    print("=" * 60)

    try:
        with open(ROUTES_FILE, 'r') as f:
            tracked_routes = json.load(f)
        print(f"📊 Loaded {len(tracked_routes)} routes")
    except Exception as e:
        print("❌ Failed to load routes:", e)
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    init_db(conn)

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    total_added = 0

    print("\n" + "📅" * 40)
    print(f"📅 Fetching data for: {yesterday} (Yesterday)")
    print("📅" * 40)

    for i, route in enumerate(tracked_routes[:ROUTES_TO_TRACK]):
        origin = route['ORIGIN']
        dest = route['DEST']

        if i > 0:
            print(f"   ⏳ Waiting {int(API_SLEEP)}s before next route...")
            time.sleep(API_SLEEP)
        
        print(f"🔄 [{i+1}/{ROUTES_TO_TRACK}] {origin}->{dest}")

        flights = fetch_with_retry(origin, dest, yesterday)

        if not flights:
            print("    ❌ No data after retries.")
            continue

        for flight in flights:
            flight_num = flight.get('flight', {}).get('number')
            if not flight_num:
                continue

            status = flight.get('flight_status')

            dep_delay = flight.get('departure', {}).get('delay') or 0
            arr_delay = flight.get('arrival', {}).get('delay') or dep_delay

            if status == 'cancelled':
                final_status = 'cancelled'
            elif status == 'landed':
                final_status = 'delayed' if arr_delay > 15 else 'on_time'
            else:
                continue

            airline_code = flight.get('airline', {}).get('iata')
            airline_name = flight.get('airline', {}).get('name')

            sched_dep = iso_to_time(flight.get('departure', {}).get('scheduled'))
            actual_dep = iso_to_time(flight.get('departure', {}).get('actual'))
            sched_arr = iso_to_time(flight.get('arrival', {}).get('scheduled'))
            actual_arr = iso_to_time(flight.get('arrival', {}).get('actual'))

            try:
                cursor.execute("""
                    INSERT INTO flights (
                        flight_date, flight_number,
                        airline_code, airline_name,
                        origin, destination,
                        scheduled_departure, actual_departure,
                        scheduled_arrival, actual_arrival,
                        departure_delay, arrival_delay,
                        status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    yesterday, flight_num,
                    airline_code, airline_name,
                    origin, dest,
                    sched_dep, actual_dep,
                    sched_arr, actual_arr,
                    dep_delay, arr_delay,
                    final_status
                ))

                total_added += 1
                print(f"    ✅ Stored: {flight_num} ({final_status}, {arr_delay} min)")

            except Exception as e:
                pass

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO recent_flights 
                    (flight_number, origin, destination, flight_date, status, delay_minutes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (flight_num, origin, dest, yesterday, final_status, arr_delay))
            except:
                pass

    conn.commit()
    
    archive_old_data(conn, days_to_keep=180)
    
    conn.close()

    print("\n" + "=" * 60)
    print("✅ DATA COLLECTION COMPLETE!")
    print(f"📅 Date: {yesterday}")
    print(f"📈 TOTAL FLIGHTS STORED: {total_added}")
    print(f"💾 Database: {DB_NAME}")
    print("=" * 60)
    
    # ✅ Verify predictions and RL learning
    verify_predictions_and_learn(yesterday)


def verify_predictions_and_learn(flight_date):
    """
    Check stored predictions against actual outcomes
    and update RL agent with rewards/penalties
    
    ✅ FIX 3: Added fresh signals for proper RL learning
    """
    print("\n" + "=" * 60)
    print("🧠 RL LEARNING: Verifying predictions vs actual outcomes")
    print("=" * 60)
    
    try:
        import prediction_tracker
        import rl_agent
        import data_fetcher  # ✅ FIX 3: Import data_fetcher for fresh signals
    except ImportError as e:
        print(f"⚠️ Could not import learning modules: {e}")
        return
    
    pending = prediction_tracker.get_pending_predictions(flight_date)
    
    if not pending:
        print(f"ℹ️ No pending predictions for {flight_date}")
        return
    
    print(f"📋 Found {len(pending)} predictions to verify")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    verified_count = 0
    learned_count = 0
    correct_predictions = 0
    
    for key, pred in pending.items():
        flight_num = pred['flight_number']
        origin = pred['origin']
        dest = pred['destination']
        predicted_prob = pred['predicted_delay_prob']
        rl_info = pred.get('rl_info', {})
        
        cursor.execute("""
            SELECT status, arrival_delay, scheduled_departure, scheduled_arrival 
            FROM flights
            WHERE flight_number = ? AND origin = ? AND destination = ? AND flight_date = ?
        """, (flight_num, origin, dest, flight_date))
        
        result = cursor.fetchone()
        
        if result:
            actual_status, actual_delay, sched_dep, sched_arr = result
            actual_delayed = actual_status in ['delayed', 'cancelled']
            
            predicted_delayed = predicted_prob > 50
            is_correct = predicted_delayed == actual_delayed
            if is_correct:
                correct_predictions += 1
            
            # ✅ FIX 3: Get fresh signals for proper RL learning
            if rl_info:
                try:
                    # Fetch current signals (weather, airport status at time of flight)
                    dep_time = f"{flight_date}T{sched_dep}" if sched_dep else f"{flight_date}T12:00:00"
                    arr_time = f"{flight_date}T{sched_arr}" if sched_arr else f"{flight_date}T14:00:00"
                    
                    current_signals = data_fetcher.get_prediction_signals(
                        origin, dest, flight_date, 
                        flight_num, dep_time, arr_time
                    )
                    
                    reward = rl_agent.record_outcome(
                        rl_info=rl_info,
                        actual_delayed=actual_delayed,
                        predicted_prob=predicted_prob,
                        current_signals=current_signals  # ✅ This is critical!
                    )
                    learned_count += 1
                    
                    result_emoji = "✅" if is_correct else "❌"
                    print(f"  {result_emoji} {flight_num}: Predicted {predicted_prob}% → Actual {'DELAYED' if actual_delayed else 'ON-TIME'} (Reward: {reward:+.2f})")
                    
                except Exception as e:
                    print(f"  ⚠️ RL learning error for {flight_num}: {e}")
            
            prediction_tracker.mark_prediction_verified(key, actual_delayed, reward if rl_info else None)
            verified_count += 1
    
    conn.close()
    
    if learned_count > 0:
        rl_agent.save_agent_state()
    
    accuracy = (correct_predictions / verified_count * 100) if verified_count > 0 else 0
    
    print(f"\n📊 Verification Summary:")
    print(f"   • Verified: {verified_count}/{len(pending)} predictions")
    print(f"   • Correct: {correct_predictions}/{verified_count} ({accuracy:.1f}%)")
    print(f"   • RL Learned: {learned_count} outcomes")
    
    stats = prediction_tracker.get_stats()
    print(f"   • Overall Accuracy: {stats['accuracy']}%")
    print("=" * 60)


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("AVIATIONSTACK_API_KEY"):
        print("❌ AviationStack API key missing in .env")
        exit(1)

    run_update()
