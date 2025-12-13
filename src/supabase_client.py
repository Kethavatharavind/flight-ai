"""
Supabase Client Helper - Cloud Database Integration
✅ Replaces SQLite with Supabase PostgreSQL
✅ Replaces JSON files with JSONB storage
✅ Retry logic with exponential backoff for network resilience
"""

import os
import json
import time
from functools import wraps
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    """
    Decorator to retry a function with exponential backoff.
    Handles transient network/DNS errors like getaddrinfo failures.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Check if it's a retryable network error
                    retryable_errors = [
                        'getaddrinfo',
                        'connection',
                        'timeout',
                        'temporary failure',
                        'name resolution',
                        'network',
                        'socket'
                    ]
                    
                    is_retryable = any(err in error_str for err in retryable_errors)
                    
                    if is_retryable and attempt < max_retries - 1:
                        logger.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Non-retryable error or last attempt, raise immediately
                        raise
            
            # Should not reach here, but just in case
            raise last_exception
        return wrapper
    return decorator

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Try to import supabase
supabase_client = None
USE_SUPABASE = False

try:
    from supabase import create_client, Client
    if SUPABASE_URL and SUPABASE_KEY:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        USE_SUPABASE = True
        logger.info("✅ Supabase connected!")
    else:
        logger.warning("⚠️ Supabase credentials missing - using local storage")
except ImportError:
    logger.warning("⚠️ Supabase not installed - using local storage")
except Exception as e:
    logger.warning(f"⚠️ Supabase connection failed: {e}")


def get_client():
    """Get Supabase client"""
    return supabase_client


def is_cloud_enabled():
    """Check if cloud storage is available"""
    return USE_SUPABASE and supabase_client is not None


# ============================================================
# FLIGHTS TABLE OPERATIONS
# ============================================================

@retry_with_backoff(max_retries=3, initial_delay=1.0)
def insert_flight(flight_data: dict):
    """Insert a flight record with retry logic"""
    if not is_cloud_enabled():
        return None
    
    try:
        result = supabase_client.table('flights').insert(flight_data).execute()
        return result.data
    except Exception as e:
        logger.error(f"❌ Insert flight failed: {e}")
        return None


def get_flights_by_route(origin: str, destination: str, limit: int = 100):
    """Get flights for a specific route"""
    if not is_cloud_enabled():
        return []
    
    try:
        result = supabase_client.table('flights')\
            .select('*')\
            .eq('origin', origin)\
            .eq('destination', destination)\
            .order('flight_date', desc=True)\
            .limit(limit)\
            .execute()
        return result.data
    except Exception as e:
        logger.error(f"❌ Get flights failed: {e}")
        return []


def get_flights_by_date(flight_date: str):
    """Get all flights for a specific date"""
    if not is_cloud_enabled():
        return []
    
    try:
        result = supabase_client.table('flights')\
            .select('*')\
            .eq('flight_date', flight_date)\
            .execute()
        return result.data
    except Exception as e:
        logger.error(f"❌ Get flights by date failed: {e}")
        return []


# ============================================================
# APP_DATA TABLE (JSON STORAGE)
# ============================================================

@retry_with_backoff(max_retries=3, initial_delay=1.0)
def save_json_data(key: str, data: dict):
    """Save JSON data to cloud with retry logic"""
    if not is_cloud_enabled():
        return False
    
    try:
        # Upsert - insert or update
        result = supabase_client.table('app_data').upsert({
            'key': key,
            'data': data,
            'updated_at': 'now()'
        }).execute()
        return True
    except Exception as e:
        logger.error(f"❌ Save JSON failed: {e}")
        return False


@retry_with_backoff(max_retries=3, initial_delay=1.0)
def load_json_data(key: str, default=None):
    """Load JSON data from cloud with retry logic"""
    if not is_cloud_enabled():
        return default
    
    try:
        result = supabase_client.table('app_data')\
            .select('data')\
            .eq('key', key)\
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]['data']
        return default
    except Exception as e:
        logger.error(f"❌ Load JSON failed: {e}")
        return default


# ============================================================
# SPECIFIC DATA HELPERS
# ============================================================

def save_q_table(q_table: dict):
    """Save RL Q-table to cloud"""
    return save_json_data('rl_q_table', q_table)


def load_q_table():
    """Load RL Q-table from cloud"""
    return load_json_data('rl_q_table', {})


def save_rl_metrics(metrics: dict):
    """Save RL metrics to cloud"""
    return save_json_data('rl_metrics', metrics)


def load_rl_metrics():
    """Load RL metrics from cloud"""
    return load_json_data('rl_metrics', {})


def save_predictions(predictions: dict):
    """Save predictions to cloud"""
    return save_json_data('predictions', predictions)


def load_predictions():
    """Load predictions from cloud"""
    return load_json_data('predictions', {})


# ============================================================
# INITIALIZATION
# ============================================================

def create_tables_if_needed():
    """
    Create tables in Supabase.
    NOTE: This should be run once manually via Supabase SQL editor.
    """
    sql = """
    -- Flights table
    CREATE TABLE IF NOT EXISTS flights (
        id SERIAL PRIMARY KEY,
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
        status TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- App data table (for JSON storage)
    CREATE TABLE IF NOT EXISTS app_data (
        key TEXT PRIMARY KEY,
        data JSONB,
        updated_at TIMESTAMP DEFAULT NOW()
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_flights_route ON flights(origin, destination);
    CREATE INDEX IF NOT EXISTS idx_flights_date ON flights(flight_date);
    CREATE INDEX IF NOT EXISTS idx_flights_number ON flights(flight_number);
    """
    print("Run this SQL in Supabase SQL Editor:")
    print(sql)
    return sql
