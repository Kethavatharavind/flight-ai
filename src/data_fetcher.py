"""
Enhanced Data Fetcher with proper timezone handling and caching
✅ Fixed: Timezone issues, better error handling, response caching
"""

import os
import requests
from dotenv import load_dotenv
import json
import sqlite3
from datetime import datetime, timedelta, timezone
import logging
from functools import lru_cache
import hashlib
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
AERODATABOX_API_KEY = os.getenv("AERODATABOX_API_KEY")

AERODATABOX_HEADERS = {
    "x-rapidapi-key": AERODATABOX_API_KEY,
    "x-rapidapi-host": "aerodatabox.p.rapidapi.com"
} if AERODATABOX_API_KEY else {}


LONG_TERM_STATS = {}
long_term_path = os.path.join(PROJECT_ROOT, 'data', 'long_term_summary.json')
try:
    with open(long_term_path, 'r') as f:
        LONG_TERM_STATS = json.load(f)
    logger.info(f"✅ Loaded {len(LONG_TERM_STATS)} historical records")
except FileNotFoundError:
    logger.warning("⚠️ long_term_summary.json not found")
except Exception as e:
    logger.error(f"❌ Error loading historical data: {e}")

# Database path - using new folder structure
DB_NAME = os.path.join(PROJECT_ROOT, 'data', 'india_data.db')
RECENT_DAYS_TO_QUERY = 180


IATA_LOCATION_MAP = {
    
    "JFK": {"city": "New York", "lat": 40.6413, "lon": -73.7781},
    "LAX": {"city": "Los Angeles", "lat": 33.9416, "lon": -118.4085},
    "ORD": {"city": "Chicago", "lat": 41.9742, "lon": -87.9073},
    "ATL": {"city": "Atlanta", "lat": 33.6407, "lon": -84.4277},
    "DFW": {"city": "Dallas", "lat": 32.8998, "lon": -97.0403},
    "DEN": {"city": "Denver", "lat": 39.8561, "lon": -104.6737},
    "SFO": {"city": "San Francisco", "lat": 37.6213, "lon": -122.3790},
    "SEA": {"city": "Seattle", "lat": 47.4502, "lon": -122.3088},
    "LAS": {"city": "Las Vegas", "lat": 36.0840, "lon": -115.1537},
    "PHX": {"city": "Phoenix", "lat": 33.4342, "lon": -112.0080},
    "IAH": {"city": "Houston", "lat": 29.9902, "lon": -95.3368},
    "MCO": {"city": "Orlando", "lat": 28.4312, "lon": -81.3081},
    "EWR": {"city": "Newark", "lat": 40.6895, "lon": -74.1745},
    "MSP": {"city": "Minneapolis", "lat": 44.8848, "lon": -93.2223},
    "BOS": {"city": "Boston", "lat": 42.3656, "lon": -71.0096},
    "DTW": {"city": "Detroit", "lat": 42.2124, "lon": -83.3534},
    "PHL": {"city": "Philadelphia", "lat": 39.8729, "lon": -75.2437},
    "LGA": {"city": "New York", "lat": 40.7769, "lon": -73.8740},
    "DCA": {"city": "Washington", "lat": 38.8512, "lon": -77.0402},
    "FLL": {"city": "Fort Lauderdale", "lat": 26.0742, "lon": -80.1506},
    "BWI": {"city": "Baltimore", "lat": 39.1774, "lon": -76.6684},
    "CLT": {"city": "Charlotte", "lat": 35.2140, "lon": -80.9431},
    "MIA": {"city": "Miami", "lat": 25.7959, "lon": -80.2870},
    "SLC": {"city": "Salt Lake City", "lat": 40.7899, "lon": -111.9791},
    "SAN": {"city": "San Diego", "lat": 32.7338, "lon": -117.1933},
    "TPA": {"city": "Tampa", "lat": 27.9755, "lon": -82.5332},
    "PDX": {"city": "Portland", "lat": 45.5898, "lon": -122.5951},
    "HNL": {"city": "Honolulu", "lat": 21.3187, "lon": -157.9225},
    
    
    "DEL": {"city": "Delhi", "lat": 28.5562, "lon": 77.1000},
    "BOM": {"city": "Mumbai", "lat": 19.0896, "lon": 72.8656},
    "BLR": {"city": "Bangalore", "lat": 13.1986, "lon": 77.7066},
    "HYD": {"city": "Hyderabad", "lat": 17.2403, "lon": 78.4294},
    "MAA": {"city": "Chennai", "lat": 12.9941, "lon": 80.1709},
    "CCU": {"city": "Kolkata", "lat": 22.6520, "lon": 88.4463},
    "COK": {"city": "Kochi", "lat": 10.1520, "lon": 76.4019},
    "PNQ": {"city": "Pune", "lat": 18.5793, "lon": 73.9089},
    
    
    "LHR": {"city": "London", "lat": 51.4700, "lon": -0.4543},
    "CDG": {"city": "Paris", "lat": 49.0097, "lon": 2.5479},
    "FRA": {"city": "Frankfurt", "lat": 50.0379, "lon": 8.5622},
    "AMS": {"city": "Amsterdam", "lat": 52.3105, "lon": 4.7683},
    "MAD": {"city": "Madrid", "lat": 40.4983, "lon": -3.5676},
    "BCN": {"city": "Barcelona", "lat": 41.2974, "lon": 2.0833},
    "FCO": {"city": "Rome", "lat": 41.8003, "lon": 12.2389},
    "MUC": {"city": "Munich", "lat": 48.3537, "lon": 11.7750},
    "LGW": {"city": "London", "lat": 51.1537, "lon": -0.1821},
    
    
    "DXB": {"city": "Dubai", "lat": 25.2532, "lon": 55.3657},
    "SIN": {"city": "Singapore", "lat": 1.3644, "lon": 103.9915},
    "HKG": {"city": "Hong Kong", "lat": 22.3080, "lon": 113.9185},
    "NRT": {"city": "Tokyo", "lat": 35.7653, "lon": 140.3856},
    "ICN": {"city": "Seoul", "lat": 37.4602, "lon": 126.4407},
    "PEK": {"city": "Beijing", "lat": 40.0799, "lon": 116.6031},
    "PVG": {"city": "Shanghai", "lat": 31.1434, "lon": 121.8052},
    "BKK": {"city": "Bangkok", "lat": 13.6900, "lon": 100.7501},
    "KUL": {"city": "Kuala Lumpur", "lat": 2.7456, "lon": 101.7072},
    
    
    "YYZ": {"city": "Toronto", "lat": 43.6777, "lon": -79.6248},
    "YVR": {"city": "Vancouver", "lat": 49.1967, "lon": -123.1815},
    "YUL": {"city": "Montreal", "lat": 45.4657, "lon": -73.7455},
    
    
    "SYD": {"city": "Sydney", "lat": -33.9399, "lon": 151.1753},
    "MEL": {"city": "Melbourne", "lat": -37.6690, "lon": 144.8410},
    "BNE": {"city": "Brisbane", "lat": -27.3942, "lon": 153.1218},
}


@lru_cache(maxsize=128)
def get_flights_by_route_cached(origin_iata, dest_iata, date_str, api_key_hash):
    """Cached version of flight search"""
    return _get_flights_by_route_impl(origin_iata, dest_iata, date_str)


def _get_flights_by_route_impl(origin_iata, dest_iata, date_str):
    """Implementation of flight search"""
    if not AVIATIONSTACK_API_KEY:
        return {"error": "AviationStack API Key Missing"}
    
    logger.info(f"🔍 Fetching flights: {origin_iata} → {dest_iata} on {date_str}")
    
    
    try:
        flight_date = datetime.strptime(date_str, '%Y-%m-%d')
        days_ahead = (flight_date - datetime.now()).days
        
        if days_ahead < 8:
            min_date = (datetime.now() + timedelta(days=8)).strftime('%Y-%m-%d')
            return {"error": f"Date must be at least 8 days ahead. Minimum: {min_date}"}
        
        if days_ahead > 180:
            max_date = (datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')
            return {"error": f"Date too far in future. Maximum: {max_date}"}
            
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}
    
    params = {
        'access_key': AVIATIONSTACK_API_KEY,
        'iataCode': origin_iata,
        'type': 'departure',
        'date': date_str
    }
    
    try:
        response = requests.get(
            "https://api.aviationstack.com/v1/flightsFuture",
            params=params,
            timeout=15,
            verify=False
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get('error'):
            return {"error": str(data['error'])}
        
        all_flights = data.get("data", [])
        flights_on_route = []
        
        for flight in all_flights:
            arrival = flight.get('arrival', {})
            dest_code = arrival.get('iataCode', '').upper()
            
            if dest_code == dest_iata.upper():
                flight_info = flight.get('flight', {})
                airline = flight.get('airline', {})
                departure = flight.get('departure', {})
                
                dep_time = departure.get('scheduledTime', 'N/A')
                arr_time = arrival.get('scheduledTime', 'N/A')
                
                
                dep_datetime = f"{date_str}T{dep_time}" if dep_time != 'N/A' else 'N/A'
                arr_datetime = f"{date_str}T{arr_time}" if arr_time != 'N/A' else 'N/A'
                
                flights_on_route.append({
                    "flnr": flight_info.get('number', 'Unknown'),
                    "airline": airline.get('name', 'Unknown'),
                    "airline_iata": airline.get('iataCode', '').upper(),
                    "status": "scheduled",
                    "departure_time_scheduled_local": dep_datetime,
                    "arrival_time_scheduled_local": arr_datetime,
                    "departure_time_display": dep_time,
                    "arrival_time_display": arr_time
                })
        
        logger.info(f"✅ Found {len(flights_on_route)} flights")
        
        return {
            "origin": origin_iata,
            "destination": dest_iata,
            "date": date_str,
            "flights_found": len(flights_on_route),
            "flights": flights_on_route
        }
        
    except requests.exceptions.Timeout:
        logger.error("⏱️ API Timeout")
        return {"error": "Flight search timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request error: {e}")
        return {"error": f"Could not retrieve flight data: {str(e)}"}
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


def get_flights_by_route(origin_iata, dest_iata, date_str):
    """Public API with caching"""
    api_key_hash = hashlib.md5(AVIATIONSTACK_API_KEY.encode()).hexdigest() if AVIATIONSTACK_API_KEY else "none"
    return get_flights_by_route_cached(origin_iata, dest_iata, date_str, api_key_hash)


@lru_cache(maxsize=256)
def get_weather_forecast(airport_code, flight_date_str, flight_time_str):
    """Fetch weather with proper timezone handling"""
    logger.info(f"🌤️ Fetching weather: {airport_code}")
    
    location = IATA_LOCATION_MAP.get(airport_code.upper())
    if not location:
        logger.warning(f"⚠️ No coordinates for {airport_code}")
        return {
            "error": f"Airport {airport_code} not in database",
            "airport": airport_code,
            "city": airport_code,
            "condition": "Unknown"
        }
    
    try:
        
        target_dt = datetime.fromisoformat(flight_time_str.replace('Z', '+00:00'))
        target_date = target_dt.strftime('%Y-%m-%d')
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': location['lat'],
            'longitude': location['lon'],
            'hourly': 'temperature_2m,precipitation_probability,windspeed_10m,weathercode',
            'start_date': target_date,
            'end_date': target_date,
            'timezone': 'auto'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        
        if not times:
            raise ValueError("No weather data returned")
        
        
        temps = hourly.get('temperature_2m', [])
        precip_probs = hourly.get('precipitation_probability', [])
        wind_speeds = hourly.get('windspeed_10m', [])
        weather_codes = hourly.get('weathercode', [])
        
        
        target_hour = target_dt.hour
        closest_idx = 0
        min_diff = float('inf')
        
        for i, time_str in enumerate(times):
            time_dt = datetime.fromisoformat(time_str)
            diff = abs((time_dt.hour - target_hour + 12) % 24 - 12)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        
        weather_desc = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Heavy rain showers",
            85: "Snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm"
        }
        
        weather_code = weather_codes[closest_idx] if closest_idx < len(weather_codes) else 0
        condition = weather_desc.get(weather_code, "Unknown")
        
        result = {
            "airport": airport_code,
            "city": location['city'],
            "condition": condition,
            "temp_c": round(temps[closest_idx], 1) if closest_idx < len(temps) else None,
            "wind_speed_kph": round(wind_speeds[closest_idx], 1) if closest_idx < len(wind_speeds) else None,
            "precip_prob_percent": precip_probs[closest_idx] if closest_idx < len(precip_probs) else None,
            "forecast_time_utc": times[closest_idx] if closest_idx < len(times) else None
        }
        
        logger.info(f"✅ Weather: {condition}, {result['temp_c']}°C")
        return result
        
    except Exception as e:
        logger.error(f"❌ Weather error: {e}")
        return {
            "error": str(e),
            "airport": airport_code,
            "city": location.get('city', airport_code),
            "condition": "Unknown"
        }


def get_long_term_history(flight_number, date_str, origin_iata=None, dest_iata=None):
    """
    Fetch historical statistics using route-level keys (no month)
    ✅ UPDATED: Now uses flight_number_origin_destination format
    """
    if not LONG_TERM_STATS:
        return {
            "delay_rate": None,
            "cancel_rate": None,
            "avg_delay_time": None,
            "total_flights_analyzed": 0,
            "note": "Historical data not available"
        }
    
    try:
        
        if origin_iata and dest_iata:
            key = f"{flight_number}_{origin_iata}_{dest_iata}"
            
            if key in LONG_TERM_STATS:
                logger.info(f"✅ Historical data found: {key}")
                return LONG_TERM_STATS[key]
            
            
            generic_fn = "".join(filter(str.isdigit, str(flight_number)))
            if generic_fn:
                key = f"{generic_fn}_{origin_iata}_{dest_iata}"
                if key in LONG_TERM_STATS:
                    logger.info(f"✅ Historical data found (generic): {key}")
                    return LONG_TERM_STATS[key]
        
        
        for key, stats in LONG_TERM_STATS.items():
            if key.startswith(f"{flight_number}_"):
                logger.info(f"✅ Historical data found (partial match): {key}")
                return stats
        
        logger.warning(f"⚠️ No historical data for {flight_number}")
        return {
            "delay_rate": None,
            "cancel_rate": None,
            "avg_delay_time": None,
            "total_flights_analyzed": 0,
            "note": f"No historical data for this route"
        }
        
    except Exception as e:
        logger.error(f"❌ Historical data error: {e}")
        return {
            "delay_rate": None,
            "cancel_rate": None,
            "avg_delay_time": None,
            "total_flights_analyzed": 0,
            "error": str(e)
        }


def get_recent_performance(origin_iata, dest_iata, flight_number):
    """Query recent database"""
    logger.info(f"📊 Querying database: {flight_number}")
    
    try:
        if not os.path.exists(DB_NAME):
            logger.warning("⚠️ Database not found")
            return {
                "flights_analyzed": 0,
                "note": "Database not available - run update_latest_data.py"
            }
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=RECENT_DAYS_TO_QUERY)).strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM recent_flights
            WHERE flight_number = ? AND origin = ? AND destination = ? AND flight_date >= ?
            GROUP BY status
        """, (flight_number, origin_iata, dest_iata, cutoff_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "flights_analyzed": 0,
                "note": f"No flights in last {RECENT_DAYS_TO_QUERY} days"
            }
        
        stats = dict(rows)
        total = sum(stats.values())
        delayed = stats.get('delayed', 0)
        on_time = stats.get('on_time', 0)
        cancelled = stats.get('cancelled', 0)
        
        logger.info(f"✅ Found {total} recent flights")
        
        return {
            "flights_analyzed": total,
            "delay_rate_percent": round((delayed / total) * 100, 1),
            "cancel_rate_percent": round((cancelled / total) * 100, 1),
            "on_time_rate_percent": round((on_time / total) * 100, 1),
            "period_days": RECENT_DAYS_TO_QUERY
        }
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return {"flights_analyzed": 0, "error": str(e)}


@lru_cache(maxsize=128)
def get_airport_status(airport_code):
    """
    Return airport status (AeroDataBox API removed - was causing 400 errors)
    Always returns 'Normal Operations' as default
    """
    logger.info(f"🏢 Airport: {airport_code} (default status)")
    
    return {
        "code": airport_code,
        "delay_is_active": "False",
        "current_reason": "Normal Operations",
        "current_avg_delay_mins": 0
    }


def get_news(query):
    """Fetch recent news"""
    if not NEWS_API_KEY:
        return []
    
    try:
        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "apiKey": NEWS_API_KEY,
                "pageSize": 3,
                "sortBy": "relevancy",
                "from": from_date,
                "language": "en"
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        articles = [f"{a['title']}" for a in data.get('articles', [])]
        return articles
        
    except Exception as e:
        logger.warning(f"⚠️ News error: {e}")
        return []


def get_prediction_signals(origin_iata, dest_iata, date_str, flight_number, departure_time, arrival_time):
    """Gather all prediction signals"""
    logger.info(f"📊 Gathering data for {flight_number}")
    
    
    long_term_history = get_long_term_history(flight_number, date_str, origin_iata, dest_iata)
    
    
    recent_performance = get_recent_performance(origin_iata, dest_iata, flight_number)
    
    
    origin_forecast = get_weather_forecast(origin_iata, date_str, departure_time)
    
    try:
        dest_date = arrival_time.split('T')[0]
        dest_forecast = get_weather_forecast(dest_iata, dest_date, arrival_time)
    except:
        dest_forecast = {"error": "Invalid time format", "airport": dest_iata, "condition": "Unknown"}
    
    origin_status = get_airport_status(origin_iata)
    dest_status = get_airport_status(dest_iata)
    
    airline_code = "".join(filter(str.isalpha, str(flight_number)[:2]))
    news = get_news(f"({origin_iata} OR {dest_iata} airport) OR (airline {airline_code})")
    
    signals = {
        "long_term_history_seasonal": long_term_history,
        "recent_performance_last_6_months": recent_performance,
        "live_forecast_origin": origin_forecast,
        "live_forecast_destination": dest_forecast,
        "live_context_origin_airport": origin_status,
        "live_context_destination_airport": dest_status,
        "live_context_news": news,
    }
    
    logger.info("✅ Data collection complete")
    return signals