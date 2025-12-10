"""
Enhanced Flask Application with proper error handling and logging
✅ FIXED: Better shutdown handling - Ctrl+C works immediately, RL saves on exit
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import sys
from datetime import datetime, timedelta
import numpy as np

# Add src folder to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules from src/
import data_fetcher
import llm_analyzer
import rl_agent
import atexit
import signal
import logging
from functools import wraps
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production-12345')


chat_context_storage = {}


request_timestamps = {}
RATE_LIMIT_WINDOW = 60  
RATE_LIMIT_MAX_REQUESTS = 50


def rate_limit(f):
    """Simple rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time.time()
        
        if client_ip not in request_timestamps:
            request_timestamps[client_ip] = []
        
        
        request_timestamps[client_ip] = [
            ts for ts in request_timestamps[client_ip]
            if current_time - ts < RATE_LIMIT_WINDOW
        ]
        
        
        if len(request_timestamps[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
            return jsonify({
                "error": "Rate limit exceeded. Please wait a moment.",
                "retry_after": RATE_LIMIT_WINDOW
            }), 429
        
        request_timestamps[client_ip].append(current_time)
        return f(*args, **kwargs)
    
    return decorated_function



AVAILABLE_ROUTES = []
try:
    routes_path = os.path.join(os.path.dirname(__file__), 'config', 'major_routes.json')
    with open(routes_path, 'r') as f:
        AVAILABLE_ROUTES = json.load(f)
    logger.info(f"Loaded {len(AVAILABLE_ROUTES)} major routes (India)")
except FileNotFoundError:
    logger.warning("major_routes.json not found in config/")
except Exception as e:
    logger.error(f"Error loading routes: {e}")


@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Serve dashboard page"""
    return send_from_directory('templates', 'dashboard.html')


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/get_available_routes', methods=['GET'])
@rate_limit
def get_available_routes():
    """Return available routes"""
    if not AVAILABLE_ROUTES:
        return jsonify({
            "error": "No routes available",
            "message": "Please run process_history.py to generate routes"
        }), 500
    
    return jsonify(AVAILABLE_ROUTES)


@app.route('/get_min_date', methods=['GET'])
def get_min_date():
    """Return minimum valid date"""
    min_date = (datetime.now() + timedelta(days=8)).strftime('%Y-%m-%d')
    max_date = (datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')
    return jsonify({
        "min_date": min_date,
        "max_date": max_date
    })


@app.route('/find_flights', methods=['POST'])
@rate_limit
def find_flights():
    """Find available flights"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        origin_iata = data.get('origin_iata', '').strip().upper()
        dest_iata = data.get('dest_iata', '').strip().upper()
        date = data.get('date', '').strip()
        
        
        if not all([origin_iata, dest_iata, date]):
            return jsonify({"error": "Missing required fields"}), 400
        
        if len(origin_iata) != 3 or len(dest_iata) != 3:
            return jsonify({"error": "Invalid airport codes"}), 400
        
        if origin_iata == dest_iata:
            return jsonify({"error": "Origin and destination must be different"}), 400
        
        
        try:
            flight_date = datetime.strptime(date, '%Y-%m-%d')
            days_ahead = (flight_date - datetime.now()).days
            
            if days_ahead < 8:
                min_date = (datetime.now() + timedelta(days=8)).strftime('%Y-%m-%d')
                return jsonify({
                    "error": f"Date must be at least 8 days ahead. Minimum: {min_date}"
                }), 400
            
            if days_ahead > 180:
                max_date = (datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')
                return jsonify({
                    "error": f"Date too far in future. Maximum: {max_date}"
                }), 400
                
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        
        logger.info(f"Flight search: {origin_iata} → {dest_iata} on {date}")
        flight_data = data_fetcher.get_flights_by_route(origin_iata, dest_iata, date)
        
        if "error" in flight_data:
            return jsonify({"error": flight_data["error"]}), 400
        
        return jsonify(flight_data)
        
    except Exception as e:
        logger.error(f"❌ Error in /find_flights: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


def clean_for_json(obj):
    """Convert numpy types to JSON-safe types"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_for_json(v) for v in obj)
    elif isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@app.route('/predict_status', methods=['POST'])
@rate_limit
def predict_status():
    """Generate flight delay prediction"""
    global chat_context_storage
    
    try:
        data = request.json
        
        logger.info(f"Received prediction request: {data}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        
        origin_iata = data.get('origin_iata', '').strip().upper()
        dest_iata = data.get('dest_iata', '').strip().upper()
        date = data.get('date', '').strip()
        flight_number = data.get('flight_number', '').strip().upper()
        departure_time = data.get('departure_time', '').strip()
        arrival_time = data.get('arrival_time', '').strip()
        
        
        required = {
            'origin_iata': origin_iata,
            'dest_iata': dest_iata,
            'date': date,
            'flight_number': flight_number,
            'departure_time': departure_time,
            'arrival_time': arrival_time
        }
        
        missing = [k for k, v in required.items() if not v]
        if missing:
            logger.error(f"❌ Missing fields: {missing}")
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing)}"
            }), 400
        
        
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Invalid date format"}), 400
        
        logger.info(f"Prediction request: {flight_number} ({origin_iata}→{dest_iata}) on {date}")
        
        
        signals = data_fetcher.get_prediction_signals(
            origin_iata, dest_iata, date, flight_number, departure_time, arrival_time
        )
        
        logger.info(" Signals gathered, generating prediction...")
        
        
        # llm_response is already a dict from the analyzer
        llm_data = llm_analyzer.predict_flight_outcome(
            signals, origin_iata, dest_iata, date, departure_time, arrival_time, flight_number
        )
        
        logger.info(f" LLM response received (keys: {list(llm_data.keys())})")

        
        
        response_data = {
            "signals": clean_for_json(signals),
            "prediction": clean_for_json(llm_data)
        }
        
        
        chat_context_storage = signals.copy()
        chat_context_storage['prediction_probabilities'] = llm_data
        chat_context_storage['flight_details'] = {
            'origin': origin_iata,
            'destination': dest_iata,
            'date': date,
            'flight_number': flight_number,
            'departure_time': departure_time,
            'arrival_time': arrival_time
        }
        
        logger.info(f" Prediction complete: {llm_data.get('probability_delay', 0)}% delay probability")
        
        return jsonify(response_data)
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error: {e}")
        logger.error(f"❌ Raw LLM response: {llm_response_str[:500]}")
        return jsonify({
            "error": "Failed to parse prediction response",
            "signals": {},
            "prediction": {
                "justification": "Internal error parsing prediction. Check server logs for details.",
                "user_friendly_summary": "We encountered an error generating your prediction. Please try again.",
                "probability_delay": 0,
                "probability_cancel": 0,
                "confidence_level": "LOW"
            }
        }), 500
        
    except Exception as e:
        logger.error(f"❌ Error in /predict_status: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "signals": {},
            "prediction": {
                "justification": f"Server error: {str(e)}",
                "user_friendly_summary": f"We encountered an error: {str(e)}. Please try again.",
                "probability_delay": 0,
                "probability_cancel": 0,
                "confidence_level": "LOW"
            }
        }), 500


@app.route('/chat', methods=['POST'])
@rate_limit
def chat():
    """Handle chatbot questions"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        if not chat_context_storage:
            return jsonify({
                "answer": "Please get a flight prediction first so I have context to answer your questions."
            })
        
        logger.info(f"💬 Chat question: {user_message[:50]}...")
        
        llm_answer = llm_analyzer.get_chat_response(user_message, chat_context_storage)
        
        return jsonify({"answer": llm_answer})
        
    except Exception as e:
        logger.error(f"❌ Error in /chat: {e}", exc_info=True)
        return jsonify({"error": f"Chat error: {str(e)}"}), 500


@app.route('/rl_stats', methods=['GET'])
def rl_stats():
    """Get RL agent statistics"""
    try:
        agent = rl_agent.get_rl_agent()
        stats = agent.get_stats()
        return jsonify(clean_for_json(stats))
    except Exception as e:
        logger.error(f"❌ Error getting RL stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "routes_loaded": len(AVAILABLE_ROUTES),
        "rl_agent_states": len(rl_agent.get_rl_agent().q_table)
    })


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500



def cleanup():
    """Cleanup on shutdown - saves RL agent state"""
    logger.info("\n" + "="*60)
    logger.info("Server shutting down - saving data...")
    logger.info("="*60)
    try:
        rl_agent.save_agent_state()
        logger.info("RL agent state saved successfully")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
    finally:
        logger.info("Goodbye!")



def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n")  
    logger.info("⚡ Received interrupt signal (Ctrl+C)")
    cleanup()
    sys.exit(0)



atexit.register(cleanup)


signal.signal(signal.SIGINT, signal_handler)  
try:
    signal.signal(signal.SIGTERM, signal_handler)  
except AttributeError:
    pass  


if __name__ == '__main__':
    print("\n" + "="*80)
    print("✈️  FLIGHT DELAY PREDICTION SYSTEM v2.0")
    print("="*80)
    print(f"🌐 Server URL: http://127.0.0.1:5000")
    print(f"🏥 Health Check: http://127.0.0.1:5000/health")
    print(f"📊 RL Stats: http://127.0.0.1:5000/rl_stats")
    print("\n🧠 Prediction System:")
    print("   ✅ ML Model (XGBoost + Random Forest) - 75%")
    print("   ✅ Weather Risk (Open-Meteo) - 25%")
    print("   ✅ RL Agent (Q-Learning) - Adjusts predictions")
    print("\n📡 Data Sources:")
    print("   ✅ Flight Database (india_data.db)")
    print("   ✅ Live Weather Forecasts")
    print("   ✅ Flight Schedules (AviationStack)")
    print("   ✅ LLM Summary (Gemini)")
    print("\n⚠️  REQUIREMENTS:")
    print(f"   • Predictions: 8-180 days in advance")
    print(f"   • Valid date range: {(datetime.now() + timedelta(days=8)).strftime('%Y-%m-%d')}")
    print(f"                    to {(datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')}")
    print("\n🔧 System Checks:")
    
    
    warnings = []
    if os.path.exists('india_data.db'):
        print("   ✅ Flight database ready")
    else:
        warnings.append("   ⚠️  india_data.db missing (run update_latest_data.py)")
        
    if os.path.exists('delay_model.pkl'):
        print("   ✅ ML model loaded")
    else:
        warnings.append("   ⚠️  ML model missing (run ml_model.py)")
        
    print(f"   ✅ {len(AVAILABLE_ROUTES)} routes available")
    
    
    api_keys_ok = True
    required_keys = ['AVIATIONSTACK_API_KEY', 'GEMINI_API_KEY']
    for key in required_keys:
        if not os.getenv(key):
            warnings.append(f"   ⚠️  {key} missing in .env")
            api_keys_ok = False
    
    if api_keys_ok:
        print("   ✅ API keys configured")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(warning)
    
    print("\n" + "="*80)
    print("🚀 Starting server...")
    print("⚠️  Press Ctrl+C to stop (RL agent will save automatically)")
    print("="*80 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        
        signal_handler(None, None)