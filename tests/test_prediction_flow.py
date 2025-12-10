import sys
import os
import json
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Mock Flask app context if needed
# But we can call functions directly

from data_fetcher import get_prediction_signals
from llm_analyzer import predict_flight_outcome

logging.basicConfig(level=logging.INFO)

def test_prediction():
    print("Testing prediction flow...")
    
    # Mock data equivalent to what dashboard.html sends
    data = {
        "origin_iata": "DEL",
        "dest_iata": "BOM",
        "date": "2025-12-25",
        "flight_number": "6E123",
        "departure_time": "2025-12-25T10:00:00",
        "arrival_time": "2025-12-25T12:00:00"
    }
    
    try:
        print("1. Fetching signals...")
        signals = get_prediction_signals(
            data['origin_iata'], data['dest_iata'], data['date'],
            data['flight_number'], data['departure_time'], data['arrival_time']
        )
        print("Signals fetched successfully.")
        
        print("2. Generating prediction...")
        prediction = predict_flight_outcome(
            signals, data['origin_iata'], data['dest_iata'], data['date'],
            data['departure_time'], data['arrival_time'], data['flight_number']
        )
        print("Prediction generated successfully.")
        print(json.dumps(prediction, indent=2))
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
