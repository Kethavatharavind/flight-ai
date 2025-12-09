"""
Render Deployment Test Script
✅ Tests all components that caused errors on Render
✅ Run this locally BEFORE pushing to catch issues early
"""

import sys
import os

def print_header(title):
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)

def print_result(name, success, message=""):
    icon = "✅" if success else "❌"
    print(f"  {icon} {name}: {message}")
    return success

def test_imports():
    """Test all module imports"""
    print_header("TESTING IMPORTS")
    all_passed = True
    
    modules = [
        ('flask', 'Flask web framework'),
        ('sklearn', 'Scikit-learn ML'),
        ('xgboost', 'XGBoost ML'),
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical computing'),
        ('requests', 'HTTP requests'),
        ('supabase', 'Supabase client'),
    ]
    
    for module, desc in modules:
        try:
            __import__(module)
            print_result(desc, True, "imported")
        except ImportError as e:
            print_result(desc, False, str(e))
            all_passed = False
    
    # Optional modules
    try:
        import torch
        print_result("PyTorch (optional)", True, "imported")
    except ImportError:
        print_result("PyTorch (optional)", True, "not installed (OK)")
    
    return all_passed

def test_sklearn_version():
    """Test sklearn version matches Render"""
    print_header("TESTING SKLEARN VERSION")
    
    import sklearn
    local_version = sklearn.__version__
    render_version = "1.7.2"  # Render installs latest
    
    print(f"  Local version:  {local_version}")
    print(f"  Render version: {render_version}")
    
    if local_version == render_version:
        print_result("Version match", True, "PERFECT!")
        return True
    else:
        print_result("Version mismatch", False, f"Local {local_version} != Render {render_version}")
        print("  💡 Fix: pip install scikit-learn==1.7.2")
        return False

def test_ml_model():
    """Test ML model loading and prediction"""
    print_header("TESTING ML MODEL")
    
    try:
        import ml_model
        model = ml_model.get_ml_model()
        
        if not model.is_trained:
            print_result("Model trained", False, "Model not trained!")
            return False
        
        print_result("Model loaded", True, f"trained={model.is_trained}")
        
        # Test prediction
        result = model.predict_delay_probability(
            origin='DEL',
            destination='BOM', 
            airline_code='6E',
            departure_time='12:00',
            flight_date='2025-12-15'
        )
        
        prob = result.get('probability_delay', -1)
        if prob >= 0:
            print_result("Prediction works", True, f"{prob}% delay probability")
            return True
        else:
            print_result("Prediction failed", False, str(result))
            return False
            
    except Exception as e:
        print_result("ML Model", False, str(e))
        # Check for specific error
        if "monotonic_cst" in str(e):
            print("  💡 This is the VERSION MISMATCH error!")
            print("  💡 Fix: del delay_model.pkl && python ml_model.py")
        return False

def test_rl_agent():
    """Test RL agent"""
    print_header("TESTING RL AGENT")
    
    try:
        import rl_agent
        agent = rl_agent.FlightPredictionRLAgent()
        
        print_result("RL Agent created", True, f"States: {len(agent.q_table)}")
        
        # Test adjustment
        test_signals = {
            'live_forecast_origin': {'condition': 'Rain'},
            'live_forecast_destination': {'condition': 'Clear'},
        }
        
        adjusted, info = agent.adjust_prediction(50, test_signals)
        print_result("Prediction adjustment", True, f"50% → {adjusted}%")
        return True
        
    except Exception as e:
        print_result("RL Agent", False, str(e))
        return False

def test_supabase():
    """Test Supabase connection"""
    print_header("TESTING SUPABASE")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        
        if not url or not key:
            print_result("Supabase credentials", False, "Missing URL or KEY in .env")
            return False
        
        print_result("Supabase credentials", True, "Found in .env")
        
        import supabase_client
        if supabase_client.is_cloud_enabled():
            print_result("Supabase connection", True, "Connected!")
            return True
        else:
            print_result("Supabase connection", False, "Not connected")
            return False
            
    except Exception as e:
        print_result("Supabase", False, str(e))
        return False

def test_gemini():
    """Test Gemini API and available models"""
    print_header("TESTING GEMINI API & MODELS")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        key1 = os.getenv('GEMINI_API_KEY')
        key2 = os.getenv('GEMINI_API_KEY_BACKUP')
        
        if not key1:
            print_result("Gemini API key", False, "Missing GEMINI_API_KEY in .env")
            return False
        
        print_result("Primary key", True, f"{key1[:10]}...{key1[-4:]}")
        if key2:
            print_result("Backup key", True, f"{key2[:10]}...{key2[-4:]}")
        
        # Try to initialize
        import google.generativeai as genai
        genai.configure(api_key=key1)
        print_result("Gemini SDK", True, "Configured")
        
        # Test each model
        import llm_analyzer
        print(f"\n  📋 Configured models: {llm_analyzer.MODEL_NAMES}")
        
        # Try to actually generate content with first model
        working_model = None
        for model_name in llm_analyzer.MODEL_NAMES:
            try:
                test_model = genai.GenerativeModel(model_name)
                response = test_model.generate_content("Say 'OK' in one word.")
                if response and response.text:
                    print_result(f"Model {model_name.split('/')[-1]}", True, "Works!")
                    working_model = model_name
                    break
            except Exception as e:
                error_msg = str(e)[:50]
                if "quota" in error_msg.lower() or "429" in error_msg:
                    print_result(f"Model {model_name.split('/')[-1]}", False, "Quota exceeded")
                else:
                    print_result(f"Model {model_name.split('/')[-1]}", False, error_msg)
        
        if working_model:
            print(f"\n  ✅ Working model found: {working_model}")
            return True
        else:
            print("\n  ⚠️ No working models found. Using fallback summary.")
            return True  # Still pass - app works without Gemini
        
    except Exception as e:
        print_result("Gemini", False, str(e))
        return False

def test_data_fetcher():
    """Test data fetcher module"""
    print_header("TESTING DATA FETCHER")
    
    try:
        import data_fetcher
        print_result("Data fetcher import", True, "imported")
        return True
    except Exception as e:
        print_result("Data fetcher", False, str(e))
        return False

def test_app_startup():
    """Test Flask app can start"""
    print_header("TESTING FLASK APP")
    
    try:
        import app
        print_result("App import", True, "Flask app loaded")
        return True
    except Exception as e:
        print_result("Flask app", False, str(e))
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 RENDER DEPLOYMENT PRE-CHECK")
    print("=" * 60)
    print("This test simulates Render startup to catch errors early.\n")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Sklearn Version", test_sklearn_version()))
    results.append(("ML Model", test_ml_model()))
    results.append(("RL Agent", test_rl_agent()))
    results.append(("Supabase", test_supabase()))
    results.append(("Gemini API", test_gemini()))
    results.append(("Data Fetcher", test_data_fetcher()))
    results.append(("Flask App", test_app_startup()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        icon = "✅" if result else "❌"
        print(f"  {icon} {name}")
    
    print(f"\n  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Safe to deploy to Render.")
    else:
        print("\n⚠️ Some tests failed. Fix issues before deploying.")
        sys.exit(1)
