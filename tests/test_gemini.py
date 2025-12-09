"""
Gemini API Model Tester
Tests each Gemini model with time delays to check availability
"""

import os
import sys
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()

# Get API keys
PRIMARY_KEY = os.getenv("GEMINI_API_KEY")
BACKUP_KEY = os.getenv("GEMINI_API_KEY_BACKUP")

# Models to test
MODELS_TO_TEST = [
    "models/gemini-2.0-flash-exp",
    "models/gemini-2.0-flash-thinking-exp-1219",
    "models/gemini-exp-1206",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.5-pro",
]

def test_model(model_name, api_key, key_label):
    """Test a single model with given API key"""
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 100,
            }
        )
        
        # Try to generate content
        response = model.generate_content("Say 'Hello' in one word.")
        
        if response and response.text:
            print(f"   ✅ {model_name.split('/')[-1]}: WORKING")
            print(f"      Response: {response.text.strip()}")
            return True
        else:
            print(f"   ❌ {model_name.split('/')[-1]}: No response")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"   ⚠️ {model_name.split('/')[-1]}: QUOTA EXCEEDED")
        elif "404" in error_msg or "not found" in error_msg.lower():
            print(f"   ❌ {model_name.split('/')[-1]}: Model not found")
        else:
            print(f"   ❌ {model_name.split('/')[-1]}: {error_msg[:60]}")
        return False

def main():
    print("\n" + "=" * 70)
    print("🧪 GEMINI API MODEL TESTER")
    print("=" * 70)
    
    # Check API keys
    if not PRIMARY_KEY:
        print("\n❌ ERROR: GEMINI_API_KEY not found in .env")
        return
    
    print(f"\n🔑 Primary Key: {PRIMARY_KEY[:10]}...{PRIMARY_KEY[-4:]}")
    if BACKUP_KEY:
        print(f"🔑 Backup Key:  {BACKUP_KEY[:10]}...{BACKUP_KEY[-4:]}")
    else:
        print("⚠️ No backup key found")
    
    # Test with primary key
    print("\n" + "=" * 70)
    print("📋 TESTING WITH PRIMARY KEY")
    print("=" * 70)
    
    working_models = []
    
    for i, model_name in enumerate(MODELS_TO_TEST, 1):
        print(f"\n[{i}/{len(MODELS_TO_TEST)}] Testing: {model_name.split('/')[-1]}")
        
        if test_model(model_name, PRIMARY_KEY, "PRIMARY"):
            working_models.append((model_name, "PRIMARY"))
        
        # Wait 2 seconds between tests to avoid rate limiting
        if i < len(MODELS_TO_TEST):
            print("   ⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    # Test with backup key if available
    if BACKUP_KEY and not working_models:
        print("\n" + "=" * 70)
        print("📋 TESTING WITH BACKUP KEY")
        print("=" * 70)
        
        for i, model_name in enumerate(MODELS_TO_TEST, 1):
            print(f"\n[{i}/{len(MODELS_TO_TEST)}] Testing: {model_name.split('/')[-1]}")
            
            if test_model(model_name, BACKUP_KEY, "BACKUP"):
                working_models.append((model_name, "BACKUP"))
            
            # Wait 2 seconds between tests
            if i < len(MODELS_TO_TEST):
                print("   ⏳ Waiting 2 seconds...")
                time.sleep(2)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if working_models:
        print(f"\n✅ Found {len(working_models)} working model(s):\n")
        for model_name, key_label in working_models:
            print(f"   • {model_name} ({key_label} key)")
        
        print("\n💡 RECOMMENDATION:")
        best_model = working_models[0][0]
        print(f"   Update llm_analyzer.py MODEL_NAMES to use: {best_model}")
    else:
        print("\n❌ No working models found!")
        print("\n💡 POSSIBLE SOLUTIONS:")
        print("   1. Wait for quota reset (midnight Pacific Time)")
        print("   2. Get a new API key from https://ai.google.dev")
        print("   3. Use backup key if available")
        print("   4. App will use fallback summaries (still works!)")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
