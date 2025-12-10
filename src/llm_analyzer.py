"""
Enhanced LLM Analyzer with ML + Weather + RL prediction formula
✅ 75% ML Model + 25% Weather Risk
✅ RL Agent adjustment based on learning
✅ Fixed: Recent trends properly integrated
✅ Fixed: Variable scope issues resolved
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
# Using DQN (Deep Q-Network) instead of Q-Learning for better generalization
try:
    from rl_agent_dqn import get_dqn_agent
    DQN_AVAILABLE = True
    dqn_agent = get_dqn_agent()
except ImportError:
    import rl_agent
    DQN_AVAILABLE = False
    dqn_agent = None
import logging

# Import ML model
try:
    import ml_model
    ML_MODEL_AVAILABLE = True
except ImportError:
    ML_MODEL_AVAILABLE = False

# Import prediction tracker for RL learning
try:
    import prediction_tracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found - using statistical predictions only")


if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

MODEL_NAMES = [
    "models/gemini-2.5-flash-lite",    # ✅ WORKING! Has quota
    "models/gemini-2.0-flash-exp",     # Fallback 1
    "models/gemini-exp-1206",          # Fallback 2
    "models/gemini-2.0-flash",         # Fallback 3
]

# API key rotation - tries primary, then backup if quota exceeded
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_BACKUP")
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]  # Remove None values

current_key_index = 0
model = None

def initialize_gemini():
    """Initialize Gemini with API key rotation"""
    global model, current_key_index
    
    for key_idx, api_key in enumerate(GEMINI_API_KEYS):
        if not api_key:
            continue
            
        genai.configure(api_key=api_key)
        
        for model_name in MODEL_NAMES:
            try:
                test_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                # Skip test call to save quota - just initialize the model
                model = test_model
                current_key_index = key_idx
                key_label = "PRIMARY" if key_idx == 0 else "BACKUP"
                logger.info(f"✅ Gemini Model: {model_name} (Key: {key_label})")
                return True
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.warning(f"⚠️ Quota exceeded for key {key_idx+1}, trying next...")
                    break  # Try next key
                logger.warning(f"Failed {model_name}: {str(e)[:60]}")
                continue
    
    logger.error("❌ All Gemini API keys exhausted!")
    return False

# Initialize on module load
if GEMINI_API_KEYS:
    initialize_gemini()




SYSTEM_PROMPT = """
You are an expert flight delay prediction analyst for India domestic flights.

**PREDICTION SYSTEM:**
Our system uses a 3-step approach to predict flight delays:

**STEP 1: ML Model (XGBoost + Random Forest)**
- Trained on India flight data (india_data.db)
- Uses: Route, airline, time of day, day of week, historical delay rates
- Weight: 75% of base prediction

**STEP 2: Weather Impact**
- Real-time weather from Open-Meteo API for both airports
- Weather risk scoring:
  • Clear: 5%
  • Cloudy: 10%
  • Rain/Shower: 25%
  • Fog/Mist: 30%
  • Storm/Thunder: 40%
  • Snow/Blizzard: 45%
- Weight: 25% of base prediction

**STEP 3: RL Agent Adjustment**
- Q-Learning agent that learns from prediction outcomes
- Adjusts prediction based on learned patterns
- Gets smarter over time as more predictions verified

**FINAL FORMULA:**
Base = 0.75 × ML_Prediction + 0.25 × Weather_Risk
Final = RL_Agent_Adjust(Base)

**NEWS IMPACT:**
- Check for flight-related news (strikes, closures, weather events)
- Warn users if serious news may cause delays/cancellations

**RESPONSE LIMITS:**
- delay_probability: 15-90% (integer)
- cancel_probability: 1-30% (integer)

**RESPONSE FORMAT (JSON ONLY — STRICTLY FOLLOW THIS):**
```json
{
  "probability_delay": <15-90>,
  "probability_cancel": <5-90>,
  "predicted_delay_range_mins": [<min>, <max>],

  "confidence_level": "HIGH|MEDIUM|LOW",
  "risk_factors": ["factor1", "factor2", ...],
  "justification": "Start with risk level emoji (🔴/🟡/🟢) then explain"
}
"""


def convert_to_native_types(obj):
    """Convert numpy types to native Python types"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def calculate_fallback_prediction(signals):
    """
    ✅ CORRECTED: Proper weighted formula implementation
    """
    hist = signals.get('long_term_history_seasonal', {})
    recent = signals.get('recent_performance_last_6_months', {})
    weather_origin = signals.get('live_forecast_origin', {})
    weather_dest = signals.get('live_forecast_destination', {})
    airport_origin = signals.get('live_context_origin_airport', {})
    airport_dest = signals.get('live_context_destination_airport', {})
    
    risk_factors = []
    
    
    
    
    has_historical = hist.get('delay_rate') is not None
    
    if has_historical:
        H = float(hist['delay_rate'])
        base_cancel = float(hist.get('cancel_rate', 0))
        avg_delay_mins = float(hist.get('avg_delay_time', 30))
        flights_count = int(hist.get('total_flights_analyzed', 0))
        confidence = min(95, 50 + flights_count)
        
        if H > 40:
            risk_factors.append(f"📊 High Historical Delays: {H:.1f}% over {flights_count} flights")
        elif H > 25:
            risk_factors.append(f"📊 Moderate Historical Delays: {H:.1f}% delay rate")
        else:
            risk_factors.append(f"✅ Good Historical Performance: {H:.1f}% delays")
    else:
        H = 22.0  
        base_cancel = 2.5
        avg_delay_mins = 30.0
        flights_count = 0
        confidence = 40
        risk_factors.append("⚠️ No Historical Data - Using industry baseline (22%)")
    
    
    
    
    if recent.get('flights_analyzed', 0) >= 5:
        R = float(recent.get('delay_rate_percent', H))
        recent_count = int(recent['flights_analyzed'])
        recent_cancel = float(recent.get('cancel_rate_percent', base_cancel))
        
        trend_diff = R - H
        
        if R > 40:
            risk_factors.append(f"📈 High Recent Delays: {R:.1f}% over {recent_count} flights")
        elif R > 25:
            risk_factors.append(f"📊 Moderate Recent Performance: {R:.1f}% delay rate")
        else:
            risk_factors.append(f"✅ Good Recent Performance: {R:.1f}% delays")
        
        
        if abs(trend_diff) > 10:
            if trend_diff > 0:
                risk_factors.append(f"📈 Worsening Trend: Up {trend_diff:.1f}% from historical avg")
            else:
                risk_factors.append(f"📉 Improving Trend: Down {abs(trend_diff):.1f}% from historical avg")
    else:
        R = H  
        recent_cancel = base_cancel
        risk_factors.append("ℹ️ Limited Recent Data - Using historical trends")
    
    
    
    
    severe_weather = ['thunderstorm', 'heavy rain', 'heavy snow', 'blizzard', 'hail', 'hurricane', 'tornado']
    moderate_weather = ['rain', 'snow', 'fog', 'drizzle', 'sleet', 'storm']
    
    origin_cond = weather_origin.get('condition', 'clear').lower()
    dest_cond = weather_dest.get('condition', 'clear').lower()
    origin_city = weather_origin.get('city', 'origin')
    dest_city = weather_dest.get('city', 'destination')
    
    W = 0.0  
    weather_cancel_risk = 0.0
    
    
    if any(w in origin_cond for w in severe_weather):
        W += 60
        weather_cancel_risk += 8
        risk_factors.append(f"⛈️ SEVERE WEATHER at {origin_city}: {origin_cond.title()}")
    elif any(w in origin_cond for w in moderate_weather):
        W += 25
        weather_cancel_risk += 3
        risk_factors.append(f"🌧️ Moderate Weather at {origin_city}: {origin_cond.title()}")
    elif 'clear' in origin_cond or 'sunny' in origin_cond:
        W += 5
    else:
        W += 10  
    
    
    if any(w in dest_cond for w in severe_weather):
        W += 50
        weather_cancel_risk += 7
        risk_factors.append(f"⛈️ SEVERE WEATHER at {dest_city}: {dest_cond.title()}")
    elif any(w in dest_cond for w in moderate_weather):
        W += 20
        weather_cancel_risk += 2
        risk_factors.append(f"🌧️ Moderate Weather at {dest_city}: {dest_cond.title()}")
    elif 'clear' in dest_cond or 'sunny' in dest_cond:
        W += 5
    else:
        W += 10
    
    
    W = W / 2.0
    
    if W <= 10:
        risk_factors.append(f"☀️ Clear Conditions: {origin_cond.title()} → {dest_cond.title()}")
    
    
    
    
    A = 0.0  
    airport_cancel_risk = 0.0
    
    origin_delayed = airport_origin.get('delay_is_active') == 'True'
    dest_delayed = airport_dest.get('delay_is_active') == 'True'
    
    if origin_delayed:
        avg_delay = int(airport_origin.get('current_avg_delay_mins', 15))
        reason = airport_origin.get('current_reason', 'Operational delays')
        
        if avg_delay > 45:
            A += 70
            airport_cancel_risk += 6
        elif avg_delay > 30:
            A += 50
            airport_cancel_risk += 4
        elif avg_delay > 20:
            A += 30
            airport_cancel_risk += 2
        elif avg_delay > 10:
            A += 15
            airport_cancel_risk += 1
        else:
            A += 5
        
        risk_factors.append(f"🔴 {origin_city} Airport Delays: {reason} (~{avg_delay} min)")
    else:
        A += 5  
    
    if dest_delayed:
        avg_delay = int(airport_dest.get('current_avg_delay_mins', 15))
        reason = airport_dest.get('current_reason', 'Operational delays')
        
        if avg_delay > 45:
            A += 60
            airport_cancel_risk += 5
        elif avg_delay > 30:
            A += 45
            airport_cancel_risk += 3
        elif avg_delay > 20:
            A += 25
            airport_cancel_risk += 2
        elif avg_delay > 10:
            A += 12
            airport_cancel_risk += 1
        else:
            A += 5
        
        risk_factors.append(f"🔴 {dest_city} Airport Delays: {reason} (~{avg_delay} min)")
    else:
        A += 5
    
    
    A = A / 2.0
    
    if not origin_delayed and not dest_delayed:
        risk_factors.append(f"✅ Both Airports Operating Normally")
    
    
    
    
    delay_prob = (0.25 * H) + (0.40 * R) + (0.20 * W) + (0.15 * A)
    
    
    cancel_prob = base_cancel + weather_cancel_risk + airport_cancel_risk
    
    
    if recent.get('flights_analyzed', 0) >= 5:
        cancel_prob = (0.6 * recent_cancel) + (0.4 * cancel_prob)
    
    
    
    
    delay_prob = max(15, min(90, delay_prob))
    cancel_prob = min(cancel_prob, delay_prob * 0.4)  
    cancel_prob = max(1, min(30, cancel_prob))
    
    delay_prob = int(round(delay_prob))
    cancel_prob = int(round(cancel_prob))
    
    
    if delay_prob > 70:
        delay_min = max(20, int(avg_delay_mins * 0.8))
        delay_max = int(avg_delay_mins * 2.5)
    elif delay_prob > 50:
        delay_min = max(15, int(avg_delay_mins * 0.7))
        delay_max = int(avg_delay_mins * 2.0)
    else:
        delay_min = max(10, int(avg_delay_mins * 0.5))
        delay_max = int(avg_delay_mins * 1.5)
    
    
    risk_level = "🔴 HIGH RISK" if delay_prob > 60 else "🟡 MODERATE RISK" if delay_prob > 35 else "🟢 LOW RISK"
    
    
    justification = f"{risk_level} — {delay_prob}% delay probability\n\n"
    justification += "📐 **Statistical Fallback (ML Model Unavailable):**\n"
    justification += f"• Historical: {H:.1f}%\n"
    justification += f"• Recent: {R:.1f}%\n"
    justification += f"• Weather Risk: {W:.1f}%\n"
    justification += f"• Airport Status: {A:.1f}%\n"
    justification += f"• **Total: {delay_prob}%**\n\n"
    justification += "🔍 **Key Factors:**\n"
    justification += "\n".join(f"• {factor}" for factor in risk_factors)
    
    confidence_level = "HIGH" if confidence > 70 else "MEDIUM" if confidence > 50 else "LOW"
    
    result = {
        "probability_delay": delay_prob,
        "probability_cancel": cancel_prob,
        "predicted_delay_range_mins": [delay_min, delay_max],
        "justification": justification,
        "confidence_level": confidence_level,
        "risk_factors": risk_factors
    }
    
    return convert_to_native_types(result)


def generate_user_friendly_summary_fallback(prediction_data, signals):
    """
    ✅ Non-LLM fallback summary generation
    """
    delay_prob = prediction_data.get('probability_delay', 0)
    cancel_prob = prediction_data.get('probability_cancel', 0)
    risk_factors = prediction_data.get('risk_factors', [])
    delay_range = prediction_data.get('predicted_delay_range_mins', [15, 45])
    
    
    weather_issues = []
    airport_issues = []
    historical_issues = []
    
    for factor in risk_factors:
        factor_lower = factor.lower()
        if any(keyword in factor_lower for keyword in ['weather', '⛈️', '🌧️', '☀️', 'rain', 'snow', 'storm', 'fog']):
            weather_issues.append(factor)
        elif any(keyword in factor_lower for keyword in ['airport', 'delays', '🔴', 'operational']):
            airport_issues.append(factor)
        elif any(keyword in factor_lower for keyword in ['historical', 'track record', '📊', 'history', 'delay rate']):
            historical_issues.append(factor)
    
    risk_emoji = "🟢" if delay_prob < 35 else "🟡" if delay_prob < 60 else "🔴"
    risk_level = "LOW RISK" if delay_prob < 35 else "MODERATE RISK" if delay_prob < 60 else "HIGH RISK"
    
    summary = f"{risk_emoji} {risk_level}\n\n"
    
    if delay_prob < 35:
        
        summary += f"Good news! Your flight has only a {delay_prob}% chance of delays. "
        
        if historical_issues and 'good' in historical_issues[0].lower():
            summary += "This route has an excellent track record with minimal delays. "
        else:
            summary += "Historical data suggests reliable performance. "
        
        if not weather_issues or any(word in str(weather_issues).lower() for word in ['clear', 'good', '☀️']):
            summary += "Weather conditions are favorable at both airports. "
        else:
            summary += f"Weather: {weather_issues[0].replace('•', '').replace('✅', '').strip()} "
        
        if not airport_issues:
            summary += "Both airports are operating normally."
        
        summary += f"\n\n⏱️ **Expected Delay (if any):** {delay_range[0]}-{delay_range[1]} minutes"
        summary += "\n\n💡 **Recommendation:** Arrive at the standard check-in time. Your flight should depart on schedule."
    
    elif delay_prob < 60:
        
        summary += f"Your flight has a {delay_prob}% chance of delays. Here's what's affecting it:\n\n"
        
        reasons_found = False
        
        if weather_issues:
            summary += f"🌤️ **Weather Factor:** {weather_issues[0].replace('•', '').strip()}\n"
            reasons_found = True
        
        if airport_issues:
            summary += f"🏢 **Airport Status:** {airport_issues[0].replace('•', '').strip()}\n"
            reasons_found = True
        
        if historical_issues:
            summary += f"📊 **Historical Pattern:** {historical_issues[0].replace('•', '').strip()}\n"
            reasons_found = True
        
        if not reasons_found:
            summary += "• Operational factors suggest potential disruptions\n"
            summary += "• Recent performance shows some variability\n"
        
        summary += f"\n⏱️ **Expected Delay (if occurs):** {delay_range[0]}-{delay_range[1]} minutes"
        summary += f"\n\n💡 **Recommendation:** Arrive 30-45 minutes earlier than usual. Monitor your flight status and weather updates. Pack snacks and entertainment just in case."
    
    else:
        
        summary += f"⚠️ High delay probability detected ({delay_prob}%). Here are the key concerns:\n\n"
        
        if weather_issues:
            summary += f"🌩️ **Weather Alert:**\n"
            for issue in weather_issues[:2]:
                summary += f"   • {issue.replace('•', '').replace('⛈️', '').replace('🌧️', '').strip()}\n"
        
        if airport_issues:
            summary += f"\n✈️ **Airport Delays:**\n"
            for issue in airport_issues[:2]:
                summary += f"   • {issue.replace('•', '').replace('🔴', '').strip()}\n"
        
        if historical_issues:
            summary += f"\n📉 **Historical Data:**\n"
            summary += f"   • {historical_issues[0].replace('•', '').strip()}\n"
        
        if not (weather_issues or airport_issues or historical_issues):
            summary += "⚠️ Multiple severe factors detected:\n"
            for i, factor in enumerate(risk_factors[:3], 1):
                summary += f"   {i}. {factor.replace('•', '').strip()}\n"
        
        if cancel_prob > 10:
            summary += f"\n⚠️ Cancellation risk is also elevated at {cancel_prob}%.\n"
        
        summary += f"\n⏱️ **Expected Delay (if occurs):** {delay_range[0]}-{delay_range[1]} minutes"
        summary += f"\n\n🚨 **Strong Recommendation:** Allow extra time (arrive 1+ hour early). Check flight status every 2-3 hours. Contact your airline proactively. Consider travel insurance if not already purchased."
    
    return summary


def generate_user_friendly_summary(prediction_data, signals):
    """
    Generate a comprehensive user-friendly summary with:
    - Full prediction breakdown (ML%, Weather%, RL%)
    - 3-day news check (today, yesterday, day before)
    - Weather explanation for both airports
    - Any flight-related issues or alerts
    """
    delay_prob = prediction_data.get('probability_delay', 0)
    cancel_prob = prediction_data.get('probability_cancel', 0)
    risk_factors = prediction_data.get('risk_factors', [])
    delay_range = prediction_data.get('predicted_delay_range_mins', [15, 45])
    breakdown = prediction_data.get('prediction_breakdown', {})
    
    # Get weather info
    weather_origin = signals.get('live_forecast_origin', {})
    weather_dest = signals.get('live_forecast_destination', {})
    
    # Get news (last 3 days)
    news_data = signals.get('live_context_news', {})
    news_articles = news_data.get('articles', []) if isinstance(news_data, dict) else []
    
    # Filter for flight/aviation related news
    flight_related_news = []
    if news_articles:
        keywords = ['flight', 'airport', 'airline', 'delay', 'cancel', 'strike', 'weather', 'storm', 'fog', 'closure']
        for article in news_articles[:10]:
            title = article.get('title', '').lower()
            desc = article.get('description', '').lower() if article.get('description') else ''
            if any(kw in title or kw in desc for kw in keywords):
                flight_related_news.append(article)
    
    if model is None:
        logger.info("No LLM available - using fallback summary")
        return generate_enhanced_fallback_summary(prediction_data, signals, breakdown, flight_related_news)
    
    try:
        # Build news section
        news_text = ""
        if flight_related_news:
            news_text = "\\n\\n**RECENT NEWS (Last 3 days) - IMPORTANT:**\\n"
            for i, article in enumerate(flight_related_news[:3], 1):
                news_text += f"{i}. {article.get('title', 'No title')}\\n"
        else:
            news_text = "\\n\\n**RECENT NEWS:** No significant flight-related news in the last 3 days."
        
        # Build breakdown section
        breakdown_text = ""
        if breakdown:
            ml_pred = breakdown.get('ml_prediction')
            breakdown_text = f"""
**PREDICTION BREAKDOWN:**
- ML Model Prediction: {ml_pred}% ({breakdown.get('ml_confidence', 'N/A')}) - trained on real India flight data
- Weather Risk Score: {breakdown.get('weather_risk', 0)}%
- Combined (75% ML + 25% Weather): {breakdown.get('combined_base', 0)}%
- RL Agent Adjustment: {breakdown.get('rl_adjustment', 0):+.0f}%
- **FINAL PREDICTION: {breakdown.get('final_prediction', delay_prob)}%**
"""
        
        factors_text = "\\n".join(f"- {factor}" for factor in risk_factors[:5])
        
        summary_prompt = f"""
You are a friendly flight assistant helping a traveler understand their flight's delay prediction.

**FLIGHT INFO:**
- Delay Probability: {delay_prob}%
- Risk Level: {'🔴 HIGH' if delay_prob > 60 else '🟡 MODERATE' if delay_prob > 35 else '🟢 LOW'}
- Expected Delay (if occurs): {delay_range[0]}-{delay_range[1]} minutes

**WEATHER:**
📍 {signals.get('live_forecast_origin', {}).get('city', 'Origin')}: {weather_origin.get('condition', 'Unknown')}, {weather_origin.get('temperature', 'N/A')}°C
📍 {signals.get('live_forecast_destination', {}).get('city', 'Destination')}: {weather_dest.get('condition', 'Unknown')}, {weather_dest.get('temperature', 'N/A')}°C

**NEWS (Last 3 Days):**
{news_text if flight_related_news else 'No significant flight-related news.'}

WRITE A FRIENDLY 2-3 PARAGRAPH SUMMARY:
1. Start with the risk emoji and tell them their delay chance ({delay_prob}%)
2. Describe the weather at BOTH airports in a friendly way
3. ONLY mention news if there's something serious (strikes, storms, closures)
4. Give simple advice based on risk level

IMPORTANT RULES:
- Do NOT explain calculations or formulas
- Do NOT mention ML, RL, percentages breakdown
- Be conversational like talking to a friend
- Keep it SHORT and SIMPLE
"""
        
        response = model.generate_content([
            "You are a helpful flight assistant who provides specific, detailed explanations with numbers and facts.",
            summary_prompt
        ])
        
        logger.info("✅ Generated comprehensive user-friendly summary with LLM")
        return response.text.strip()
        
    except Exception as e:
        logger.warning(f"LLM summary generation failed: {e}")
        logger.info("Using non-LLM fallback summary")
        return generate_enhanced_fallback_summary(prediction_data, signals, breakdown, flight_related_news)


def generate_enhanced_fallback_summary(prediction_data, signals, breakdown, flight_related_news):
    """Simple fallback summary when LLM unavailable - no calculation details"""
    delay_prob = prediction_data.get('probability_delay', 0)
    delay_range = prediction_data.get('predicted_delay_range_mins', [15, 45])
    
    weather_origin = signals.get('live_forecast_origin', {})
    weather_dest = signals.get('live_forecast_destination', {})
    origin_city = signals.get('live_forecast_origin', {}).get('city', 'Origin')
    dest_city = signals.get('live_forecast_destination', {}).get('city', 'Destination')
    
    risk_emoji = "🟢" if delay_prob < 35 else "🟡" if delay_prob < 60 else "🔴"
    risk_level = "LOW" if delay_prob < 35 else "MODERATE" if delay_prob < 60 else "HIGH"
    
    summary = f"{risk_emoji} **{risk_level} RISK** - {delay_prob}% chance of delay\n\n"
    
    # Weather section
    summary += f"**Weather:**\n"
    summary += f"• {origin_city}: {weather_origin.get('condition', 'Unknown')}, {weather_origin.get('temperature', 'N/A')}°C\n"
    summary += f"• {dest_city}: {weather_dest.get('condition', 'Unknown')}, {weather_dest.get('temperature', 'N/A')}°C\n\n"
    
    # News only if important
    if flight_related_news:
        summary += "**⚠️ Important News:**\n"
        for article in flight_related_news[:2]:
            summary += f"• {article.get('title', 'News alert')}\n"
        summary += "\n"
    
    # Simple advice
    if delay_prob < 35:
        summary += f"✅ Your flight looks good! If a delay happens, expect {delay_range[0]}-{delay_range[1]} minutes."
    elif delay_prob < 60:
        summary += f"⏰ Moderate delay chance. If delayed, expect {delay_range[0]}-{delay_range[1]} minutes. Arrive 30 min early."
    else:
        summary += f"🚨 High delay risk! Expect {delay_range[0]}-{delay_range[1]} minutes if delayed. Arrive 1 hour early."
    
    return summary


def predict_flight_outcome(signals, origin, dest, date, dep_time, arr_time, flight_number):
    """
    Main prediction function with clear flow:
    STEP 1: ML Model gives base prediction
    STEP 2: Add weather impact %
    STEP 3: RL adjusts final prediction
    STEP 4: Save prediction for later verification
    """
    logger.info(f"🎯 Generating prediction for {flight_number}")
    
    # ========================================
    # STEP 1: Get ML Model base prediction
    # ========================================
    ml_prediction = None
    ml_confidence = 'N/A'
    
    if ML_MODEL_AVAILABLE:
        try:
            airline_code = ''.join(filter(str.isalpha, str(flight_number)[:2]))
            ml_result = ml_model.predict_with_ml(
                origin=origin,
                destination=dest,
                airline_code=airline_code,
                departure_time=dep_time,
                flight_date=date
            )
            if ml_result.get('probability_delay') is not None:
                ml_prediction = ml_result['probability_delay']
                ml_confidence = ml_result.get('confidence', 'MEDIUM')
                logger.info(f"🧠 STEP 1 - ML Model: {ml_prediction}% delay ({ml_confidence} confidence)")
        except Exception as e:
            logger.warning(f"⚠️ ML prediction failed: {e}")
    
    # ========================================
    # STEP 2: Calculate Weather Impact %
    # ========================================
    weather_origin = signals.get('live_forecast_origin', {})
    weather_dest = signals.get('live_forecast_destination', {})
    
    # Calculate weather risk score (0-100)
    weather_risk = 0
    origin_condition = weather_origin.get('condition', 'Unknown').lower()
    dest_condition = weather_dest.get('condition', 'Unknown').lower()
    
    # Origin weather impact
    if 'storm' in origin_condition or 'thunder' in origin_condition:
        weather_risk += 40
    elif 'rain' in origin_condition or 'shower' in origin_condition:
        weather_risk += 25
    elif 'fog' in origin_condition or 'mist' in origin_condition:
        weather_risk += 30
    elif 'snow' in origin_condition:
        weather_risk += 45
    elif 'cloud' in origin_condition:
        weather_risk += 10
    else:
        weather_risk += 5
    
    # Destination weather impact
    if 'storm' in dest_condition or 'thunder' in dest_condition:
        weather_risk += 40
    elif 'rain' in dest_condition or 'shower' in dest_condition:
        weather_risk += 25
    elif 'fog' in dest_condition or 'mist' in dest_condition:
        weather_risk += 30
    elif 'snow' in dest_condition:
        weather_risk += 45
    elif 'cloud' in dest_condition:
        weather_risk += 10
    else:
        weather_risk += 5
    
    weather_risk = min(weather_risk, 80)  # Cap at 80%
    logger.info(f"🌤️ STEP 2 - Weather Impact: {weather_risk}%")
    
    # ========================================
    # Combine ML + Weather for base probability
    # ========================================
    if ml_prediction is not None:
        # 75% ML + 25% Weather
        base_delay_prob = int(0.75 * ml_prediction + 0.25 * weather_risk)
    else:
        # Fallback: Use statistical formula
        fallback = calculate_fallback_prediction(signals)
        base_delay_prob = fallback['probability_delay']
    
    logger.info(f"📊 Combined (ML+Weather): {base_delay_prob}%")
    
    # ========================================
    # STEP 3: RL Agent Adjustment (Using DQN if available)
    # ========================================
    if DQN_AVAILABLE and dqn_agent:
        adjusted_delay_prob, rl_info = dqn_agent.adjust_prediction(
            base_delay_prob, signals, date, dep_time
        )
        logger.info(f"🧠 Using DQN Agent (Neural Network)")
    else:
        adjusted_delay_prob, rl_info = rl_agent.apply_rl_adjustment(
            base_delay_prob, signals, date, dep_time
        )
        logger.info(f"📊 Using Q-Learning Agent (Fallback)")
    rl_adjustment = adjusted_delay_prob - base_delay_prob
    logger.info(f"🤖 STEP 3 - RL Adjustment: {base_delay_prob}% → {adjusted_delay_prob}% (Δ{rl_adjustment:+.0f}%)")
    
    # ========================================
    # Build final prediction with breakdown
    # ========================================
    fallback_prediction = calculate_fallback_prediction(signals)
    fallback_prediction['probability_delay'] = int(adjusted_delay_prob)
    fallback_prediction['probability_cancel'] = int(min(
        fallback_prediction['probability_cancel'],
        int(adjusted_delay_prob * 0.4)
    ))
    
    # Store component breakdown for summary
    fallback_prediction['prediction_breakdown'] = {
        'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence,
        'weather_risk': weather_risk,
        'weather_origin': weather_origin.get('condition', 'Unknown'),
        'weather_dest': weather_dest.get('condition', 'Unknown'),
        'combined_base': base_delay_prob,
        'rl_adjustment': rl_adjustment,
        'final_prediction': int(adjusted_delay_prob)
    }
    
    # Add to justification
    fallback_prediction['justification'] += f"\n\n🧠 **Prediction Breakdown:**\n"
    if ml_prediction is not None:
        fallback_prediction['justification'] += f"• ML Model: {ml_prediction}% ({ml_confidence})\n"
    fallback_prediction['justification'] += f"• Weather Risk: {weather_risk}%\n"
    fallback_prediction['justification'] += f"• Combined Base: {base_delay_prob}%\n"
    fallback_prediction['justification'] += f"• RL Adjustment: {rl_adjustment:+.0f}%\n"
    fallback_prediction['justification'] += f"• **Final Prediction: {int(adjusted_delay_prob)}%**"
    
    fallback_prediction['rl_metadata'] = convert_to_native_types(rl_info)
    if ml_prediction is not None:
        fallback_prediction['ml_prediction'] = ml_prediction
    
    # ========================================
    # STEP 4: Store prediction for RL learning
    # ========================================
    if TRACKER_AVAILABLE:
        try:
            prediction_tracker.store_prediction(
                flight_number=flight_number,
                origin=origin,
                destination=dest,
                flight_date=date,
                predicted_delay_prob=int(adjusted_delay_prob),
                rl_info=convert_to_native_types(rl_info)
            )
            logger.info(f"📝 STEP 4 - Stored prediction for later verification")
        except Exception as e:
            logger.warning(f"⚠️ Could not store prediction for tracking: {e}")
    
    # ========================================
    # Generate user-friendly summary
    # ========================================
    logger.info("📝 Generating user-friendly summary...")
    fallback_prediction['user_friendly_summary'] = generate_user_friendly_summary(
        fallback_prediction, signals
    )
    logger.info(f"✅ User-friendly summary generated (length: {len(fallback_prediction['user_friendly_summary'])})")
    
    
    if model is not None:
        try:
            user_prompt = f"""
**FLIGHT:** {flight_number}
**ROUTE:** {origin} → {dest}
**DATE:** {date}
**TIMES:** Depart {dep_time} | Arrive {arr_time}

**HISTORICAL DATA:**
{json.dumps(signals.get('long_term_history_seasonal', {}), indent=2)}

**RECENT TRENDS (Last 6 months):**
{json.dumps(signals.get('recent_performance_last_6_months', {}), indent=2)}

**WEATHER FORECAST:**
Origin: {json.dumps(signals.get('live_forecast_origin', {}), indent=2)}
Destination: {json.dumps(signals.get('live_forecast_destination', {}), indent=2)}

**AIRPORT STATUS:**
Origin: {json.dumps(signals.get('live_context_origin_airport', {}), indent=2)}
Destination: {json.dumps(signals.get('live_context_destination_airport', {}), indent=2)}
 

Provide prediction in required JSON format.
"""
            
            response = model.generate_content([SYSTEM_PROMPT, user_prompt])
            response_text = ""

            if hasattr(response, "candidates") and response.candidates:
                for cand in response.candidates:
                    if hasattr(cand, "content") and cand.content:
                        for part in cand.content.parts:
                            if hasattr(part, "text"):
                                response_text += part.text

            response_text = response_text.strip()

            if not response_text:
                raise ValueError("Gemini returned empty response (finish_reason blocked or filtered)")

            
            
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            prediction_data = json.loads(response_text)
            
            
            required_fields = ['probability_delay', 'probability_cancel', 'justification']
            if not all(field in prediction_data for field in required_fields):
                raise ValueError("Missing required fields")
            
            
            llm_base_prob = int(prediction_data['probability_delay'])
            adjusted_prob, rl_info = rl_agent.apply_rl_adjustment(
                llm_base_prob, signals, date, dep_time
            )
            
            prediction_data['probability_delay'] = int(adjusted_prob)
            prediction_data['probability_cancel'] = int(min(
                prediction_data['probability_cancel'],
                int(adjusted_prob * 0.4)
            ))
            
            
            logger.info("📝 Generating user-friendly summary for LLM prediction...")
            prediction_data['user_friendly_summary'] = generate_user_friendly_summary(
                prediction_data, signals
            )
            logger.info(f"✅ User-friendly summary generated (length: {len(prediction_data['user_friendly_summary'])})")
            
            logger.info(f"✅ LLM+RL Prediction: {adjusted_prob}% delay")
            
            return convert_to_native_types(prediction_data)
            
        except Exception as e:
            logger.warning(f"LLM error: {e} - Using statistical prediction")
    
    logger.info(f"✅ Statistical+RL Prediction: {adjusted_delay_prob}% delay")
    return fallback_prediction


def get_chat_response(message, context):
    """Handle follow-up questions"""
    logger.info("💬 Processing chat question")
    
    pred = context.get('prediction_probabilities', {})
    
    if model is None:
        return f"Based on the prediction: Delay probability is {pred.get('probability_delay', 'N/A')}%. The analysis considers historical data, recent trends, weather conditions, and airport status. See the detailed justification above for specifics."
    
    chat_prompt = f"""
Answer this question about a flight prediction concisely:

PREDICTION: {json.dumps(pred, indent=2)}
CONTEXT: {json.dumps(context, indent=2)[:1000]}

USER QUESTION: {message}

Provide a clear 2-3 sentence answer.
"""
    
    try:
        response = model.generate_content(["You are a helpful flight analyst.", chat_prompt])
        return response.text
    except Exception as e:
        logger.warning(f"Chat error: {e}")
        return f"I'm having trouble processing that. The prediction shows {pred.get('probability_delay', 'N/A')}% delay probability. Check the detailed justification for more information."