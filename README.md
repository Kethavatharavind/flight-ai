# ✈️ Flight Delay Prediction AI

AI-powered flight delay prediction system for Indian domestic flights using Machine Learning, Reinforcement Learning, and Real-Time Data.

## 🚀 Live Demo
**https://flight-ai-f4vr.onrender.com**

## 🧠 Features
- **ML Model**: XGBoost + Random Forest ensemble (70% accuracy)
- **RL Agent**: Q-Learning with 34 learned states
- **Real-Time Data**: Weather, airport status, news
- **Gemini AI**: Natural language summaries
- **Cloud Storage**: Supabase for persistent learning

## 📁 Project Structure
```
FLIGHT_AI/
├── app.py              # Flask web application
├── ml_model.py         # XGBoost/RF ML model
├── rl_agent.py         # Q-Learning RL agent
├── rl_agent_dqn.py     # Deep Q-Network agent
├── llm_analyzer.py     # Gemini LLM integration
├── data_fetcher.py     # External API calls
├── supabase_client.py  # Cloud database
├── prediction_tracker.py # Track predictions
├── update_latest_data.py # Daily data updater
├── templates/          # HTML templates
├── static/             # CSS/JS assets
├── delay_model.pkl     # Trained ML model
└── requirements.txt    # Python dependencies
```

## 🛠️ Setup

### 1. Clone & Install
```bash
git clone https://github.com/Kethavatharavind/flight-ai.git
cd flight-ai
pip install -r requirements.txt
```

### 2. Environment Variables
Create `.env` file:
```
GEMINI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
AVIATIONSTACK_API_KEY=your_key
```

### 3. Run Locally
```bash
python app.py
```

### 4. Daily Updates
```bash
python update_latest_data.py  # Fetch new flight data
python ml_model.py            # Retrain model
```

## 🧪 Testing
```bash
python render_test.py  # Pre-deployment check
python test.py         # Model benchmarking
```

## 📊 Model Performance
- **XGBoost**: 69.78% accuracy
- **Random Forest**: 68.86% accuracy
- **Ensemble**: 70.36% accuracy
- **RL Agent**: 34 states learned

## 🌐 Deployment
Deployed on Render with auto-deploy from GitHub.


