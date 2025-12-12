"""
Production-Ready Reinforcement Learning Agent for Flight Delay Prediction
✅ FIXED: Proper state transitions, epsilon decay, probabilistic rewards, fresh signals
"""

import numpy as np
import json
import os
from datetime import datetime
import logging

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try to import Supabase cloud storage
try:
    from supabase_client import is_cloud_enabled, save_q_table as cloud_save_q, load_q_table as cloud_load_q
    from supabase_client import save_rl_metrics as cloud_save_metrics, load_rl_metrics as cloud_load_metrics
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightPredictionRLAgent:
    """
    Advanced Q-Learning agent with:
    - Proper state transitions with fresh signals
    - Experience-based epsilon decay
    - Probabilistic reward structure (Brier score)
    - Rich state representation (weather, time, delays)
    """
    
    
    DELAY_THRESHOLD_LOW = 15
    DELAY_THRESHOLD_MEDIUM = 30
    DELAY_THRESHOLD_HIGH = 50
    RECENT_DELAY_LOW = 20
    RECENT_DELAY_MEDIUM = 40
    
    
    EPSILON_DECAY_RATE = 500
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 epsilon_start=0.3, epsilon_min=0.05,
                 q_table_file=None, metrics_file=None):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        
        # Use new folder structure for data files
        self.q_table_file = q_table_file or os.path.join(PROJECT_ROOT, 'data', 'rl_q_table.json')
        self.metrics_file = metrics_file or os.path.join(PROJECT_ROOT, 'data', 'rl_metrics.json')
        
        
        self.actions = [-15, -10, -5, 0, 5, 10, 15]
        
        
        self.q_table = self._load_q_table()
        self.metrics = self._load_metrics()
        
        
        self._update_epsilon_from_experience()
        
        logger.info(f"✅ RL Agent initialized | States: {len(self.q_table)} | Epsilon: {self.epsilon:.3f}")
    
    def _load_q_table(self):
        """Load Q-table from cloud or disk"""
        # Try cloud first
        if CLOUD_AVAILABLE and is_cloud_enabled():
            try:
                data = cloud_load_q()
                if data:
                    logger.info(f"☁️ Loaded Q-table from cloud: {len(data)} states")
                    return data
            except Exception as e:
                logger.warning(f"⚠️ Cloud load failed: {e}")
        
        # Fallback to local file
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'r') as f:
                    data = json.load(f)
                    if 'q_table' in data:
                        logger.info(f"📂 Loaded Q-table from file: {len(data.get('q_table', {}))} states")
                        return data.get('q_table', {})
                    return data
            except Exception as e:
                logger.warning(f"⚠️ Could not load Q-table: {e}")
                return {}
        
        logger.info("🆕 No Q-table found, starting fresh")
        return {}
    
    def _load_metrics(self):
        """Load learning metrics from cloud or file, return defaults if doesn't exist"""
        # Try cloud first (like _load_q_table does)
        if CLOUD_AVAILABLE and is_cloud_enabled():
            try:
                data = cloud_load_metrics()
                if data:
                    if 'brier_score_history' not in data:
                        data['brier_score_history'] = []
                    logger.info(f"☁️ Loaded metrics from cloud: {data.get('total_learning_episodes', 0)} episodes")
                    return data
            except Exception as e:
                logger.warning(f"⚠️ Cloud metrics load failed: {e}")
        
        # Fallback to local file
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics = json.load(f)
                    
                    if 'brier_score_history' not in metrics:
                        metrics['brier_score_history'] = []
                    logger.info(f"📊 Loaded metrics from file: {metrics.get('total_learning_episodes', 0)} episodes")
                    return metrics
            except Exception as e:
                logger.warning(f"⚠️ Could not load metrics: {e}, using defaults")
        
        logger.info("🆕 No metrics found (cloud or file), starting fresh")
        
        return {
            'total_predictions': 0,
            'total_learning_episodes': 0,
            'avg_reward': 0.0,
            'accuracy_history': [],
            'brier_score_history': []
        }
    
    def _update_epsilon_from_experience(self):
        """Calculate epsilon based on total experience (predictions + learning)"""
        total_experience = self.metrics['total_predictions'] + self.metrics['total_learning_episodes']
        if total_experience > 0:
            
            self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
                           np.exp(-total_experience / self.EPSILON_DECAY_RATE)
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def save_q_table(self):
        """Save Q-table to cloud and disk"""
        # Save to cloud
        if CLOUD_AVAILABLE and is_cloud_enabled():
            try:
                cloud_save_q(self.q_table)
                logger.info(f"☁️ Q-table saved to cloud | States: {len(self.q_table)}")
            except Exception as e:
                logger.warning(f"⚠️ Cloud save failed: {e}")
        
        # Also save to local file as backup
        try:
            os.makedirs(os.path.dirname(self.q_table_file), exist_ok=True)
            with open(self.q_table_file, 'w') as f:
                json.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f, indent=2)
            logger.info(f"💾 Q-table saved to file | Epsilon: {self.epsilon:.3f}")
        except Exception as e:
            logger.error(f"❌ Failed to save Q-table locally: {e}")
    
    def save_metrics(self):
        """Save learning metrics to cloud and disk"""
        # Save to cloud first
        if CLOUD_AVAILABLE and is_cloud_enabled():
            try:
                cloud_save_metrics(self.metrics)
                logger.info(f"☁️ Metrics saved to cloud | Episodes: {self.metrics.get('total_learning_episodes', 0)}")
            except Exception as e:
                logger.warning(f"⚠️ Cloud metrics save failed: {e}")
        
        # Also save to local file as backup
        try:
            directory = os.path.dirname(self.metrics_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"📊 Metrics saved to file | Episodes: {self.metrics.get('total_learning_episodes', 0)}")
        except Exception as e:
            logger.error(f"❌ Failed to save metrics locally: {e}")
    
    def get_state(self, signals, flight_date=None, flight_time=None):
        """
        Enhanced state representation with temporal features
        
        Args:
            signals: Dictionary with historical, weather, airport data
            flight_date: ISO format date string
            flight_time: ISO format time string
            
        Returns:
            State string representing current situation
        """
        # Use recent database performance for delay rate
        recent = signals.get('recent_performance_last_6_months', {})
        weather_origin = signals.get('live_forecast_origin', {})
        weather_dest = signals.get('live_forecast_destination', {})
        airport_origin = signals.get('live_context_origin_airport', {})
        airport_dest = signals.get('live_context_destination_airport', {})
        
        # Categorize delay rate from database
        delay_rate = recent.get('delay_rate_percent')
        if delay_rate is None:
            delay_bucket = 'unknown'
        elif delay_rate < self.DELAY_THRESHOLD_LOW:
            delay_bucket = 'low'
        elif delay_rate < self.DELAY_THRESHOLD_MEDIUM:
            delay_bucket = 'medium'
        elif delay_rate < self.DELAY_THRESHOLD_HIGH:
            delay_bucket = 'high'
        else:
            delay_bucket = 'very_high'
            
        
        recent_delay = recent.get('delay_rate_percent')
        if recent_delay is None:
            recent_bucket = 'recent_unknown'
        elif recent_delay < self.RECENT_DELAY_LOW:
            recent_bucket = 'recent_low'
        elif recent_delay < self.RECENT_DELAY_MEDIUM:
            recent_bucket = 'recent_medium'
        else:
            recent_bucket = 'recent_high'
        
        
        severe_keywords = ['thunderstorm', 'heavy rain', 'heavy snow', 'blizzard', 'hail', 'hurricane']
        moderate_keywords = ['rain', 'snow', 'storm', 'fog', 'drizzle', 'sleet']
        
        origin_weather = weather_origin.get('condition', '').lower()
        dest_weather = weather_dest.get('condition', '').lower()
        
        if any((word in origin_weather) or (word in dest_weather) for word in severe_keywords):
            weather_status = 'severe'
        elif any((word in origin_weather) or (word in dest_weather) for word in moderate_keywords):
            weather_status = 'moderate'
        else:
            weather_status = 'clear'
        
        
        origin_delayed = airport_origin.get('delay_is_active') == 'True'
        dest_delayed = airport_dest.get('delay_is_active') == 'True'
        
        if origin_delayed and dest_delayed:
            airport_status = 'both_delayed'
        elif origin_delayed or dest_delayed:
            airport_status = 'one_delayed'
        else:
            airport_status = 'normal'
        
        
        time_period = 'unknown'
        if flight_time:
            try:
                hour = int(flight_time.split('T')[1].split(':')[0])
                if 5 <= hour < 12:
                    time_period = 'morning'
                elif 12 <= hour < 17:
                    time_period = 'afternoon'
                elif 17 <= hour < 21:
                    time_period = 'evening'
                else:
                    time_period = 'night'
            except (IndexError, ValueError, AttributeError) as e:
                logger.warning(f"⚠️ Invalid flight_time format: {flight_time} - {e}")
        
        
        day_type = 'unknown'
        if flight_date:
            try:
                date_obj = datetime.fromisoformat(flight_date.split('T')[0])
                weekday = date_obj.weekday()
                if weekday < 5:
                    day_type = 'weekday'
                else:
                    day_type = 'weekend'
            except (ValueError, AttributeError) as e:
                logger.warning(f"⚠️ Invalid flight_date format: {flight_date} - {e}")
        
        
        state = f"{delay_bucket}_{recent_bucket}_{weather_status}_{airport_status}_{time_period}_{day_type}"
        return state
    
    def get_q_values(self, state):
        """Get Q-values for a state, initialize if not exists"""
        if state not in self.q_table:
            
            self.q_table[state] = [np.random.uniform(-0.1, 0.1) for _ in self.actions]
        return self.q_table[state]
    
    def choose_action(self, state, training=False):
        """
        Epsilon-greedy action selection
        
        Args:
            state: Current state string
            training: If True, use epsilon for exploration
            
        Returns:
            (action_index, action_value) tuple
        """
        
        if training and np.random.random() < self.epsilon:
            
            action_idx = np.random.randint(len(self.actions))
            logger.debug(f"🎲 Exploring: random action {self.actions[action_idx]}")
        else:
            
            q_values = self.get_q_values(state)
            action_idx = int(np.argmax(q_values))
            logger.debug(f"🎯 Exploiting: best action {self.actions[action_idx]} (Q={q_values[action_idx]:.3f})")
        
        return action_idx, self.actions[action_idx]
    
    def adjust_prediction(self, base_probability, signals, flight_date=None, flight_time=None):
        """
        Apply RL adjustment to base ML prediction
        
        Args:
            base_probability: Base prediction from ML model (0-100)
            signals: Dictionary with all signal data
            flight_date: ISO format date
            flight_time: ISO format time
            
        Returns:
            (adjusted_probability, rl_info) tuple
        """
        
        state = self.get_state(signals, flight_date, flight_time)
        
        
        action_idx, action_value = self.choose_action(state, training=False)
        
        
        adjusted_prob = base_probability + action_value
        adjusted_prob = max(15, min(90, adjusted_prob))  
        
        
        self.metrics['total_predictions'] += 1
        self._update_epsilon_from_experience()
        
        
        rl_info = {
            'state': state,
            'action_index': int(action_idx),
            'action_value': int(action_value),
            'base_probability': int(base_probability),
            'adjusted_probability': int(adjusted_prob),
            'q_values': [float(q) for q in self.get_q_values(state)],
            'epsilon': float(self.epsilon),
            'flight_date': flight_date,
            'flight_time': flight_time
        }
        
        logger.info(
            f"🤖 RL Prediction | State: {state[:30]}... | "
            f"Base: {base_probability}% → Adjusted: {adjusted_prob}% (Δ{action_value:+d}%) | "
            f"Epsilon: {self.epsilon:.3f}"
        )
        
        return int(adjusted_prob), rl_info
    
    def update_q_value(self, state, action_idx, reward, next_state):
        """
        Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.get_q_values(state)[action_idx]
        next_max_q = max(self.get_q_values(next_state))
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[state][action_idx] = new_q
        
        logger.debug(
            f"📈 Q-Update | Q({state[:20]}..., {self.actions[action_idx]}) | "
            f"{current_q:.3f} → {new_q:.3f} (Δ{new_q-current_q:+.3f})"
        )
    
    def learn_from_outcome(self, rl_info, actual_delayed, predicted_prob, current_signals=None):
        """
        Learn from actual flight outcome using Brier score-based rewards
        
        Args:
            rl_info: Dictionary from adjust_prediction() call
            actual_delayed: Boolean - was flight actually delayed?
            predicted_prob: The probability we predicted (0-100)
            current_signals: FRESH signals from when flight landed (optional but recommended)
            
        Returns:
            reward value
        """
        state = rl_info['state']
        action_idx = rl_info['action_index']
        
        
        predicted = predicted_prob / 100.0
        actual = 1.0 if actual_delayed else 0.0
        
        
        brier_score = (predicted - actual) ** 2
        
        
        
        
        reward = 1.0 - 2.0 * brier_score
        
        
        if actual_delayed and predicted > 0.7:
            reward += 0.3  
        elif not actual_delayed and predicted < 0.3:
            reward += 0.3  
        
        
        if actual_delayed and predicted < 0.3:
            reward -= 0.3  
        elif not actual_delayed and predicted > 0.7:
            reward -= 0.3  
        
        
        if current_signals:
            
            next_state = self.get_state(
                current_signals,
                rl_info.get('flight_date'),
                rl_info.get('flight_time')
            )
            logger.debug("✅ Using fresh signals for next_state")
        else:
            
            next_state = state
            logger.warning("⚠️ No fresh signals provided, using same state (suboptimal)")
        
        
        self.update_q_value(state, action_idx, reward, next_state)
        
        
        self.metrics['total_learning_episodes'] += 1
        old_avg = self.metrics.get('avg_reward', 0.0)
        n = self.metrics['total_learning_episodes']
        self.metrics['avg_reward'] = (old_avg * (n - 1) + reward) / n
        
        
        self.metrics['brier_score_history'].append(float(brier_score))
        if len(self.metrics['brier_score_history']) > 100:
            self.metrics['brier_score_history'] = self.metrics['brier_score_history'][-100:]
        
        
        self._update_epsilon_from_experience()
        
        
        if self.metrics['total_learning_episodes'] % 10 == 0:
            self.save_q_table()
            self.save_metrics()
        
        
        avg_brier = np.mean(self.metrics['brier_score_history']) if self.metrics['brier_score_history'] else 0
        
        logger.info(
            f"🎯 RL Learning | Episode: {self.metrics['total_learning_episodes']} | "
            f"Actual: {'DELAYED' if actual_delayed else 'ON-TIME'} | "
            f"Predicted: {predicted_prob}% | "
            f"Reward: {reward:+.2f} | Avg Reward: {self.metrics['avg_reward']:.3f} | "
            f"Brier: {brier_score:.3f} (Avg: {avg_brier:.3f}) | "
            f"Epsilon: {self.epsilon:.3f}"
        )
        
        return reward
    
    def get_stats(self):
        """Get comprehensive statistics about the agent"""
        if not self.q_table:
            return {
                'total_states': 0,
                'avg_q_value': 0,
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                **self.metrics
            }
        
        all_q_values = []
        for state_q_values in self.q_table.values():
            all_q_values.extend(state_q_values)
        
        
        brier_history = self.metrics.get('brier_score_history', [])
        avg_brier = np.mean(brier_history) if brier_history else 0
        
        return {
            'total_states': len(self.q_table),
            'avg_q_value': float(np.mean(all_q_values)) if all_q_values else 0,
            'max_q_value': float(np.max(all_q_values)) if all_q_values else 0,
            'min_q_value': float(np.min(all_q_values)) if all_q_values else 0,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': float(self.epsilon),
            'epsilon_min': self.epsilon_min,
            'avg_brier_score': float(avg_brier),
            **self.metrics
        }



_rl_agent = None


def get_rl_agent():
    """Get or create the global RL agent instance"""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = FlightPredictionRLAgent()
    return _rl_agent


def apply_rl_adjustment(base_probability, signals, flight_date=None, flight_time=None):
    """
    Apply RL adjustment to base ML prediction
    
    Args:
        base_probability: Base prediction from ML model (0-100)
        signals: Dictionary with all signal data
        flight_date: ISO format date
        flight_time: ISO format time
        
    Returns:
        (adjusted_probability, rl_info) tuple
    """
    agent = get_rl_agent()
    return agent.adjust_prediction(base_probability, signals, flight_date, flight_time)


def record_outcome(rl_info, actual_delayed, predicted_prob, current_signals=None):
    """
    Record actual outcome for learning
    
    Args:
        rl_info: Dictionary from apply_rl_adjustment() call
        actual_delayed: Boolean - was flight actually delayed?
        predicted_prob: The probability we predicted (0-100)
        current_signals: FRESH signals from when flight landed (IMPORTANT!)
        
    Returns:
        reward value
    """
    agent = get_rl_agent()
    return agent.learn_from_outcome(rl_info, actual_delayed, predicted_prob, current_signals)


def save_agent_state():
    """Save agent state on shutdown"""
    agent = get_rl_agent()
    agent.save_q_table()
    agent.save_metrics()
    logger.info("✅ RL Agent state saved on shutdown")


def get_agent_stats():
    """Get current agent statistics"""
    agent = get_rl_agent()
    return agent.get_stats()


