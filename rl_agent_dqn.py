"""
Deep Q-Network (DQN) Agent for Flight Delay Prediction
✅ Uses PyTorch for Deep Reinforcement Learning
✅ Experience Replay for stable training
✅ Target Network for convergence
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    """Deep Q-Network Architecture"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience Replay Buffer to break temporal correlations"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Advanced Deep Q-Learning Agent
    Replaces tabular Q-learning with Neural Network approximation
    """
    def __init__(self, state_dim=15, learning_rate=0.001):
        self.state_dim = state_dim
        # Actions: -20%, -10%, 0%, +10%, +20% adjustment
        self.actions = [-20, -10, 0, 10, 20]
        self.action_dim = len(self.actions)
        
        # Networks
        self.policy_net = DQN(state_dim, self.action_dim).to(device)
        self.target_net = DQN(state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update_frequency = 100
        self.steps_done = 0
        
        self.model_file = 'dqn_model.pth'
        self.load_model()

    def get_state_vector(self, base_prob, signals, day_of_week=0, hour=12):
        """
        Convert complex dictionary inputs into a normalized vector for the Neural Network.
        Vector Size: 15
        [0]: Base Probability (0-1)
        [1]: Weather Origin Code (0-5)
        [2]: Weather Dest Code (0-5)
        [3]: Airport Origin Status (0-1)
        [4]: Airport Dest Status (0-1)
        [5]: Historical Seasonal Delay Rate (0-1)
        [6]: Recent Delay Rate (0-1)
        [7]: Day of Week (0-6 normalized)
        [8]: Hour of Day (0-23 normalized)
        [9-14]: Reserved/Embedding placeholders
        """
        vec = np.zeros(self.state_dim)
        
        # 1. Base Probability
        vec[0] = base_prob / 100.0
        
        # 2. Weather Encoding (Simple mapping)
        weather_map = {'Clear': 0, 'Cloudy': 0.2, 'Rain': 0.6, 'Storm': 1.0, 'Fog': 0.8}
        w_origin = signals.get('live_forecast_origin', {}).get('condition', 'Clear')
        w_dest = signals.get('live_forecast_destination', {}).get('condition', 'Clear')
        vec[1] = weather_map.get(w_origin, 0.4)
        vec[2] = weather_map.get(w_dest, 0.4)
        
        # 3. Airport Context
        c_origin = signals.get('live_context_origin_airport', {}).get('delay_is_active', 'False')
        c_dest = signals.get('live_context_destination_airport', {}).get('delay_is_active', 'False')
        vec[3] = 1.0 if str(c_origin) == 'True' else 0.0
        vec[4] = 1.0 if str(c_dest) == 'True' else 0.0
        
        # 4. History pattern
        hist_rate = signals.get('long_term_history_seasonal', {}).get('delay_rate', 0)
        recent_rate = signals.get('recent_performance_last_6_months', {}).get('delay_rate_percent', 0)
        vec[5] = float(hist_rate) / 100.0
        vec[6] = float(recent_rate) / 100.0
        
        # 5. Time
        vec[7] = day_of_week / 7.0
        vec[8] = hour / 24.0
        
        return torch.FloatTensor(vec).to(device)

    def adjust_prediction(self, base_probability, signals, flight_date=None, flight_time=None):
        """Main interface for using the agent"""
        try:
            # Parse time
            dt = datetime.now()
            if flight_time:
                try:
                    dt = datetime.fromisoformat(flight_time)
                except:
                    pass
            
            state = self.get_state_vector(base_probability, signals, dt.weekday(), dt.hour)
            
            # Select action (Epsilon Greedy)
            if random.random() < self.epsilon:
                action_idx = random.randint(0, self.action_dim - 1)
                logger.info(f"🎲 Random Action: {self.actions[action_idx]}%")
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state)
                    action_idx = q_values.argmax().item()
                    logger.info(f"🧠 AI Action: {self.actions[action_idx]}% (Q: {q_values.max():.2f})")
            
            adjustment = self.actions[action_idx]
            final_prob = max(0, min(100, base_probability + adjustment))
            
            rl_info = {
                'state_vector': state.tolist(),
                'action_idx': action_idx,
                'action_value': adjustment,
                'original_prob': base_probability
            }
            
            return final_prob, rl_info
            
        except Exception as e:
            logger.error(f"DQN Error: {e}")
            return base_probability, {}

    def learn(self, batch_size=32):
        """Train the neural network using Experience Replay"""
        if len(self.memory) < batch_size:
            return 0
        
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Q(s,a)
        current_q = self.policy_net(states).gather(1, actions)
        
        # Max Q(s', a') from target network
        with torch.no_grad():
            next_max_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_max_q * (1 - dones))
        
        # Optimization
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def record_outcome(self, rl_info, actual_delayed, predicted_prob, current_signals):
        """Store experience and invoke learning"""
        if not rl_info or 'state_vector' not in rl_info:
            return 0
        
        # Reward Calculation (Brier Score improvement)
        # If delayed (1) and prob high -> Good
        # If delayed (1) and prob low -> Bad
        target = 1.0 if actual_delayed else 0.0
        prob_normalized = predicted_prob / 100.0
        
        # Reward: Improvement over baseline? 
        # Or simple accuracy reward:
        reward = 1.0 - (target - prob_normalized)**2  # Max 1.0, Min 0.0
        
        # Penalty for being wrong side of 50%
        pred_class = prob_normalized > 0.5
        if pred_class != actual_delayed:
            reward -= 0.5
            
        # Store in memory
        state = torch.FloatTensor(rl_info['state_vector'])
        action = rl_info['action_idx']
        
        # Next state? For flight predictions, episodes are often length 1 (contextual bandit style)
        # But we can treat the next prediction as the next state
        # For now, next_state can be same as state (stateless assumption) or zeros
        next_state = state # Placeholder
        
        self.memory.add(state, action, reward, next_state, done=True)
        
        # Learn
        loss = self.learn(self.batch_size)
        self.save_model()
        
        return reward

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_file)
        
    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                self.policy_net.load_state_dict(torch.load(self.model_file, map_location=device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info(f"✅ DQN model loaded from {self.model_file}")
            except Exception as e:
                logger.error(f"Failed to load DQN model: {e}")

    def get_stats(self):
        return {
            'type': 'Deep Q-Network (PyTorch)',
            'memory_size': len(self.memory),
            'epsilon': self.epsilon,
            'device': str(device)
        }

# Factory
def get_dqn_agent():
    return DQNAgent()

if __name__ == "__main__":
    # Test
    agent = DQNAgent()
    test_signals = {'live_forecast_origin': {'condition': 'Storm'}}
    prob, info = agent.adjust_prediction(50, test_signals)
    print(f"Prediction: {prob}%")
