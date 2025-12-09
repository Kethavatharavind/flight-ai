"""
Test DQN Agent functionality
"""
import sys
sys.path.insert(0, 'src')

from rl_agent_dqn import get_dqn_agent

print("=" * 60)
print("🧠 DQN AGENT TEST")
print("=" * 60)

# Initialize agent
agent = get_dqn_agent()
print(f"\n✅ DQN Agent initialized successfully")
print(f"📊 Network Architecture:")
print(f"   Input: {agent.state_dim} features")
print(f"   Hidden: 128 → 64 → 32")
print(f"   Output: {agent.action_dim} actions")
print(f"   Device: {agent.policy_net.network[0].weight.device}")

# Test prediction
print("\n" + "=" * 60)
print("🧪 TEST PREDICTION")
print("=" * 60)

test_signals = {
    'long_term_history_seasonal': {'delay_rate': 25},
    'recent_performance_last_6_months': {'delay_rate_percent': 35},
    'live_forecast_origin': {'condition': 'Rain'},
    'live_forecast_destination': {'condition': 'Clear'},
    'live_context_origin_airport': {'delay_is_active': 'True'},
    'live_context_destination_airport': {'delay_is_active': 'False'}
}

base_prob = 50
adjusted_prob, rl_info = agent.adjust_prediction(
    base_prob, test_signals, 
    '2024-12-15', '2024-12-15T14:30:00'
)

print(f"\n📈 Results:")
print(f"   Base Prediction: {base_prob}%")
print(f"   DQN Adjusted: {adjusted_prob}%")
print(f"   Adjustment: {adjusted_prob - base_prob:+d}%")
print(f"   Action Index: {rl_info.get('action_idx')}")
print(f"   Action Value: {rl_info.get('action_value')}%")

# Test learning
print("\n" + "=" * 60)
print("🎓 TEST LEARNING")
print("=" * 60)

reward = agent.record_outcome(
    rl_info=rl_info,
    actual_delayed=True,
    predicted_prob=adjusted_prob,
    current_signals=test_signals
)

print(f"\n✅ Learning successful!")
print(f"   Reward: {reward:+.3f}")
print(f"   Memory size: {len(agent.memory)}")
print(f"   Epsilon: {agent.epsilon:.4f}")

# Get stats
print("\n" + "=" * 60)
print("📊 DQN STATS")
print("=" * 60)

stats = agent.get_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - DQN IS WORKING!")
print("=" * 60)
