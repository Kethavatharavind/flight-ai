"""
Model Comparison & Benchmarking Tool
✅ Compare XGBoost vs BiLSTM
✅ Compare Q-Learning vs DQN
✅ Generate performance reports
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

# Import existing models
import ml_model
import rl_agent

# Import new models
try:
    from ml_model_advanced import get_bilstm_model
    BILSTM_AVAILABLE = True
except:
    BILSTM_AVAILABLE = False
    print("⚠️ BiLSTM not available")

try:
    from rl_agent_dqn import get_dqn_agent
    DQN_AVAILABLE = True
except:
    DQN_AVAILABLE = False
    print("⚠️ DQN not available")


class ModelComparator:
    """
    Compare performance of different models
    """
    
    def __init__(self):
        # Load existing models
        self.xgboost_model = ml_model.get_ml_model()
        self.qlearning_agent = rl_agent.FlightPredictionRLAgent()
        
        # Load new models if available
        self.bilstm_model = get_bilstm_model() if BILSTM_AVAILABLE else None
        self.dqn_agent = get_dqn_agent() if DQN_AVAILABLE else None
        
        self.results = []
    
    def benchmark_ml_models(self, test_flights):
        """
        Compare ML models on test flights
        
        Args:
            test_flights: List of dicts with keys:
                - origin, destination, airline_code
                - departure_time, flight_date
                - actual_delayed (ground truth)
        """
        print("\n" + "=" * 60)
        print("🧪 ML MODEL COMPARISON")
        print("=" * 60)
        
        xgboost_results = []
        bilstm_results = []
        
        for i, flight in enumerate(test_flights, 1):
            print(f"\n[{i}/{len(test_flights)}] Testing: {flight['origin']} → {flight['destination']}")
            
            # XGBoost prediction
            start = time.time()
            xgb_result = self.xgboost_model.predict_delay_probability(
                origin=flight['origin'],
                destination=flight['destination'],
                airline_code=flight.get('airline_code', '6E'),
                departure_time=flight.get('departure_time', '12:00'),
                flight_date=flight['flight_date']
            )
            xgb_time = time.time() - start
            
            xgb_prob = xgb_result.get('probability_delay', 0)
            xgb_correct = (xgb_prob > 50) == flight['actual_delayed']
            
            xgboost_results.append({
                'probability': xgb_prob,
                'correct': xgb_correct,
                'inference_time': xgb_time
            })
            
            print(f"  XGBoost: {xgb_prob:.1f}% | {'✅' if xgb_correct else '❌'} | {xgb_time*1000:.1f}ms")
            
            # BiLSTM prediction
            if self.bilstm_model:
                start = time.time()
                bilstm_result = self.bilstm_model.predict_delay_probability(
                    origin=flight['origin'],
                    destination=flight['destination'],
                    airline_code=flight.get('airline_code', '6E'),
                    departure_time=flight.get('departure_time', '12:00'),
                    flight_date=flight['flight_date']
                )
                bilstm_time = time.time() - start
                
                bilstm_prob = bilstm_result.get('probability_delay', 0)
                bilstm_correct = (bilstm_prob > 50) == flight['actual_delayed']
                
                bilstm_results.append({
                    'probability': bilstm_prob,
                    'correct': bilstm_correct,
                    'inference_time': bilstm_time
                })
                
                print(f"  BiLSTM:  {bilstm_prob:.1f}% | {'✅' if bilstm_correct else '❌'} | {bilstm_time*1000:.1f}ms")
        
        # Calculate statistics
        print("\n" + "=" * 60)
        print("📊 ML MODEL RESULTS")
        print("=" * 60)
        
        xgb_accuracy = np.mean([r['correct'] for r in xgboost_results]) * 100
        xgb_avg_time = np.mean([r['inference_time'] for r in xgboost_results]) * 1000
        
        print(f"\n🔷 XGBoost + Random Forest:")
        print(f"   Accuracy:        {xgb_accuracy:.2f}%")
        print(f"   Avg Inference:   {xgb_avg_time:.2f}ms")
        
        if bilstm_results:
            bilstm_accuracy = np.mean([r['correct'] for r in bilstm_results]) * 100
            bilstm_avg_time = np.mean([r['inference_time'] for r in bilstm_results]) * 1000
            
            print(f"\n🔶 BiLSTM + Attention:")
            print(f"   Accuracy:        {bilstm_accuracy:.2f}%")
            print(f"   Avg Inference:   {bilstm_avg_time:.2f}ms")
            
            print(f"\n📈 Improvement:")
            print(f"   Accuracy Gain:   {bilstm_accuracy - xgb_accuracy:+.2f}%")
            print(f"   Speed Change:    {bilstm_avg_time - xgb_avg_time:+.2f}ms")
        
        print("=" * 60)
        
        return {
            'xgboost': xgboost_results,
            'bilstm': bilstm_results if bilstm_results else None
        }
    
    def benchmark_rl_agents(self, test_scenarios):
        """
        Compare RL agents
        
        Args:
            test_scenarios: List of dicts with:
                - base_probability
                - signals (weather, airport status, etc.)
                - actual_delayed (ground truth)
        """
        print("\n" + "=" * 60)
        print("🎮 RL AGENT COMPARISON")
        print("=" * 60)
        
        qlearning_results = []
        dqn_results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n[{i}/{len(test_scenarios)}] Base Prob: {scenario['base_probability']}%")
            
            # Q-Learning adjustment
            start = time.time()
            ql_adjusted, ql_info = self.qlearning_agent.adjust_prediction(
                base_probability=scenario['base_probability'],
                signals=scenario['signals'],
                flight_date=scenario.get('flight_date'),
                flight_time=scenario.get('flight_time')
            )
            ql_time = time.time() - start
            
            ql_correct = (ql_adjusted > 50) == scenario['actual_delayed']
            
            qlearning_results.append({
                'adjusted_prob': ql_adjusted,
                'adjustment': ql_adjusted - scenario['base_probability'],
                'correct': ql_correct,
                'inference_time': ql_time
            })
            
            print(f"  Q-Learning: {scenario['base_probability']}% → {ql_adjusted}% | "
                  f"{'✅' if ql_correct else '❌'} | {ql_time*1000:.1f}ms")
            
            # DQN adjustment
            if self.dqn_agent:
                start = time.time()
                dqn_adjusted, dqn_info = self.dqn_agent.adjust_prediction(
                    base_probability=scenario['base_probability'],
                    signals=scenario['signals'],
                    flight_date=scenario.get('flight_date'),
                    flight_time=scenario.get('flight_time')
                )
                dqn_time = time.time() - start
                
                dqn_correct = (dqn_adjusted > 50) == scenario['actual_delayed']
                
                dqn_results.append({
                    'adjusted_prob': dqn_adjusted,
                    'adjustment': dqn_adjusted - scenario['base_probability'],
                    'correct': dqn_correct,
                    'inference_time': dqn_time
                })
                
                print(f"  DQN:        {scenario['base_probability']}% → {dqn_adjusted}% | "
                      f"{'✅' if dqn_correct else '❌'} | {dqn_time*1000:.1f}ms")
        
        # Statistics
        print("\n" + "=" * 60)
        print("📊 RL AGENT RESULTS")
        print("=" * 60)
        
        ql_accuracy = np.mean([r['correct'] for r in qlearning_results]) * 100
        ql_avg_adjustment = np.mean([abs(r['adjustment']) for r in qlearning_results])
        ql_avg_time = np.mean([r['inference_time'] for r in qlearning_results]) * 1000
        
        print(f"\n🔷 Q-Learning (Tabular):")
        print(f"   Accuracy:        {ql_accuracy:.2f}%")
        print(f"   Avg Adjustment:  {ql_avg_adjustment:.2f}%")
        print(f"   Avg Inference:   {ql_avg_time:.2f}ms")
        print(f"   States Explored: {len(self.qlearning_agent.q_table)}")
        
        if dqn_results:
            dqn_accuracy = np.mean([r['correct'] for r in dqn_results]) * 100
            dqn_avg_adjustment = np.mean([abs(r['adjustment']) for r in dqn_results])
            dqn_avg_time = np.mean([r['inference_time'] for r in dqn_results]) * 1000
            
            print(f"\n🔶 DQN (Neural Network):")
            print(f"   Accuracy:        {dqn_accuracy:.2f}%")
            print(f"   Avg Adjustment:  {dqn_avg_adjustment:.2f}%")
            print(f"   Avg Inference:   {dqn_avg_time:.2f}ms")
            print(f"   Replay Buffer:   {len(self.dqn_agent.memory)}")
            
            print(f"\n📈 Improvement:")
            print(f"   Accuracy Gain:   {dqn_accuracy - ql_accuracy:+.2f}%")
            print(f"   Speed Change:    {dqn_avg_time - ql_avg_time:+.2f}ms")
        
        print("=" * 60)
        
        return {
            'qlearning': qlearning_results,
            'dqn': dqn_results if dqn_results else None
        }
    
    def generate_report(self, output_file='model_comparison_report.json'):
        """Generate comprehensive comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': {
                'ml': {
                    'xgboost': True,
                    'bilstm': BILSTM_AVAILABLE
                },
                'rl': {
                    'qlearning': True,
                    'dqn': DQN_AVAILABLE
                }
            },
            'summary': {
                'xgboost': ml_model.get_ml_model().get_stats() if hasattr(ml_model.get_ml_model(), 'get_stats') else {},
                'qlearning': {'states': len(self.qlearning_agent.q_table)},
                'bilstm': self.bilstm_model.get_stats() if self.bilstm_model else None,
                'dqn': self.dqn_agent.get_stats() if self.dqn_agent else None
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {output_file}")
        return report


def create_test_data():
    """Generate test data for comparison"""
    test_flights = [
        {
            'origin': 'DEL', 'destination': 'BOM',
            'airline_code': '6E', 'departure_time': '06:00',
            'flight_date': '2025-12-15', 'actual_delayed': True
        },
        {
            'origin': 'BLR', 'destination': 'HYD',
            'airline_code': 'AI', 'departure_time': '14:30',
            'flight_date': '2025-12-16', 'actual_delayed': False
        },
        {
            'origin': 'MAA', 'destination': 'CCU',
            'airline_code': 'SG', 'departure_time': '18:00',
            'flight_date': '2025-12-17', 'actual_delayed': True
        }
    ]
    
    test_scenarios = [
        {
            'base_probability': 45,
            'signals': {
                'long_term_history_seasonal': {'delay_rate': 35},
                'recent_performance_last_6_months': {'delay_rate_percent': 40},
                'live_forecast_origin': {'condition': 'Rain'},
                'live_forecast_destination': {'condition': 'Clear'},
                'live_context_origin_airport': {'delay_is_active': 'True'},
                'live_context_destination_airport': {'delay_is_active': 'False'}
            },
            'flight_date': '2025-12-15',
            'flight_time': '2025-12-15T06:00',
            'actual_delayed': True
        },
        {
            'base_probability': 30,
            'signals': {
                'long_term_history_seasonal': {'delay_rate': 20},
                'recent_performance_last_6_months': {'delay_rate_percent': 25},
                'live_forecast_origin': {'condition': 'Clear'},
                'live_forecast_destination': {'condition': 'Cloudy'},
                'live_context_origin_airport': {'delay_is_active': 'False'},
                'live_context_destination_airport': {'delay_is_active': 'False'}
            },
            'flight_date': '2025-12-16',
            'flight_time': '2025-12-16T14:30',
            'actual_delayed': False
        }
    ]
    
    return test_flights, test_scenarios


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 FLIGHT AI MODEL COMPARISON TOOL")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔷 XGBoost + RF:  {'✅' if ml_model.get_ml_model().is_trained else '❌'}")
    print(f"🔶 BiLSTM:        {'✅' if BILSTM_AVAILABLE else '❌'}")
    print(f"🔷 Q-Learning:    ✅")
    print(f"🔶 DQN:           {'✅' if DQN_AVAILABLE else '❌'}")
    print("=" * 60)
    
    # Create comparator
    comparator = ModelComparator()
    
    # Generate test data
    test_flights, test_scenarios = create_test_data()
    
    # Run comparisons
    print("\n🚀 Starting ML model comparison...")
    ml_results = comparator.benchmark_ml_models(test_flights)
    
    print("\n🚀 Starting RL agent comparison...")
    rl_results = comparator.benchmark_rl_agents(test_scenarios)
    
    # Generate report
    print("\n📝 Generating comprehensive report...")
    report = comparator.generate_report()
    
    print("\n✅ Comparison complete!")
    print(f"📊 View detailed report: model_comparison_report.json")