# 🤖 RL Parlay Generation System

A reinforcement learning system that learns to generate optimal NFL parlays by analyzing 3 years of historical projections vs actual outcomes.

## 🎯 Overview

This system uses **Proximal Policy Optimization (PPO)** to train an AI agent that learns to:
- Select the best players for parlay legs
- Choose optimal prop types (passing yards, rushing yards, receiving yards, receptions, DK points)
- Set realistic lines (50-90% of projections)
- Create diverse, high-value parlay combinations

## 🏗️ System Architecture

### 1. Data Collection (`rl_parlay_data_collector.py`)
- Fetches 3 years of historical NFL data from SportsData.io API
- Collects projections, actual outcomes, and game context
- Creates training dataset with accuracy metrics and hit rates

### 2. RL Environment (`rl_parlay_environment.py`)
- Defines the parlay generation environment
- State space: Player features + Game context + Parlay state
- Action space: [Player selection, Prop type, Line multiplier]
- Reward function: Based on hit rate, confidence, diversity

### 3. RL Agent (`rl_parlay_agent.py`)
- PPO-based policy network
- Learns optimal parlay generation strategies
- Evaluates and improves over time

### 4. Training Pipeline (`rl_parlay_trainer.py`)
- Complete training pipeline
- Data collection → Preparation → Training → Evaluation
- Saves trained models and results

### 5. GUI Interface (`rl_parlay_gui.py`)
- User-friendly interface for parlay generation
- Load data, select models, generate parlays
- Analyze results and export data

## 🚀 Quick Start

### 1. Setup
```bash
cd rl_parlay_system
python setup.py
```

### 2. Run Demo
```bash
python main.py demo
```

### 3. Collect Historical Data
```bash
python main.py collect --api-key YOUR_API_KEY --years 2022 2023 2024
```

### 4. Train the RL Agent
```bash
python main.py train --api-key YOUR_API_KEY --episodes 1000
```

### 5. Generate Parlays
```bash
python main.py gui
```

## 📊 Data Collection

The system collects comprehensive historical data:

### Projections Data
- Player projections for each week
- DK points, passing yards, rushing yards, receiving yards, receptions
- TDs, interceptions, fumbles
- Salary, injury status, weather conditions

### Actual Outcomes
- Real performance data for each player
- Game results and context
- Accuracy metrics and hit rates

### Game Context
- Weather conditions (temperature, wind, humidity)
- Stadium type (indoor/outdoor, surface)
- Game totals and spreads
- Team matchups

## 🧠 RL Training Process

### State Representation
- **Player Features**: Projections, historical accuracy, consistency
- **Game Context**: Weather, stadium, matchup data
- **Parlay State**: Current legs, diversity, progress

### Action Space
- **Player Selection**: Choose from available players
- **Prop Type**: 5 types (passing_yds, rushing_yds, receiving_yds, receptions, dk_points)
- **Line Multiplier**: 5 options (0.5, 0.6, 0.7, 0.8, 0.9)

### Reward Function
- **Hit Rate Reward**: Higher hit rate = higher reward
- **Confidence Reward**: More confident predictions = higher reward
- **Diversity Reward**: Different prop types and teams = higher reward
- **Diminishing Returns**: Penalty for too many legs

## 🎯 Parlay Generation

The trained agent generates parlays by:

1. **Analyzing Available Players**: Considers projections, historical accuracy, and game context
2. **Selecting Optimal Props**: Chooses prop types with highest hit rates
3. **Setting Realistic Lines**: Uses 70% of projections for safer lines
4. **Ensuring Diversity**: Mixes different players, teams, and prop types
5. **Optimizing Value**: Balances hit rate with potential payout

## 📈 Performance Metrics

The system tracks several key metrics:

- **Hit Rate**: Percentage of parlays that would win
- **Expected Value**: Average profit/loss per $100 bet
- **Diversity Score**: Mix of players, teams, and prop types
- **Confidence Level**: Certainty of predictions
- **Odds Accuracy**: How well estimated odds match reality

## 🔧 Configuration

### Training Parameters
```python
# Agent settings
learning_rate = 3e-4
gamma = 0.99
eps_clip = 0.2
k_epochs = 4

# Training settings
num_episodes = 1000
batch_size = 64
max_legs = 4
```

### Data Collection
```python
# Years to collect
years = [2022, 2023, 2024]

# API settings
api_key = "YOUR_SPORTSDATA_IO_KEY"
```

## 📁 File Structure

```
rl_parlay_system/
├── __init__.py                    # Package init
├── main.py                       # Main entry point
├── setup.py                      # Setup script
├── rl_parlay_data_collector.py   # Data collection
├── rl_parlay_environment.py      # RL environment
├── rl_parlay_agent.py           # PPO agent
├── rl_parlay_trainer.py         # Training pipeline
├── rl_parlay_gui.py             # GUI interface
├── rl_parlay_demo.py            # Demo script
├── requirements_rl.txt           # Dependencies
├── RL_PARLAY_README.md          # This file
├── rl_training_data/            # Collected data
├── rl_models/                   # Trained models
└── rl_results/                  # Training results
```

## 🎮 GUI Usage

### 1. Data Loading Tab
- Load NFL data file
- View data statistics and sample
- Prepare data for RL training

### 2. Model Management Tab
- Load trained RL models
- Test model performance
- View training progress

### 3. Parlay Generation Tab
- Set generation parameters
- Generate multiple parlays
- View detailed results

### 4. Results Analysis Tab
- Analyze generated parlays
- Export results to JSON
- View performance statistics

## 🔬 Advanced Features

### Custom Reward Functions
Modify the reward function in `rl_parlay_environment.py` to focus on different objectives:
- Higher hit rates
- Better expected value
- More diverse parlays
- Specific prop types

### Model Architecture
Customize the neural network in `rl_parlay_agent.py`:
- Adjust hidden layer sizes
- Add attention mechanisms
- Implement different architectures

### Data Augmentation
Enhance training data in `rl_parlay_data_collector.py`:
- Add more historical years
- Include additional features
- Handle missing data better

## 🚨 Important Notes

### Data Quality
- Ensure high-quality historical data
- Handle missing values appropriately
- Validate projection accuracy

### Model Training
- Train on sufficient data (3+ years recommended)
- Monitor for overfitting
- Validate on unseen data

### Betting Disclaimer
- This is for educational/research purposes
- Always bet responsibly
- Past performance doesn't guarantee future results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please use responsibly and in accordance with applicable laws and regulations.

## 🆘 Support

For issues or questions:
1. Check the documentation
2. Review the code comments
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Parlay Generating! 🏈🤖**
