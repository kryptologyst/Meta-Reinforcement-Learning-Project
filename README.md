# Meta-Reinforcement Learning Project

A comprehensive implementation of meta-reinforcement learning algorithms with support for multiple environments and state-of-the-art techniques.

## Features

- **Multiple Meta-RL Algorithms**: RL², MAML, and integration with Stable-Baselines3 (PPO, SAC, TD3, DQN)
- **Diverse Environments**: Bandit tasks, Grid World, and classic control environments
- **Modern Architecture**: Built with PyTorch, Gymnasium, and Stable-Baselines3
- **Interactive Interfaces**: Command-line interface and Streamlit web app
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Configuration Management**: YAML/JSON configuration files
- **Checkpointing**: Automatic model saving and loading
- **Visualization**: Real-time training curves and performance metrics
- **Testing**: Comprehensive unit tests and integration tests

## 📁 Project Structure

```
0260_Meta-reinforcement_learning/
├── src/
│   ├── agents/           # Meta-RL agents and algorithms
│   ├── envs/             # Custom environments
│   └── utils/             # Training utilities and configuration
├── config/               # Configuration files
├── tests/                # Unit and integration tests
├── notebooks/            # Jupyter notebooks for analysis
├── logs/                 # Training logs and TensorBoard files
├── checkpoints/          # Model checkpoints
├── train.py              # Command-line training script
├── app.py                # Streamlit web interface
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Meta-Reinforcement-Learning-Project.git
   cd Meta-Reinforcement-Learning-Project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Command Line Interface

Train a meta-RL agent with default settings:

```bash
python train.py
```

Train with custom configuration:

```bash
python train.py --agent rl2 --env bandit --episodes 5000 --hidden-dim 128
```

Train with a configuration file:

```bash
python train.py --config config/grid_world.yaml
```

### Streamlit Web Interface

Launch the interactive web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` to access the interface.

## Supported Algorithms

### Meta-RL Algorithms

- **RL² (Reinforcement Learning with Reinforcement Learning)**: Uses memory-augmented policies to learn across tasks
- **MAML (Model-Agnostic Meta-Learning)**: Learns good initializations for quick adaptation

### Stable-Baselines3 Integration

- **PPO**: Proximal Policy Optimization
- **SAC**: Soft Actor-Critic
- **TD3**: Twin Delayed Deep Deterministic Policy Gradient
- **DQN**: Deep Q-Network

## Supported Environments

### Custom Meta-RL Environments

- **Bandit Tasks**: Multi-armed bandit with varying reward probabilities
- **Grid World**: Navigation tasks with changing goal locations

### Classic Control Environments

- **CartPole**: Pole balancing task
- **MountainCar**: Car climbing task

## Configuration

Configuration files are written in YAML format and support nested parameters:

```yaml
experiment:
  name: "my_experiment"
  seed: 42

agent:
  type: "rl2"
  hidden_dim: 64
  lr: 0.001

env:
  type: "bandit"
  n_arms: 2

training:
  episodes: 3000
  adaptation_steps: 5
```

### Command Line Options

```bash
python train.py --help
```

Key options:
- `--agent`: Agent type (rl2, maml, ppo, sac, td3, dqn)
- `--env`: Environment type (bandit, grid_world, cartpole)
- `--episodes`: Number of training episodes
- `--config`: Path to configuration file

## Monitoring and Logging

### TensorBoard

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

### Weights & Biases

Enable W&B logging by setting `use_wandb: true` in your config or using `--use-wandb` flag.

### Checkpoints

Models are automatically saved every 500 episodes by default. Checkpoints include:
- Model weights
- Training metrics
- Configuration
- Metadata

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/test_components.py::TestEnvironments -v
pytest tests/test_components.py::TestAgents -v
pytest tests/test_components.py::TestIntegration -v
```

## Examples

### Basic Bandit Training

```python
from src.utils import Config, MetaRLTrainer
from src.envs import create_bandit_env
from src.agents import create_agent

# Create configuration
config = Config()
config.set("env.type", "bandit")
config.set("agent.type", "rl2")
config.set("training.episodes", 1000)

# Train agent
trainer = MetaRLTrainer(config)
trainer.train()
```

### Custom Environment

```python
from src.envs import OneStepBanditEnv, MetaRLWrapper

# Create custom bandit with specific reward probabilities
env = OneStepBanditEnv(n_arms=3, reward_probs=[0.8, 0.6, 0.4])
wrapped_env = MetaRLWrapper(env)

# Use in training
obs, info = wrapped_env.reset()
action = 0
obs, reward, terminated, truncated, info = wrapped_env.step(action)
```

### Custom Agent Configuration

```python
from src.agents import RL2Agent

config = {
    "input_dim": 3,
    "hidden_dim": 128,
    "output_dim": 2,
    "lr": 0.0005,
    "gamma": 0.95,
    "entropy_coef": 0.02
}

agent = RL2Agent(config)
```

## Research Applications

This project provides a foundation for meta-RL research with:

- **Few-shot Learning**: Agents learn to adapt quickly to new tasks
- **Transfer Learning**: Knowledge transfer across related tasks
- **Continual Learning**: Learning new tasks without forgetting old ones
- **Multi-task Learning**: Simultaneous learning across task distributions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for environment interfaces
- Stable-Baselines3 for RL algorithm implementations
- PyTorch for deep learning framework
- Streamlit for web interface
- The meta-RL research community for foundational algorithms

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

 
# Meta-Reinforcement-Learning-Project
