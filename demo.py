#!/usr/bin/env python3
"""
Demo script for meta-reinforcement learning project.

This script demonstrates the key features of the meta-RL system
with a quick training run and visualization.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import Config, MetaRLTrainer
from src.envs import create_bandit_env
from src.agents import create_agent


def main():
    """Run a quick demo of the meta-RL system."""
    print("🧠 Meta-Reinforcement Learning Demo")
    print("=" * 40)
    
    # Create a simple configuration for demo
    config = Config()
    config.set("experiment.name", "demo_experiment")
    config.set("training.episodes", 500)  # Shorter for demo
    config.set("training.eval_frequency", 50)
    config.set("training.log_frequency", 25)
    config.set("logging.use_tensorboard", False)
    config.set("logging.use_wandb", False)
    
    print(f"Agent: {config.get('agent.type')}")
    print(f"Environment: {config.get('env.type')}")
    print(f"Episodes: {config.get('training.episodes')}")
    print(f"Hidden Dimension: {config.get('agent.hidden_dim')}")
    print(f"Learning Rate: {config.get('agent.lr')}")
    print()
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = MetaRLTrainer(config)
    
    # Run training
    print("Starting training...")
    trainer.train()
    
    # Analyze results
    rewards = trainer.reward_history
    losses = [h.get('loss', 0) for h in trainer.loss_history]
    
    print("\n📊 Training Results:")
    print(f"Total episodes: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Final reward: {rewards[-1]:.3f}")
    print(f"Best reward: {np.max(rewards):.3f}")
    print(f"Average loss: {np.mean(losses):.4f}")
    
    # Create simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Reward curve
    ax1.plot(rewards, alpha=0.7, linewidth=1)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Loss curve
    ax2.plot(losses, color='red', alpha=0.7, linewidth=1)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("demo_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Results saved to: {plot_path}")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("• Run 'python train.py --help' for more options")
    print("• Try 'streamlit run app.py' for interactive interface")
    print("• Check 'notebooks/analysis.ipynb' for detailed analysis")


if __name__ == "__main__":
    main()
