#!/usr/bin/env python3
"""
Main training script for meta-reinforcement learning experiments.

This script provides a command-line interface for training meta-RL agents
with various algorithms and environments.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import Config, MetaRLTrainer, create_default_config, load_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train meta-reinforcement learning agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    
    # Experiment settings
    parser.add_argument(
        "--experiment-name", "-n",
        type=str,
        default="meta_rl_experiment",
        help="Name of the experiment"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Agent settings
    parser.add_argument(
        "--agent", "-a",
        type=str,
        choices=["rl2", "maml", "ppo", "sac", "td3", "dqn"],
        default="rl2",
        help="Type of agent to train"
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of the neural network"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    
    # Environment settings
    parser.add_argument(
        "--env", "-e",
        type=str,
        choices=["bandit", "grid_world", "cartpole"],
        default="bandit",
        help="Type of environment"
    )
    
    parser.add_argument(
        "--n-arms",
        type=int,
        default=2,
        help="Number of arms for bandit environment"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Size of grid for grid world environment"
    )
    
    # Training settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=3000,
        help="Number of training episodes"
    )
    
    parser.add_argument(
        "--adaptation-steps",
        type=int,
        default=5,
        help="Number of adaptation steps per task"
    )
    
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=100,
        help="Frequency of evaluation (in episodes)"
    )
    
    # Logging settings
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints"
    )
    
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        default=True,
        help="Use TensorBoard for logging"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config.set("experiment.name", args.experiment_name)
    config.set("experiment.seed", args.seed)
    
    config.set("agent.type", args.agent)
    config.set("agent.hidden_dim", args.hidden_dim)
    config.set("agent.lr", args.lr)
    
    config.set("env.type", args.env)
    config.set("env.n_arms", args.n_arms)
    config.set("env.size", args.grid_size)
    
    config.set("training.episodes", args.episodes)
    config.set("training.adaptation_steps", args.adaptation_steps)
    config.set("training.eval_frequency", args.eval_frequency)
    
    config.set("logging.log_dir", args.log_dir)
    config.set("logging.save_dir", args.save_dir)
    config.set("logging.use_wandb", args.use_wandb)
    config.set("logging.use_tensorboard", args.use_tensorboard)
    
    # Print configuration
    print("=" * 60)
    print("Meta-Reinforcement Learning Training")
    print("=" * 60)
    print(f"Experiment: {config.get('experiment.name')}")
    print(f"Agent: {config.get('agent.type')}")
    print(f"Environment: {config.get('env.type')}")
    print(f"Episodes: {config.get('training.episodes')}")
    print(f"Seed: {config.get('experiment.seed')}")
    print("=" * 60)
    
    # Create trainer
    trainer = MetaRLTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
    
    # Start training
    try:
        trainer.train()
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.logger.log_warning("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        trainer.logger.log_error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
