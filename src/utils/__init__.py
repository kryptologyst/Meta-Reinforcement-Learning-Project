"""
Training utilities and configuration management for meta-RL.

This module provides training loops, logging, checkpointing, and configuration
management for meta-reinforcement learning experiments.
"""

import os
import json
import yaml
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from src.agents import create_agent
from src.envs import create_bandit_env, create_grid_world_env, create_cartpole_env


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "experiment": {
                "name": "meta_rl_experiment",
                "seed": 42,
                "device": "auto"
            },
            "agent": {
                "type": "rl2",
                "input_dim": 3,
                "hidden_dim": 64,
                "output_dim": 2,
                "lr": 1e-3,
                "gamma": 0.99,
                "entropy_coef": 0.01,
                "value_coef": 0.5
            },
            "env": {
                "type": "bandit",
                "n_arms": 2,
                "size": 5,
                "max_steps": 50
            },
            "training": {
                "episodes": 3000,
                "adaptation_steps": 5,
                "eval_frequency": 100,
                "save_frequency": 500,
                "log_frequency": 10
            },
            "logging": {
                "use_wandb": False,
                "use_tensorboard": True,
                "log_dir": "logs",
                "save_dir": "checkpoints"
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
            with open(config_path, "r") as f:
                loaded_config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        self._update_config(self.config, loaded_config)
    
    def _update_config(self, base_config: Dict[str, Any], update_config: Dict[str, Any]) -> None:
        """Recursively update configuration."""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def save_config(self, save_path: str) -> None:
        """Save configuration to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == ".yaml" or save_path.suffix == ".yml":
            with open(save_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif save_path.suffix == ".json":
            with open(save_path, "w") as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


class Logger:
    """Unified logging system."""
    
    def __init__(self, config: Config):
        """Initialize logger."""
        self.config = config
        self.log_dir = Path(config.get("logging.log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logging
        self.setup_python_logging()
        
        # Setup TensorBoard
        if config.get("logging.use_tensorboard", True) and TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(self.log_dir / "tensorboard")
        else:
            self.tb_writer = None
        
        # Setup Weights & Biases
        if config.get("logging.use_wandb", False) and WANDB_AVAILABLE:
            wandb.init(
                project=config.get("experiment.name", "meta_rl"),
                config=config.config,
                dir=str(self.log_dir)
            )
        else:
            if WANDB_AVAILABLE:
                wandb.init(mode="disabled")
    
    def setup_python_logging(self) -> None:
        """Setup Python logging."""
        log_file = self.log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("MetaRL")
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)
        
        if WANDB_AVAILABLE:
            wandb.log({name: value}, step=step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int) -> None:
        """Log a histogram."""
        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)
        
        if WANDB_AVAILABLE:
            wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def log_figure(self, name: str, figure: plt.Figure, step: int) -> None:
        """Log a matplotlib figure."""
        if self.tb_writer:
            self.tb_writer.add_figure(name, figure, step)
        
        if WANDB_AVAILABLE:
            wandb.log({name: wandb.Image(figure)}, step=step)
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if WANDB_AVAILABLE:
            wandb.finish()


class CheckpointManager:
    """Checkpoint management system."""
    
    def __init__(self, config: Config):
        """Initialize checkpoint manager."""
        self.config = config
        self.save_dir = Path(config.get("logging.save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_counter = 0
    
    def save_checkpoint(
        self,
        agent: Any,
        episode: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a checkpoint."""
        checkpoint_name = f"checkpoint_ep{episode:06d}.pkl"
        checkpoint_path = self.save_dir / checkpoint_name
        
        checkpoint_data = {
            "episode": episode,
            "agent_state": agent.policy.state_dict() if hasattr(agent, 'policy') else None,
            "metrics": metrics,
            "config": self.config.config,
            "metadata": metadata or {}
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        self.checkpoint_counter += 1
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint."""
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint path."""
        checkpoints = list(self.save_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return None
        
        # Sort by episode number
        checkpoints.sort(key=lambda x: int(x.stem.split("_ep")[1]))
        return str(checkpoints[-1])


class MetaRLTrainer:
    """Main training class for meta-RL experiments."""
    
    def __init__(self, config: Config):
        """Initialize trainer."""
        self.config = config
        self.logger = Logger(config)
        self.checkpoint_manager = CheckpointManager(config)
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize environment and agent
        self.env = self._create_environment()
        self.agent = self._create_agent()
        
        # Training state
        self.episode = 0
        self.reward_history = []
        self.loss_history = []
        
        self.logger.log_info(f"Initialized MetaRL trainer with config: {config.config}")
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.get("experiment.seed", 42)
        
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _create_environment(self):
        """Create environment based on config."""
        env_type = self.config.get("env.type", "bandit")
        
        if env_type == "bandit":
            return create_bandit_env(
                n_arms=self.config.get("env.n_arms", 2)
            )
        elif env_type == "grid_world":
            return create_grid_world_env(
                size=self.config.get("env.size", 5),
                max_steps=self.config.get("env.max_steps", 50)
            )
        elif env_type == "cartpole":
            return create_cartpole_env()
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    def _create_agent(self):
        """Create agent based on config."""
        agent_config = self.config.get("agent", {})
        agent_type = agent_config.get("type", "rl2")
        
        return create_agent(agent_type, agent_config)
    
    def train(self) -> None:
        """Main training loop."""
        episodes = self.config.get("training.episodes", 3000)
        adaptation_steps = self.config.get("training.adaptation_steps", 5)
        eval_frequency = self.config.get("training.eval_frequency", 100)
        save_frequency = self.config.get("training.save_frequency", 500)
        log_frequency = self.config.get("training.log_frequency", 10)
        
        self.logger.log_info(f"Starting training for {episodes} episodes")
        
        for episode in range(episodes):
            self.episode = episode
            
            # Sample new task
            self.env.sample_task()
            
            # Adapt to new task
            self.agent.adapt(self.env, adaptation_steps)
            
            # Collect rollouts
            rollouts = self._collect_rollouts()
            
            # Update agent
            loss_metrics = self.agent.update(rollouts)
            
            # Log metrics
            episode_reward = sum(r["reward"] for r in rollouts)
            self.reward_history.append(episode_reward)
            self.loss_history.append(loss_metrics)
            
            if episode % log_frequency == 0:
                self.logger.log_scalar("reward/episode_reward", episode_reward, episode)
                self.logger.log_scalar("loss/total_loss", loss_metrics.get("loss", 0), episode)
                self.logger.log_scalar("loss/policy_loss", loss_metrics.get("policy_loss", 0), episode)
                self.logger.log_scalar("loss/value_loss", loss_metrics.get("value_loss", 0), episode)
                self.logger.log_scalar("loss/entropy", loss_metrics.get("entropy", 0), episode)
                
                self.logger.log_info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Loss={loss_metrics.get('loss', 0):.4f}"
                )
            
            # Evaluation
            if episode % eval_frequency == 0:
                self._evaluate()
            
            # Save checkpoint
            if episode % save_frequency == 0:
                self._save_checkpoint()
        
        self.logger.log_info("Training completed")
        self._final_evaluation()
        self.logger.close()
    
    def _collect_rollouts(self) -> List[Dict[str, Any]]:
        """Collect rollouts from the environment."""
        rollouts = []
        obs, _ = self.env.reset()
        
        for _ in range(self.config.get("training.adaptation_steps", 5)):
            action = self.agent.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            rollouts.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            })
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        return rollouts
    
    def _evaluate(self) -> None:
        """Evaluate the agent."""
        eval_episodes = 10
        eval_rewards = []
        
        for _ in range(eval_episodes):
            self.env.sample_task()
            self.agent.adapt(self.env, 5)
            
            episode_reward = 0
            obs, _ = self.env.reset()
            
            for _ in range(10):  # Evaluation episodes
                action = self.agent.act(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        avg_eval_reward = np.mean(eval_rewards)
        std_eval_reward = np.std(eval_rewards)
        
        self.logger.log_scalar("eval/avg_reward", avg_eval_reward, self.episode)
        self.logger.log_scalar("eval/std_reward", std_eval_reward, self.episode)
        
        self.logger.log_info(
            f"Evaluation at episode {self.episode}: "
            f"Avg Reward={avg_eval_reward:.2f} ± {std_eval_reward:.2f}"
        )
    
    def _save_checkpoint(self) -> None:
        """Save a checkpoint."""
        metrics = {
            "episode": self.episode,
            "avg_reward": np.mean(self.reward_history[-100:]) if self.reward_history else 0,
            "total_loss": np.mean([h.get("loss", 0) for h in self.loss_history[-100:]]) if self.loss_history else 0
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.agent, self.episode, metrics
        )
        
        self.logger.log_info(f"Saved checkpoint: {checkpoint_path}")
    
    def _final_evaluation(self) -> None:
        """Perform final evaluation and create plots."""
        self.logger.log_info("Performing final evaluation...")
        
        # Create learning curve plot
        self._plot_learning_curves()
        
        # Final evaluation
        self._evaluate()
    
    def _plot_learning_curves(self) -> None:
        """Create and log learning curve plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward curve
        axes[0, 0].plot(self.reward_history)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)
        
        # Smoothed reward curve
        if len(self.reward_history) > 10:
            window_size = min(100, len(self.reward_history) // 10)
            smoothed_rewards = np.convolve(
                self.reward_history, 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
            axes[0, 1].plot(smoothed_rewards)
            axes[0, 1].set_title(f"Smoothed Rewards (window={window_size})")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Smoothed Reward")
            axes[0, 1].grid(True)
        
        # Loss curves
        if self.loss_history:
            losses = [h.get("loss", 0) for h in self.loss_history]
            policy_losses = [h.get("policy_loss", 0) for h in self.loss_history]
            value_losses = [h.get("value_loss", 0) for h in self.loss_history]
            
            axes[1, 0].plot(losses, label="Total Loss")
            axes[1, 0].plot(policy_losses, label="Policy Loss")
            axes[1, 0].plot(value_losses, label="Value Loss")
            axes[1, 0].set_title("Training Losses")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Entropy
            entropies = [h.get("entropy", 0) for h in self.loss_history]
            axes[1, 1].plot(entropies)
            axes[1, 1].set_title("Policy Entropy")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Entropy")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.logger.log_dir / "learning_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb/tensorboard
        self.logger.log_figure("learning_curves", fig, self.episode)
        
        plt.close()
        
        self.logger.log_info(f"Learning curves saved to {plot_path}")


def create_default_config() -> Config:
    """Create a default configuration."""
    return Config()


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    return Config(config_path)
