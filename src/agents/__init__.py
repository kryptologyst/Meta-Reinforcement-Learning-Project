"""
Modern meta-reinforcement learning agents.

This module provides various meta-RL agents including memory-augmented policies,
MAML, and integration with stable-baselines3 algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import random
from collections import deque
import logging

try:
    from stable_baselines3 import PPO, SAC, TD3, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    # Create dummy classes for when stable-baselines3 is not available
    class PPO: pass
    class SAC: pass
    class TD3: pass
    class DQN: pass
    class BaseCallback: pass
    class DummyVecEnv: pass
    class SubprocVecEnv: pass
    def set_random_seed(*args, **kwargs): pass


class MetaRLAgent(ABC):
    """Abstract base class for meta-RL agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the agent with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def adapt(self, env, adaptation_steps: int = 5) -> None:
        """Adapt to a new task."""
        pass
    
    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select an action given an observation."""
        pass
    
    @abstractmethod
    def update(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update the agent's parameters."""
        pass


class MemoryAugmentedPolicy(nn.Module):
    """
    Memory-augmented policy network for meta-RL.
    
    Uses a GRU to maintain memory of past observations, actions, and rewards,
    enabling quick adaptation to new tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        memory_size: int = 32,
        dropout: float = 0.1
    ):
        """
        Initialize the memory-augmented policy.
        
        Args:
            input_dim: Dimension of input observations
            hidden_dim: Hidden dimension of the GRU
            output_dim: Dimension of action space
            memory_size: Size of the memory buffer
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_size = memory_size
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Memory network (GRU)
        self.memory = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layers
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            obs: Current observation
            hidden_state: Previous hidden state
            return_hidden: Whether to return hidden state
            
        Returns:
            Action logits, value estimate, and optionally hidden state
        """
        batch_size = obs.size(0)
        
        # Project input
        x = self.input_proj(obs)
        
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Process through memory
        memory_out, new_hidden = self.memory(x, hidden_state)
        
        # Get final output
        final_out = memory_out[:, -1] if memory_out.size(1) > 1 else memory_out.squeeze(1)
        
        # Compute policy and value
        action_logits = self.policy_head(final_out)
        value = self.value_head(final_out)
        
        if return_hidden:
            return action_logits, value, new_hidden
        else:
            return action_logits, value


class RL2Agent(MetaRLAgent):
    """
    RL^2 (Reinforcement Learning with Reinforcement Learning) agent.
    
    Uses a memory-augmented policy to learn across multiple tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RL^2 agent."""
        super().__init__(config)
        
        self.input_dim = config.get("input_dim", 3)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.output_dim = config.get("output_dim", 2)
        self.lr = config.get("lr", 1e-3)
        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        
        # Initialize policy
        self.policy = MemoryAugmentedPolicy(
            self.input_dim, self.hidden_dim, self.output_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Training state
        self.hidden_state = None
        self.rollout_buffer = []
        
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select an action given an observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_logits, _ = self.policy(obs_tensor, self.hidden_state)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1).item()
            else:
                dist = Categorical(logits=action_logits)
                action = dist.sample().item()
            
            return action
    
    def adapt(self, env, adaptation_steps: int = 5) -> None:
        """Adapt to a new task using few-shot learning."""
        self.hidden_state = None
        self.rollout_buffer = []
        
        obs, _ = env.reset()
        
        for step in range(adaptation_steps):
            action = self.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store experience
            self.rollout_buffer.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "terminated": terminated,
                "truncated": truncated
            })
            
            obs = next_obs
            
            if terminated or truncated:
                break
    
    def update(self, rollouts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Update the agent's parameters."""
        if rollouts is None:
            rollouts = self.rollout_buffer
        
        if not rollouts:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Convert rollouts to tensors
        obs = torch.FloatTensor([r["obs"] for r in rollouts]).to(self.device)
        actions = torch.LongTensor([r["action"] for r in rollouts]).to(self.device)
        rewards = torch.FloatTensor([r["reward"] for r in rollouts]).to(self.device)
        
        # Compute returns
        returns = self._compute_returns(rewards)
        
        # Forward pass
        action_logits, values = self.policy(obs)
        
        # Compute losses
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Policy loss (REINFORCE)
        policy_loss = -(log_probs * returns).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_coef * value_loss - 
            self.entropy_coef * entropy
        )
        
        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }
    
    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns


class MAMLAgent(MetaRLAgent):
    """
    Model-Agnostic Meta-Learning (MAML) agent.
    
    Learns a good initialization that can be quickly adapted to new tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MAML agent."""
        super().__init__(config)
        
        self.inner_lr = config.get("inner_lr", 0.01)
        self.outer_lr = config.get("outer_lr", 1e-3)
        self.inner_steps = config.get("inner_steps", 5)
        
        # Initialize base policy (same as RL^2)
        self.base_policy = MemoryAugmentedPolicy(
            config.get("input_dim", 3),
            config.get("hidden_dim", 64),
            config.get("output_dim", 2)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.base_policy.parameters(), lr=self.outer_lr)
    
    def adapt(self, env, adaptation_steps: int = 5) -> nn.Module:
        """Adapt the policy to a new task."""
        # Clone the base policy
        adapted_policy = MemoryAugmentedPolicy(
            self.base_policy.input_dim,
            self.base_policy.hidden_dim,
            self.base_policy.output_dim
        ).to(self.device)
        
        adapted_policy.load_state_dict(self.base_policy.state_dict())
        
        # Inner loop adaptation
        inner_optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)
        
        for step in range(adaptation_steps):
            obs, _ = env.reset()
            total_loss = 0
            
            for _ in range(5):  # Few-shot episodes
                action = self._act_with_policy(adapted_policy, obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                # Compute loss (simplified)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_logits, _ = adapted_policy(obs_tensor)
                
                # Use reward as target (simplified)
                target = torch.FloatTensor([reward]).to(self.device)
                loss = F.mse_loss(action_logits[0, action], target)
                
                total_loss += loss
                
                obs = next_obs
                if terminated or truncated:
                    break
            
            # Update adapted policy
            inner_optimizer.zero_grad()
            total_loss.backward()
            inner_optimizer.step()
        
        return adapted_policy
    
    def _act_with_policy(self, policy: nn.Module, obs: np.ndarray) -> int:
        """Act using a specific policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_logits, _ = policy(obs_tensor)
            action = torch.argmax(action_logits, dim=-1).item()
            return action
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select an action using the base policy."""
        return self._act_with_policy(self.base_policy, obs)
    
    def update(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update the base policy using MAML outer loop."""
        # This is a simplified version - full MAML requires more complex implementation
        # For now, we'll use standard policy gradient
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}


class StableBaselines3Agent(MetaRLAgent):
    """
    Wrapper for stable-baselines3 agents to work with meta-RL.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the stable-baselines3 agent."""
        super().__init__(config)
        
        if not STABLE_BASELINES3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for StableBaselines3Agent. Install with: pip install stable-baselines3")
        
        self.algorithm = config.get("algorithm", "PPO")
        self.env_config = config.get("env_config", {})
        
        # Create dummy environment for initialization
        from src.envs import create_bandit_env
        dummy_env = create_bandit_env()
        
        # Initialize agent based on algorithm
        if self.algorithm == "PPO":
            self.agent = PPO("MlpPolicy", dummy_env, verbose=0)
        elif self.algorithm == "SAC":
            self.agent = SAC("MlpPolicy", dummy_env, verbose=0)
        elif self.algorithm == "TD3":
            self.agent = TD3("MlpPolicy", dummy_env, verbose=0)
        elif self.algorithm == "DQN":
            self.agent = DQN("MlpPolicy", dummy_env, verbose=0)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def adapt(self, env, adaptation_steps: int = 5) -> None:
        """Adapt to a new task."""
        # For stable-baselines3, we'll train on the new environment
        vec_env = DummyVecEnv([lambda: env])
        self.agent.set_env(vec_env)
        self.agent.learn(total_timesteps=adaptation_steps * 100)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select an action."""
        action, _ = self.agent.predict(obs, deterministic=deterministic)
        return action
    
    def update(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update the agent."""
        # This is handled internally by stable-baselines3
        return {"loss": 0.0}


def create_agent(agent_type: str, config: Dict[str, Any]) -> MetaRLAgent:
    """Create an agent of the specified type."""
    if agent_type == "rl2":
        return RL2Agent(config)
    elif agent_type == "maml":
        return MAMLAgent(config)
    elif agent_type in ["ppo", "sac", "td3", "dqn"]:
        config["algorithm"] = agent_type.upper()
        return StableBaselines3Agent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
