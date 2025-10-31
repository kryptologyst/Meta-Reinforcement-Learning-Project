"""
Modern environments for meta-reinforcement learning.

This module provides various environments suitable for meta-RL experiments,
including bandit tasks, grid worlds, and classic control environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class MetaRLEnvironment(ABC):
    """Abstract base class for meta-RL environments."""
    
    def __init__(self, task_params: Optional[Dict[str, Any]] = None):
        """Initialize the environment with task parameters."""
        self.task_params = task_params or {}
        self.reset()
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observation."""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        pass
    
    @abstractmethod
    def sample_task(self) -> None:
        """Sample a new task from the task distribution."""
        pass


class OneStepBanditEnv(MetaRLEnvironment):
    """
    One-step bandit environment for meta-RL experiments.
    
    This environment simulates a multi-armed bandit where the agent
    must learn to identify the best arm quickly across different tasks.
    """
    
    def __init__(self, n_arms: int = 2, reward_probs: Optional[List[float]] = None):
        """
        Initialize the bandit environment.
        
        Args:
            n_arms: Number of arms in the bandit
            reward_probs: Fixed reward probabilities (if None, will be sampled)
        """
        self.n_arms = n_arms
        self.reward_probs = reward_probs
        self.action_space = spaces.Discrete(n_arms)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )  # [prev_action, prev_reward, timestep]
        super().__init__()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and sample a new task."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.sample_task()
        self.prev_action = 0.0
        self.prev_reward = 0.0
        self.timestep = 0.0
        
        obs = np.array([self.prev_action, self.prev_reward, self.timestep], dtype=np.float32)
        info = {"reward_probs": self.reward_probs.copy()}
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the bandit environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space {self.action_space}")
        
        # Calculate reward
        reward = 1.0 if np.random.random() < self.reward_probs[action] else 0.0
        
        # Update state
        self.prev_action = float(action)
        self.prev_reward = reward
        self.timestep = min(1.0, self.timestep + 0.2)  # Normalize timestep
        
        obs = np.array([self.prev_action, self.prev_reward, self.timestep], dtype=np.float32)
        terminated = True  # One-step environment
        truncated = False
        info = {"reward_probs": self.reward_probs.copy()}
        
        return obs, reward, terminated, truncated, info
    
    def sample_task(self) -> None:
        """Sample new reward probabilities for the bandit."""
        if self.reward_probs is None:
            # Sample random probabilities
            self.reward_probs = [random.random() for _ in range(self.n_arms)]
        else:
            # Use provided probabilities
            self.reward_probs = self.reward_probs.copy()


class GridWorldEnv(MetaRLEnvironment):
    """
    Grid world environment for meta-RL experiments.
    
    The agent must navigate to a goal in a grid world, with different
    goal locations across tasks.
    """
    
    def __init__(self, size: int = 5, max_steps: int = 50):
        """
        Initialize the grid world environment.
        
        Args:
            size: Size of the grid (size x size)
            max_steps: Maximum number of steps per episode
        """
        self.size = size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(size * size + 2,), dtype=np.float32
        )  # flattened grid + agent_pos
        
        self.goal_pos = None
        self.agent_pos = None
        self.steps = 0
        super().__init__()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and sample a new task."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.sample_task()
        self.agent_pos = [0, 0]  # Start at top-left
        self.steps = 0
        
        obs = self._get_observation()
        info = {"goal_pos": self.goal_pos.copy(), "agent_pos": self.agent_pos.copy()}
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the grid world."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space {self.action_space}")
        
        # Move agent
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        
        self.steps += 1
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            terminated = True
        else:
            reward = -0.01  # Small negative reward for each step
            terminated = False
        
        truncated = self.steps >= self.max_steps
        
        obs = self._get_observation()
        info = {"goal_pos": self.goal_pos.copy(), "agent_pos": self.agent_pos.copy()}
        
        return obs, reward, terminated, truncated, info
    
    def sample_task(self) -> None:
        """Sample a new goal position."""
        self.goal_pos = [
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1)
        ]
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        # Create grid representation
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1.0  # Agent position
        grid[self.goal_pos[0], self.goal_pos[1]] = 0.5    # Goal position
        
        # Flatten grid and add agent position
        obs = np.concatenate([
            grid.flatten(),
            [self.agent_pos[0] / self.size, self.agent_pos[1] / self.size]
        ])
        
        return obs


class MetaRLWrapper(gym.Wrapper):
    """
    Wrapper to convert MetaRLEnvironment to gymnasium format.
    """
    
    def __init__(self, env: MetaRLEnvironment):
        """Initialize the wrapper."""
        # Convert MetaRLEnvironment to gymnasium format first
        gym_env = self._convert_to_gym_env(env)
        super().__init__(gym_env)
        self.meta_env = env
    
    def _convert_to_gym_env(self, env: MetaRLEnvironment):
        """Convert MetaRLEnvironment to gymnasium.Env."""
        class GymEnvWrapper(gym.Env):
            def __init__(self, meta_env):
                self.meta_env = meta_env
                self.action_space = meta_env.action_space
                self.observation_space = meta_env.observation_space
            
            def reset(self, seed=None, options=None):
                return self.meta_env.reset(seed)
            
            def step(self, action):
                return self.meta_env.step(action)
            
            def sample_task(self):
                return self.meta_env.sample_task()
        
        return GymEnvWrapper(env)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        return self.meta_env.reset(seed)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        return self.meta_env.step(action)
    
    def sample_task(self) -> None:
        """Sample a new task."""
        self.meta_env.sample_task()


def create_bandit_env(n_arms: int = 2, reward_probs: Optional[List[float]] = None) -> MetaRLWrapper:
    """Create a bandit environment."""
    return MetaRLWrapper(OneStepBanditEnv(n_arms, reward_probs))


def create_grid_world_env(size: int = 5, max_steps: int = 50) -> MetaRLWrapper:
    """Create a grid world environment."""
    return MetaRLWrapper(GridWorldEnv(size, max_steps))


def create_cartpole_env() -> gym.Env:
    """Create a CartPole environment for comparison."""
    return gym.make("CartPole-v1")


def create_mountain_car_env() -> gym.Env:
    """Create a MountainCar environment for comparison."""
    return gym.make("MountainCar-v0")
