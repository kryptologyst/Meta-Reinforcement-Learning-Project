"""
Unit tests for meta-reinforcement learning components.

This module contains comprehensive tests for environments, agents, and utilities.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.envs import OneStepBanditEnv, GridWorldEnv, MetaRLWrapper, create_bandit_env, create_grid_world_env
from src.agents import RL2Agent, MAMLAgent, MemoryAugmentedPolicy, create_agent
from src.utils import Config, Logger, CheckpointManager


class TestEnvironments:
    """Test cases for environments."""
    
    def test_bandit_env_initialization(self):
        """Test bandit environment initialization."""
        env = OneStepBanditEnv(n_arms=3)
        assert env.n_arms == 3
        assert env.action_space.n == 3
        assert env.observation_space.shape == (3,)
    
    def test_bandit_env_reset(self):
        """Test bandit environment reset."""
        env = OneStepBanditEnv(n_arms=2, reward_probs=[0.7, 0.3])
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (3,)
        assert obs.dtype == np.float32
        assert "reward_probs" in info
        assert info["reward_probs"] == [0.7, 0.3]
    
    def test_bandit_env_step(self):
        """Test bandit environment step."""
        env = OneStepBanditEnv(n_arms=2, reward_probs=[1.0, 0.0])  # Deterministic
        obs, info = env.reset()
        
        # Test action 0 (should always give reward 1)
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 1.0
        assert terminated == True
        assert truncated == False
    
    def test_grid_world_env_initialization(self):
        """Test grid world environment initialization."""
        env = GridWorldEnv(size=5, max_steps=50)
        assert env.size == 5
        assert env.max_steps == 50
        assert env.action_space.n == 4
        assert env.observation_space.shape == (27,)  # 5*5 + 2
    
    def test_grid_world_env_reset(self):
        """Test grid world environment reset."""
        env = GridWorldEnv(size=3, max_steps=10)
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (11,)  # 3*3 + 2
        assert obs.dtype == np.float32
        assert "goal_pos" in info
        assert "agent_pos" in info
        assert info["agent_pos"] == [0, 0]  # Start at top-left
    
    def test_grid_world_env_step(self):
        """Test grid world environment step."""
        env = GridWorldEnv(size=3, max_steps=10)
        obs, info = env.reset()
        
        # Move right
        obs, reward, terminated, truncated, info = env.step(3)
        assert reward == -0.01  # Small negative reward
        assert terminated == False
        assert truncated == False
        assert info["agent_pos"] == [0, 1]  # Moved right
    
    def test_meta_rl_wrapper(self):
        """Test MetaRL wrapper."""
        env = OneStepBanditEnv(n_arms=2)
        wrapped_env = MetaRLWrapper(env)
        
        obs, info = wrapped_env.reset()
        assert obs.shape == (3,)
        
        obs, reward, terminated, truncated, info = wrapped_env.step(0)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


class TestAgents:
    """Test cases for agents."""
    
    def test_memory_augmented_policy_initialization(self):
        """Test memory augmented policy initialization."""
        policy = MemoryAugmentedPolicy(
            input_dim=3,
            hidden_dim=32,
            output_dim=2
        )
        
        assert policy.input_dim == 3
        assert policy.hidden_dim == 32
        assert policy.output_dim == 2
    
    def test_memory_augmented_policy_forward(self):
        """Test memory augmented policy forward pass."""
        policy = MemoryAugmentedPolicy(
            input_dim=3,
            hidden_dim=32,
            output_dim=2
        )
        
        obs = torch.randn(1, 3)
        action_logits, value = policy(obs)
        
        assert action_logits.shape == (1, 2)
        assert value.shape == (1, 1)
    
    def test_rl2_agent_initialization(self):
        """Test RL^2 agent initialization."""
        config = {
            "input_dim": 3,
            "hidden_dim": 32,
            "output_dim": 2,
            "lr": 1e-3
        }
        
        agent = RL2Agent(config)
        assert agent.input_dim == 3
        assert agent.hidden_dim == 32
        assert agent.output_dim == 2
    
    def test_rl2_agent_act(self):
        """Test RL^2 agent action selection."""
        config = {
            "input_dim": 3,
            "hidden_dim": 32,
            "output_dim": 2,
            "lr": 1e-3
        }
        
        agent = RL2Agent(config)
        obs = np.array([0.0, 0.0, 0.0])
        
        action = agent.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < 2
    
    def test_maml_agent_initialization(self):
        """Test MAML agent initialization."""
        config = {
            "input_dim": 3,
            "hidden_dim": 32,
            "output_dim": 2,
            "inner_lr": 0.01,
            "outer_lr": 1e-3
        }
        
        agent = MAMLAgent(config)
        assert agent.inner_lr == 0.01
        assert agent.outer_lr == 1e-3
    
    def test_create_agent(self):
        """Test agent creation function."""
        config = {
            "input_dim": 3,
            "hidden_dim": 32,
            "output_dim": 2
        }
        
        # Test RL^2 agent
        rl2_agent = create_agent("rl2", config)
        assert isinstance(rl2_agent, RL2Agent)
        
        # Test MAML agent
        maml_agent = create_agent("maml", config)
        assert isinstance(maml_agent, MAMLAgent)


class TestUtils:
    """Test cases for utilities."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = Config()
        assert config.get("experiment.name") == "meta_rl_experiment"
        assert config.get("agent.type") == "rl2"
    
    def test_config_set_get(self):
        """Test configuration set and get methods."""
        config = Config()
        
        config.set("test.value", 42)
        assert config.get("test.value") == 42
        
        config.set("agent.hidden_dim", 128)
        assert config.get("agent.hidden_dim") == 128
    
    def test_config_load_save(self, tmp_path):
        """Test configuration loading and saving."""
        config = Config()
        config.set("test.value", 42)
        
        # Save config
        config_path = tmp_path / "test_config.yaml"
        config.save_config(str(config_path))
        
        # Load config
        new_config = Config(str(config_path))
        assert new_config.get("test.value") == 42
    
    def test_logger_initialization(self, tmp_path):
        """Test logger initialization."""
        config = Config()
        config.set("logging.log_dir", str(tmp_path))
        config.set("logging.use_wandb", False)
        config.set("logging.use_tensorboard", False)
        
        logger = Logger(config)
        assert logger.log_dir == tmp_path
        logger.close()
    
    def test_checkpoint_manager(self, tmp_path):
        """Test checkpoint manager."""
        config = Config()
        config.set("logging.save_dir", str(tmp_path))
        
        manager = CheckpointManager(config)
        assert manager.save_dir == tmp_path
        
        # Mock agent for testing
        mock_agent = Mock()
        mock_agent.policy = Mock()
        mock_agent.policy.state_dict.return_value = {"weight": torch.tensor([1.0])}
        
        # Test checkpoint saving
        checkpoint_path = manager.save_checkpoint(
            mock_agent,
            episode=100,
            metrics={"reward": 0.5}
        )
        
        assert Path(checkpoint_path).exists()
        
        # Test checkpoint loading
        checkpoint_data = manager.load_checkpoint(checkpoint_path)
        assert checkpoint_data["episode"] == 100
        assert checkpoint_data["metrics"]["reward"] == 0.5


class TestIntegration:
    """Integration tests."""
    
    def test_bandit_training_loop(self):
        """Test a simple training loop with bandit environment."""
        env = create_bandit_env(n_arms=2)
        config = {
            "input_dim": 3,
            "hidden_dim": 32,
            "output_dim": 2,
            "lr": 1e-3
        }
        agent = create_agent("rl2", config)
        
        # Simple training loop
        for episode in range(10):
            env.sample_task()
            agent.adapt(env, adaptation_steps=3)
            
            rollouts = []
            obs, _ = env.reset()
            for _ in range(3):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                rollouts.append({
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
            
            # Update agent
            loss_metrics = agent.update(rollouts)
            assert isinstance(loss_metrics, dict)
            assert "loss" in loss_metrics
    
    def test_grid_world_training_loop(self):
        """Test a simple training loop with grid world environment."""
        env = create_grid_world_env(size=3, max_steps=10)
        config = {
            "input_dim": 11,  # 3*3 + 2
            "hidden_dim": 32,
            "output_dim": 4,
            "lr": 1e-3
        }
        agent = create_agent("rl2", config)
        
        # Simple training loop
        for episode in range(5):
            env.sample_task()
            agent.adapt(env, adaptation_steps=3)
            
            rollouts = []
            obs, _ = env.reset()
            for _ in range(3):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                rollouts.append({
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
            
            # Update agent
            loss_metrics = agent.update(rollouts)
            assert isinstance(loss_metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
