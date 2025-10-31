#!/usr/bin/env python3
"""
Test script to verify the meta-RL system works correctly.

This script runs basic tests to ensure all components are functioning.
"""

import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from src.envs import create_bandit_env, create_grid_world_env
        from src.agents import create_agent
        from src.utils import Config, MetaRLTrainer
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_environment_creation():
    """Test environment creation."""
    print("\nTesting environment creation...")
    
    try:
        from src.envs import create_bandit_env, create_grid_world_env
        
        # Test bandit environment
        bandit_env = create_bandit_env(n_arms=2)
        obs, info = bandit_env.reset()
        action = 0
        obs, reward, terminated, truncated, info = bandit_env.step(action)
        print("✅ Bandit environment works")
        
        # Test grid world environment
        grid_env = create_grid_world_env(size=3, max_steps=10)
        obs, info = grid_env.reset()
        action = 0
        obs, reward, terminated, truncated, info = grid_env.step(action)
        print("✅ Grid world environment works")
        
        return True
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test agent creation."""
    print("\nTesting agent creation...")
    
    try:
        from src.agents import create_agent
        
        # Test RL^2 agent
        config = {
            "input_dim": 3,
            "hidden_dim": 32,
            "output_dim": 2,
            "lr": 1e-3
        }
        
        agent = create_agent("rl2", config)
        obs = [0.0, 0.0, 0.0]
        action = agent.act(obs)
        print("✅ RL^2 agent works")
        
        # Test MAML agent
        agent = create_agent("maml", config)
        action = agent.act(obs)
        print("✅ MAML agent works")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from src.utils import Config
        
        config = Config()
        config.set("test.value", 42)
        assert config.get("test.value") == 42
        
        config.set("agent.hidden_dim", 128)
        assert config.get("agent.hidden_dim") == 128
        
        print("✅ Configuration system works")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        traceback.print_exc()
        return False


def test_simple_training():
    """Test a simple training loop."""
    print("\nTesting simple training loop...")
    
    try:
        from src.envs import create_bandit_env
        from src.agents import create_agent
        
        # Create environment and agent
        env = create_bandit_env(n_arms=2)
        config = {
            "input_dim": 3,
            "hidden_dim": 16,  # Smaller for quick test
            "output_dim": 2,
            "lr": 1e-3
        }
        agent = create_agent("rl2", config)
        
        # Simple training loop
        for episode in range(5):
            env.sample_task()
            agent.adapt(env, adaptation_steps=2)
            
            rollouts = []
            obs, _ = env.reset()
            for _ in range(2):
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
        
        print("✅ Simple training loop works")
        return True
    except Exception as e:
        print(f"❌ Simple training failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Meta-RL System Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_environment_creation,
        test_agent_creation,
        test_configuration,
        test_simple_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("• Run 'python demo.py' for a quick demonstration")
        print("• Run 'python train.py --help' for training options")
        print("• Run 'streamlit run app.py' for interactive interface")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
