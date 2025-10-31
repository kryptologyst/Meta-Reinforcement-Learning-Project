"""
Streamlit interface for meta-reinforcement learning experiments.

This module provides an interactive web interface for training meta-RL agents,
visualizing results, and comparing different algorithms.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import time
import threading
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import Config, MetaRLTrainer, create_default_config
from src.agents import create_agent
from src.envs import create_bandit_env, create_grid_world_env, create_cartpole_env


# Page configuration
st.set_page_config(
    page_title="Meta-Reinforcement Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitTrainer:
    """Streamlit-compatible trainer for real-time updates."""
    
    def __init__(self, config: Config, progress_bar, status_text):
        self.config = config
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.trainer = None
        self.is_training = False
        self.training_data = {
            "episodes": [],
            "rewards": [],
            "losses": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": []
        }
    
    def start_training(self):
        """Start training in a separate thread."""
        self.is_training = True
        self.trainer = MetaRLTrainer(self.config)
        
        # Override trainer's logging methods to update Streamlit
        original_log_info = self.trainer.logger.log_info
        original_log_scalar = self.trainer.logger.log_scalar
        
        def log_info_with_streamlit(message: str):
            original_log_info(message)
            self.status_text.text(message)
        
        def log_scalar_with_streamlit(name: str, value: float, step: int):
            original_log_scalar(name, value, step)
            
            # Update training data
            if name == "reward/episode_reward":
                self.training_data["episodes"].append(step)
                self.training_data["rewards"].append(value)
            elif name == "loss/total_loss":
                self.training_data["losses"].append(value)
            elif name == "loss/policy_loss":
                self.training_data["policy_losses"].append(value)
            elif name == "loss/value_loss":
                self.training_data["value_losses"].append(value)
            elif name == "loss/entropy":
                self.training_data["entropies"].append(value)
            
            # Update progress bar
            episodes = self.config.get("training.episodes", 3000)
            progress = min(step / episodes, 1.0)
            self.progress_bar.progress(progress)
        
        self.trainer.logger.log_info = log_info_with_streamlit
        self.trainer.logger.log_scalar = log_scalar_with_streamlit
        
        # Start training thread
        training_thread = threading.Thread(target=self._train_loop)
        training_thread.daemon = True
        training_thread.start()
    
    def _train_loop(self):
        """Training loop that runs in a separate thread."""
        try:
            self.trainer.train()
            self.is_training = False
            self.status_text.text("Training completed!")
        except Exception as e:
            self.is_training = False
            self.status_text.text(f"Training failed: {str(e)}")
    
    def stop_training(self):
        """Stop training."""
        self.is_training = False
        if self.trainer:
            self.trainer.logger.close()


def create_config_from_sidebar() -> Config:
    """Create configuration from sidebar inputs."""
    config = create_default_config()
    
    # Experiment settings
    config.set("experiment.name", st.sidebar.text_input("Experiment Name", "meta_rl_experiment"))
    config.set("experiment.seed", st.sidebar.number_input("Random Seed", value=42, min_value=0))
    
    # Agent settings
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        ["rl2", "maml", "ppo", "sac", "td3", "dqn"],
        index=0
    )
    config.set("agent.type", agent_type)
    
    config.set("agent.hidden_dim", st.sidebar.number_input("Hidden Dimension", value=64, min_value=8, max_value=512))
    config.set("agent.lr", st.sidebar.number_input("Learning Rate", value=1e-3, min_value=1e-6, max_value=1e-1, format="%.2e"))
    
    # Environment settings
    env_type = st.sidebar.selectbox(
        "Environment Type",
        ["bandit", "grid_world", "cartpole"],
        index=0
    )
    config.set("env.type", env_type)
    
    if env_type == "bandit":
        config.set("env.n_arms", st.sidebar.number_input("Number of Arms", value=2, min_value=2, max_value=10))
    elif env_type == "grid_world":
        config.set("env.size", st.sidebar.number_input("Grid Size", value=5, min_value=3, max_value=10))
        config.set("env.max_steps", st.sidebar.number_input("Max Steps", value=50, min_value=10, max_value=200))
    
    # Training settings
    config.set("training.episodes", st.sidebar.number_input("Episodes", value=3000, min_value=100, max_value=10000))
    config.set("training.adaptation_steps", st.sidebar.number_input("Adaptation Steps", value=5, min_value=1, max_value=20))
    config.set("training.eval_frequency", st.sidebar.number_input("Evaluation Frequency", value=100, min_value=10, max_value=1000))
    
    # Logging settings
    config.set("logging.use_wandb", st.sidebar.checkbox("Use Weights & Biases", value=False))
    config.set("logging.use_tensorboard", st.sidebar.checkbox("Use TensorBoard", value=True))
    
    return config


def plot_training_curves(training_data: Dict[str, List[float]]) -> plt.Figure:
    """Create training curve plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    episodes = training_data["episodes"]
    
    if episodes:
        # Reward curve
        axes[0, 0].plot(episodes, training_data["rewards"], alpha=0.7)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Smoothed reward curve
        if len(training_data["rewards"]) > 10:
            window_size = min(50, len(training_data["rewards"]) // 10)
            smoothed_rewards = np.convolve(
                training_data["rewards"],
                np.ones(window_size) / window_size,
                mode='valid'
            )
            smoothed_episodes = episodes[window_size-1:]
            axes[0, 1].plot(smoothed_episodes, smoothed_rewards, color='orange', linewidth=2)
            axes[0, 1].set_title(f"Smoothed Rewards (window={window_size})")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Smoothed Reward")
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss curves
        if training_data["losses"]:
            axes[1, 0].plot(episodes[:len(training_data["losses"])], training_data["losses"], label="Total Loss", alpha=0.7)
            if training_data["policy_losses"]:
                axes[1, 0].plot(episodes[:len(training_data["policy_losses"])], training_data["policy_losses"], label="Policy Loss", alpha=0.7)
            if training_data["value_losses"]:
                axes[1, 0].plot(episodes[:len(training_data["value_losses"])], training_data["value_losses"], label="Value Loss", alpha=0.7)
            axes[1, 0].set_title("Training Losses")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        if training_data["entropies"]:
            axes[1, 1].plot(episodes[:len(training_data["entropies"])], training_data["entropies"], color='green', alpha=0.7)
            axes[1, 1].set_title("Policy Entropy")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Entropy")
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Meta-Reinforcement Learning</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    config = create_config_from_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training")
        
        # Training controls
        col_train1, col_train2, col_train3 = st.columns(3)
        
        with col_train1:
            start_training = st.button("🚀 Start Training", key="start")
        
        with col_train2:
            stop_training = st.button("⏹️ Stop Training", key="stop")
        
        with col_train3:
            reset_training = st.button("🔄 Reset", key="reset")
        
        # Training status
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Initialize session state
        if 'trainer' not in st.session_state:
            st.session_state.trainer = None
        if 'training_data' not in st.session_state:
            st.session_state.training_data = {
                "episodes": [],
                "rewards": [],
                "losses": [],
                "policy_losses": [],
                "value_losses": [],
                "entropies": []
            }
        
        # Handle training controls
        if start_training and st.session_state.trainer is None:
            st.session_state.trainer = StreamlitTrainer(config, progress_bar, status_text)
            st.session_state.trainer.start_training()
            st.rerun()
        
        if stop_training and st.session_state.trainer:
            st.session_state.trainer.stop_training()
            st.session_state.trainer = None
            st.rerun()
        
        if reset_training:
            st.session_state.trainer = None
            st.session_state.training_data = {
                "episodes": [],
                "rewards": [],
                "losses": [],
                "policy_losses": [],
                "value_losses": [],
                "entropies": []
            }
            progress_bar.progress(0)
            status_text.text("Ready to train")
            st.rerun()
        
        # Training curves
        st.header("Training Progress")
        
        if st.session_state.trainer and st.session_state.trainer.training_data["episodes"]:
            training_data = st.session_state.trainer.training_data
        else:
            training_data = st.session_state.training_data
        
        if training_data["episodes"]:
            fig = plot_training_curves(training_data)
            st.pyplot(fig)
        else:
            st.info("Start training to see progress curves")
    
    with col2:
        st.header("Metrics")
        
        if training_data["episodes"]:
            # Current metrics
            if training_data["rewards"]:
                latest_reward = training_data["rewards"][-1]
                avg_reward = np.mean(training_data["rewards"][-10:]) if len(training_data["rewards"]) >= 10 else np.mean(training_data["rewards"])
                
                st.metric("Latest Reward", f"{latest_reward:.3f}")
                st.metric("Avg Reward (10)", f"{avg_reward:.3f}")
            
            if training_data["losses"]:
                latest_loss = training_data["losses"][-1]
                st.metric("Latest Loss", f"{latest_loss:.4f}")
            
            if training_data["entropies"]:
                latest_entropy = training_data["entropies"][-1]
                st.metric("Latest Entropy", f"{latest_entropy:.4f}")
            
            # Episode count
            st.metric("Episodes Completed", len(training_data["episodes"]))
        else:
            st.info("No training data available")
        
        # Configuration summary
        st.header("Configuration")
        st.json({
            "Agent": config.get("agent.type"),
            "Environment": config.get("env.type"),
            "Episodes": config.get("training.episodes"),
            "Hidden Dim": config.get("agent.hidden_dim"),
            "Learning Rate": config.get("agent.lr")
        })
    
    # Auto-refresh during training
    if st.session_state.trainer and st.session_state.trainer.is_training:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
