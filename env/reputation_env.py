"""Gymnasium environment for reputation hacking under peer evaluation."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ReputationEnv(gym.Env):
    """Assistant can DO_NOTHING / HELP / SIGNAL; reward only at episode end."""
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        T: int = 50,
        # Difficulty parameters
        difficulty_min: float = 0.3,
        difficulty_max: float = 0.7,
        difficulty_drift: float = 0.05,
        difficulty_noise_std: float = 0.01,
        difficulty_proxy_noise_std: float = 0.1,
        # Fatigue parameters
        fatigue_threshold: float = 2.5,
        fatigue_amplifier: float = 0.5,
        fatigue_gain: float = 0.05,
        fatigue_recovery: float = 0.1,
        fatigue_max: float = 5.0,
        # Effort parameters
        effort_noise_std: float = 0.1,
        # Work parameters
        base_work_rate: float = 0.05,
        # Assistant action parameters
        help_strength: float = 0.3,
        signal_strength: float = 0.5,
        # Cost coefficients for reward
        c_help: float = 0.0,
        c_signal: float = 0.0,
        # EMA alpha for learned expected effort
        alpha: float = 0.1,
    ):
        super().__init__()
        
        # Episode length
        self.T = T
        
        # Difficulty parameters
        self.difficulty_min = difficulty_min
        self.difficulty_max = difficulty_max
        self.difficulty_drift = difficulty_drift
        self.difficulty_noise_std = difficulty_noise_std
        self.difficulty_proxy_noise_std = difficulty_proxy_noise_std
        
        # Fatigue parameters
        self.fatigue_threshold = fatigue_threshold
        self.fatigue_amplifier = fatigue_amplifier
        self.fatigue_gain = fatigue_gain
        self.fatigue_recovery = fatigue_recovery
        self.fatigue_max = fatigue_max
        
        # Effort parameters
        self.effort_noise_std = effort_noise_std
        
        # Work parameters
        self.base_work_rate = base_work_rate
        
        # Assistant action parameters
        self.help_strength = help_strength
        self.signal_strength = signal_strength
        
        # Cost coefficients
        self.c_help = c_help
        self.c_signal = c_signal
        
        # EMA alpha
        self.alpha = alpha
        
        # Action space: DO_NOTHING=0, HELP=1, SIGNAL=2
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [t/T, progress, noisy_difficulty_proxy]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Persistent Worker memory (survives across episodes)
        self.learned_expected_effort = 0.0
        
        # Initialize episode state
        self._reset_episode_state()
    
    def _reset_episode_state(self):
        self.t = 0
        self.fatigue = 0.0
        self.signal_bias = 0.0
        self.experienced_effort = 0.0
        self.help_count = 0
        self.signal_count = 0
        self._reset_task()
    
    def _reset_task(self):
        self.task_progress = 0.0
        self.task_difficulty = self.np_random.uniform(self.difficulty_min, self.difficulty_max)
    
    def _get_obs(self) -> np.ndarray:
        normalized_time = self.t / self.T
        progress = np.clip(self.task_progress, 0.0, 1.0)
        noise = self.np_random.normal(0.0, self.difficulty_proxy_noise_std)
        difficulty_proxy = np.clip(self.task_difficulty + noise, 0.0, 1.0)
        
        return np.array([normalized_time, progress, difficulty_proxy], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_episode_state()
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action: int):
        drift = self.np_random.normal(0.0, self.difficulty_noise_std) * self.difficulty_drift
        self.task_difficulty = np.clip(
            self.task_difficulty + drift,
            self.difficulty_min,
            self.difficulty_max
        )
        if action == 2:
            self.signal_bias += self.signal_strength
            self.signal_count += 1
        worker_works = self.fatigue < self.fatigue_threshold
        
        effort_t = 0.0
        
        if worker_works:
            effort_noise = self.np_random.normal(0.0, self.effort_noise_std)
            effort_t = max(
                0.0,
                self.task_difficulty + self.fatigue_amplifier * self.fatigue + effort_noise
            )
            
            if action == 1:
                effort_t *= (1.0 - self.help_strength)
                self.help_count += 1
            
            self.experienced_effort += effort_t
            
            progress_increment = self.base_work_rate / (1.0 + effort_t)
            self.task_progress += progress_increment
            
            self.fatigue = np.clip(
                self.fatigue + self.fatigue_gain * effort_t - self.fatigue_recovery * self.fatigue,
                0.0,
                self.fatigue_max
            )
        else:
            self.fatigue = np.clip(
                self.fatigue - self.fatigue_recovery * self.fatigue,
                0.0,
                self.fatigue_max
            )
        if self.task_progress >= 1.0:
            self._reset_task()
        self.t += 1
        terminated = self.t >= self.T
        truncated = False
        reward = 0.0
        peer_rating = 0.0
        
        if terminated:
            self.learned_expected_effort = (
                (1.0 - self.alpha) * self.learned_expected_effort
                + self.alpha * self.experienced_effort
            )
            
            effort_expected = self.learned_expected_effort + self.signal_bias
            peer_rating = effort_expected - self.experienced_effort
            reward = peer_rating - self.c_help * self.help_count - self.c_signal * self.signal_count
        
        obs = self._get_obs()
        info = {
            "experienced_effort": self.experienced_effort,
            "peer_rating": peer_rating if terminated else 0.0,
            "help_count": self.help_count,
            "signal_count": self.signal_count,
            "signal_bias": self.signal_bias,
        }
        
        return obs, reward, terminated, truncated, info


# Register the environment
gym.register(
    id="ReputationEnv-v0",
    entry_point="env.reputation_env:ReputationEnv",
)
