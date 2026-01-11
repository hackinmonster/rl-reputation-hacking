"""Train PPO on ReputationEnv."""

import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env.reputation_env import ReputationEnv

# Quick sanity check (10k steps instead of 100k)
QUICK_TEST = True


class EpisodeMetricsCallback(BaseCallback):
    """Log one row per finished episode to a CSV file."""

    def __init__(self, csv_path: str = "training_metrics.csv", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.csv_path = csv_path
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "timesteps",
                "episode_reward",
                "peer_rating",
                "experienced_effort",
                "help_count",
                "signal_count",
                "signal_bias",
            ],
        )
        self._writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones") or []
        rewards = self.locals.get("rewards") or []

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            terminal_reward = float(rewards[i]) if i < len(rewards) else 0.0

            self._writer.writerow(
                {
                    "timesteps": int(self.num_timesteps),
                    "episode_reward": terminal_reward,
                    "peer_rating": float(info.get("peer_rating", 0.0)),
                    "experienced_effort": float(info.get("experienced_effort", 0.0)),
                    "help_count": int(info.get("help_count", 0)),
                    "signal_count": int(info.get("signal_count", 0)),
                    "signal_bias": float(info.get("signal_bias", 0.0)),
                }
            )

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def main():
    env = DummyVecEnv([lambda: ReputationEnv()])
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        gamma=0.97,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs={
            "net_arch": [64, 64],
        },
    )
    
    # Set timesteps based on mode
    if QUICK_TEST:
        total_timesteps = 10_000
        print("*** QUICK_TEST MODE: Training for only 10,000 timesteps ***")
    else:
        total_timesteps = 100_000
    
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    callback = EpisodeMetricsCallback(csv_path="training_metrics.csv")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the trained model
    model_path = "trained_model"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
