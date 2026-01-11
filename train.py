"""Train PPO on ReputationEnv."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.reputation_env import ReputationEnv

# Quick sanity check (10k steps instead of 100k)
QUICK_TEST = False


def main():
    env = DummyVecEnv([lambda: ReputationEnv()])
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
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
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model_path = "trained_model"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
