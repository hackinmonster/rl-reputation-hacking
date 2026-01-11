import functools
import json
from pathlib import Path

def log_step_full(path="step_trace.jsonl"):
    path = Path(path)

    def decorator(step_fn):
        @functools.wraps(step_fn)
        def wrapper(self, action):
            # Call original step
            obs, reward, terminated, truncated, info = step_fn(self, action)

            # Build full state dump
            record = {
                "t": self.t,
                "action": int(action),
                "worker_works": bool(self.fatigue < self.fatigue_threshold),
                "terminated": terminated,

                # Task state
                "task_difficulty": float(self.task_difficulty),
                "task_progress": float(self.task_progress),

                # Effort & fatigue
                "experienced_effort": float(self.experienced_effort),
                "fatigue": float(self.fatigue),

                # Reputation dynamics
                "signal_bias": float(self.signal_bias),
                "signal_count": int(self.signal_count),
                "help_count": int(self.help_count),

                # Evaluator memory
                "learned_expected_effort": float(self.learned_expected_effort),

                # Reward
                "reward": float(reward),

                # Raw info dict (if anything extra)
                "info": info,
            }

            # Add episode-end fields
            if terminated:
                effort_expected = self.learned_expected_effort + self.signal_bias
                record.update({
                    "effort_expected": float(effort_expected),
                    "peer_rating": float(info.get("peer_rating", 0.0)),
                    "EPISODE_END": True
                })

            # Append as JSON line
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")

            return obs, reward, terminated, truncated, info

        return wrapper
    return decorator
