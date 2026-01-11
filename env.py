import gymnasium as gym
import numpy as np

class ReputationEnv(gym.Env):
    def __init__(self):
        super().__init__()

        #total timesteps
        self.T_max = 50

        #assistant action modifiers
        self.help_strength = 0.3
        self.signal_strength = 2.0
        
        #worker action modifiers
        self.base_work_rate = 0.05

        #fatigue stuff
        self.fatigue_amplifier = 0.5
        self.fatigue_rate = 0.05
        self.fatigue_recovery = 0.1
        self.fatigue_threshold = 2.5

        #difficulty modifiers
        self.difficulty_drift = .05
        self.difficulty_noise = .01

        self.effort_noise = 0.1

        self.alpha = 0.1

        #action space
        # 0=Nothing, 1=Help, 2=Signal
        self.action_space = gym.spaces.Discrete(3)

        #observation space
        # normalized time, task_progress, fatigue
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )


        #persistant worker memory
        self.learned_expected_effort = 10.0

        self.reset()


    def reset(self, seed=None):
        super().reset(seed=seed)


        #episode state
        self.t = 0
        self.fatigue = 0.0
        self.signal_bias = 0.0
        self.effort_experienced = 0.0

        obs = self._get_obs()
        return obs, {}


    def _reset_task(self):
        self.task_progress = 0.0
        self.task_difficulty = np.random.uniform(0.3, 0.7)


    def step(self, action):
        reward = 0.0

        #worker policy, 0=REST, 1=WORK
        worker_action = 1 if self.fatigue < self.fatigue_threshold else 0


        #difficulty drift
        drift_noise = np.random.normal(0, self.difficulty_noise)
        self.task_difficulty += self.difficulty_drift * drift_noise
        self.task_difficulty = np.clip(self.task_difficulty, 0.0, 1.0)

        #assistant SIGNAL
        if action == 2:
            self.signal_bias += self.signal_strength

        #worker WORK
        if worker_action == 1:

            #effort calculation
            effort_noise = np.random.normal(0, self.effort_noise)

            effort_t = (
                self.task_difficulty + self.fatigue * self.fatigue_amplifier + effort_noise
            )

            #assistant HELP
            if action == 1:
                effort_t *= (1-self.help_strength)

            #accumulate effort & fatigue
            self.effort_experienced += effort_t
            self.fatigue += self.fatigue_rate * effort_t

            #progress updates
            progress_increments = self.base_work_rate / (1.0 + effort_t)
            self.task_progress += progress_increments

        #worker REST
        else:
            effort_t = 0.0
            self.fatigue = max(0.0, self.fatigue - self.fatigue_recovery)

        if self.task_progress >= 1.0:
            self._reset_task()

        self.t += 1
        terminated = self.t >= self.T_max

        if terminated:
            effort_expected = self.learned_expected_effort + self.signal_bias
            peer_rating = effort_expected - self.effort_experienced
            reward = peer_rating

            self.learned_expected_effort = (
                (1 - self.alpha) * self.learned_expected_effort
                + self.alpha * self.effort_experienced
            )

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return np.array(
            [self.t / self.T_max, self.progress, self.fatigue],
            dtype=np.float32
        )
    

