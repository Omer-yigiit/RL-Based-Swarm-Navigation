import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SwarmNavEnv(MultiAgentEnv):
    def __init__(self, config):
        self.num_agents = config.get("num_agents", 4)
        self.arena_size = config.get("arena_size", 10.0)
        self.max_steps = config.get("max_steps", 300)

        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        # ---- OBS & ACTION SPACE (PER AGENT!) ----
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.goal_threshold = 0.5
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.steps = 0

        self.positions = {
            aid: np.random.uniform(-self.arena_size, self.arena_size, size=2)
            for aid in self.agent_ids
        }
        self.velocities = {
            aid: np.zeros(2, dtype=np.float32)
            for aid in self.agent_ids
        }
        self.goals = {
            aid: np.random.uniform(-self.arena_size, self.arena_size, size=2)
            for aid in self.agent_ids
        }

        obs = {
            aid: self._get_obs(aid)
            for aid in self.agent_ids
        }
        return obs, {}

    def step(self, actions):
        self.steps += 1

        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}

        for aid, action in actions.items():
            # --- previous distance ---
            prev_dist = np.linalg.norm(self.positions[aid] - self.goals[aid])

            # --- dynamics ---
            self.velocities[aid] = np.clip(action, -1, 1)
            self.positions[aid] += self.velocities[aid] * 0.1

            # --- new distance ---
            new_dist = np.linalg.norm(self.positions[aid] - self.goals[aid])

            # --- reward ---
            reward = 0.0
            reward += (prev_dist - new_dist) * 5.0
            reward -= 0.01
            reward -= 0.001 * np.linalg.norm(self.velocities[aid])

            done = False
            if new_dist < self.goal_threshold:
                reward += 5.0
                done = True

            obs[aid] = self._get_obs(aid)
            rewards[aid] = reward
            terminated[aid] = done
            truncated[aid] = False
            infos[aid] = {}

        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = self.steps >= self.max_steps

        return obs, rewards, terminated, truncated, infos

    def _get_obs(self, aid):
        return np.concatenate([
            self.positions[aid],
            self.velocities[aid],
            self.goals[aid] - self.positions[aid],
        ]).astype(np.float32)
