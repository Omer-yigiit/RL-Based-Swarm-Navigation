import numpy as np
import math
import random
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SwarmNavEnv(MultiAgentEnv):
    """
    Swarm-grade multi-agent navigation environment.
    - Continuous control
    - Hard obstacle collisions
    - Local communication (positions + velocities)
    - MAPPO compatible
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.num_agents = config.get("num_agents", 4)
        self.max_steps = config.get("max_steps", 300)
        self.arena_size = config.get("arena_size", 10.0)

        self.agent_radius = 0.25
        self.obstacle_radius = 0.6
        self.n_obstacles = 5

        self.dt = 0.1
        self.max_speed = 1.5
        self.comm_radius = 3.0
        self.max_neighbors = 2

        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # === Observation Space ===
        obs_dim = (
            2 +   # self position
            2 +   # goal position
            2 +   # self velocity
            3 +   # nearest obstacle (dir_x, dir_y, dist)
            self.max_neighbors * 4  # neighbor rel_pos (2) + rel_vel (2)
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # === Action Space (acceleration) ===
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0

        self.positions = np.random.uniform(
            -self.arena_size / 2,
            self.arena_size / 2,
            size=(self.num_agents, 2)
        ).astype(np.float32)

        self.velocities = np.zeros((self.num_agents, 2), dtype=np.float32)

        self.goal = np.random.uniform(
            -self.arena_size / 2,
            self.arena_size / 2,
            size=(2,)
        ).astype(np.float32)

        self.obstacles = []
        for _ in range(self.n_obstacles):
            ox = random.uniform(-self.arena_size / 2, self.arena_size / 2)
            oy = random.uniform(-self.arena_size / 2, self.arena_size / 2)
            self.obstacles.append([ox, oy, self.obstacle_radius])

        obs = {a: self._get_obs(i) for i, a in enumerate(self.agents)}
        return obs, {}

    # ------------------------------------------------------------

    def _get_obs(self, idx):
        sx, sy = self.positions[idx]
        vx, vy = self.velocities[idx]

        # ---- nearest obstacle ----
        min_dist = 1e9
        obs_dir = np.zeros(2)
        obs_dist = 0.0

        for ox, oy, r in self.obstacles:
            vec = np.array([ox - sx, oy - sy])
            d = np.linalg.norm(vec) - r
            if d < min_dist:
                min_dist = d
                obs_dir = vec / (np.linalg.norm(vec) + 1e-6)
                obs_dist = d

        # ---- neighbor communication ----
        neighbors = []
        for j in range(self.num_agents):
            if j == idx:
                continue
            rel_pos = self.positions[j] - self.positions[idx]
            d = np.linalg.norm(rel_pos)
            if d < self.comm_radius:
                rel_vel = self.velocities[j] - self.velocities[idx]
                neighbors.append([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]])

        neighbors = neighbors[:self.max_neighbors]
        while len(neighbors) < self.max_neighbors:
            neighbors.append([0, 0, 0, 0])

        neighbors = np.array(neighbors).flatten()

        obs = np.concatenate([
            self.positions[idx],
            self.goal,
            self.velocities[idx],
            obs_dir,
            np.array([obs_dist]),
            neighbors
        ])

        return obs.astype(np.float32)

    # ------------------------------------------------------------

    def step(self, action_dict):
        self.steps += 1

        prev_positions = self.positions.copy()
        prev_dist = np.linalg.norm(self.positions - self.goal, axis=1)

        # === dynamics ===
        for i, a in enumerate(self.agents):
            acc = np.clip(action_dict[a], -1.0, 1.0)
            self.velocities[i] += acc * self.dt

            speed = np.linalg.norm(self.velocities[i])
            if speed > self.max_speed:
                self.velocities[i] *= self.max_speed / speed

            self.positions[i] += self.velocities[i] * self.dt

        rewards, obs, terminated, truncated, infos = {}, {}, {}, {}, {}

        for i, a in enumerate(self.agents):
            reward = -0.01
            collided = False

            # ---- obstacle collision (HARD) ----
            for ox, oy, r in self.obstacles:
                d = np.linalg.norm(self.positions[i] - np.array([ox, oy]))
                if d < r + self.agent_radius:
                    self.positions[i] = prev_positions[i]
                    self.velocities[i] *= 0.0
                    reward -= 20.0
                    collided = True

            # ---- agent-agent collision ----
            for j in range(self.num_agents):
                if i == j:
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < 2 * self.agent_radius:
                    self.positions[i] = prev_positions[i]
                    self.velocities[i] *= 0.0
                    reward -= 10.0
                    collided = True

            # ---- goal reward ----
            dist = np.linalg.norm(self.positions[i] - self.goal)
            reward += (prev_dist[i] - dist) * 2.0

            if dist < 0.4:
                reward += 10.0
                terminated[a] = True
            else:
                terminated[a] = False

            rewards[a] = reward
            obs[a] = self._get_obs(i)
            infos[a] = {"collision": collided}

        truncated["__all__"] = self.steps >= self.max_steps
        terminated["__all__"] = all(terminated.values())

        return obs, rewards, terminated, truncated, infos
    
    def render(self):
        import matplotlib.pyplot as plt

        if not hasattr(self, '_fig'):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6,6))

        ax = self._ax
        ax.clear()

        ax.set_xlim(-self.arena_size/ 2, self.arena_size / 2)
        ax.set_ylim(-self.arena_size/ 2, self.arena_size / 2)
        ax.set_aspect("equal")
        ax.set_title(f"Step: {self.steps}")
        # ---- obstacles ----
        for ox, oy, r in self.obstacles:
            circle = plt.Circle((ox, oy), r, color='gray', alpha= 0.6)
            ax.add_artist(circle)
        
        # ---- agents ----
        for i in range(self.num_agents):
            x ,y = self.positions[i]
            vx, vy = self.velocities[i]

            ax.plot(x,y, "bo")
            ax.arrow(
                x, y, vx*0.3, vy*0.3,
                head_width = 0.1, color = "blue"
                 
            )
        # ---- goal ----
        ax.plot(self.goal[0], self.goal[1], "rx", markersize=12, linewidth=3)
        plt.pause(0.001)
    
if __name__ == "__main__":
    import time

    env = SwarmNavEnv({
        "num_agents": 4,
        "max_steps": 300,
        "dt": 0.05,
        "speed": 1.0
    })

    obs, _ = env.reset()

    terminated = {"__all__": False}
    truncated = {"__all__": False}

    print("SwarmNavEnv test başladı...")

    while not terminated["__all__"] and not truncated["__all__"]:
        actions = {
            a: env.action_space.sample()
            for a in env.agents
        }

        obs, rewards, terminated, truncated, infos = env.step(actions)
        env.render()
        time.sleep(0.03)

    print("Test bitti.")

