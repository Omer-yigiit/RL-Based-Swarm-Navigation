"""
==============================================================================
 RL REAL-WORLD TRAINER — Swarm Multi-Agent Live GPU Training
==============================================================================
 Trains 4 Swarm Robots simultaneously with a shared policy (multi-agent).

 3-PHASE OPERATION:
   ┌─────────────┐    [T]    ┌─────────────┐    [E]    ┌─────────────┐
   │    SETUP    │ ────────► │  PLANNING   │ ────────► │  EXECUTION  │
   │ Save start  │           │ Virtual sim │           │ Real motors │
   │  positions  │           │  No motors  │           │  per-robot  │
   └─────────────┘           └─────────────┘           └─────────────┘
               Press [H] at any time to return to SETUP
==============================================================================
"""

import cv2
import numpy as np
import threading
import time
import math
import serial
import os
import random
import csv
from datetime import datetime
from collections import deque
from pupil_apriltags import Detector

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
from torch.distributions import Normal

# ==============================================================================
#  LOGGING & COLOR SYSTEM
# ==============================================================================
class Log:
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BLUE   = '\033[94m'
    RESET  = '\033[0m'
    BOLD   = '\033[1m'

    @staticmethod
    def success(msg):  print(f"{Log.GREEN}[SUCCESS]{Log.RESET} {msg}")
    @staticmethod
    def error(msg):    print(f"{Log.RED}[ERROR]{Log.RESET} {msg}")
    @staticmethod
    def warning(msg):  print(f"{Log.YELLOW}[WARNING]{Log.RESET} {msg}")
    @staticmethod
    def info(msg):     print(f"{Log.CYAN}[INFO]{Log.RESET} {msg}")
    @staticmethod
    def plan(msg):     print(f"{Log.BLUE}[PLAN]{Log.RESET} {msg}")
    @staticmethod
    def training(ep, step, rew, rid, ls, rs):
        print(f"{Log.CYAN}[TRAIN]{Log.RESET} R:{rid} | Ep:{Log.BOLD}{ep}{Log.RESET} "
              f"Step:{step} | Reward:{Log.GREEN}{rew:.2f}{Log.RESET} | Motor L:{ls} R:{rs}")

# ==============================================================================
#  PPO (PROXIMAL POLICY OPTIMIZATION) LEARNING ENGINE
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Log.info(f"PyTorch Device: {device}")

class RolloutBuffer:
    def __init__(self):
        self.actions      = []
        self.states       = []
        self.logprobs     = []
        self.rewards      = []
        self.is_terminals = []
        self.returns      = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.returns[:]

    def compute_returns(self, gamma):
        discounted_reward = 0
        returns = []
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        self.returns = returns


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=16, action_dim=2, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean   = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic       = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features    = self.shared(obs)
        action_mean = self.actor_mean(features)
        action_std  = self.actor_logstd.exp().expand_as(action_mean)
        value       = self.critic(features)
        return action_mean, action_std, value

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_mean, action_std, _ = self.forward(state)
        dist          = Normal(action_mean, action_std)
        action        = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy()[0], action_logprob.detach().cpu().numpy()[0]

    def evaluate(self, state, action):
        action_mean, action_std, state_value = self.forward(state)
        dist            = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy    = dist.entropy().sum(dim=-1)
        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.gamma    = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy     = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer  = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state  = np.array(state, dtype=np.float32)
        action, action_logprob = self.policy_old.act(state)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return np.clip(action, -1.0, 1.0)

    def select_action_deterministic(self, state):
        """Returns mean action only — no exploration (used in EXECUTION phase)."""
        state = np.array(state, dtype=np.float32)
        with torch.no_grad():
            state_t     = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_mean, _, _ = self.policy(state_t)
            action      = action_mean.cpu().numpy()[0]
        return np.clip(action, -1.0, 1.0)

    def update(self, memory):
        rewards = torch.tensor(memory.returns, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states   = torch.tensor(np.array(memory.states)).to(device)
        old_actions  = torch.tensor(np.array(memory.actions)).to(device)
        old_logprobs = torch.tensor(np.array(memory.logprobs)).to(device)

        total_loss = 0
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            advantages   = rewards - state_values.detach()
            ratios       = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss  = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.mean().item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return total_loss / self.K_epochs

# ==============================================================================
#  CAMERA & HARDWARE SETUP
# ==============================================================================
class LatencyFreeCamera:
    def __init__(self, source):
        self.cap     = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok, self.frame = self.cap.read()
        self.running = True
        if self.cap.isOpened():
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        else:
            self.running = False

    def _update(self):
        while self.running:
            self.ok, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


at_detector   = Detector(families='tag36h11', nthreads=6)
LOGIC_GRID_MAX = 400.0
ROBOTS         = [0, 1, 2, 3]
ROBOT_COLORS   = {0: (255, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0), 3: (255, 255, 0)}

# Arena boundary (trapezoid / perspective-corrected)
WALL_POLYGON = [
    (350.0,  50.0),
    (350.0, 350.0),
    ( 50.0, 290.0),
    ( 50.0, 110.0)
]
WALL_MIN_X = min(p[0] for p in WALL_POLYGON)
WALL_MAX_X = max(p[0] for p in WALL_POLYGON)
WALL_MIN_Y = min(p[1] for p in WALL_POLYGON)
WALL_MAX_Y = max(p[1] for p in WALL_POLYGON)

def is_point_in_polygon(px, py):
    inside = False
    n = len(WALL_POLYGON)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = WALL_POLYGON[i]
        xj, yj = WALL_POLYGON[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-6) + xi):
            inside = not inside
    return inside

# RL Environment constants
ARENA_SIZE = 3.0
POS_SCALE  = ARENA_SIZE / 2.0
MAX_DIST   = ARENA_SIZE * 1.41
RL_MOTOR_SCALE = 20

virtual_target    = None
virtual_obstacles = []
EMA_ALPHA         = 0.3
WAITING_FOR_TARGET = False   # True after [T] is pressed — waiting for mouse click

# Track which robots have reached the goal (separate sets for each phase)
plan_robots_at_goal      = set()   # Virtual robots that reached goal during PLANNING
execution_robots_at_goal = set()   # Real robots that reached goal during EXECUTION


def generate_environment(only_obstacles=False):
    """Regenerates obstacles (and optionally the target)."""
    global virtual_target, virtual_obstacles
    virtual_obstacles.clear()

    active_positions = [(robot_states[rid]["nx"], robot_states[rid]["ny"])
                        for rid in ROBOTS if robot_states[rid]["nx"] is not None]

    # Generate target only if forced (only_obstacles=False) or if no target exists
    if not only_obstacles or virtual_target is None:
        for _ in range(100):
            tx = random.uniform(WALL_MIN_X + 20, WALL_MAX_X - 20)
            ty = random.uniform(WALL_MIN_Y + 20, WALL_MAX_Y - 20)
            if not is_point_in_polygon(tx, ty):
                continue
            ok = all(math.sqrt((tx - rx)**2 + (ty - ry)**2) > 80
                     for rx, ry in active_positions)
            if ok:
                virtual_target = (tx, ty)
                break
        if virtual_target is None:
            virtual_target = (200.0, 200.0)

    tx, ty = virtual_target

    # ── Obstacle generation ────────────────────────────────────────────────────
    # Small obstacles (r=5-10, max 2) — placed so they don't block the direct path
    N_OBSTACLES = 2
    for _ in range(N_OBSTACLES):
        for _ in range(80):
            ox = random.uniform(WALL_MIN_X + 20, WALL_MAX_X - 20)
            oy = random.uniform(WALL_MIN_Y + 20, WALL_MAX_Y - 20)
            if not is_point_in_polygon(ox, oy):
                continue
            r = random.uniform(5, 10)   # Small obstacle radius

            # Must not be too close to the target
            if math.sqrt((ox - tx)**2 + (oy - ty)**2) < (r + 70):
                continue

            # Must not be too close to any robot
            robot_conflict = any(
                math.sqrt((ox - rx)**2 + (oy - ry)**2) < (r + 45)
                for rx, ry in active_positions
            )
            if robot_conflict:
                continue

            # Must not block the direct path from any robot to the target
            # (point-to-line-segment distance check)
            blocks_path = False
            for rx, ry in active_positions:
                seg_len = math.sqrt((tx - rx)**2 + (ty - ry)**2)
                if seg_len < 1:
                    continue
                cross = abs((oy - ry) * (tx - rx) - (ox - rx) * (ty - ry)) / seg_len
                dot   = ((ox - rx) * (tx - rx) + (oy - ry) * (ty - ry)) / (seg_len * seg_len)
                if 0.15 <= dot <= 0.85 and cross < (r + 25):
                    blocks_path = True
                    break
            if blocks_path:
                continue

            # Must not overlap other obstacles
            too_close = any(
                math.sqrt((vox - ox)**2 + (voy - oy)**2) < (r + vr + 25)
                for vox, voy, vr in virtual_obstacles
            )
            if too_close:
                continue

            virtual_obstacles.append((ox, oy, r))
            break


def mouse_callback(event, x, y, flags, param):
    global virtual_target, fw, fh, WAITING_FOR_TARGET
    if event == cv2.EVENT_LBUTTONDOWN:
        if 'fw' not in globals() or 'fh' not in globals():
            return
        lx = (x / fw) * LOGIC_GRID_MAX
        ly = (y / fh) * LOGIC_GRID_MAX
        if not is_point_in_polygon(lx, ly):
            Log.warning("Selected point is outside the arena! Click inside the boundary.")
            return
        virtual_target = (lx, ly)
        generate_environment(only_obstacles=True)
        Log.info(f"Target set: ({lx:.1f}, {ly:.1f})")

        if WAITING_FOR_TARGET:
            # Target selected after [T] — now start PLANNING
            WAITING_FOR_TARGET = False
            _start_planning()


robot_states = {rid: {
    "nx": None, "ny": None, "angle": None, "base_angle": None,
    "corners": None, "logic_corners": [], "radius": 15.0,
    "last_seen_time": time.time(), "RL_STATE": "COMPUTE",
    "action_send_time": 0, "current_action": [0.0, 0.0],
    "ep_reward": 0, "ep_steps": 0, "prev_d": None,
    "wall_bounce_target": (200.0, 200.0), "episode": 1
} for rid in ROBOTS}


def get_calibrated_angle(rid, raw_angle):
    """
    Converts the raw AprilTag angle to the robot's forward direction.
    The dx/dy formula (corners[3+2] - corners[0+1]) gives the backward direction;
    adding +180° corrects this to the forward direction.
    No relative base_angle calibration — absolute [0,360] range used,
    consistent with PLANNING virtual angles (also [0,360]).
    """
    return (raw_angle + 180) % 360

# ==============================================================================
#  OBSERVATION, REWARD & VIRTUAL ENVIRONMENT LOGIC
# ==============================================================================

def ray_segment_intersect(px, py, dx, dy, x1, y1, x2, y2):
    s_dx = x2 - x1
    s_dy = y2 - y1
    det  = dx * s_dy - dy * s_dx
    if abs(det) < 1e-6:
        return float('inf')
    diff_x = px - x1
    diff_y = py - y1
    u = (dx * diff_y - dy * diff_x) / det
    t = (s_dx * diff_y - s_dy * diff_x) / det
    if 0.0 <= u <= 1.0 and t > 0.0:
        return t
    return float('inf')


def get_ray_wall_intersection(nx, ny, rad):
    dx, dy = math.cos(rad), math.sin(rad)
    t_min  = float('inf')
    n = len(WALL_POLYGON)
    for i in range(n):
        p1 = WALL_POLYGON[i]
        p2 = WALL_POLYGON[(i + 1) % n]
        t  = ray_segment_intersect(nx, ny, dx, dy, p1[0], p1[1], p2[0], p2[1])
        if t < t_min:
            t_min = t
    return t_min


def compute_virtual_sensors(nx, ny, heading_deg, rid, override_positions=None):
    """
    override_positions: {rid: (nx, ny)} — used in PLANNING phase with virtual positions.
    Returns 8 sensor readings (0=clear, 1=blocked) for 8 directions around the robot.
    """
    sensors       = [0.0] * 8
    sensor_angles = [0, 45, 90, 135, 180, -135, -90, -45]
    max_sens_dist = 60.0

    active_obstacles = list(virtual_obstacles)
    for oid in ROBOTS:
        if oid != rid:
            if override_positions and oid in override_positions:
                onx, ony = override_positions[oid]
                active_obstacles.append((onx, ony, robot_states[oid]["radius"]))
            elif robot_states[oid]["nx"] is not None:
                active_obstacles.append((robot_states[oid]["nx"],
                                         robot_states[oid]["ny"],
                                         robot_states[oid]["radius"]))

    for i, sa in enumerate(sensor_angles):
        rad    = math.radians(heading_deg + sa)
        t_wall = get_ray_wall_intersection(nx, ny, rad)
        if t_wall < max_sens_dist:
            sensors[i] = max(sensors[i], 1.0 - t_wall / max_sens_dist)
        for (ox, oy, r) in active_obstacles:
            dist = math.sqrt((ox - nx)**2 + (oy - ny)**2)
            if dist < max_sens_dist + r:
                angle_to_obs = math.degrees(math.atan2(oy - ny, ox - nx))
                rel_angle    = (angle_to_obs - heading_deg + 180) % 360 - 180
                diff         = abs((rel_angle - sa + 180) % 360 - 180)
                if diff < 25:
                    val = max(0.0, 1.0 - max(0, dist - r) / max_sens_dist)
                    sensors[i] = max(sensors[i], val)

    return sensors


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.sqrt((px - (x1 + t * dx))**2 + (py - (y1 + t * dy))**2)


def _build_obs_reward(rid, nx, ny, angle, prev_dist=None, prev_action=None, override_positions=None):
    """
    Core observation + reward computation.
    nx, ny, angle: robot position (real or virtual).
    Returns: (obs, reward, done, dist_to_target)
    """
    if nx is None or virtual_target is None:
        return None, 0.0, False, None

    px = (nx / LOGIC_GRID_MAX) * ARENA_SIZE - (ARENA_SIZE / 2.0)
    py = (ny / LOGIC_GRID_MAX) * ARENA_SIZE - (ARENA_SIZE / 2.0)
    gx = (virtual_target[0] / LOGIC_GRID_MAX) * ARENA_SIZE - (ARENA_SIZE / 2.0)
    gy = (virtual_target[1] / LOGIC_GRID_MAX) * ARENA_SIZE - (ARENA_SIZE / 2.0)

    dist_logic  = math.sqrt((virtual_target[0] - nx)**2 + (virtual_target[1] - ny)**2)
    dist_metric = min(math.sqrt((gx - px)**2 + (gy - py)**2), MAX_DIST)

    target_angle_rad = math.atan2(gy - py, gx - px)
    heading_rad      = math.radians(angle) if angle is not None else 0.0
    rel_angle        = target_angle_rad - heading_rad
    while rel_angle >  math.pi: rel_angle -= 2 * math.pi
    while rel_angle < -math.pi: rel_angle += 2 * math.pi

    sensors = compute_virtual_sensors(nx, ny, angle if angle is not None else 0.0,
                                      rid, override_positions)
    pa_l = prev_action[0] if prev_action is not None else 0.0
    pa_r = prev_action[1] if prev_action is not None else 0.0

    obs = np.array([
        px / POS_SCALE, py / POS_SCALE, gx / POS_SCALE, gy / POS_SCALE,
        dist_metric / MAX_DIST, rel_angle / math.pi,
        sensors[0], sensors[1], sensors[2], sensors[3],
        sensors[4], sensors[5], sensors[6], sensors[7],
        pa_l, pa_r
    ], dtype=np.float32)
    obs = np.nan_to_num(obs, nan=0.0)

    robot_radius = robot_states[rid]["radius"]
    reward = -0.1
    done   = False

    # Progress reward
    if prev_dist is not None:
        progress = prev_dist - dist_logic
        reward  += progress * 1.0

    # GOAL reached
    if dist_logic < (robot_radius + 15):
        reward += 100.0
        done    = True

    # OBSTACLE collision
    for (ox, oy, r) in virtual_obstacles:
        if math.sqrt((ox - nx)**2 + (oy - ny)**2) < (r + robot_radius - 2):
            reward -= 50.0
            done    = True
            break

    # INTER-ROBOT proximity penalty (virtual or real positions)
    for oid in ROBOTS:
        if oid != rid:
            if override_positions and oid in override_positions:
                onx, ony = override_positions[oid]
            elif robot_states[oid]["nx"] is not None:
                onx, ony = robot_states[oid]["nx"], robot_states[oid]["ny"]
            else:
                continue
            dist_o = math.sqrt((onx - nx)**2 + (ony - ny)**2)
            if dist_o < 35.0:
                reward -= (35.0 - dist_o)
            if dist_o < 22.0:
                reward -= 50.0
                done    = True
                break

    # WALL collision penalty
    if not is_point_in_polygon(nx, ny):
        reward -= 50.0
        done    = True

    return obs, reward, done, dist_logic


def get_observation_and_reward(rid, prev_dist=None, prev_action=None):
    """Compute obs/reward from real camera position (EXECUTION phase)."""
    st = robot_states[rid]
    return _build_obs_reward(rid, st["nx"], st["ny"], st["angle"], prev_dist, prev_action)


def get_virtual_obs_reward(rid, vst, prev_dist=None, prev_action=None, all_vst=None):
    """
    Compute obs/reward from virtual robot position (PLANNING phase).
    all_vst: all virtual robot states (for other robots' virtual positions).
    """
    override = {}
    if all_vst:
        for oid, ovst in all_vst.items():
            if oid != rid:
                override[oid] = (ovst["nx"], ovst["ny"])
    return _build_obs_reward(rid, vst["nx"], vst["ny"], vst["angle"],
                             prev_dist, prev_action, override)

# ==============================================================================
#  VIRTUAL PHYSICS ENGINE (PLANNING PHASE)
# ==============================================================================
SIM_DT          = 0.04    # Virtual step duration (seconds)
SIM_SPEED_SCALE = 26.0    # RL action → pixels/step


def virtual_step(vst, action):
    """
    Applies the RL action to the virtual robot state.
    Sends NO command to real robots.

    Turning convention (matches real differential drive):
      Left motor (action[0]) > Right motor (action[1])  →  turn right  →  angle increases (CW)
      Right motor (action[1]) > Left motor (action[0])  →  turn left   →  angle decreases (CCW)
    """
    ls = float(np.clip(action[0], -1.0, 1.0)) * SIM_SPEED_SCALE
    rs = float(np.clip(action[1], -1.0, 1.0)) * SIM_SPEED_SCALE
    v  = (ls + rs) / 2.0          # Forward speed (pixels/step)
    w  = (ls - rs) / 20.0         # Angular velocity: left>right → turn right → angle increases ✓
    angle_rad     = math.radians(vst["angle"])
    vst["nx"]    += v * math.cos(angle_rad) * SIM_DT
    vst["ny"]    += v * math.sin(angle_rad) * SIM_DT
    vst["angle"]  = (vst["angle"] + math.degrees(w * SIM_DT)) % 360
    # Clamp to arena bounds
    vst["nx"] = float(np.clip(vst["nx"], WALL_MIN_X + 15, WALL_MAX_X - 15))
    vst["ny"] = float(np.clip(vst["ny"], WALL_MIN_Y + 15, WALL_MAX_Y - 15))

# ==============================================================================
#  SERIAL PORT & ROBOT COMMUNICATION
# ==============================================================================
GATEWAY_PORT   = 'COM7'
gateway_serial = None

def connect_serial_port():
    global gateway_serial
    try:
        gateway_serial = serial.Serial(GATEWAY_PORT, 115200, timeout=0.1)
        gateway_serial.dtr = False
        gateway_serial.rts = False
        Log.success(f"Connected to serial port {GATEWAY_PORT}!")
    except:
        pass

def send_robot_command(rid, left, right):
    if gateway_serial and gateway_serial.is_open:
        cmd = f"<{rid},{int(left)},{int(right)}>\n"
        gateway_serial.write(cmd.encode('utf-8'))
        gateway_serial.flush()

connect_serial_port()

def stop_all_robots():
    for rid in ROBOTS:
        send_robot_command(rid, 0, 0)

# ==============================================================================
#  3-PHASE SYSTEM VARIABLES
# ==============================================================================
# SYSTEM_PHASE: 'SETUP' | 'PLANNING' | 'EXECUTION'
SYSTEM_PHASE = 'SETUP'

# Start positions recorded from camera: {rid: (nx, ny, angle)}
start_positions = {}

# Virtual robot states used during PLANNING:
# {rid: {nx, ny, angle, prev_d, action, ep_steps, episode, ep_reward}}
virtual_robot_states = {}

# PLANNING parameters
PLANNING_MIN_EPISODES  = 20     # Minimum episodes before convergence check
PLANNING_MAX_EPISODES  = 500    # Maximum episodes (safety cap if no convergence)
PPO_UPDATE_TIMESTEP    = 1600   # Update PPO every N virtual steps (buffer size)
MAX_EP_STEPS           = 400    # Maximum steps per episode

# Convergence criteria
CONVERGENCE_WINDOW     = 20     # Number of recent episodes for moving average
CONVERGENCE_THRESHOLD  = 60.0   # Avg reward above this → "learned"
CONVERGENCE_STREAK     = 3      # Consecutive PPO updates above threshold needed

planning_total_episodes = 0     # Total episodes completed in PLANNING phase
plan_update_pending     = False
convergence_streak_count = 0    # Consecutive updates above convergence threshold


def _banner(message, color=None):
    """Prints a wide, eye-catching banner to the console."""
    color = color or Log.BOLD
    line  = "═" * 60
    print(f"\n{color}{line}")
    for row in message.split('\n'):
        padding = max(0, (60 - len(row)) // 2)
        print(f"{' ' * padding}{row}")
    print(f"{line}{Log.RESET}\n")


def save_start_positions():
    """Record the current camera position of every visible robot."""
    global start_positions
    start_positions.clear()
    found = 0
    for rid in ROBOTS:
        st = robot_states[rid]
        if st["nx"] is not None:
            start_positions[rid] = (
                st["nx"], st["ny"],
                st["angle"] if st["angle"] is not None else 0.0
            )
            Log.success(f"[SETUP] Robot {rid} start position: "
                        f"({st['nx']:.1f}, {st['ny']:.1f}, {st['angle']}°)")
            found += 1
        else:
            Log.warning(f"[SETUP] Robot {rid} NOT visible — start position not saved!")
    return found


def init_virtual_states():
    """
    Initialise virtual states at the start of PLANNING.
    Positions come from recorded start positions; angles are randomised.
    """
    global virtual_robot_states
    virtual_robot_states.clear()
    for rid in ROBOTS:
        if rid in start_positions:
            nx, ny, _ = start_positions[rid]
        else:
            nx, ny = 200.0, 200.0
        # Randomise heading for training robustness
        angle = random.uniform(0, 360)
        virtual_robot_states[rid] = {
            "nx": nx, "ny": ny, "angle": angle,
            "prev_d": None, "action": [0.0, 0.0],
            "ep_steps": 0, "episode": 0, "ep_reward": 0.0
        }
    Log.plan(f"Virtual states (random heading) created: {list(virtual_robot_states.keys())}")


def enter_planning():
    """
    SETUP → Waiting-for-target transition.
    Saves robot positions, stops motors, then waits for the user to click a target.
    """
    global WAITING_FOR_TARGET

    found = save_start_positions()
    if found == 0:
        Log.error("No robots detected! Planning not started.")
        return
    stop_all_robots()

    WAITING_FOR_TARGET = True
    _banner(
        "✔  SETUP COMPLETE\n"
        "Robot positions saved.\n"
        ">>> LEFT CLICK ON SCREEN TO SELECT TARGET <<<",
        Log.YELLOW
    )
    Log.info("Waiting for target selection... Left-click to set the target.")


def _start_planning():
    """
    Starts the PLANNING phase after the target has been selected.
    Called by mouse_callback.
    """
    global SYSTEM_PHASE, planning_total_episodes, plan_update_pending
    global global_time_step, convergence_streak_count, plan_robots_at_goal

    init_virtual_states()
    planning_total_episodes  = 0
    convergence_streak_count = 0
    plan_update_pending      = False
    plan_robots_at_goal.clear()
    for rm in robot_memories.values():
        rm.clear()
    train_memory.clear()
    global_time_step = 0
    SYSTEM_PHASE = 'PLANNING'
    phase_log.append({"t": time.time() - _plot_start,
                      "from_p": "SETUP", "to_p": "PLANNING", "ep_count": 0})

    _banner(
        f"► PLANNING STARTED\n"
        f"Target: ({virtual_target[0]:.0f}, {virtual_target[1]:.0f})  |  "
        f"Obstacles: {len(virtual_obstacles)}\n"
        f"Convergence: last {CONVERGENCE_WINDOW} ep avg >= {CONVERGENCE_THRESHOLD:.0f} "
        f"(x{CONVERGENCE_STREAK})\n"
        f"Motors OFF — Robots stationary",
        Log.BLUE
    )


def enter_execution():
    """PLANNING → EXECUTION transition."""
    global SYSTEM_PHASE, execution_robots_at_goal

    _banner(
        f"✔  PLANNING COMPLETE\n"
        f"{planning_total_episodes} virtual episodes finished.\n"
        f"SWITCHING TO EXECUTION PHASE...",
        Log.GREEN
    )

    stop_all_robots()
    execution_robots_at_goal.clear()
    # Reset real robot states
    for rid in ROBOTS:
        robot_states[rid]["RL_STATE"]        = "COMPUTE"
        robot_states[rid]["prev_d"]          = None
        robot_states[rid]["current_action"]  = [0.0, 0.0]
        robot_states[rid]["action_send_time"] = 0
        robot_states[rid]["ep_steps"]        = 0
        robot_states[rid]["ep_reward"]       = 0
    SYSTEM_PHASE = 'EXECUTION'
    phase_log.append({"t": time.time() - _plot_start,
                      "from_p": "PLANNING", "to_p": "EXECUTION",
                      "ep_count": planning_total_episodes})

    _banner(
        f"► EXECUTION STARTED\n"
        f"Learned policy is now active!\n"
        f"Sending individual motor commands per robot ID.\n"
        f"R0  R1  R2  R3  →  TARGET!",
        Log.GREEN
    )


def enter_setup():
    """Return to SETUP from any phase."""
    global SYSTEM_PHASE, plan_robots_at_goal, execution_robots_at_goal
    previous_phase = SYSTEM_PHASE
    stop_all_robots()
    SYSTEM_PHASE = 'SETUP'
    phase_log.append({"t": time.time() - _plot_start,
                      "from_p": previous_phase, "to_p": "SETUP",
                      "ep_count": planning_total_episodes})
    start_positions.clear()
    virtual_robot_states.clear()
    plan_robots_at_goal.clear()
    execution_robots_at_goal.clear()

    _banner(
        f"↺  SYSTEM RESET\n"
        f"Previous phase: {previous_phase}\n"
        f"RETURNED TO SETUP MODE\n"
        f"Position robots, then press [T].",
        Log.CYAN
    )


def execute_learned_policy(rid):
    """
    EXECUTION phase: generate and send a per-robot motor command using the learned policy.
    No exploration — deterministic mean action.
    """
    global execution_robots_at_goal
    st = robot_states[rid]
    if st["nx"] is None:
        return

    # Get observation from real camera position
    obs, _, done, d = get_observation_and_reward(rid, st["prev_d"], st["current_action"])
    if obs is None:
        return

    # Deterministic action (no exploration)
    action = ppo_agent.select_action_deterministic(obs)
    ls = int(np.clip(action[0] * RL_MOTOR_SCALE, -30, 30))
    rs = int(np.clip(action[1] * RL_MOTOR_SCALE, -30, 30))
    send_robot_command(rid, ls, rs)
    execution_log.append({
        "t": time.time() - _plot_start, "rid": rid,
        "nx": st["nx"] or 0.0, "ny": st["ny"] or 0.0,
        "dist": d if d is not None else 0.0,
        "motor_l": ls, "motor_r": rs, "done": False
    })

    st["current_action"] = action
    st["prev_d"]         = d
    st["ep_steps"]      += 1

    if st["ep_steps"] % 6 == 0:
        Log.info(f"[EXEC] R{rid} | Step:{st['ep_steps']} | "
                 f"Dist to target:{d:.1f} | L:{ls} R:{rs}")

    if done:
        if execution_log and execution_log[-1]["rid"] == rid:
            execution_log[-1]["done"] = True
        execution_robots_at_goal.add(rid)
        Log.success(f"[EXEC] Robot {rid} REACHED TARGET! "
                    f"({len(execution_robots_at_goal)}/{len(ROBOTS)} robots at goal)")
        st["ep_steps"] = 0
        st["prev_d"]   = None

        # If all visible robots reached goal, refresh obstacles only
        active_robots = {r for r in ROBOTS if robot_states[r]["nx"] is not None}
        if active_robots and execution_robots_at_goal >= active_robots:
            Log.success("[EXEC] ALL ROBOTS REACHED TARGET! Refreshing obstacles...")
            generate_environment(only_obstacles=True)
            execution_robots_at_goal.clear()

# ==============================================================================
#  TRAINING & RUNTIME VARIABLES
# ==============================================================================
camera = LatencyFreeCamera("http://192.168.1.177:8080/video")

ppo_agent     = PPO(obs_dim=16, action_dim=2, lr=0.0003)
train_memory  = RolloutBuffer()
robot_memories = {rid: RolloutBuffer() for rid in ROBOTS}

MODEL_SAVE_PATH = "live_model_weights.pt"
if os.path.exists(MODEL_SAVE_PATH):
    try:
        ppo_agent.policy.load_state_dict(
            torch.load(MODEL_SAVE_PATH, map_location=device))
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())
        Log.info(f"[{MODEL_SAVE_PATH}] loaded.")
        Log.warning("NOTE: Physics fix applied (w=(ls-rs)/20 and absolute angle).")
        Log.warning("Old model may have been trained with incorrect physics. "
                    "Consider deleting it and retraining:")
        Log.warning(f"  del {MODEL_SAVE_PATH}")
    except Exception as e:
        Log.warning("Saved model incompatible with current architecture. "
                    "Starting fresh training...")

e_stop_active         = False
global_time_step      = 0
last_loss             = 0.0
update_count          = 0
selected_manual_robot = 0
manual_motor_active   = False
manual_last_cmd       = 0.0
episode_rewards_history = []

# ═══════════════════════════════════════════════════════════════════════
#  LOGGING STRUCTURES FOR PLOTS  (25 graphs generated when simulation ends)
# ═══════════════════════════════════════════════════════════════════════
update_log    = []  # Per PPO update:   {update, loss, avg_r20, global_step, plan_ep}
episode_log   = []  # Per episode:      {ep, rid, reward, steps, phase}
execution_log = []  # Per EXEC step:    {t, rid, nx, ny, dist, motor_l, motor_r, done}
phase_log     = []  # Phase transitions:{t, from_p, to_p, ep_count}
virt_pos_log  = []  # Virtual positions:{ep, rid, nx, ny}
_plot_start   = time.time()

# CSV Logger
CSV_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
csv_file   = open(CSV_LOG_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'phase', 'update', 'loss',
                     'avg_reward_20', 'buffer_size', 'global_step', 'plan_episode'])
Log.info(f"Training log: {CSV_LOG_PATH}")

cv2.namedWindow('MULTI_AGENT_RL', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MULTI_AGENT_RL', 900, 700)
cv2.setMouseCallback('MULTI_AGENT_RL', mouse_callback)
FONT = cv2.FONT_HERSHEY_SIMPLEX

_banner(
    "◉  SYSTEM READY\n"
    "STARTED IN SETUP MODE\n"
    "Place robots inside the arena.\n"
    "[T] → Start planning   [WASD] Manual drive\n"
    "[SPACE] E-Stop   [Q] Quit",
    Log.CYAN
)

try:
    while True:
        current_time = time.time()

        # ──────────────────────────────────────────────
        # 1. CAMERA & TAG TRACKING
        # ──────────────────────────────────────────────
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # Update global fw, fh (used by mouse callback)
        fh, fw = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags   = at_detector.detect(gray)

        seen_this_frame = set()
        for tag in tags:
            rid = tag.tag_id
            if rid in ROBOTS:
                seen_this_frame.add(rid)
                st = robot_states[rid]
                st["last_seen_time"] = current_time
                cx  = int(tag.center[0])
                cy  = int(tag.center[1])
                raw_nx = (cx / fw) * LOGIC_GRID_MAX
                raw_ny = (cy / fh) * LOGIC_GRID_MAX
                if st["nx"] is None:
                    st["nx"], st["ny"] = raw_nx, raw_ny
                else:
                    st["nx"] = EMA_ALPHA * raw_nx + (1 - EMA_ALPHA) * st["nx"]
                    st["ny"] = EMA_ALPHA * raw_ny + (1 - EMA_ALPHA) * st["ny"]

                corners = tag.corners.astype(int)
                st["corners"] = corners
                dx = (corners[3][0] + corners[2][0]) / 2.0 - \
                     (corners[0][0] + corners[1][0]) / 2.0
                dy = (corners[3][1] + corners[2][1]) / 2.0 - \
                     (corners[0][1] + corners[1][1]) / 2.0
                raw_deg    = int(math.degrees(math.atan2(dy, dx))) % 360
                st["angle"] = get_calibrated_angle(rid, raw_deg)

                p_width    = math.sqrt((corners[0][0] - corners[1][0])**2 +
                                       (corners[0][1] - corners[1][1])**2)
                st["radius"] = max((p_width / fw) * LOGIC_GRID_MAX * 0.707, 10.0)

                l_corners = []
                for pt in tag.corners:
                    l_corners.append(((pt[0] / fw) * LOGIC_GRID_MAX,
                                      (pt[1] / fh) * LOGIC_GRID_MAX))
                st["logic_corners"] = l_corners

        for rid in ROBOTS:
            if rid not in seen_this_frame:
                robot_states[rid]["corners"] = None

        if virtual_target is None:
            any_valid = next((st for st in robot_states.values()
                              if st["nx"] is not None), None)
            if any_valid:
                generate_environment()

        # ──────────────────────────────────────────────
        # 2A. PLANNING PHASE — Virtual Simulation (No Motors)
        # ──────────────────────────────────────────────
        if SYSTEM_PHASE == 'PLANNING' and not e_stop_active and not plan_update_pending:

            for rid in ROBOTS:
                if rid not in virtual_robot_states:
                    continue
                vst = virtual_robot_states[rid]

                # Get observation (from virtual position)
                obs, _, _, _ = get_virtual_obs_reward(
                    rid, vst, vst["prev_d"], vst["action"], virtual_robot_states)
                if obs is None:
                    continue

                # Select RL action (with exploration — training)
                action     = ppo_agent.select_action(obs, robot_memories[rid])
                vst["action"] = action

                # Virtual step (NO real motor command)
                virtual_step(vst, action)
                if len(virt_pos_log) < 120000:
                    virt_pos_log.append({"ep": vst["episode"], "rid": rid,
                                         "nx": vst["nx"], "ny": vst["ny"]})

                # New obs + reward
                _, reward, done, new_d = get_virtual_obs_reward(
                    rid, vst, vst["prev_d"], action, virtual_robot_states)

                robot_memories[rid].rewards.append(reward)
                robot_memories[rid].is_terminals.append(done)

                vst["prev_d"]    = new_d
                vst["ep_reward"] += reward
                vst["ep_steps"]  += 1
                global_time_step  += 1

                # If virtual robot reached target, refresh obstacles
                if done and new_d is not None and new_d < (robot_states[rid]["radius"] + 15):
                    Log.plan(f"[PLAN] R{rid} reached virtual target! Refreshing obstacles...")
                    generate_environment(only_obstacles=True)
                    plan_robots_at_goal.clear()

                # Episode ended?
                if done or vst["ep_steps"] >= MAX_EP_STEPS:
                    Log.plan(f"[PLAN] R{rid} Ep:{vst['episode']} DONE | "
                             f"Reward:{vst['ep_reward']:.1f}")

                    robot_memories[rid].compute_returns(ppo_agent.gamma)
                    train_memory.states.extend(robot_memories[rid].states)
                    train_memory.actions.extend(robot_memories[rid].actions)
                    train_memory.logprobs.extend(robot_memories[rid].logprobs)
                    train_memory.rewards.extend(robot_memories[rid].rewards)
                    train_memory.is_terminals.extend(robot_memories[rid].is_terminals)
                    train_memory.returns.extend(robot_memories[rid].returns)
                    robot_memories[rid].clear()

                    episode_rewards_history.append(vst["ep_reward"])
                    episode_log.append({
                        "ep": planning_total_episodes, "rid": rid,
                        "reward": vst["ep_reward"], "steps": vst["ep_steps"],
                        "phase": "PLANNING"
                    })
                    planning_total_episodes += 1
                    vst["episode"]   += 1
                    vst["ep_reward"]  = 0.0
                    vst["ep_steps"]   = 0
                    vst["prev_d"]     = None

                    # Reset to start position with randomised heading
                    if rid in start_positions:
                        vst["nx"], vst["ny"], _ = start_positions[rid]
                    else:
                        vst["nx"], vst["ny"] = 200.0, 200.0
                    vst["angle"] = random.uniform(0, 360)

                # Trigger PPO update when buffer is full
                if len(train_memory.states) >= PPO_UPDATE_TIMESTEP:
                    plan_update_pending = True

            # PPO network update
            if plan_update_pending:
                plan_update_pending = False
                Log.plan(f"--- PPO UPDATE | Buffer:{len(train_memory.states)} | "
                         f"TotalEp:{planning_total_episodes} ---")
                try:
                    last_loss = ppo_agent.update(train_memory)
                    train_memory.clear()
                    update_count += 1
                    torch.save(ppo_agent.policy.state_dict(), MODEL_SAVE_PATH)
                    if update_count % 20 == 0:
                        backup_path = MODEL_SAVE_PATH.replace('.pt', f'_v{update_count}.pt')
                        torch.save(ppo_agent.policy.state_dict(), backup_path)
                        Log.info(f"Backup saved: {backup_path}")

                    avg_r20 = (np.mean(episode_rewards_history[-CONVERGENCE_WINDOW:])
                               if len(episode_rewards_history) >= CONVERGENCE_WINDOW
                               else None)
                    avg_display = avg_r20 if avg_r20 is not None else 0.0
                    csv_writer.writerow([
                        datetime.now().isoformat(), 'PLANNING', update_count,
                        f'{last_loss:.4f}', f'{avg_display:.1f}',
                        len(train_memory.states), global_time_step,
                        planning_total_episodes
                    ])
                    csv_file.flush()
                    update_log.append({
                        "update": update_count, "loss": last_loss,
                        "avg_r20": avg_display, "global_step": global_time_step,
                        "plan_ep": planning_total_episodes
                    })

                    # Convergence check
                    if avg_r20 is not None and planning_total_episodes >= PLANNING_MIN_EPISODES:
                        if avg_r20 >= CONVERGENCE_THRESHOLD:
                            convergence_streak_count += 1
                            Log.plan(f"UPDATE DONE | Loss:{last_loss:.4f} | "
                                     f"AvgR({CONVERGENCE_WINDOW}):{avg_r20:.1f} "
                                     f"[CONVERGENCE {convergence_streak_count}/{CONVERGENCE_STREAK}]")
                        else:
                            convergence_streak_count = 0
                            Log.plan(f"UPDATE DONE | Loss:{last_loss:.4f} | "
                                     f"AvgR({CONVERGENCE_WINDOW}):{avg_r20:.1f} "
                                     f"(Target: >={CONVERGENCE_THRESHOLD:.0f})")
                    else:
                        convergence_streak_count = 0
                        Log.plan(f"UPDATE DONE | Loss:{last_loss:.4f} | "
                                 f"AvgR:{avg_display:.1f} "
                                 f"(Min {PLANNING_MIN_EPISODES} eps required, "
                                 f"current:{planning_total_episodes})")

                except Exception as e:
                    Log.error(f"PPO UPDATE ERROR: {e}")
                    train_memory.clear()

            # Convergence detection: switch to EXECUTION if learned sufficiently
            converged = (planning_total_episodes >= PLANNING_MIN_EPISODES and
                         convergence_streak_count >= CONVERGENCE_STREAK)
            # Safety cap: switch if max episodes exceeded
            max_exceeded = planning_total_episodes >= PLANNING_MAX_EPISODES

            if converged or max_exceeded:
                reason = (
                    f"{CONVERGENCE_STREAK} consecutive updates avg>={CONVERGENCE_THRESHOLD:.0f}"
                    if converged
                    else f"Max episodes ({PLANNING_MAX_EPISODES}) exceeded"
                )
                Log.success(f"[CONVERGENCE] Criterion: {reason}")
                enter_execution()

        # ──────────────────────────────────────────────
        # 2B. EXECUTION PHASE — Per-Robot Deterministic Command
        # ──────────────────────────────────────────────
        elif SYSTEM_PHASE == 'EXECUTION' and not e_stop_active:
            for rid in ROBOTS:
                st = robot_states[rid]
                if st["nx"] is None:
                    if (current_time - st["action_send_time"]) > 0.5:
                        send_robot_command(rid, 0, 0)
                        st["action_send_time"] = current_time
                    continue

                # Stop robot if it goes outside the arena boundary
                if not is_point_in_polygon(st["nx"], st["ny"]):
                    Log.warning(f"[EXEC] Robot {rid} out of bounds! Stopping...")
                    send_robot_command(rid, 0, 0)
                    continue

                # Limit command rate to ~20 Hz
                if (current_time - st["action_send_time"]) >= 0.05:
                    execute_learned_policy(rid)
                    st["action_send_time"] = current_time

        # ──────────────────────────────────────────────
        # 2C. SETUP PHASE — No motor commands sent
        # (Robots are positioned manually by the user)
        # ──────────────────────────────────────────────

        # ──────────────────────────────────────────────
        # 3. USER INTERFACE (DRAWING)
        # ──────────────────────────────────────────────
        display = frame.copy()

        # Draw arena boundary
        poly_pts = []
        for (lx, ly) in WALL_POLYGON:
            poly_pts.append([int((lx / LOGIC_GRID_MAX) * fw),
                             int((ly / LOGIC_GRID_MAX) * fh)])
        cv2.polylines(display, [np.array(poly_pts)], True, (0, 0, 255), 2)
        cv2.putText(display, "ARENA WALL",
                    (poly_pts[0][0] - 20, poly_pts[0][1] - 10),
                    FONT, 0.5, (0, 0, 255), 2)

        # Draw real robots
        for rid in ROBOTS:
            st = robot_states[rid]
            if st["nx"] is not None:
                cx  = int((st["nx"] / LOGIC_GRID_MAX) * fw)
                cy  = int((st["ny"] / LOGIC_GRID_MAX) * fh)
                clr = ROBOT_COLORS[rid]

                is_danger = False
                for oid in ROBOTS:
                    if oid != rid and robot_states[oid]["nx"] is not None:
                        dist_o = math.sqrt((robot_states[oid]["nx"] - st["nx"])**2 +
                                           (robot_states[oid]["ny"] - st["ny"])**2)
                        if dist_o < 35.0:
                            is_danger = True
                            break

                zone_clr        = (0, 0, 255) if is_danger else clr
                safe_radius_px  = int((35.0 / LOGIC_GRID_MAX) * fw)
                cv2.circle(display, (cx, cy), safe_radius_px, zone_clr, 1, cv2.LINE_AA)

                if st["corners"] is not None:
                    cv2.polylines(display, [st["corners"]], True, clr, 2)
                else:
                    cv2.circle(display, (cx, cy), int(st["radius"]), clr, 2)

                phase_short = {
                    "SETUP": "SET", "PLANNING": "PLAN", "EXECUTION": "EXEC"
                }.get(SYSTEM_PHASE, "?")
                cv2.putText(display, f"R{rid}[{phase_short}]",
                            (cx - 20, cy - 25), FONT, 0.5, clr, 2)

                if st["angle"] is not None:
                    arad = math.radians(st["angle"])
                    tx_  = int(cx + math.cos(arad) * 30)
                    ty_  = int(cy + math.sin(arad) * 30)
                    cv2.arrowedLine(display, (cx, cy), (tx_, ty_), clr, 3, tipLength=0.3)

        # Draw start position markers (★)
        for rid, (spx, spy, _) in start_positions.items():
            sx  = int((spx / LOGIC_GRID_MAX) * fw)
            sy  = int((spy / LOGIC_GRID_MAX) * fh)
            clr = ROBOT_COLORS[rid]
            cv2.drawMarker(display, (sx, sy), clr, cv2.MARKER_STAR, 20, 2)
            cv2.putText(display, f"S{rid}", (sx + 8, sy - 8), FONT, 0.4, clr, 1)

        # Draw virtual robots during PLANNING phase
        if SYSTEM_PHASE == 'PLANNING' and virtual_robot_states:
            for rid, vst in virtual_robot_states.items():
                vx  = int((vst["nx"] / LOGIC_GRID_MAX) * fw)
                vy  = int((vst["ny"] / LOGIC_GRID_MAX) * fh)
                clr = ROBOT_COLORS[rid]
                # Ghost robot (dashed circle effect)
                cv2.circle(display, (vx, vy), 12, clr, 1, cv2.LINE_AA)
                cv2.putText(display, f"v{rid}", (vx + 5, vy - 5), FONT, 0.35, clr, 1)
                # Heading arrow
                varad = math.radians(vst["angle"])
                vtx   = int(vx + math.cos(varad) * 15)
                vty   = int(vy + math.sin(varad) * 15)
                cv2.arrowedLine(display, (vx, vy), (vtx, vty), clr, 1, tipLength=0.4)

        # Draw target and obstacles
        if virtual_target is not None:
            tx_ = int((virtual_target[0] / LOGIC_GRID_MAX) * fw)
            ty_ = int((virtual_target[1] / LOGIC_GRID_MAX) * fh)
            cv2.drawMarker(display, (tx_, ty_), (255, 200, 0), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(display, "TARGET",
                        (tx_ - 20, ty_ - 20), FONT, 0.6, (255, 200, 0), 2)

        for (ox, oy, r) in virtual_obstacles:
            ecx = int((ox / LOGIC_GRID_MAX) * fw)
            ecy = int((oy / LOGIC_GRID_MAX) * fh)
            er  = int((r  / LOGIC_GRID_MAX) * fw)
            cv2.circle(display, (ecx, ecy), er, (0, 0, 255), 2)
            cv2.putText(display, "OBS",
                        (ecx - 15, ecy), FONT, 0.4, (0, 0, 255), 1)

        # ── Top-left info panel ──
        cv2.rectangle(display, (0, 0), (320, 240), (20, 20, 20), -1)

        # Phase label
        if e_stop_active:
            phase_str, phase_clr = "E-STOP ACTIVE!", (0, 0, 255)
        elif WAITING_FOR_TARGET:
            phase_str, phase_clr = ">>> LEFT CLICK TO SELECT TARGET <<<", (0, 200, 255)
        elif SYSTEM_PHASE == 'SETUP':
            phase_str, phase_clr = "SETUP — [T] to start Planning", (0, 200, 255)
        elif SYSTEM_PHASE == 'PLANNING':
            phase_str, phase_clr = (
                f"PLANNING — Virtual Sim "
                f"({planning_total_episodes}/{PLANNING_MAX_EPISODES} ep)",
                (0, 255, 0)
            )
        else:  # EXECUTION
            phase_str, phase_clr = "EXECUTION — Policy Active", (255, 150, 0)

        cv2.putText(display, f"PHASE: {phase_str}",
                    (10, 25), FONT, 0.5, phase_clr, 2)

        # Big overlay prompt when waiting for target
        if WAITING_FOR_TARGET:
            overlay = display.copy()
            cv2.rectangle(overlay,
                          (fw // 2 - 280, fh // 2 - 40),
                          (fw // 2 + 280, fh // 2 + 40),
                          (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)
            cv2.putText(display, "LEFT CLICK: SELECT TARGET",
                        (fw // 2 - 230, fh // 2 + 12),
                        FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(display,
                    f"BUFFER: {len(train_memory.states)}/{PPO_UPDATE_TIMESTEP}",
                    (10, 50), FONT, 0.5, (100, 255, 255), 1)
        cv2.putText(display,
                    f"LOSS: {last_loss:.4f} | UPD: {update_count}",
                    (10, 75), FONT, 0.5, (0, 200, 255), 1)
        avg_r = np.mean(episode_rewards_history[-20:]) if episode_rewards_history else 0.0
        cv2.putText(display,
                    f"AVG REWARD(20): {avg_r:.1f}",
                    (10, 100), FONT, 0.5, (0, 255, 200), 1)

        y_off = 125
        for rid in ROBOTS:
            st      = robot_states[rid]
            vis     = "OK" if st["nx"] is not None else "--"
            start_ok = "✓" if rid in start_positions else "✗"
            cv2.putText(display,
                        f"R{rid}: {vis} | Start:{start_ok} | "
                        f"Ep:{st['episode']} | Rew:{st['ep_reward']:.1f}",
                        (10, y_off), FONT, 0.38, ROBOT_COLORS[rid], 1)
            y_off += 20

        # Bottom status bar
        if SYSTEM_PHASE == 'SETUP':
            cv2.putText(display,
                        f"MANUAL: Robot:{selected_manual_robot} (0-3 select, WASD drive)",
                        (10, fh - 45), FONT, 0.5, (255, 150, 150), 2)
        cv2.putText(display,
                    "[T]Plan [E]Execute [H]Setup [SPACE]E-Stop [R]Map [Q]Quit",
                    (10, fh - 20), FONT, 0.45, (255, 255, 255), 1)

        cv2.imshow('MULTI_AGENT_RL', display)
        key = cv2.waitKey(1) & 0xFF

        # ──────────────────────────────────────────────
        # 4. KEYBOARD CONTROLS
        # ──────────────────────────────────────────────
        if key == ord('q'):
            break

        elif key == ord(' '):   # E-STOP toggle
            e_stop_active = not e_stop_active
            if e_stop_active:
                stop_all_robots()
                Log.error("!!! E-STOP ACTIVE !!! Press [SPACE] to resume.")
            else:
                Log.success("E-Stop released.")

        elif key == ord('t'):   # SETUP → Wait for target → PLANNING
            if not e_stop_active:
                if SYSTEM_PHASE == 'SETUP' and not WAITING_FOR_TARGET:
                    enter_planning()
                elif WAITING_FOR_TARGET:
                    Log.warning("Waiting for target — left-click to set it, or [H] to cancel.")
                elif SYSTEM_PHASE == 'PLANNING':
                    Log.warning("Already in PLANNING phase. Press [E] to switch to EXECUTION.")
                else:
                    Log.warning("EXECUTION is active. Press [H] to return to SETUP first.")

        elif key == ord('e'):   # PLANNING → EXECUTION (manual override)
            if not e_stop_active:
                if SYSTEM_PHASE == 'PLANNING':
                    Log.success(f"Manual EXECUTION switch! "
                                f"({planning_total_episodes} eps completed)")
                    enter_execution()
                elif SYSTEM_PHASE == 'SETUP':
                    Log.warning("Run [T] to start PLANNING first!")
                else:
                    Log.info("Already in EXECUTION phase.")

        elif key == ord('h'):   # → SETUP (reset)
            WAITING_FOR_TARGET = False
            enter_setup()

        elif key == ord('c'):
            for rid in ROBOTS:
                robot_states[rid]["base_angle"] = None
            Log.info("All robot base angles reset.")

        elif key == ord('r'):
            generate_environment()
            Log.info("Environment refreshed.")

        elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
            selected_manual_robot = int(chr(key))

        # Manual WASD — only in SETUP phase
        if SYSTEM_PHASE == 'SETUP' and not e_stop_active:
            if key in [ord('w'), ord('a'), ord('s'), ord('d')]:
                if   key == ord('w'): send_robot_command(selected_manual_robot,  20,  20)
                elif key == ord('s'): send_robot_command(selected_manual_robot, -20, -20)
                elif key == ord('a'): send_robot_command(selected_manual_robot, -18,  18)
                elif key == ord('d'): send_robot_command(selected_manual_robot,  18, -18)
                manual_motor_active = True
                manual_last_cmd     = current_time
            else:
                if manual_motor_active and (current_time - manual_last_cmd > 0.15):
                    send_robot_command(selected_manual_robot, 0, 0)
                    manual_motor_active = False

finally:
    stop_all_robots()
    if gateway_serial and gateway_serial.is_open:
        gateway_serial.close()
    camera.stop()
    csv_file.close()
    cv2.destroyAllWindows()
    Log.success(f"System safely shut down. Log: {CSV_LOG_PATH}")

    # ==========================================================================
    #  AUTOMATIC PLOT GENERATION — 25 Graphs
    # ==========================================================================
    try:
        print("\n" + "=" * 65)
        print("  SIMULATION ENDED — GENERATING 25 PLOTS...")
        print("=" * 65)

        OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_plots")
        os.makedirs(OUT_DIR, exist_ok=True)

        ROBOT_HEX = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#ff7f00"}
        C_PLAN    = "#1f77b4"
        C_EXEC    = "#2ca02c"
        C_LOSS    = "#d62728"
        C_REW     = "#2ca02c"
        C_THRESH  = "#ff7f0e"

        plt.rcParams.update({
            "figure.facecolor": "white", "axes.facecolor": "white",
            "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
            "axes.labelsize": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
            "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
            "grid.color": "#dddddd", "grid.linewidth": 0.8, "grid.linestyle": "--",
            "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 10, "legend.framealpha": 0.85,
            "lines.linewidth": 2.0, "figure.dpi": 150,
            "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
        })

        def _save(fig, name):
            fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
            plt.close(fig)
            print(f"  [OK] {name}.png")

        def _mav(arr, w=20):
            if len(arr) == 0: return np.array([])
            w = min(w, len(arr))
            return np.convolve(arr, np.ones(w) / w, mode='valid')

        def _wall(ax, color="blue", ls="--", lw=2):
            wx = [p[0] for p in WALL_POLYGON] + [WALL_POLYGON[0][0]]
            wy = [p[1] for p in WALL_POLYGON] + [WALL_POLYGON[0][1]]
            ax.plot(wx, wy, color=color, ls=ls, lw=lw, label="Arena Boundary")

        N_ep    = len(episode_rewards_history)
        N_upd   = len(update_log)
        N_exec  = len(execution_log)
        N_ph    = len(phase_log)
        N_eplog = len(episode_log)
        N_virt  = len(virt_pos_log)

        # ── 1. Episode Reward Curve ────────────────────────────────────────────
        print("[1/25] Episode reward curve...")
        if N_ep > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            eps = np.arange(1, N_ep + 1)
            ax.plot(eps, episode_rewards_history,
                    color=C_REW, alpha=0.25, lw=1, label="Raw Reward")
            if N_ep >= 5:
                w  = min(20, N_ep // 2)
                ma = _mav(episode_rewards_history, w)
                ax.plot(np.arange(w, N_ep + 1), ma, color=C_REW, lw=2.5,
                        label=f"Moving Avg ({w} ep)")
            ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=2, ls="--",
                       label=f"Convergence Threshold ({CONVERGENCE_THRESHOLD:.0f})")
            ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
            ax.set_title("PPO Training — Episode Reward Curve"); ax.legend()
            _save(fig, "01_episode_reward_curve")
        else: print("  [SKIP] No data")

        # ── 2. PPO Loss Curve ──────────────────────────────────────────────────
        print("[2/25] PPO loss curve...")
        if N_upd > 0:
            upd  = [u["update"] for u in update_log]
            loss = [u["loss"]   for u in update_log]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(upd, loss, alpha=0.15, color=C_LOSS)
            ax.plot(upd, loss, color=C_LOSS, lw=2.5, label="PPO Loss")
            ax.set_xlabel("PPO Update #"); ax.set_ylabel("Loss")
            ax.set_title("PPO — Loss per Update"); ax.legend()
            _save(fig, "02_ppo_loss_curve")
        else: print("  [SKIP] No data")

        # ── 3. Avg Reward per Update + Convergence ─────────────────────────────
        print("[3/25] Average reward per update...")
        if N_upd > 0:
            upd   = [u["update"]  for u in update_log]
            avg_r = [u["avg_r20"] for u in update_log]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(upd, avg_r, alpha=0.15, color=C_REW)
            ax.plot(upd, avg_r, color=C_REW, lw=2.5,
                    label=f"Avg Reward (last {CONVERGENCE_WINDOW} ep)")
            ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=2, ls="--",
                       label=f"Convergence Threshold ({CONVERGENCE_THRESHOLD:.0f})")
            ax.set_xlabel("PPO Update #"); ax.set_ylabel("Average Reward")
            ax.set_title("PPO — Average Reward per Update & Convergence Analysis")
            ax.legend()
            _save(fig, "03_avg_reward_per_update")
        else: print("  [SKIP] No data")

        # ── 4. Global Timestep vs Avg Reward ──────────────────────────────────
        print("[4/25] Global timestep vs reward...")
        if N_upd > 0:
            gsteps = [u["global_step"] for u in update_log]
            avg_r  = [u["avg_r20"]    for u in update_log]
            fig, ax = plt.subplots(figsize=(10, 5))
            sc = ax.scatter(gsteps, avg_r, c=np.arange(len(gsteps)),
                            cmap="viridis", s=60, zorder=5)
            ax.plot(gsteps, avg_r, color="#999999", lw=1, alpha=0.5)
            fig.colorbar(sc, ax=ax, label="Update #")
            ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=1.5, ls="--",
                       label=f"Threshold={CONVERGENCE_THRESHOLD:.0f}")
            ax.set_xlabel("Global Timestep"); ax.set_ylabel("Average Reward")
            ax.set_title("PPO — Learning Progress vs Global Timestep"); ax.legend()
            _save(fig, "04_global_step_vs_reward")
        else: print("  [SKIP] No data")

        # ── 5. Episodes per Update ─────────────────────────────────────────────
        print("[5/25] Episodes per update...")
        if N_upd > 0:
            upd      = [u["update"]  for u in update_log]
            plan_eps = [u["plan_ep"] for u in update_log]
            fig, ax  = plt.subplots(figsize=(10, 5))
            ax.bar(upd, plan_eps, color=C_PLAN, alpha=0.8, label="Episodes Completed")
            ax.axhline(PLANNING_MIN_EPISODES, color=C_THRESH, lw=2, ls="--",
                       label=f"Min Ep ({PLANNING_MIN_EPISODES})")
            ax.axhline(PLANNING_MAX_EPISODES, color=C_LOSS,   lw=2, ls=":",
                       label=f"Max Ep ({PLANNING_MAX_EPISODES})")
            ax.set_xlabel("PPO Update #"); ax.set_ylabel("Episode Count")
            ax.set_title("PPO — Episodes Completed per Update"); ax.legend()
            _save(fig, "05_episodes_per_update")
        else: print("  [SKIP] No data")

        # ── 6. Per-Robot Episode Rewards ───────────────────────────────────────
        print("[6/25] Per-robot episode rewards...")
        if N_eplog > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTS:
                r_data = [(e["ep"], e["reward"])
                          for e in episode_log
                          if e["rid"] == rid and e["phase"] == "PLANNING"]
                if r_data:
                    xs, ys = zip(*r_data)
                    ax.plot(xs, ys, color=ROBOT_HEX[rid], alpha=0.35, lw=1)
                    if len(ys) >= 5:
                        w  = min(10, len(ys) // 2)
                        ma = _mav(list(ys), w)
                        ax.plot(xs[len(ys) - len(ma):], ma,
                                color=ROBOT_HEX[rid], lw=2.5, label=f"Robot {rid}")
            ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=1.5, ls="--",
                       label=f"Threshold ({CONVERGENCE_THRESHOLD:.0f})")
            ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
            ax.set_title("PLANNING — Per-Robot Episode Reward Curves"); ax.legend()
            _save(fig, "06_per_robot_rewards")
        else: print("  [SKIP] No data")

        # ── 7. Episode Length Histogram ────────────────────────────────────────
        print("[7/25] Episode length histogram...")
        psteps = [e["steps"] for e in episode_log if e["phase"] == "PLANNING"]
        if psteps:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(psteps, bins=min(40, len(set(psteps))),
                    color=C_PLAN, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(psteps), color=C_THRESH, lw=2, ls="--",
                       label=f"Mean = {np.mean(psteps):.0f}")
            ax.axvline(MAX_EP_STEPS, color=C_LOSS, lw=2, ls=":",
                       label=f"Max = {MAX_EP_STEPS}")
            ax.set_xlabel("Episode Length (steps)"); ax.set_ylabel("Frequency")
            ax.set_title("PLANNING — Episode Length Distribution"); ax.legend()
            _save(fig, "07_episode_length_histogram")
        else: print("  [SKIP] No data")

        # ── 8. Cumulative Episode Reward ───────────────────────────────────────
        print("[8/25] Cumulative reward curve...")
        if N_ep > 0:
            eps = np.arange(1, N_ep + 1)
            cum = np.cumsum(episode_rewards_history)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(eps, cum, alpha=0.2, color=C_REW)
            ax.plot(eps, cum, color=C_REW, lw=2.5)
            ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Total Reward")
            ax.set_title("PPO Training — Cumulative Total Reward")
            _save(fig, "08_cumulative_reward")
        else: print("  [SKIP] No data")

        # ── 9. PLANNING Virtual Position Heatmap ──────────────────────────────
        print("[9/25] PLANNING position heatmap...")
        if N_virt > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            vnx = [v["nx"] for v in virt_pos_log]
            vny = [v["ny"] for v in virt_pos_log]
            h = ax.hist2d(vnx, vny, bins=40,
                          range=[[0, LOGIC_GRID_MAX], [0, LOGIC_GRID_MAX]],
                          cmap="YlOrRd")
            fig.colorbar(h[3], ax=ax, label="Visit Count")
            _wall(ax, color="blue")
            if virtual_target:
                ax.scatter(*virtual_target, c="cyan", s=200, marker="*",
                           zorder=5, label="Last Target", edgecolors="k")
            ax.set_xlim(0, LOGIC_GRID_MAX); ax.set_ylim(0, LOGIC_GRID_MAX)
            ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_title("PLANNING — Virtual Robot Position Density"); ax.legend(fontsize=9)
            _save(fig, "09_planning_heatmap")
        else: print("  [SKIP] No data")

        # ── 10. EXECUTION Real Trajectory ─────────────────────────────────────
        print("[10/25] EXECUTION real trajectory...")
        if N_exec > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            _wall(ax, color="red")
            for rid in ROBOTS:
                rl = [e for e in execution_log if e["rid"] == rid]
                if rl:
                    xs = [e["nx"] for e in rl]; ys = [e["ny"] for e in rl]
                    ax.plot(xs, ys, color=ROBOT_HEX[rid], lw=2, alpha=0.85,
                            label=f"Robot {rid}")
                    ax.plot(xs[0],  ys[0],  "o", color=ROBOT_HEX[rid],
                            ms=10, markeredgecolor="k", zorder=5)
                    ax.plot(xs[-1], ys[-1], "s", color=ROBOT_HEX[rid],
                            ms=10, markeredgecolor="k", zorder=5)
            if virtual_target:
                ax.scatter(*virtual_target, c="gold", s=250, marker="*",
                           zorder=6, label="Target", edgecolors="k")
            for (ox, oy, r) in virtual_obstacles:
                ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.4))
            ax.set_xlim(0, LOGIC_GRID_MAX); ax.set_ylim(0, LOGIC_GRID_MAX)
            ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_title("EXECUTION — Real Robot Trajectory with Learned Policy")
            ax.legend(fontsize=9)
            _save(fig, "10_execution_trajectory")
        else: print("  [SKIP] No data")

        # ── 11. EXECUTION Distance to Target Time Series ───────────────────────
        print("[11/25] EXECUTION distance to target time series...")
        if N_exec > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTS:
                rl = [e for e in execution_log if e["rid"] == rid]
                if rl:
                    ts = [e["t"]    for e in rl]; ds = [e["dist"] for e in rl]
                    ax.plot(ts, ds, color=ROBOT_HEX[rid], lw=2, alpha=0.85,
                            label=f"Robot {rid}")
                    done_t = [e["t"]    for e in rl if e["done"]]
                    done_d = [e["dist"] for e in rl if e["done"]]
                    if done_t:
                        ax.scatter(done_t, done_d, color=ROBOT_HEX[rid],
                                   s=120, marker="*", zorder=5)
            ax.axhline(15, color="#888888", lw=1.5, ls="--",
                       label="Goal threshold (~15 px)")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance to Target")
            ax.set_title("EXECUTION — Distance to Target Time Series"); ax.legend()
            _save(fig, "11_execution_dist_to_target")
        else: print("  [SKIP] No data")

        # ── 12. EXECUTION Left Motor Time Series ──────────────────────────────
        print("[12/25] EXECUTION left motor time series...")
        if N_exec > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTS:
                rl = [e for e in execution_log if e["rid"] == rid]
                if rl:
                    ax.plot([e["t"] for e in rl], [e["motor_l"] for e in rl],
                            color=ROBOT_HEX[rid], lw=1.8, alpha=0.85,
                            label=f"Robot {rid}")
            ax.axhline(0, color="#aaaaaa", lw=1, ls="--")
            ax.set_ylim(-35, 35)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Left Motor Command")
            ax.set_title("EXECUTION — Left Motor Command Time Series"); ax.legend()
            _save(fig, "12_execution_motor_left")
        else: print("  [SKIP] No data")

        # ── 13. EXECUTION Right Motor Time Series ─────────────────────────────
        print("[13/25] EXECUTION right motor time series...")
        if N_exec > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTS:
                rl = [e for e in execution_log if e["rid"] == rid]
                if rl:
                    ax.plot([e["t"] for e in rl], [e["motor_r"] for e in rl],
                            color=ROBOT_HEX[rid], lw=1.8, alpha=0.85,
                            label=f"Robot {rid}")
            ax.axhline(0, color="#aaaaaa", lw=1, ls="--")
            ax.set_ylim(-35, 35)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Right Motor Command")
            ax.set_title("EXECUTION — Right Motor Command Time Series"); ax.legend()
            _save(fig, "13_execution_motor_right")
        else: print("  [SKIP] No data")

        # ── 14. Motor Action Space Scatter (L vs R) ────────────────────────────
        print("[14/25] Motor action space scatter...")
        if N_exec > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            for rid in ROBOTS:
                rl = [e for e in execution_log if e["rid"] == rid]
                if rl:
                    ax.scatter([e["motor_l"] for e in rl],
                               [e["motor_r"] for e in rl],
                               color=ROBOT_HEX[rid], s=10, alpha=0.4,
                               label=f"Robot {rid}")
            ax.axhline(0, color="#aaaaaa", lw=1, ls="--")
            ax.axvline(0, color="#aaaaaa", lw=1, ls="--")
            ax.set_xlim(-35, 35); ax.set_ylim(-35, 35); ax.set_aspect("equal")
            ax.set_xlabel("Left Motor"); ax.set_ylabel("Right Motor")
            ax.set_title("EXECUTION — Motor Action Space Distribution")
            ax.legend(fontsize=9)
            _save(fig, "14_motor_action_space")
        else: print("  [SKIP] No data")

        # ── 15. Virtual Arena Map ──────────────────────────────────────────────
        print("[15/25] Virtual arena map...")
        fig, ax = plt.subplots(figsize=(7, 7))
        wx = [p[0] for p in WALL_POLYGON] + [WALL_POLYGON[0][0]]
        wy = [p[1] for p in WALL_POLYGON] + [WALL_POLYGON[0][1]]
        ax.fill(wx, wy, alpha=0.06, color="blue")
        ax.plot(wx, wy, "b-", lw=2.5, label="Arena Boundary")
        for (ox, oy, r) in virtual_obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.65, zorder=3))
        if virtual_obstacles:
            ax.scatter([], [], color="red", alpha=0.8, label="Obstacles", s=100)
        if virtual_target:
            ax.scatter(*virtual_target, c="gold", s=300, marker="*",
                       zorder=5, label="Target", edgecolors="k", lw=1)
        for rid, (spx, spy, _) in start_positions.items():
            ax.scatter(spx, spy, color=ROBOT_HEX.get(rid, "gray"), s=120,
                       marker="o", zorder=4, label=f"R{rid} Start")
        ax.set_xlim(0, LOGIC_GRID_MAX); ax.set_ylim(0, LOGIC_GRID_MAX)
        ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title("RL Environment — Virtual Arena Map"); ax.legend(fontsize=9)
        _save(fig, "15_arena_map")

        # ── 16. Episode Reward vs Length Scatter ──────────────────────────────
        print("[16/25] Episode reward vs length scatter...")
        plan_rl = [(e["reward"], e["steps"])
                   for e in episode_log if e["phase"] == "PLANNING"]
        if plan_rl:
            rews_, stps_ = zip(*plan_rl)
            fig, ax = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(stps_, rews_, c=np.arange(len(rews_)),
                            cmap="plasma", s=20, alpha=0.5)
            fig.colorbar(sc, ax=ax, label="Episode Number")
            ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=1.5, ls="--",
                       label=f"Threshold ({CONVERGENCE_THRESHOLD:.0f})")
            ax.set_xlabel("Episode Length (steps)"); ax.set_ylabel("Episode Reward")
            ax.set_title("PLANNING — Episode Reward vs. Length"); ax.legend()
            _save(fig, "16_reward_vs_steps_scatter")
        else: print("  [SKIP] No data")

        # ── 17. Phase Timeline ─────────────────────────────────────────────────
        print("[17/25] Phase transition timeline...")
        if N_ph > 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            phase_clr = {
                "SETUP":     "#aec7e8",
                "PLANNING":  "#1f77b4",
                "EXECUTION": "#2ca02c"
            }
            t_max = phase_log[-1]["t"] + 5
            prev_t, prev_phase = 0.0, "SETUP"
            for entry in phase_log:
                dur = entry["t"] - prev_t
                ax.barh(0, dur, left=prev_t,
                        color=phase_clr.get(prev_phase, "gray"),
                        height=0.6, edgecolor="white", lw=0.5)
                if dur > 1:
                    ax.text(prev_t + dur / 2, 0, prev_phase,
                            ha="center", va="center", fontsize=9,
                            color="white", fontweight="bold")
                prev_t, prev_phase = entry["t"], entry["to_p"]
            ax.barh(0, t_max - prev_t, left=prev_t,
                    color=phase_clr.get(prev_phase, "gray"),
                    height=0.6, edgecolor="white", lw=0.5)
            if t_max - prev_t > 1:
                ax.text(prev_t + (t_max - prev_t) / 2, 0, prev_phase,
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
            ax.set_xlim(0, t_max); ax.set_xlabel("Time (s)"); ax.set_yticks([])
            ax.set_title("System Phase Transition Timeline")
            patches = [mpatches.Patch(color=v, label=k) for k, v in phase_clr.items()]
            ax.legend(handles=patches, loc="upper right")
            _save(fig, "17_phase_timeline")
        else: print("  [SKIP] No data")

        # ── 18. PLANNING Virtual Trajectory (Last 5 Episodes) ─────────────────
        print("[18/25] PLANNING virtual trajectory (last episodes)...")
        if N_virt > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            _wall(ax, color="blue")
            all_eps  = sorted(set(v["ep"] for v in virt_pos_log))
            last_eps = all_eps[-min(5, len(all_eps)):]
            for i, ep_num in enumerate(last_eps):
                alp = 0.25 + 0.75 * (i / max(len(last_eps) - 1, 1))
                for rid in ROBOTS:
                    seg = [(v["nx"], v["ny"])
                           for v in virt_pos_log
                           if v["ep"] == ep_num and v["rid"] == rid]
                    if len(seg) > 1:
                        xs_, ys_ = zip(*seg)
                        ax.plot(xs_, ys_, color=ROBOT_HEX[rid], lw=1.5, alpha=alp)
            if virtual_target:
                ax.scatter(*virtual_target, c="gold", s=250, marker="*",
                           zorder=6, label="Target", edgecolors="k")
            for rid in ROBOTS:
                ax.plot([], [], color=ROBOT_HEX[rid], lw=2, label=f"Robot {rid}")
            ax.set_xlim(0, LOGIC_GRID_MAX); ax.set_ylim(0, LOGIC_GRID_MAX)
            ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_title(f"PLANNING — Last {len(last_eps)} Episode Virtual Trajectories")
            ax.legend(fontsize=9)
            _save(fig, "18_virtual_trajectory")
        else: print("  [SKIP] No data")

        # ── 19. PPO Loss Distribution Histogram ───────────────────────────────
        print("[19/25] PPO loss distribution histogram...")
        if N_upd > 0:
            losses = [u["loss"] for u in update_log]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(losses, bins=min(30, N_upd),
                    color=C_LOSS, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(losses), color=C_THRESH, lw=2, ls="--",
                       label=f"Mean = {np.mean(losses):.4f}")
            ax.set_xlabel("PPO Loss Value"); ax.set_ylabel("Frequency")
            ax.set_title("PPO — Loss Distribution (All Updates)"); ax.legend()
            _save(fig, "19_loss_histogram")
        else: print("  [SKIP] No data")

        # ── 20. Per-Robot Cumulative Reward ───────────────────────────────────
        print("[20/25] Per-robot cumulative reward...")
        if N_eplog > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTS:
                r_data = [(e["ep"], e["reward"])
                          for e in episode_log
                          if e["rid"] == rid and e["phase"] == "PLANNING"]
                if r_data:
                    eps_, rews_ = zip(*r_data)
                    ax.plot(eps_, np.cumsum(rews_),
                            color=ROBOT_HEX[rid], lw=2.5, label=f"Robot {rid}")
            ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
            ax.set_title("PLANNING — Per-Robot Cumulative Reward"); ax.legend()
            _save(fig, "20_per_robot_cumulative_reward")
        else: print("  [SKIP] No data")

        # ── 21. Reward Uncertainty Band (Mean ± Std) ──────────────────────────
        print("[21/25] Reward uncertainty band...")
        if N_ep >= 10:
            rh  = np.array(episode_rewards_history)
            eps = np.arange(1, N_ep + 1)
            w   = min(20, N_ep // 3)
            mus, stds = [], []
            for i in range(len(rh)):
                win = rh[max(0, i - w + 1):i + 1]
                mus.append(np.mean(win)); stds.append(np.std(win))
            mus = np.array(mus); stds = np.array(stds)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(eps, mus - stds, mus + stds,
                            alpha=0.2, color=C_REW, label="±1 Std")
            ax.plot(eps, mus, color=C_REW, lw=2.5, label=f"Rolling Mean (w={w})")
            ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=1.5, ls="--",
                       label=f"Threshold ({CONVERGENCE_THRESHOLD:.0f})")
            ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
            ax.set_title("PPO — Reward Uncertainty Band (Mean ± Std)"); ax.legend()
            _save(fig, "21_reward_band")
        else: print("  [SKIP] Insufficient data (<10 ep)")

        # ── 22. EXECUTION Position Heatmap ────────────────────────────────────
        print("[22/25] EXECUTION position heatmap...")
        if N_exec > 1:
            ux = [e["nx"] for e in execution_log]
            uy = [e["ny"] for e in execution_log]
            fig, ax = plt.subplots(figsize=(7, 7))
            h = ax.hist2d(ux, uy, bins=30,
                          range=[[0, LOGIC_GRID_MAX], [0, LOGIC_GRID_MAX]],
                          cmap="Blues")
            fig.colorbar(h[3], ax=ax, label="Visit Count")
            _wall(ax, color="red")
            if virtual_target:
                ax.scatter(*virtual_target, c="gold", s=250, marker="*",
                           zorder=6, label="Target", edgecolors="k")
            ax.set_xlim(0, LOGIC_GRID_MAX); ax.set_ylim(0, LOGIC_GRID_MAX)
            ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_title("EXECUTION — Robot Position Density Map"); ax.legend(fontsize=9)
            _save(fig, "22_execution_heatmap")
        else: print("  [SKIP] No data")

        # ── 23. Reward Boxplot (Learning Phases) ──────────────────────────────
        print("[23/25] Episode reward boxplot (learning phases)...")
        if N_ep >= 10:
            n_groups = min(5, N_ep // 10)
            if n_groups >= 2:
                rh   = episode_rewards_history
                gsz  = N_ep // n_groups
                groups = [rh[i * gsz:(i + 1) * gsz] for i in range(n_groups)]
                labels = [f"Ep {i*gsz+1}–{(i+1)*gsz}" for i in range(n_groups)]
                fig, ax = plt.subplots(figsize=(10, 5))
                bp = ax.boxplot(groups, labels=labels, patch_artist=True)
                colors_ = plt.cm.viridis(np.linspace(0.2, 0.9, n_groups))
                for patch, clr in zip(bp['boxes'], colors_):
                    patch.set_facecolor(clr); patch.set_alpha(0.75)
                ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=1.5, ls="--",
                           label=f"Threshold ({CONVERGENCE_THRESHOLD:.0f})")
                ax.set_xlabel("Training Phase"); ax.set_ylabel("Episode Reward")
                ax.set_title("PPO — Reward Distribution Across Learning Phases")
                ax.legend(); plt.xticks(rotation=15)
                _save(fig, "23_reward_boxplot")
            else: print("  [SKIP] Insufficient data")
        else: print("  [SKIP] Insufficient data (<10 ep)")

        # ── 24. PLANNING vs EXECUTION Reward Comparison ───────────────────────
        print("[24/25] PLANNING vs EXECUTION reward comparison...")
        plan_rews_ = [e["reward"] for e in episode_log if e["phase"] == "PLANNING"]
        exec_rews_ = [e["reward"] for e in episode_log if e["phase"] == "EXECUTION"]
        if plan_rews_ or exec_rews_:
            data_, lbls_ = [], []
            if plan_rews_: data_.append(plan_rews_); lbls_.append("PLANNING\n(Virtual)")
            if exec_rews_: data_.append(exec_rews_); lbls_.append("EXECUTION\n(Real)")
            fig, ax = plt.subplots(figsize=(7, 5))
            bp = ax.boxplot(data_, labels=lbls_, patch_artist=True, widths=0.4)
            for patch, clr in zip(bp['boxes'], [C_PLAN, C_EXEC][:len(data_)]):
                patch.set_facecolor(clr); patch.set_alpha(0.75)
            ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=1.5, ls="--",
                       label=f"Threshold ({CONVERGENCE_THRESHOLD:.0f})")
            ax.set_ylabel("Episode Reward"); ax.legend()
            ax.set_title("PLANNING vs EXECUTION — Reward Distribution Comparison")
            _save(fig, "24_plan_vs_execution_reward")
        else: print("  [SKIP] No data")

        # ── 25. Best / Worst 5 Episodes ───────────────────────────────────────
        print("[25/25] Best / worst 5 episodes comparison...")
        if N_ep >= 10:
            rh     = episode_rewards_history
            sidx   = np.argsort(rh)
            worst5 = sidx[:5]; best5 = sidx[-5:]
            eps    = np.arange(1, N_ep + 1)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(eps, rh, color="#cccccc", s=15, alpha=0.5,
                       label="All Episodes")
            ax.scatter(worst5 + 1, [rh[i] for i in worst5],
                       color=C_LOSS, s=100, zorder=5, marker="v", label="Worst 5")
            ax.scatter(best5 + 1,  [rh[i] for i in best5],
                       color=C_REW,  s=100, zorder=5, marker="^", label="Best 5")
            ax.axhline(CONVERGENCE_THRESHOLD, color=C_THRESH, lw=1.5, ls="--",
                       label=f"Threshold ({CONVERGENCE_THRESHOLD:.0f})")
            ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
            ax.set_title("PPO — Best / Worst 5 Episodes Comparison"); ax.legend()
            _save(fig, "25_best_worst_episodes")
        else: print("  [SKIP] Insufficient data (<10 ep)")

        # ── Summary ────────────────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print("  ALL PLOTS SAVED SUCCESSFULLY!")
        print(f"  Output directory : {OUT_DIR}")
        print(f"  Total Episodes   : {N_ep}")
        print(f"  PPO Updates      : {N_upd}")
        print(f"  Execution Steps  : {N_exec}")
        print(f"  Virtual Positions: {N_virt}")
        print("=" * 65)

    except Exception as _ge:
        print(f"\n[PLOT ERROR] {_ge}")
        import traceback; traceback.print_exc()
