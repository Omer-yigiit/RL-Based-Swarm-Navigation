

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
#  LOG & RENK SISTEMI
# ==============================================================================
class Log:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    @staticmethod
    def basari(msg): print(f"{Log.GREEN}[BASARI]{Log.RESET} {msg}")
    @staticmethod
    def hata(msg): print(f"{Log.RED}[HATA]{Log.RESET} {msg}")
    @staticmethod
    def uyari(msg): print(f"{Log.YELLOW}[UYARI]{Log.RESET} {msg}")
    @staticmethod
    def bilgi(msg): print(f"{Log.CYAN}[BILGI]{Log.RESET} {msg}")
    @staticmethod
    def plan(msg): print(f"{Log.BLUE}[PLAN]{Log.RESET} {msg}")
    @staticmethod
    def egitim(ep, step, rew, rid, ls, rs):
        print(f"{Log.CYAN}[EGITIM]{Log.RESET} R:{rid} | Ep:{Log.BOLD}{ep}{Log.RESET} Step:{step} | Odul:{Log.GREEN}{rew:.2f}{Log.RESET} | Motor: L:{ls} R:{rs}")

# ==============================================================================
#  PPO (PROXIMAL POLICY OPTIMIZATION) ÖĞRENME MOTORU
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Log.bilgi(f"PyTorch Cihazi: {device}")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.returns = []
    
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
            if is_terminal: discounted_reward = 0
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
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        value = self.critic(features)
        return action_mean, action_std, value

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_mean, action_std, _ = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy()[0], action_logprob.detach().cpu().numpy()[0]

    def evaluate(self, state, action):
        action_mean, action_std, state_value = self.forward(state)
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        return action_logprobs, state_value, dist_entropy

class PPO:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = np.array(state, dtype=np.float32)
        action, action_logprob = self.policy_old.act(state)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        action = np.clip(action, -1.0, 1.0)
        return action

    def select_action_deterministic(self, state):
        """Exploration olmadan, sadece mean aksiyonu döner (UYGULAMA fazı için)."""
        state = np.array(state, dtype=np.float32)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_mean, _, _ = self.policy(state_t)
            action = action_mean.cpu().numpy()[0]
        return np.clip(action, -1.0, 1.0)

    def update(self, memory):
        rewards = torch.tensor(memory.returns, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.tensor(np.array(memory.states)).to(device)
        old_actions = torch.tensor(np.array(memory.actions)).to(device)
        old_logprobs = torch.tensor(np.array(memory.logprobs)).to(device)

        total_loss = 0
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            advantages = rewards - state_values.detach()
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.mean().item()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        return total_loss / self.K_epochs

# ==============================================================================
#  KAMERA VE DONANIM AYARLARI
# ==============================================================================
class GecikmesizKamera:
    def __init__(self, kaynak):
        self.kamera = cv2.VideoCapture(kaynak)
        self.kamera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.durum, self.kare = self.kamera.read()
        self.calisiyor = True
        if self.kamera.isOpened():
            self.thread = threading.Thread(target=self.guncelle, daemon=True)
            self.thread.start()
        else: self.calisiyor = False

    def guncelle(self):
        while self.calisiyor: self.durum, self.kare = self.kamera.read()
    def oku(self): return self.kare
    def durdur(self):
        self.calisiyor = False
        if hasattr(self, 'kamera') and self.kamera: self.kamera.release()

at_detector = Detector(families='tag36h11', nthreads=6)
LOGIC_GRID_MAX = 400.0
ROBOTLAR = [0, 1, 2, 3]
ROBOT_RENKLERI = {0: (255, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0), 3: (255, 255, 0)}

# Sınırlar (Trapez/Yamuk perspektif)
WALL_POLYGON = [
    (350.0, 50.0),
    (350.0, 350.0),
    (50.0, 290.0),
    (50.0, 110.0)
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

# RL Çevre Değişkenleri
ARENA_SIZE = 3.0
POS_SCALE = ARENA_SIZE / 2.0
MAX_DIST = ARENA_SIZE * 1.41
RL_MOTOR_SCALE = 20

virtual_target    = None
virtual_obstacles = []
EMA_ALPHA         = 0.3
HEDEF_BEKLENIYOR  = False   # [T] sonrasi hedef tiklanmasi bekleniyor

# Hedefe ulaşan robotları takip eder (PLANLAMA ve UYGULAMA için ayrı)
plan_robots_at_goal  = set()   # PLANLAMA fazında hedefe varan sanal robot ID'leri
uygulama_robots_at_goal = set()  # UYGULAMA fazında hedefe varan gerçek robot ID'leri

def generate_environment(only_obstacles=False):
    """Engelleri (ve isteğe bağlı olarak hedefi) yeniler."""
    global virtual_target, virtual_obstacles
    virtual_obstacles.clear()

    aktif_pozlar = [(robot_states[rid]["nx"], robot_states[rid]["ny"])
                    for rid in ROBOTLAR if robot_states[rid]["nx"] is not None]

    # Hedef üret — yalnızca zorunluysa (only_obstacles=False ve target yoksa)
    if not only_obstacles or virtual_target is None:
        for _ in range(100):
            tx = random.uniform(WALL_MIN_X + 20, WALL_MAX_X - 20)
            ty = random.uniform(WALL_MIN_Y + 20, WALL_MAX_Y - 20)
            if not is_point_in_polygon(tx, ty): continue
            ok = all(math.sqrt((tx-rx)**2+(ty-ry)**2) > 80 for rx,ry in aktif_pozlar)
            if ok:
                virtual_target = (tx, ty)
                break
        if virtual_target is None:
            virtual_target = (200.0, 200.0)

    tx, ty = virtual_target

    # ── Engel üretimi ─────────────────────────────────────────────────────────
    # Küçük (r=5-10) ve az (maks 2) engel — direkt yolu kesmeyecek şekilde
    N_ENGEL = 2
    for _ in range(N_ENGEL):
        for _ in range(80):
            ox = random.uniform(WALL_MIN_X + 20, WALL_MAX_X - 20)
            oy = random.uniform(WALL_MIN_Y + 20, WALL_MAX_Y - 20)
            if not is_point_in_polygon(ox, oy): continue
            r = random.uniform(5, 10)          # Küçük engel

            # Hedefe çok yakın olmasın
            if math.sqrt((ox-tx)**2+(oy-ty)**2) < (r + 70): continue

            # Robotlara çok yakın olmasın
            robot_conflict = any(
                math.sqrt((ox-rx)**2+(oy-ry)**2) < (r + 45)
                for rx, ry in aktif_pozlar
            )
            if robot_conflict: continue

            # Her robot-hedef çiftinin direkt yolunu kesmemeye çalış
            # (Çizgi-nokta mesafesi kontrolü)
            blocks_path = False
            for rx, ry in aktif_pozlar:
                # Robottan hedefe doğru çizginin engele olan mesafesi
                seg_len = math.sqrt((tx-rx)**2+(ty-ry)**2)
                if seg_len < 1: continue
                # Nokta-çizgi mesafesi
                cross = abs((oy-ry)*(tx-rx) - (ox-rx)*(ty-ry)) / seg_len
                # Engel, çizgi üzerinde mi? (parametre t ∈ [0.15, 0.85])
                dot = ((ox-rx)*(tx-rx)+(oy-ry)*(ty-ry)) / (seg_len*seg_len)
                if 0.15 <= dot <= 0.85 and cross < (r + 25):
                    blocks_path = True
                    break
            if blocks_path: continue

            # Diğer engellerle çakışmasın
            too_close = any(
                math.sqrt((vox-ox)**2+(voy-oy)**2) < (r + vr + 25)
                for vox, voy, vr in virtual_obstacles
            )
            if too_close: continue

            virtual_obstacles.append((ox, oy, r))
            break

def mouse_callback(event, x, y, flags, param):
    global virtual_target, fw, fh, HEDEF_BEKLENIYOR
    if event == cv2.EVENT_LBUTTONDOWN:
        if 'fw' not in globals() or 'fh' not in globals(): return
        lx = (x / fw) * LOGIC_GRID_MAX
        ly = (y / fh) * LOGIC_GRID_MAX
        if not is_point_in_polygon(lx, ly):
            Log.uyari("Secilen nokta arena disinda! Gecerli bir noktaya tiklayin.")
            return
        virtual_target = (lx, ly)
        generate_environment(only_obstacles=True)
        Log.bilgi(f"Hedef belirlendi: ({lx:.1f}, {ly:.1f})")

        if HEDEF_BEKLENIYOR:
            # [T] sonrasi hedef bekleniyor — şimdi PLANLAMA başlatılabilir
            HEDEF_BEKLENIYOR = False
            _baslat_planlama()

robot_states = {rid: {
    "nx": None, "ny": None, "angle": None, "base_angle": None,
    "corners": None, "logic_corners": [], "radius": 15.0,
    "last_seen_time": time.time(), "RL_STATE": "COMPUTE",
    "action_send_time": 0, "current_action": [0.0, 0.0],
    "ep_reward": 0, "ep_steps": 0, "prev_d": None,
    "wall_bounce_target": (200.0, 200.0), "episode": 1
} for rid in ROBOTLAR}

def get_calibrated_angle(rid, raw_angle):
    """
    Ham AprilTag açısını robot'un ileri yönüne dönüştürür.
    rlkodu'ndaki dx/dy hesabı (köşeler[3+2] - köşeler[0+1]) geriye yönü
    verir; +180° ile ileri yöne çevriliyor.
    base_angle göreli kalibrasyonu kaldırıldı — mutlak [0,360] kullanılıyor.
    Bu, PLANLAMA sanal açılarıyla (da [0,360]) tutarlıdır.
    """
    return (raw_angle + 180) % 360

# ==============================================================================
#  OBSERVATION, REWARD & VIRTUAL ENVIRONMENT LOGIC
# ==============================================================================

def ray_segment_intersect(px, py, dx, dy, x1, y1, x2, y2):
    s_dx = x2 - x1
    s_dy = y2 - y1
    det = dx * s_dy - dy * s_dx
    if abs(det) < 1e-6: return float('inf')
    diff_x = px - x1
    diff_y = py - y1
    u = (dx * diff_y - dy * diff_x) / det
    t = (s_dx * diff_y - s_dy * diff_x) / det
    if 0.0 <= u <= 1.0 and t > 0.0: return t
    return float('inf')

def get_ray_wall_intersection(nx, ny, rad):
    dx, dy = math.cos(rad), math.sin(rad)
    t_min = float('inf')
    n = len(WALL_POLYGON)
    for i in range(n):
        p1 = WALL_POLYGON[i]
        p2 = WALL_POLYGON[(i + 1) % n]
        t = ray_segment_intersect(nx, ny, dx, dy, p1[0], p1[1], p2[0], p2[1])
        if t < t_min: t_min = t
    return t_min

def compute_virtual_sensors(nx, ny, heading_deg, rid, override_positions=None):
    """
    override_positions: {rid: (nx, ny)} — PLANLAMA fazında sanal konumlar kullanılır.
    """
    sensors = [0.0] * 8
    sensor_angles = [0, 45, 90, 135, 180, -135, -90, -45]
    max_sens_dist = 60.0
    
    active_obstacles = []
    for (ox, oy, r) in virtual_obstacles:
        active_obstacles.append((ox, oy, r))
    for oid in ROBOTLAR:
        if oid != rid:
            if override_positions and oid in override_positions:
                onx, ony = override_positions[oid]
                active_obstacles.append((onx, ony, robot_states[oid]["radius"]))
            elif robot_states[oid]["nx"] is not None:
                active_obstacles.append((robot_states[oid]["nx"], robot_states[oid]["ny"], robot_states[oid]["radius"]))
            
    for i, sa in enumerate(sensor_angles):
        rad = math.radians(heading_deg + sa)
        t_wall = get_ray_wall_intersection(nx, ny, rad)
        if t_wall < max_sens_dist:
            sensors[i] = max(sensors[i], 1.0 - t_wall / max_sens_dist)
        for (ox, oy, r) in active_obstacles:
            dist = math.sqrt((ox - nx)**2 + (oy - ny)**2)
            if dist < max_sens_dist + r:
                angle_to_obs = math.degrees(math.atan2(oy - ny, ox - nx))
                rel_angle = (angle_to_obs - heading_deg + 180) % 360 - 180
                diff = abs((rel_angle - sa + 180) % 360 - 180)
                if diff < 25:
                    val = max(0.0, 1.0 - max(0, dist - r) / max_sens_dist)
                    sensors[i] = max(sensors[i], val)
                    
    return sensors

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0: return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.sqrt((px - (x1 + t * dx))**2 + (py - (y1 + t * dy))**2)

def _build_obs_reward(rid, nx, ny, angle, prev_dist=None, prev_action=None, override_positions=None):
    """
    Ortak observation + reward hesaplama çekirdeği.
    nx, ny, angle: robotun konumu (gerçek veya sanal).
    """
    if nx is None or virtual_target is None:
        return None, 0.0, False, None

    px = (nx / LOGIC_GRID_MAX) * ARENA_SIZE - (ARENA_SIZE / 2.0)
    py = (ny / LOGIC_GRID_MAX) * ARENA_SIZE - (ARENA_SIZE / 2.0)
    gx = (virtual_target[0] / LOGIC_GRID_MAX) * ARENA_SIZE - (ARENA_SIZE / 2.0)
    gy = (virtual_target[1] / LOGIC_GRID_MAX) * ARENA_SIZE - (ARENA_SIZE / 2.0)

    dist_logic = math.sqrt((virtual_target[0] - nx)**2 + (virtual_target[1] - ny)**2)
    dist_metric = min(math.sqrt((gx - px)**2 + (gy - py)**2), MAX_DIST)

    target_angle_rad = math.atan2(gy - py, gx - px)
    heading_rad = math.radians(angle) if angle is not None else 0.0
    rel_angle = target_angle_rad - heading_rad
    while rel_angle > math.pi: rel_angle -= 2 * math.pi
    while rel_angle < -math.pi: rel_angle += 2 * math.pi

    sensors = compute_virtual_sensors(nx, ny, angle if angle is not None else 0.0, rid, override_positions)
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
    done = False

    if prev_dist is not None:
        progress = prev_dist - dist_logic
        reward += progress * 1.0

    # HEDEF
    if dist_logic < (robot_radius + 15):
        reward += 100.0
        done = True

    # SANAL ENGELLER
    for (ox, oy, r) in virtual_obstacles:
        if math.sqrt((ox - nx)**2 + (oy - ny)**2) < (r + robot_radius - 2):
            reward -= 50.0
            done = True
            break

    # DİĞER ROBOTLARLA YAKINLAŞMA CEZASI (sanal veya gerçek pozisyon)
    for oid in ROBOTLAR:
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
                done = True
                break

    # DUVAR ÇARPMA CEZASI
    if not is_point_in_polygon(nx, ny):
        reward -= 50.0
        done = True

    return obs, reward, done, dist_logic

def get_observation_and_reward(rid, prev_dist=None, prev_action=None):
    """Gerçek kamera pozisyonu ile obs/reward (UYGULAMA fazı)."""
    st = robot_states[rid]
    return _build_obs_reward(rid, st["nx"], st["ny"], st["angle"], prev_dist, prev_action)

def get_virtual_obs_reward(rid, vst, prev_dist=None, prev_action=None, all_vst=None):
    """
    Sanal robot pozisyonu ile obs/reward (PLANLAMA fazı).
    all_vst: tüm sanal robot state'leri (diğer robotların sanal konumu için).
    """
    override = {}
    if all_vst:
        for oid, ovst in all_vst.items():
            if oid != rid:
                override[oid] = (ovst["nx"], ovst["ny"])
    return _build_obs_reward(rid, vst["nx"], vst["ny"], vst["angle"], prev_dist, prev_action, override)

# ==============================================================================
#  SANAL FİZİK MOTORU (PLANLAMA FAZI)
# ==============================================================================
SIM_DT = 0.04           # Sanal adım süresi (saniye)
SIM_SPEED_SCALE = 26.0  # RL aksiyon → piksel/adım (Robot yavaşladığı için bu da düşürüldü)

def virtual_step(vst, action):
    """
    RL aksiyonunu sanal robot konumuna uygular.
    Gerçek robota HİÇBİR komut göndermez.

    Dönüş yönü sözleşmesi (gerçek diferansiyel sürüşle eşleşir):
      Sol motor (action[0]) > Sağ motor (action[1])  →  sağa dön  →  açı artar (saat yönü)
      Sağ motor (action[1]) > Sol motor (action[0])  →  sola dön  →  açı azalır (saat yönünün tersi)
    """
    ls = float(np.clip(action[0], -1.0, 1.0)) * SIM_SPEED_SCALE
    rs = float(np.clip(action[1], -1.0, 1.0)) * SIM_SPEED_SCALE
    v  = (ls + rs) / 2.0           # İleri hız (piksel/adım)
    w  = (ls - rs) / 20.0          # Açısal hız: sol>sağ → sağa dön → açı artar ✓
    angle_rad = math.radians(vst["angle"])
    vst["nx"]   += v * math.cos(angle_rad) * SIM_DT
    vst["ny"]   += v * math.sin(angle_rad) * SIM_DT
    vst["angle"] = (vst["angle"] + math.degrees(w * SIM_DT)) % 360
    # Sınır içinde tut
    vst["nx"] = float(np.clip(vst["nx"], WALL_MIN_X + 15, WALL_MAX_X - 15))
    vst["ny"] = float(np.clip(vst["ny"], WALL_MIN_Y + 15, WALL_MAX_Y - 15))

# ==============================================================================
#  SERİ PORT & ROBOT COMM
# ==============================================================================
GATEWAY_PORT = 'COM7'
gateway_serial = None
def baglan_seri_port():
    global gateway_serial
    try:
        gateway_serial = serial.Serial(GATEWAY_PORT, 115200, timeout=0.1)
        gateway_serial.dtr, gateway_serial.rts = False, False
        Log.basari(f"{GATEWAY_PORT} Seri Porta Baglandi!")
    except: pass

def send_robot_command(rid, left, right):
    if gateway_serial and gateway_serial.is_open:
        komut = f"<{rid},{int(left)},{int(right)}>\n"
        gateway_serial.write(komut.encode('utf-8'))
        gateway_serial.flush()

baglan_seri_port()

def durdur_tum_robotlar():
    for rid in ROBOTLAR: send_robot_command(rid, 0, 0)

# ==============================================================================
#  3 FAZLI SİSTEM DEĞİŞKENLERİ
# ==============================================================================
# SISTEM_FAZI: 'HAZIRLIK' | 'PLANLAMA' | 'UYGULAMA'
SISTEM_FAZI = 'HAZIRLIK'

# Kameradan kaydedilen başlangıç pozisyonları: {rid: (nx, ny, angle)}
start_positions = {}

# Sanal robot state'leri (PLANLAMA fazında kullanılır)
# {rid: {nx, ny, angle, prev_d, action, ep_steps, episode, ep_reward}}
virtual_robot_states = {}

# PLANLAMA parametreleri
PLANLAMA_MIN_EPISODE      = 20     # En az bu kadar episode yapilmadan bitmez
PLANLAMA_EPISODE_HEDEFI   = 500    # Maksimum episode (yakinsama olmazsa bu sinirda biter)
PPO_UPDATE_TIMESTEP       = 1600   # Her N sanal adim (train_memory boyutu) olunca guncelle
MAX_EP_STEPS              = 400    # Bir episode'da max adim

# Yakinsama (Convergence) kriterleri
YAKINSAMA_ORT_PENCERE     = 20     # Son kac episode'un ortalamasi alinsin
YAKINSAMA_ESIK            = 60.0   # Ortalama odul bu esigi gecerse "ogrendi" sayilir
YAKINSAMA_ARDISIK         = 3      # Kac ust uste PPO guncellemesi esigi korumali

planlama_toplam_episode   = 0      # PLANLAMA fazinda tamamlanan toplam episode
plan_update_pending       = False
yakinsama_ardisik_sayac   = 0      # Esigi gecen ardisik guncelleme sayisi


def _banner(mesaj, renk=None):
    """Konsolda geniş, göz alıcı banner yazar."""
    renk = renk or Log.BOLD
    cizgi = "═" * 60
    print(f"\n{renk}{cizgi}")
    for satir in mesaj.split('\n'):
        bosluk = max(0, (60 - len(satir)) // 2)
        print(f"{' ' * bosluk}{satir}")
    print(f"{cizgi}{Log.RESET}\n")


def kaydet_baslangic_pozisyonlari():
    """Kamerada görünen tüm robotların anlık konumunu start_positions'a kaydet."""
    global start_positions
    start_positions.clear()
    bulunan = 0
    for rid in ROBOTLAR:
        st = robot_states[rid]
        if st["nx"] is not None:
            start_positions[rid] = (st["nx"], st["ny"], st["angle"] if st["angle"] is not None else 0.0)
            Log.basari(f"[HAZIRLIK] Robot {rid} START konumu: ({st['nx']:.1f}, {st['ny']:.1f}, {st['angle']}°)")
            bulunan += 1
        else:
            Log.uyari(f"[HAZIRLIK] Robot {rid} kamerada YOK — baslangic kaydedilemedi!")
    return bulunan


def init_virtual_states():
    """
    PLANLAMA başlarken sanal state'leri başlangıç pozisyonlarından
    ama RASTGELE açılardan oluştur.
    """
    global virtual_robot_states
    virtual_robot_states.clear()
    for rid in ROBOTLAR:
        if rid in start_positions:
            nx, ny, _ = start_positions[rid]
        else:
            nx, ny = 200.0, 200.0
        
        # Açıyı randomize et
        angle = random.uniform(0, 360)
        
        virtual_robot_states[rid] = {
            "nx": nx, "ny": ny, "angle": angle,
            "prev_d": None, "action": [0.0, 0.0],
            "ep_steps": 0, "episode": 0, "ep_reward": 0.0
        }
    Log.plan(f"Sanal state'ler (Rastgele Acili) olusturuldu: {list(virtual_robot_states.keys())}")


def gec_planlama():
    """
    HAZIRLIK → Hedef Bekleme geçişi.
    Robot pozisyonları kaydedilir, motorlar durdurulur.
    Kullanıcı ekrana tıklayarak hedefi seçene kadar beklenir.
    """
    global HEDEF_BEKLENIYOR

    bulunan = kaydet_baslangic_pozisyonlari()
    if bulunan == 0:
        Log.hata("Hic robot bulunamadi! Planlama baslatilmiyor.")
        return
    durdur_tum_robotlar()

    HEDEF_BEKLENIYOR = True
    _banner(
        "✔  HAZIRLIK TAMAMLANDI\n"
        "Robot konumlari kaydedildi.\n"
        ">>> EKRANDA HEDEFE TIKLAYARAK SECIN <<<",
        Log.YELLOW
    )
    Log.bilgi("Hedef seciminizi bekliyor... Sol tikla hedefi belirleyin.")


def _baslat_planlama():
    """
    Hedef seçildikten sonra PLANLAMA fazını başlatır.
    mouse_callback tarafından çağrılır.
    """
    global SISTEM_FAZI, planlama_toplam_episode, plan_update_pending, global_time_step, yakinsama_ardisik_sayac, plan_robots_at_goal

    init_virtual_states()
    planlama_toplam_episode = 0
    yakinsama_ardisik_sayac = 0
    plan_update_pending = False
    plan_robots_at_goal.clear()
    for rm in robot_memories.values(): rm.clear()
    train_memory.clear()
    global_time_step = 0
    SISTEM_FAZI = 'PLANLAMA'
    phase_log.append({"t": time.time()-_graf_start, "from_p": "HAZIRLIK", "to_p": "PLANLAMA", "ep_count": 0})

    _banner(
        f"► PLANLAMA BASLADI\n"
        f"Hedef: ({virtual_target[0]:.0f}, {virtual_target[1]:.0f})  |  Engel: {len(virtual_obstacles)}\n"
        f"Yakinsama: Son {YAKINSAMA_ORT_PENCERE} ep ort. >= {YAKINSAMA_ESIK:.0f} (x{YAKINSAMA_ARDISIK})\n"
        f"Motorlar KAPALI — Robotlar hareketsiz",
        Log.BLUE
    )



def gec_uygulama():
    """PLANLAMA → UYGULAMA geçişi."""
    global SISTEM_FAZI, uygulama_robots_at_goal

    _banner(
        f"✔  PLANLAMA TAMAMLANDI\n"
        f"{planlama_toplam_episode} sanal episode tamamlandi.\n"
        f"SISTEMI UYGULAMA FAZINA GECIRILIYOR...",
        Log.GREEN
    )

    durdur_tum_robotlar()
    uygulama_robots_at_goal.clear()
    # Gerçek robot state'lerini sıfırla
    for rid in ROBOTLAR:
        robot_states[rid]["RL_STATE"] = "COMPUTE"
        robot_states[rid]["prev_d"] = None
        robot_states[rid]["current_action"] = [0.0, 0.0]
        robot_states[rid]["action_send_time"] = 0
        robot_states[rid]["ep_steps"] = 0
        robot_states[rid]["ep_reward"] = 0
    SISTEM_FAZI = 'UYGULAMA'
    phase_log.append({"t": time.time()-_graf_start, "from_p": "PLANLAMA", "to_p": "UYGULAMA", "ep_count": planlama_toplam_episode})

    _banner(
        f"► UYGULAMA BASLADI\n"
        f"Ogrrenilmis policy aktif!\n"
        f"Her robot ID'ye ayri motor komutu gonderiliyor.\n"
        f"R0  R1  R2  R3  →  HEDEFE!",
        Log.GREEN
    )


def gec_hazirlik():
    """Herhangi bir fazdan HAZIRLIK'a dön."""
    global SISTEM_FAZI, plan_robots_at_goal, uygulama_robots_at_goal
    onceki_faz = SISTEM_FAZI
    durdur_tum_robotlar()
    SISTEM_FAZI = 'HAZIRLIK'
    phase_log.append({"t": time.time()-_graf_start, "from_p": onceki_faz, "to_p": "HAZIRLIK", "ep_count": planlama_toplam_episode})
    start_positions.clear()
    virtual_robot_states.clear()
    plan_robots_at_goal.clear()
    uygulama_robots_at_goal.clear()

    _banner(
        f"↺  SISTEM SIFIRLANDI\n"
        f"Onceki faz: {onceki_faz}\n"
        f"HAZIRLIK MODUNA DONULDU\n"
        f"Robotlari konumlandirin, ardindan [T] ye basin.",
        Log.CYAN
    )


def execute_learned_policy(rid):
    """
    UYGULAMA fazında: öğrenilmiş policy ile robot ID'ye özgü motor komutu üret ve gönder.
    Exploration YOK (deterministic mean aksiyon).
    """
    global uygulama_robots_at_goal
    st = robot_states[rid]
    if st["nx"] is None:
        return

    # Gerçek kamera pozisyonu ile obs al
    obs, _, done, d = get_observation_and_reward(rid, st["prev_d"], st["current_action"])
    if obs is None:
        return

    # Deterministik aksiyon (exploration yok)
    action = ppo_agent.select_action_deterministic(obs)
    ls = int(np.clip(action[0] * RL_MOTOR_SCALE, -30, 30))
    rs = int(np.clip(action[1] * RL_MOTOR_SCALE, -30, 30))
    send_robot_command(rid, ls, rs)
    uygulama_log.append({"t": time.time()-_graf_start, "rid": rid,
                          "nx": st["nx"] or 0.0, "ny": st["ny"] or 0.0,
                          "dist": d if d is not None else 0.0,
                          "motor_l": ls, "motor_r": rs, "done": False})

    st["current_action"] = action
    st["prev_d"] = d
    st["ep_steps"] += 1

    if st["ep_steps"] % 6 == 0:
        Log.bilgi(f"[UYGULAMA] R{rid} | Adim:{st['ep_steps']} | Hedef uzaklik:{d:.1f} | L:{ls} R:{rs}")

    if done:
        if uygulama_log and uygulama_log[-1]["rid"] == rid:
            uygulama_log[-1]["done"] = True
        uygulama_robots_at_goal.add(rid)
        Log.basari(f"[UYGULAMA] Robot {rid} HEDEFE ULASTI! ({len(uygulama_robots_at_goal)}/{len(ROBOTLAR)} robot hedede)")
        st["ep_steps"] = 0
        st["prev_d"] = None

        # Tüm aktif (kamerada görünen) robotlar hedefe ulaştıysa yeni hedef üret
        aktif_robotlar = {r for r in ROBOTLAR if robot_states[r]["nx"] is not None}
        if aktif_robotlar and uygulama_robots_at_goal >= aktif_robotlar:
            Log.basari("[UYGULAMA] TUM ROBOTLAR HEDEFE ULASTI! Sadece engeller yenileniyor...")
            generate_environment(only_obstacles=True)
            uygulama_robots_at_goal.clear()

# ==============================================================================
#  EĞİTİM & ÇALIŞMA DEĞİŞKENLERİ
# ==============================================================================
kamera = GecikmesizKamera("http://192.168.1.177:8080/video")

ppo_agent = PPO(obs_dim=16, action_dim=2, lr=0.0003)
train_memory = RolloutBuffer()
robot_memories = {rid: RolloutBuffer() for rid in ROBOTLAR}

MODEL_SAVE_PATH = "live_model_weights.pt"
if os.path.exists(MODEL_SAVE_PATH):
    try:
        ppo_agent.policy.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())
        Log.bilgi(f"[{MODEL_SAVE_PATH}] yuklendi.")
        Log.uyari("UYARI: Fizik duzeltmesi yapildi (w=(ls-rs)/20 ve mutlak aci).")
        Log.uyari("Onceki model yanlis fizikle egitilmis olabilir. Silip yeniden egitmek onerilir:")
        Log.uyari(f"  del {MODEL_SAVE_PATH}")
    except Exception as e:
        Log.uyari("Gecmisten gelen veri uyumsuz. Sifirdan egitim baslatiliyor...")

e_stop_active   = False
global_time_step = 0
last_loss        = 0.0
update_count     = 0
selected_manual_robot = 0
manual_motor_active   = False
manual_last_cmd       = 0.0
episode_rewards_history = []

# ═══════════════════════════════════════════════════════════════════════
#  GRAFİK İÇİN LOG YAPILARI  (simülasyon bitince 25 grafik üretilir)
# ═══════════════════════════════════════════════════════════════════════
update_log   = []  # PPO güncelleme başına: {update, loss, avg_r20, global_step, plan_ep}
episode_log  = []  # Episode başına:        {ep, rid, reward, steps, phase}
uygulama_log = []  # UYGULAMA adımı:        {t, rid, nx, ny, dist, motor_l, motor_r, done}
phase_log    = []  # Faz geçişleri:         {t, from_p, to_p, ep_count}
virt_pos_log = []  # Sanal pozisyon:        {ep, rid, nx, ny}
_graf_start  = time.time()

# CSV Logger
CSV_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_file   = open(CSV_LOG_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'faz', 'update', 'loss', 'avg_reward_20', 'buffer_size', 'global_step', 'plan_episode'])
Log.bilgi(f"Egitim logu kaydediliyor: {CSV_LOG_PATH}")

cv2.namedWindow('MULTI_AGENT_RL', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MULTI_AGENT_RL', 900, 700)
cv2.setMouseCallback('MULTI_AGENT_RL', mouse_callback)
FONT = cv2.FONT_HERSHEY_SIMPLEX

_banner(
    "◉  SISTEM HAZIR\n"
    "HAZIRLIK MODUNDA BASLADI\n"
    "Robotlari arena icine yerlestirin.\n"
    "[T] → Planlamaya gec   [WASD] Manuel sur\n"
    "[SPACE] E-Stop   [Q] Cikis",
    Log.CYAN
)

try:
    while True:
        simdiki_zaman = time.time()

        # ──────────────────────────────────────────────
        # 1. KAMERA VE TAG TAKİBİ
        # ──────────────────────────────────────────────
        frame = kamera.oku()
        if frame is None:
            time.sleep(0.01)
            continue

        # Global fw, fh gunclle (mouse callback icin)
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray)

        seen_this_frame = set()
        for tag in tags:
            rid = tag.tag_id
            if rid in ROBOTLAR:
                seen_this_frame.add(rid)
                st = robot_states[rid]
                st["last_seen_time"] = simdiki_zaman
                cx, cy = int(tag.center[0]), int(tag.center[1])
                raw_nx = (cx / fw) * LOGIC_GRID_MAX
                raw_ny = (cy / fh) * LOGIC_GRID_MAX
                if st["nx"] is None:
                    st["nx"], st["ny"] = raw_nx, raw_ny
                else:
                    st["nx"] = EMA_ALPHA * raw_nx + (1 - EMA_ALPHA) * st["nx"]
                    st["ny"] = EMA_ALPHA * raw_ny + (1 - EMA_ALPHA) * st["ny"]

                corners = tag.corners.astype(int)
                st["corners"] = corners
                dx = (corners[3][0] + corners[2][0])/2.0 - (corners[0][0] + corners[1][0])/2.0
                dy = (corners[3][1] + corners[2][1])/2.0 - (corners[0][1] + corners[1][1])/2.0
                raw_deg = int(math.degrees(math.atan2(dy, dx))) % 360
                st["angle"] = get_calibrated_angle(rid, raw_deg)

                p_width = math.sqrt((corners[0][0] - corners[1][0])**2 + (corners[0][1] - corners[1][1])**2)
                st["radius"] = max((p_width / fw) * LOGIC_GRID_MAX * 0.707, 10.0)

                l_corners = []
                for pt in tag.corners:
                    l_corners.append(((pt[0] / fw) * LOGIC_GRID_MAX, (pt[1] / fh) * LOGIC_GRID_MAX))
                st["logic_corners"] = l_corners

        for rid in ROBOTLAR:
            if rid not in seen_this_frame:
                robot_states[rid]["corners"] = None

        if virtual_target is None:
            any_valid = next((st for st in robot_states.values() if st["nx"] is not None), None)
            if any_valid: generate_environment()

        # ──────────────────────────────────────────────
        # 2A. PLANLAMA FAZI — Sanal Simülasyon (Motor Yok)
        # ──────────────────────────────────────────────
        if SISTEM_FAZI == 'PLANLAMA' and not e_stop_active and not plan_update_pending:

            for rid in ROBOTLAR:
                if rid not in virtual_robot_states:
                    continue
                vst = virtual_robot_states[rid]

                # Obs al (sanal konumdan)
                obs, _, _, _ = get_virtual_obs_reward(rid, vst, vst["prev_d"], vst["action"], virtual_robot_states)
                if obs is None:
                    continue

                # RL aksiyonu seç (exploration ile — eğitim)
                action = ppo_agent.select_action(obs, robot_memories[rid])
                vst["action"] = action

                # Sanal adım (motor komutu YOK)
                virtual_step(vst, action)
                if len(virt_pos_log) < 120000:
                    virt_pos_log.append({"ep": vst["episode"], "rid": rid, "nx": vst["nx"], "ny": vst["ny"]})

                # Yeni obs + ödül
                _, reward, done, new_d = get_virtual_obs_reward(rid, vst, vst["prev_d"], action, virtual_robot_states)

                robot_memories[rid].rewards.append(reward)
                robot_memories[rid].is_terminals.append(done)

                vst["prev_d"]    = new_d
                vst["ep_reward"] += reward
                vst["ep_steps"]  += 1
                global_time_step += 1

                # Hedef bulduysa log yaz
                if done and new_d is not None and new_d < (robot_states[rid]["radius"] + 15):
                    Log.plan(f"[PLAN] R{rid} sanal hedefe ulasti! Engeller yenileniyor...")
                    generate_environment(only_obstacles=True)
                    plan_robots_at_goal.clear() # Tum robotlarin ulasmasini beklemeye gerek yok

                # Episode bitti?
                if done or vst["ep_steps"] >= MAX_EP_STEPS:
                    Log.plan(f"[PLAN] R{rid} Ep:{vst['episode']} TAMAM | Odul:{vst['ep_reward']:.1f}")
                    
                    robot_memories[rid].compute_returns(ppo_agent.gamma)
                    train_memory.states.extend(robot_memories[rid].states)
                    train_memory.actions.extend(robot_memories[rid].actions)
                    train_memory.logprobs.extend(robot_memories[rid].logprobs)
                    train_memory.rewards.extend(robot_memories[rid].rewards)
                    train_memory.is_terminals.extend(robot_memories[rid].is_terminals)
                    train_memory.returns.extend(robot_memories[rid].returns)
                    robot_memories[rid].clear()
                    
                    episode_rewards_history.append(vst["ep_reward"])
                    episode_log.append({"ep": planlama_toplam_episode, "rid": rid,
                                        "reward": vst["ep_reward"], "steps": vst["ep_steps"], "phase": "PLANLAMA"})
                    planlama_toplam_episode += 1
                    vst["episode"]  += 1
                    vst["ep_reward"] = 0.0
                    vst["ep_steps"]  = 0
                    vst["prev_d"]    = None

                    # Konumu sabit tut, ama açıyı randomize et
                    if rid in start_positions:
                        vst["nx"], vst["ny"], _ = start_positions[rid]
                    else:
                        vst["nx"], vst["ny"] = 200.0, 200.0
                    vst["angle"] = random.uniform(0, 360)

                # Global PPO güncelleme tetikleyici
                if len(train_memory.states) >= PPO_UPDATE_TIMESTEP:
                    plan_update_pending = True

            # PPO ağ güncellemesi
            if plan_update_pending:
                plan_update_pending = False
                Log.plan(f"--- PPO GUNCELLEME | Buffer:{len(train_memory.states)} | TotalEp:{planlama_toplam_episode} ---")
                try:
                    last_loss = ppo_agent.update(train_memory)
                    train_memory.clear()
                    update_count += 1
                    torch.save(ppo_agent.policy.state_dict(), MODEL_SAVE_PATH)
                    if update_count % 20 == 0:
                        backup_path = MODEL_SAVE_PATH.replace('.pt', f'_v{update_count}.pt')
                        torch.save(ppo_agent.policy.state_dict(), backup_path)
                        Log.bilgi(f"Yedek kaydedildi: {backup_path}")
                    avg_r20 = np.mean(episode_rewards_history[-YAKINSAMA_ORT_PENCERE:]) if len(episode_rewards_history) >= YAKINSAMA_ORT_PENCERE else None
                    avg_display = avg_r20 if avg_r20 is not None else 0.0
                    csv_writer.writerow([datetime.now().isoformat(), 'PLANLAMA', update_count, f'{last_loss:.4f}',
                                         f'{avg_display:.1f}', len(train_memory.states), global_time_step, planlama_toplam_episode])
                    csv_file.flush()
                    update_log.append({"update": update_count, "loss": last_loss, "avg_r20": avg_display,
                                       "global_step": global_time_step, "plan_ep": planlama_toplam_episode})

                    # Yakınsama kontrolü
                    if avg_r20 is not None and planlama_toplam_episode >= PLANLAMA_MIN_EPISODE:
                        if avg_r20 >= YAKINSAMA_ESIK:
                            yakinsama_ardisik_sayac += 1
                            Log.plan(f"UPDATE BITTI | Loss:{last_loss:.4f} | AvgR({YAKINSAMA_ORT_PENCERE}):{avg_r20:.1f} "
                                     f"[YAKINSAMA {yakinsama_ardisik_sayac}/{YAKINSAMA_ARDISIK}]")
                        else:
                            yakinsama_ardisik_sayac = 0
                            Log.plan(f"UPDATE BITTI | Loss:{last_loss:.4f} | AvgR({YAKINSAMA_ORT_PENCERE}):{avg_r20:.1f} "
                                     f"(Hedef: >={YAKINSAMA_ESIK:.0f})")
                    else:
                        yakinsama_ardisik_sayac = 0
                        Log.plan(f"UPDATE BITTI | Loss:{last_loss:.4f} | AvgR:{avg_display:.1f} "
                                 f"(Min {PLANLAMA_MIN_EPISODE} ep bekleniyor, simdi:{planlama_toplam_episode})")

                except Exception as e:
                    Log.hata(f"PPO UPDATE HATASI: {e}")
                    train_memory.clear()

            # Yakınsama tespiti: yeterince iyi öğrenildiyse UYGULAMA'ya geç
            ogrenildi = (
                planlama_toplam_episode >= PLANLAMA_MIN_EPISODE and
                yakinsama_ardisik_sayac >= YAKINSAMA_ARDISIK
            )
            # Maksimum episode sınırına ulaşıldıysa da geç (güvenlik)
            max_asimi = planlama_toplam_episode >= PLANLAMA_EPISODE_HEDEFI

            if ogrenildi or max_asimi:
                neden = (f"{YAKINSAMA_ARDISIK} ust uste guncelleme ort>={YAKINSAMA_ESIK:.0f} saglandi"
                         if ogrenildi else f"Max episode ({PLANLAMA_EPISODE_HEDEFI}) asimi")
                Log.basari(f"[YAKINSAMА] Kriter: {neden}")
                gec_uygulama()   # Banner ve log gec_uygulama() icinde

        # ──────────────────────────────────────────────
        # 2B. UYGULAMA FAZI — Per-Robot Deterministik Komut
        # ──────────────────────────────────────────────
        elif SISTEM_FAZI == 'UYGULAMA' and not e_stop_active:
            for rid in ROBOTLAR:
                st = robot_states[rid]
                if st["nx"] is None:
                    if (simdiki_zaman - st["action_send_time"]) > 0.5:
                        send_robot_command(rid, 0, 0)
                        st["action_send_time"] = simdiki_zaman
                    continue

                # Sadece duvar dışında olan robotu durdur ve uyar
                if not is_point_in_polygon(st["nx"], st["ny"]):
                    Log.uyari(f"[UYGULAMA] Robot {rid} sinir disi! Durduruluyor...")
                    send_robot_command(rid, 0, 0)
                    continue

                # Komut gönderme hızını sınırla (20Hz)
                if (simdiki_zaman - st["action_send_time"]) >= 0.05:
                    execute_learned_policy(rid)
                    st["action_send_time"] = simdiki_zaman

        # ──────────────────────────────────────────────
        # 2C. HAZIRLIK FAZI — Robotları durdurucu
        # ──────────────────────────────────────────────
        # (HAZIRLIK fazında motor komutu gönderilmez — robotlar elle yerleştirilir)

        # ──────────────────────────────────────────────
        # 3. KULLANICI ARAYÜZÜ (DRAWING)
        # ──────────────────────────────────────────────
        display = frame.copy()

        # Duvarları Çiz
        poly_pts = []
        for (lx, ly) in WALL_POLYGON:
            poly_pts.append([int((lx / LOGIC_GRID_MAX) * fw), int((ly / LOGIC_GRID_MAX) * fh)])
        cv2.polylines(display, [np.array(poly_pts)], True, (0, 0, 255), 2)
        cv2.putText(display, "SANAL DUVAR", (poly_pts[0][0] - 20, poly_pts[0][1] - 10), FONT, 0.5, (0, 0, 255), 2)

        # Robot görselleştirme (gerçek)
        for rid in ROBOTLAR:
            st = robot_states[rid]
            if st["nx"] is not None:
                cx = int((st["nx"] / LOGIC_GRID_MAX) * fw)
                cy = int((st["ny"] / LOGIC_GRID_MAX) * fh)
                clr = ROBOT_RENKLERI[rid]

                is_danger = False
                for oid in ROBOTLAR:
                    if oid != rid and robot_states[oid]["nx"] is not None:
                        dist_o = math.sqrt((robot_states[oid]["nx"] - st["nx"])**2 + (robot_states[oid]["ny"] - st["ny"])**2)
                        if dist_o < 35.0:
                            is_danger = True
                            break

                zone_clr = (0, 0, 255) if is_danger else clr
                safe_radius_px = int((35.0 / LOGIC_GRID_MAX) * fw)
                cv2.circle(display, (cx, cy), safe_radius_px, zone_clr, 1, cv2.LINE_AA)

                if st["corners"] is not None:
                    cv2.polylines(display, [st["corners"]], True, clr, 2)
                else:
                    cv2.circle(display, (cx, cy), int(st["radius"]), clr, 2)

                # Faz bilgisini etikete ekle
                faz_kisa = {"HAZIRLIK": "HAZ", "PLANLAMA": "PLAN", "UYGULAMA": "UYG"}.get(SISTEM_FAZI, "?")
                cv2.putText(display, f"R{rid}[{faz_kisa}]", (cx - 20, cy - 25), FONT, 0.5, clr, 2)

                if st["angle"] is not None:
                    arad = math.radians(st["angle"])
                    tx_ = int(cx + math.cos(arad) * 30)
                    ty_ = int(cy + math.sin(arad) * 30)
                    cv2.arrowedLine(display, (cx, cy), (tx_, ty_), clr, 3, tipLength=0.3)

        # Başlangıç pozisyonları (★ sembolü)
        for rid, (spx, spy, _) in start_positions.items():
            sx = int((spx / LOGIC_GRID_MAX) * fw)
            sy = int((spy / LOGIC_GRID_MAX) * fh)
            clr = ROBOT_RENKLERI[rid]
            cv2.drawMarker(display, (sx, sy), clr, cv2.MARKER_STAR, 20, 2)
            cv2.putText(display, f"S{rid}", (sx + 8, sy - 8), FONT, 0.4, clr, 1)

        # PLANLAMA fazında sanal robotları da göster
        if SISTEM_FAZI == 'PLANLAMA' and virtual_robot_states:
            for rid, vst in virtual_robot_states.items():
                vx = int((vst["nx"] / LOGIC_GRID_MAX) * fw)
                vy = int((vst["ny"] / LOGIC_GRID_MAX) * fh)
                clr = ROBOT_RENKLERI[rid]
                # Hayalet (yarı şeffaf görünüm için noktalı daire)
                cv2.circle(display, (vx, vy), 12, clr, 1, cv2.LINE_AA)
                cv2.putText(display, f"v{rid}", (vx + 5, vy - 5), FONT, 0.35, clr, 1)
                # Açı oku
                varad = math.radians(vst["angle"])
                vtx = int(vx + math.cos(varad) * 15)
                vty = int(vy + math.sin(varad) * 15)
                cv2.arrowedLine(display, (vx, vy), (vtx, vty), clr, 1, tipLength=0.4)

        # Hedef ve engeller
        if virtual_target is not None:
            tx_ = int((virtual_target[0] / LOGIC_GRID_MAX) * fw)
            ty_ = int((virtual_target[1] / LOGIC_GRID_MAX) * fh)
            cv2.drawMarker(display, (tx_, ty_), (255, 200, 0), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(display, "HEDEF", (tx_ - 20, ty_ - 20), FONT, 0.6, (255, 200, 0), 2)

        for (ox, oy, r) in virtual_obstacles:
            ecx = int((ox / LOGIC_GRID_MAX) * fw)
            ecy = int((oy / LOGIC_GRID_MAX) * fh)
            er = int((r / LOGIC_GRID_MAX) * fw)
            cv2.circle(display, (ecx, ecy), er, (0, 0, 255), 2)
            cv2.putText(display, "ENGEL", (ecx - 15, ecy), FONT, 0.4, (0, 0, 255), 1)

        # ── Sol üst bilgi paneli ──
        cv2.rectangle(display, (0, 0), (320, 240), (20, 20, 20), -1)

        # Faza göre renk ve mesaj
        if e_stop_active:
            faz_str, faz_clr = "ACIL DURUS!", (0, 0, 255)
        elif HEDEF_BEKLENIYOR:
            faz_str, faz_clr = ">>> EKRANDA HEDEFE TIKLAYARAK SECIN <<<", (0, 200, 255)
        elif SISTEM_FAZI == 'HAZIRLIK':
            faz_str, faz_clr = "HAZIRLIK — [T] Planlamaya Gec", (0, 200, 255)
        elif SISTEM_FAZI == 'PLANLAMA':
            faz_str, faz_clr = f"PLANLAMA — Sanal Sim ({planlama_toplam_episode}/{PLANLAMA_EPISODE_HEDEFI} ep)", (0, 255, 0)
        else:  # UYGULAMA
            faz_str, faz_clr = "UYGULAMA — Policy Aktif", (255, 150, 0)

        cv2.putText(display, f"FAZ: {faz_str}", (10, 25), FONT, 0.5, faz_clr, 2)

        # Hedef bekleme modunda ekranın ortasına büyük uyarı
        if HEDEF_BEKLENIYOR:
            overlay = display.copy()
            cv2.rectangle(overlay, (fw//2-280, fh//2-40), (fw//2+280, fh//2+40), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)
            cv2.putText(display, "SOL TIKLA: HEDEFI SEC", (fw//2-220, fh//2+12),
                        FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(display, f"BUFFER: {len(train_memory.states)}/{PPO_UPDATE_TIMESTEP}", (10, 50), FONT, 0.5, (100, 255, 255), 1)
        cv2.putText(display, f"LOSS: {last_loss:.4f} | UPD: {update_count}", (10, 75), FONT, 0.5, (0, 200, 255), 1)
        avg_r = np.mean(episode_rewards_history[-20:]) if episode_rewards_history else 0.0
        cv2.putText(display, f"AVG REWARD(20): {avg_r:.1f}", (10, 100), FONT, 0.5, (0, 255, 200), 1)

        y_off = 125
        for rid in ROBOTLAR:
            st = robot_states[rid]
            v = "VAR" if st["nx"] is not None else "YOK"
            start_ok = "✓" if rid in start_positions else "✗"
            cv2.putText(display, f"R{rid}: {v} | Start:{start_ok} | Ep:{st['episode']} | Rew:{st['ep_reward']:.1f}",
                        (10, y_off), FONT, 0.38, ROBOT_RENKLERI[rid], 1)
            y_off += 20

        # Alt bilgi çubuğu
        if SISTEM_FAZI == 'HAZIRLIK':
            cv2.putText(display, f"MANUEL: Robot:{selected_manual_robot} (0-3 sec, WASD sur)",
                        (10, fh - 45), FONT, 0.5, (255, 150, 150), 2)
        cv2.putText(display, "[T]Planla [E]Uygula [H]Hazirlik [SPACE]E-Stop [R]Map [C]Kalib [Q]Cikis",
                    (10, fh - 20), FONT, 0.45, (255, 255, 255), 1)

        cv2.imshow('MULTI_AGENT_RL', display)
        tus = cv2.waitKey(1) & 0xFF

        # ──────────────────────────────────────────────
        # 4. KLAVYE KONTROLÜ
        # ──────────────────────────────────────────────
        if tus == ord('q'):
            break

        elif tus == ord(' '):   # E-STOP
            e_stop_active = not e_stop_active
            if e_stop_active:
                durdur_tum_robotlar()
                Log.hata("!!! ACIL DURUS AKTIF !!! [SPACE] ile devam et.")
            else:
                Log.basari("Acil durus kaldirildi.")

        elif tus == ord('t'):   # HAZIRLIK → Hedef Bekleme → PLANLAMA
            if not e_stop_active:
                if SISTEM_FAZI == 'HAZIRLIK' and not HEDEF_BEKLENIYOR:
                    gec_planlama()
                elif HEDEF_BEKLENIYOR:
                    Log.uyari("Hedef bekleniyor — Sol tikla hedefi sec, veya [H] ile iptal et.")
                elif SISTEM_FAZI == 'PLANLAMA':
                    Log.uyari("Zaten PLANLAMA fazindasiniz. [E] ile UYGULAMA'ya gecin.")
                else:
                    Log.uyari("UYGULAMA aktif. [H] ile once HAZIRLIK'a donun.")

        elif tus == ord('e'):   # PLANLAMA → UYGULAMA (elle geçiş)
            if not e_stop_active:
                if SISTEM_FAZI == 'PLANLAMA':
                    Log.basari(f"Elle UYGULAMA gecisi! ({planlama_toplam_episode} ep tamamlandi)")
                    gec_uygulama()
                elif SISTEM_FAZI == 'HAZIRLIK':
                    Log.uyari("Once [T] ile PLANLAMA'yi calistirin!")
                else:
                    Log.bilgi("Zaten UYGULAMA fazindasiniz.")

        elif tus == ord('h'):   # → HAZIRLIK (sıfırla)
            HEDEF_BEKLENIYOR = False
            gec_hazirlik()

        elif tus == ord('c'):
            for rid in ROBOTLAR: robot_states[rid]["base_angle"] = None
            Log.bilgi("Tum robotlarin acilari sifirlandi.")

        elif tus == ord('r'):
            generate_environment()
            Log.bilgi("Harita yenilendi.")

        elif tus in [ord('0'), ord('1'), ord('2'), ord('3')]:
            selected_manual_robot = int(chr(tus))

        # Manuel WASD — yalnızca HAZIRLIK fazında
        if SISTEM_FAZI == 'HAZIRLIK' and not e_stop_active:
            if tus in [ord('w'), ord('a'), ord('s'), ord('d')]:
                if tus == ord('w'): send_robot_command(selected_manual_robot, 20, 20)
                elif tus == ord('s'): send_robot_command(selected_manual_robot, -20, -20)
                elif tus == ord('a'): send_robot_command(selected_manual_robot, -18, 18)
                elif tus == ord('d'): send_robot_command(selected_manual_robot, 18, -18)
                manual_motor_active = True
                manual_last_cmd = simdiki_zaman
            else:
                if manual_motor_active and (simdiki_zaman - manual_last_cmd > 0.15):
                    send_robot_command(selected_manual_robot, 0, 0)
                    manual_motor_active = False

finally:
    durdur_tum_robotlar()
    if gateway_serial and gateway_serial.is_open: gateway_serial.close()
    kamera.durdur()
    csv_file.close()
    cv2.destroyAllWindows()
    Log.basari(f"Sistem guvenle kapatildi. Log: {CSV_LOG_PATH}")

    # ==========================================================================
    #  OTOMATİK GRAFİK ÜRETİMİ — 25 Grafik
    # ==========================================================================
    try:
        print("\n" + "="*65)
        print("  SİMÜLASYON BİTTİ — 25 GRAFİK ÜRETİLİYOR...")
        print("="*65)

        OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_grafikler")
        os.makedirs(OUT_DIR, exist_ok=True)

        ROBOT_HEX  = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#ff7f00"}
        C_PLAN     = "#1f77b4"
        C_UYG      = "#2ca02c"
        C_LOSS     = "#d62728"
        C_REW      = "#2ca02c"
        C_THRESH   = "#ff7f0e"

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
            return np.convolve(arr, np.ones(w)/w, mode='valid')

        def _wall(ax, color="blue", ls="--", lw=2):
            wx = [p[0] for p in WALL_POLYGON] + [WALL_POLYGON[0][0]]
            wy = [p[1] for p in WALL_POLYGON] + [WALL_POLYGON[0][1]]
            ax.plot(wx, wy, color=color, ls=ls, lw=lw, label="Arena Sınırı")

        N_ep    = len(episode_rewards_history)
        N_upd   = len(update_log)
        N_uyg   = len(uygulama_log)
        N_ph    = len(phase_log)
        N_eplog = len(episode_log)
        N_virt  = len(virt_pos_log)

        # ── 1. Episode Ödül Eğrisi ─────────────────────────────────────────────
        print("[1/25] Episode ödül eğrisi...")
        if N_ep > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            eps = np.arange(1, N_ep+1)
            ax.plot(eps, episode_rewards_history, color=C_REW, alpha=0.25, lw=1, label="Ham Ödül")
            if N_ep >= 5:
                w = min(20, N_ep//2)
                ma = _mav(episode_rewards_history, w)
                ax.plot(np.arange(w, N_ep+1), ma, color=C_REW, lw=2.5, label=f"Hareketli Ort. ({w} ep)")
            ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=2, ls="--", label=f"Yakınsama Eşiği ({YAKINSAMA_ESIK:.0f})")
            ax.set_xlabel("Episode"); ax.set_ylabel("Toplam Ödül")
            ax.set_title("PPO Eğitimi — Episode Ödül Eğrisi"); ax.legend()
            _save(fig, "01_episode_reward_curve")
        else: print("  [ATLA] Veri yok")

        # ── 2. PPO Loss Eğrisi ─────────────────────────────────────────────────
        print("[2/25] PPO loss eğrisi...")
        if N_upd > 0:
            upd  = [u["update"] for u in update_log]
            loss = [u["loss"]   for u in update_log]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(upd, loss, alpha=0.15, color=C_LOSS)
            ax.plot(upd, loss, color=C_LOSS, lw=2.5, label="PPO Loss")
            ax.set_xlabel("PPO Güncelleme #"); ax.set_ylabel("Loss")
            ax.set_title("PPO — Güncelleme Başına Kayıp Fonksiyonu"); ax.legend()
            _save(fig, "02_ppo_loss_curve")
        else: print("  [ATLA] Veri yok")

        # ── 3. Güncelleme Başına Avg Reward + Yakınsama ────────────────────────
        print("[3/25] Güncelleme başına ortalama ödül...")
        if N_upd > 0:
            upd   = [u["update"]  for u in update_log]
            avg_r = [u["avg_r20"] for u in update_log]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(upd, avg_r, alpha=0.15, color=C_REW)
            ax.plot(upd, avg_r, color=C_REW, lw=2.5, label=f"Ort. Ödül (son {YAKINSAMA_ORT_PENCERE} ep)")
            ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=2, ls="--", label=f"Yakınsama Eşiği ({YAKINSAMA_ESIK:.0f})")
            ax.set_xlabel("PPO Güncelleme #"); ax.set_ylabel("Ortalama Ödül")
            ax.set_title("PPO — Güncelleme Başına Ortalama Ödül & Yakınsama Analizi"); ax.legend()
            _save(fig, "03_avg_reward_per_update")
        else: print("  [ATLA] Veri yok")

        # ── 4. Global Timestep vs Avg Reward ───────────────────────────────────
        print("[4/25] Global timestep vs ödül...")
        if N_upd > 0:
            gsteps = [u["global_step"] for u in update_log]
            avg_r  = [u["avg_r20"]    for u in update_log]
            fig, ax = plt.subplots(figsize=(10, 5))
            sc = ax.scatter(gsteps, avg_r, c=np.arange(len(gsteps)), cmap="viridis", s=60, zorder=5)
            ax.plot(gsteps, avg_r, color="#999999", lw=1, alpha=0.5)
            fig.colorbar(sc, ax=ax, label="Güncelleme #")
            ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=1.5, ls="--", label=f"Eşik={YAKINSAMA_ESIK:.0f}")
            ax.set_xlabel("Global Timestep"); ax.set_ylabel("Ortalama Ödül")
            ax.set_title("PPO — Global Timestep'e Göre Öğrenme İlerlemesi"); ax.legend()
            _save(fig, "04_global_step_vs_reward")
        else: print("  [ATLA] Veri yok")

        # ── 5. Güncelleme Başına Tamamlanan Episode ────────────────────────────
        print("[5/25] Güncelleme başına episode sayısı...")
        if N_upd > 0:
            upd      = [u["update"]   for u in update_log]
            plan_eps = [u["plan_ep"]  for u in update_log]
            fig, ax  = plt.subplots(figsize=(10, 5))
            ax.bar(upd, plan_eps, color=C_PLAN, alpha=0.8, label="Tamamlanan Episode")
            ax.axhline(PLANLAMA_MIN_EPISODE,    color=C_THRESH, lw=2, ls="--", label=f"Min Ep ({PLANLAMA_MIN_EPISODE})")
            ax.axhline(PLANLAMA_EPISODE_HEDEFI, color=C_LOSS,   lw=2, ls=":",  label=f"Max Ep ({PLANLAMA_EPISODE_HEDEFI})")
            ax.set_xlabel("PPO Güncelleme #"); ax.set_ylabel("Episode Sayısı")
            ax.set_title("PPO — Her Güncellemede Tamamlanan Episode Sayısı"); ax.legend()
            _save(fig, "05_episodes_per_update")
        else: print("  [ATLA] Veri yok")

        # ── 6. Robot Bazlı Episode Ödülleri ───────────────────────────────────
        print("[6/25] Robot bazlı episode ödülleri...")
        if N_eplog > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTLAR:
                r_data = [(e["ep"], e["reward"]) for e in episode_log if e["rid"]==rid and e["phase"]=="PLANLAMA"]
                if r_data:
                    xs, ys = zip(*r_data)
                    ax.plot(xs, ys, color=ROBOT_HEX[rid], alpha=0.35, lw=1)
                    if len(ys) >= 5:
                        w = min(10, len(ys)//2)
                        ma = _mav(list(ys), w)
                        ax.plot(xs[len(ys)-len(ma):], ma, color=ROBOT_HEX[rid], lw=2.5, label=f"Robot {rid}")
            ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=1.5, ls="--", label=f"Eşik ({YAKINSAMA_ESIK:.0f})")
            ax.set_xlabel("Episode"); ax.set_ylabel("Episode Ödülü")
            ax.set_title("PLANLAMA — Robot Bazlı Episode Ödül Eğrileri"); ax.legend()
            _save(fig, "06_per_robot_rewards")
        else: print("  [ATLA] Veri yok")

        # ── 7. Episode Uzunluğu Histogramı ────────────────────────────────────
        print("[7/25] Episode uzunluğu histogramı...")
        psteps = [e["steps"] for e in episode_log if e["phase"]=="PLANLAMA"]
        if psteps:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(psteps, bins=min(40, len(set(psteps))), color=C_PLAN, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(psteps), color=C_THRESH, lw=2, ls="--", label=f"Ort. = {np.mean(psteps):.0f}")
            ax.axvline(MAX_EP_STEPS, color=C_LOSS, lw=2, ls=":", label=f"Max = {MAX_EP_STEPS}")
            ax.set_xlabel("Episode Uzunluğu (adım)"); ax.set_ylabel("Frekans")
            ax.set_title("PLANLAMA — Episode Uzunluğu Dağılımı"); ax.legend()
            _save(fig, "07_episode_length_histogram")
        else: print("  [ATLA] Veri yok")

        # ── 8. Kümülatif Episode Ödülü ─────────────────────────────────────────
        print("[8/25] Kümülatif ödül eğrisi...")
        if N_ep > 0:
            eps = np.arange(1, N_ep+1)
            cum = np.cumsum(episode_rewards_history)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(eps, cum, alpha=0.2, color=C_REW)
            ax.plot(eps, cum, color=C_REW, lw=2.5)
            ax.set_xlabel("Episode"); ax.set_ylabel("Kümülatif Toplam Ödül")
            ax.set_title("PPO Eğitimi — Kümülatif Toplam Ödül")
            _save(fig, "08_cumulative_reward")
        else: print("  [ATLA] Veri yok")

        # ── 9. PLANLAMA Sanal Pozisyon Isı Haritası ───────────────────────────
        print("[9/25] PLANLAMA pozisyon ısı haritası...")
        if N_virt > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            vnx = [v["nx"] for v in virt_pos_log]
            vny = [v["ny"] for v in virt_pos_log]
            h = ax.hist2d(vnx, vny, bins=40, range=[[0,LOGIC_GRID_MAX],[0,LOGIC_GRID_MAX]], cmap="YlOrRd")
            fig.colorbar(h[3], ax=ax, label="Ziyaret Sayısı")
            _wall(ax, color="blue")
            if virtual_target: ax.scatter(*virtual_target, c="cyan", s=200, marker="*", zorder=5, label="Son Hedef", edgecolors="k")
            ax.set_xlim(0,LOGIC_GRID_MAX); ax.set_ylim(0,LOGIC_GRID_MAX)
            ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_title("PLANLAMA — Sanal Robot Pozisyon Yoğunluğu"); ax.legend(fontsize=9)
            _save(fig, "09_planlama_heatmap")
        else: print("  [ATLA] Veri yok")

        # ── 10. UYGULAMA Gerçek Trajektori ────────────────────────────────────
        print("[10/25] UYGULAMA gerçek trajektori...")
        if N_uyg > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            _wall(ax, color="red")
            for rid in ROBOTLAR:
                rl = [e for e in uygulama_log if e["rid"]==rid]
                if rl:
                    xs = [e["nx"] for e in rl]; ys = [e["ny"] for e in rl]
                    ax.plot(xs, ys, color=ROBOT_HEX[rid], lw=2, alpha=0.85, label=f"Robot {rid}")
                    ax.plot(xs[0],  ys[0],  "o", color=ROBOT_HEX[rid], ms=10, markeredgecolor="k", zorder=5)
                    ax.plot(xs[-1], ys[-1], "s", color=ROBOT_HEX[rid], ms=10, markeredgecolor="k", zorder=5)
            if virtual_target: ax.scatter(*virtual_target, c="gold", s=250, marker="*", zorder=6, label="Hedef", edgecolors="k")
            for (ox,oy,r) in virtual_obstacles:
                ax.add_patch(plt.Circle((ox,oy), r, color="red", alpha=0.4))
            ax.set_xlim(0,LOGIC_GRID_MAX); ax.set_ylim(0,LOGIC_GRID_MAX)
            ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_title("UYGULAMA — Öğrenilmiş Policy ile Gerçek Robot Trajektorisi"); ax.legend(fontsize=9)
            _save(fig, "10_uygulama_trajectory")
        else: print("  [ATLA] Veri yok")

        # ── 11. UYGULAMA Hedef Uzaklığı Zaman Serisi ──────────────────────────
        print("[11/25] UYGULAMA hedef uzaklığı zaman serisi...")
        if N_uyg > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTLAR:
                rl = [e for e in uygulama_log if e["rid"]==rid]
                if rl:
                    ts = [e["t"] for e in rl]; ds = [e["dist"] for e in rl]
                    ax.plot(ts, ds, color=ROBOT_HEX[rid], lw=2, alpha=0.85, label=f"Robot {rid}")
                    done_t = [e["t"]    for e in rl if e["done"]]
                    done_d = [e["dist"] for e in rl if e["done"]]
                    if done_t: ax.scatter(done_t, done_d, color=ROBOT_HEX[rid], s=120, marker="*", zorder=5)
            ax.axhline(15, color="#888888", lw=1.5, ls="--", label="Hedef eşiği (~15 px)")
            ax.set_xlabel("Zaman (s)"); ax.set_ylabel("Hedefe Uzaklık")
            ax.set_title("UYGULAMA — Robot Hedef Uzaklığı Zaman Serisi"); ax.legend()
            _save(fig, "11_uygulama_dist_to_target")
        else: print("  [ATLA] Veri yok")

        # ── 12. UYGULAMA Sol Motor Zaman Serisi ───────────────────────────────
        print("[12/25] UYGULAMA sol motor zaman serisi...")
        if N_uyg > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTLAR:
                rl = [e for e in uygulama_log if e["rid"]==rid]
                if rl:
                    ax.plot([e["t"] for e in rl], [e["motor_l"] for e in rl],
                            color=ROBOT_HEX[rid], lw=1.8, alpha=0.85, label=f"Robot {rid}")
            ax.axhline(0, color="#aaaaaa", lw=1, ls="--")
            ax.set_ylim(-35, 35); ax.set_xlabel("Zaman (s)"); ax.set_ylabel("Sol Motor Komutu")
            ax.set_title("UYGULAMA — Sol Motor Komutu Zaman Serisi"); ax.legend()
            _save(fig, "12_uygulama_motor_left")
        else: print("  [ATLA] Veri yok")

        # ── 13. UYGULAMA Sağ Motor Zaman Serisi ───────────────────────────────
        print("[13/25] UYGULAMA sağ motor zaman serisi...")
        if N_uyg > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTLAR:
                rl = [e for e in uygulama_log if e["rid"]==rid]
                if rl:
                    ax.plot([e["t"] for e in rl], [e["motor_r"] for e in rl],
                            color=ROBOT_HEX[rid], lw=1.8, alpha=0.85, label=f"Robot {rid}")
            ax.axhline(0, color="#aaaaaa", lw=1, ls="--")
            ax.set_ylim(-35, 35); ax.set_xlabel("Zaman (s)"); ax.set_ylabel("Sağ Motor Komutu")
            ax.set_title("UYGULAMA — Sağ Motor Komutu Zaman Serisi"); ax.legend()
            _save(fig, "13_uygulama_motor_right")
        else: print("  [ATLA] Veri yok")

        # ── 14. Motor Aksiyon Uzayı Scatter (L vs R) ──────────────────────────
        print("[14/25] Motor aksiyon uzayı scatter...")
        if N_uyg > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            for rid in ROBOTLAR:
                rl = [e for e in uygulama_log if e["rid"]==rid]
                if rl:
                    ax.scatter([e["motor_l"] for e in rl], [e["motor_r"] for e in rl],
                               color=ROBOT_HEX[rid], s=10, alpha=0.4, label=f"Robot {rid}")
            ax.axhline(0, color="#aaaaaa", lw=1, ls="--")
            ax.axvline(0, color="#aaaaaa", lw=1, ls="--")
            ax.set_xlim(-35,35); ax.set_ylim(-35,35); ax.set_aspect("equal")
            ax.set_xlabel("Sol Motor"); ax.set_ylabel("Sağ Motor")
            ax.set_title("UYGULAMA — Motor Aksiyon Uzayı Dağılımı"); ax.legend(fontsize=9)
            _save(fig, "14_motor_action_space")
        else: print("  [ATLA] Veri yok")

        # ── 15. Sanal Arena Haritası ───────────────────────────────────────────
        print("[15/25] Sanal arena haritası...")
        fig, ax = plt.subplots(figsize=(7, 7))
        wx = [p[0] for p in WALL_POLYGON] + [WALL_POLYGON[0][0]]
        wy = [p[1] for p in WALL_POLYGON] + [WALL_POLYGON[0][1]]
        ax.fill(wx, wy, alpha=0.06, color="blue")
        ax.plot(wx, wy, "b-", lw=2.5, label="Arena Sınırı")
        for (ox,oy,r) in virtual_obstacles:
            ax.add_patch(plt.Circle((ox,oy), r, color="red", alpha=0.65, zorder=3))
        if virtual_obstacles: ax.scatter([],[],color="red",alpha=0.8,label="Engeller",s=100)
        if virtual_target: ax.scatter(*virtual_target, c="gold", s=300, marker="*", zorder=5, label="Hedef", edgecolors="k", lw=1)
        for rid,(spx,spy,_) in start_positions.items():
            ax.scatter(spx, spy, color=ROBOT_HEX.get(rid,"gray"), s=120, marker="o", zorder=4, label=f"R{rid} Başlangıç")
        ax.set_xlim(0,LOGIC_GRID_MAX); ax.set_ylim(0,LOGIC_GRID_MAX); ax.set_aspect("equal")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title("RL Ortamı — Sanal Arena Haritası"); ax.legend(fontsize=9)
        _save(fig, "15_arena_map")

        # ── 16. Episode Ödülü vs Uzunluğu Scatter ─────────────────────────────
        print("[16/25] Episode ödül vs uzunluk scatter...")
        plan_rl = [(e["reward"],e["steps"]) for e in episode_log if e["phase"]=="PLANLAMA"]
        if plan_rl:
            rews_, stps_ = zip(*plan_rl)
            fig, ax = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(stps_, rews_, c=np.arange(len(rews_)), cmap="plasma", s=20, alpha=0.5)
            fig.colorbar(sc, ax=ax, label="Episode Numarası")
            ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=1.5, ls="--", label=f"Eşik ({YAKINSAMA_ESIK:.0f})")
            ax.set_xlabel("Episode Uzunluğu (adım)"); ax.set_ylabel("Episode Ödülü")
            ax.set_title("PLANLAMA — Episode Ödülü vs. Uzunluğu"); ax.legend()
            _save(fig, "16_reward_vs_steps_scatter")
        else: print("  [ATLA] Veri yok")

        # ── 17. Faz Geçiş Zaman Çizelgesi ─────────────────────────────────────
        print("[17/25] Faz geçiş zaman çizelgesi...")
        if N_ph > 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            faz_clr = {"HAZIRLIK": "#aec7e8", "PLANLAMA": "#1f77b4", "UYGULAMA": "#2ca02c"}
            t_max = phase_log[-1]["t"] + 5
            prev_t, prev_faz = 0.0, "HAZIRLIK"
            for entry in phase_log:
                dur = entry["t"] - prev_t
                ax.barh(0, dur, left=prev_t, color=faz_clr.get(prev_faz,"gray"), height=0.6, edgecolor="white", lw=0.5)
                if dur > 1: ax.text(prev_t+dur/2, 0, prev_faz, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
                prev_t, prev_faz = entry["t"], entry["to_p"]
            ax.barh(0, t_max-prev_t, left=prev_t, color=faz_clr.get(prev_faz,"gray"), height=0.6, edgecolor="white", lw=0.5)
            if t_max-prev_t > 1: ax.text(prev_t+(t_max-prev_t)/2, 0, prev_faz, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            ax.set_xlim(0, t_max); ax.set_xlabel("Zaman (s)"); ax.set_yticks([])
            ax.set_title("Sistem Faz Geçiş Zaman Çizelgesi")
            patches = [mpatches.Patch(color=v, label=k) for k,v in faz_clr.items()]
            ax.legend(handles=patches, loc="upper right")
            _save(fig, "17_phase_timeline")
        else: print("  [ATLA] Veri yok")

        # ── 18. PLANLAMA Sanal Trajektori (Son 5 Episode) ──────────────────────
        print("[18/25] PLANLAMA sanal trajektori (son episodeler)...")
        if N_virt > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            _wall(ax, color="blue")
            all_eps = sorted(set(v["ep"] for v in virt_pos_log))
            last_eps = all_eps[-min(5, len(all_eps)):]
            for i, ep_num in enumerate(last_eps):
                alp = 0.25 + 0.75*(i/max(len(last_eps)-1,1))
                for rid in ROBOTLAR:
                    seg = [(v["nx"],v["ny"]) for v in virt_pos_log if v["ep"]==ep_num and v["rid"]==rid]
                    if len(seg) > 1:
                        xs_,ys_ = zip(*seg)
                        ax.plot(xs_, ys_, color=ROBOT_HEX[rid], lw=1.5, alpha=alp)
            if virtual_target: ax.scatter(*virtual_target, c="gold", s=250, marker="*", zorder=6, label="Hedef", edgecolors="k")
            for rid in ROBOTLAR: ax.plot([],[],color=ROBOT_HEX[rid],lw=2,label=f"Robot {rid}")
            ax.set_xlim(0,LOGIC_GRID_MAX); ax.set_ylim(0,LOGIC_GRID_MAX); ax.set_aspect("equal")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_title(f"PLANLAMA — Son {len(last_eps)} Episode Sanal Trajektori"); ax.legend(fontsize=9)
            _save(fig, "18_virtual_trajectory")
        else: print("  [ATLA] Veri yok")

        # ── 19. PPO Loss Dağılımı Histogramı ──────────────────────────────────
        print("[19/25] PPO loss dağılımı histogramı...")
        if N_upd > 0:
            losses = [u["loss"] for u in update_log]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(losses, bins=min(30,N_upd), color=C_LOSS, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(losses), color=C_THRESH, lw=2, ls="--", label=f"Ort. = {np.mean(losses):.4f}")
            ax.set_xlabel("PPO Loss Değeri"); ax.set_ylabel("Frekans")
            ax.set_title("PPO — Loss Dağılımı (Tüm Güncellemeler)"); ax.legend()
            _save(fig, "19_loss_histogram")
        else: print("  [ATLA] Veri yok")

        # ── 20. Robot Bazlı Kümülatif Ödül ────────────────────────────────────
        print("[20/25] Robot bazlı kümülatif ödül...")
        if N_eplog > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rid in ROBOTLAR:
                r_data = [(e["ep"],e["reward"]) for e in episode_log if e["rid"]==rid and e["phase"]=="PLANLAMA"]
                if r_data:
                    eps_,rews_ = zip(*r_data)
                    ax.plot(eps_, np.cumsum(rews_), color=ROBOT_HEX[rid], lw=2.5, label=f"Robot {rid}")
            ax.set_xlabel("Episode"); ax.set_ylabel("Kümülatif Ödül")
            ax.set_title("PLANLAMA — Robot Bazlı Kümülatif Ödül"); ax.legend()
            _save(fig, "20_per_robot_cumulative_reward")
        else: print("  [ATLA] Veri yok")

        # ── 21. Ödül Belirsizlik Bandı (Ort ± Std) ────────────────────────────
        print("[21/25] Ödül belirsizlik bandı...")
        if N_ep >= 10:
            rh = np.array(episode_rewards_history)
            eps = np.arange(1, N_ep+1)
            w   = min(20, N_ep//3)
            mus, stds = [], []
            for i in range(len(rh)):
                win = rh[max(0,i-w+1):i+1]
                mus.append(np.mean(win)); stds.append(np.std(win))
            mus = np.array(mus); stds = np.array(stds)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(eps, mus-stds, mus+stds, alpha=0.2, color=C_REW, label="±1 Std")
            ax.plot(eps, mus, color=C_REW, lw=2.5, label=f"Kayan Ort. (w={w})")
            ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=1.5, ls="--", label=f"Eşik ({YAKINSAMA_ESIK:.0f})")
            ax.set_xlabel("Episode"); ax.set_ylabel("Ödül")
            ax.set_title("PPO — Ödül Belirsizlik Bandı (Ortalama ± Std)"); ax.legend()
            _save(fig, "21_reward_band")
        else: print("  [ATLA] Yetersiz veri (<10 ep)")

        # ── 22. UYGULAMA Pozisyon Isı Haritası ────────────────────────────────
        print("[22/25] UYGULAMA pozisyon ısı haritası...")
        if N_uyg > 1:
            ux = [e["nx"] for e in uygulama_log]
            uy = [e["ny"] for e in uygulama_log]
            fig, ax = plt.subplots(figsize=(7, 7))
            h = ax.hist2d(ux, uy, bins=30, range=[[0,LOGIC_GRID_MAX],[0,LOGIC_GRID_MAX]], cmap="Blues")
            fig.colorbar(h[3], ax=ax, label="Ziyaret Sayısı")
            _wall(ax, color="red")
            if virtual_target: ax.scatter(*virtual_target, c="gold", s=250, marker="*", zorder=6, label="Hedef", edgecolors="k")
            ax.set_xlim(0,LOGIC_GRID_MAX); ax.set_ylim(0,LOGIC_GRID_MAX); ax.set_aspect("equal")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_title("UYGULAMA — Robot Pozisyon Yoğunluk Haritası"); ax.legend(fontsize=9)
            _save(fig, "22_uygulama_heatmap")
        else: print("  [ATLA] Veri yok")

        # ── 23. Ödül Kutu Grafiği (Öğrenme Evreleri) ──────────────────────────
        print("[23/25] Episode ödül kutu grafiği (öğrenme evreleri)...")
        if N_ep >= 10:
            n_groups = min(5, N_ep//10)
            if n_groups >= 2:
                rh = episode_rewards_history
                gsz = N_ep//n_groups
                groups = [rh[i*gsz:(i+1)*gsz] for i in range(n_groups)]
                labels = [f"Ep {i*gsz+1}–{(i+1)*gsz}" for i in range(n_groups)]
                fig, ax = plt.subplots(figsize=(10, 5))
                bp = ax.boxplot(groups, labels=labels, patch_artist=True)
                colors_ = plt.cm.viridis(np.linspace(0.2,0.9,n_groups))
                for patch,clr in zip(bp['boxes'], colors_):
                    patch.set_facecolor(clr); patch.set_alpha(0.75)
                ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=1.5, ls="--", label=f"Eşik ({YAKINSAMA_ESIK:.0f})")
                ax.set_xlabel("Eğitim Evresi"); ax.set_ylabel("Episode Ödülü")
                ax.set_title("PPO — Öğrenme Evreleri Boyunca Ödül Dağılımı")
                ax.legend(); plt.xticks(rotation=15)
                _save(fig, "23_reward_boxplot")
            else: print("  [ATLA] Yetersiz veri")
        else: print("  [ATLA] Yetersiz veri (<10 ep)")

        # ── 24. PLANLAMA vs UYGULAMA Ödül Karşılaştırması ─────────────────────
        print("[24/25] PLANLAMA vs UYGULAMA ödül karşılaştırması...")
        plan_rews_ = [e["reward"] for e in episode_log if e["phase"]=="PLANLAMA"]
        uyg_rews_  = [e["reward"] for e in episode_log if e["phase"]=="UYGULAMA"]
        if plan_rews_ or uyg_rews_:
            data_, lbls_ = [], []
            if plan_rews_: data_.append(plan_rews_); lbls_.append("PLANLAMA\n(Sanal)")
            if uyg_rews_:  data_.append(uyg_rews_);  lbls_.append("UYGULAMA\n(Gerçek)")
            fig, ax = plt.subplots(figsize=(7, 5))
            bp = ax.boxplot(data_, labels=lbls_, patch_artist=True, widths=0.4)
            for patch,clr in zip(bp['boxes'], [C_PLAN,C_UYG][:len(data_)]):
                patch.set_facecolor(clr); patch.set_alpha(0.75)
            ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=1.5, ls="--", label=f"Eşik ({YAKINSAMA_ESIK:.0f})")
            ax.set_ylabel("Episode Ödülü"); ax.legend()
            ax.set_title("PLANLAMA vs UYGULAMA — Ödül Dağılımı Karşılaştırması")
            _save(fig, "24_plan_vs_uygulama_reward")
        else: print("  [ATLA] Veri yok")

        # ── 25. En İyi / En Kötü 5 Episode ────────────────────────────────────
        print("[25/25] En iyi / en kötü 5 episode karşılaştırması...")
        if N_ep >= 10:
            rh   = episode_rewards_history
            sidx = np.argsort(rh)
            worst5 = sidx[:5]; best5 = sidx[-5:]
            eps = np.arange(1, N_ep+1)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(eps, rh, color="#cccccc", s=15, alpha=0.5, label="Tüm Episodeler")
            ax.scatter(worst5+1, [rh[i] for i in worst5], color=C_LOSS,  s=100, zorder=5, marker="v", label="En Kötü 5")
            ax.scatter(best5+1,  [rh[i] for i in best5],  color=C_REW,   s=100, zorder=5, marker="^", label="En İyi 5")
            ax.axhline(YAKINSAMA_ESIK, color=C_THRESH, lw=1.5, ls="--", label=f"Eşik ({YAKINSAMA_ESIK:.0f})")
            ax.set_xlabel("Episode"); ax.set_ylabel("Toplam Ödül")
            ax.set_title("PPO — En İyi / En Kötü 5 Episode Karşılaştırması"); ax.legend()
            _save(fig, "25_best_worst_episodes")
        else: print("  [ATLA] Yetersiz veri (<10 ep)")

        # ── Özet ──────────────────────────────────────────────────────────────
        print("\n" + "="*65)
        print("  TÜM GRAFİKLER BAŞARIYLA KAYDEDİLDİ!")
        print(f"  Çıktı dizini : {OUT_DIR}")
        print(f"  Toplam Episode      : {N_ep}")
        print(f"  PPO Güncelleme      : {N_upd}")
        print(f"  UYGULAMA Adım Sayısı: {N_uyg}")
        print(f"  Sanal Konum Kaydı   : {N_virt}")
        print("="*65)

    except Exception as _ge:
        print(f"\n[GRAFİK HATA] {_ge}")
        import traceback; traceback.print_exc()
