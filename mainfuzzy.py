"""
==============================================================================
 SWARM OS v4.0 - OBSTACLE PLACEMENT + TRIANGLE / LINE FORMATION
==============================================================================
 LEFT CLICK   → Place obstacle on arena (max 10). Right-click clears all.
 [T]          → Toggle movement ON / OFF (formation starts / stops)
 [1]          → Switch to LINE formation
 [2]          → Switch to TRIANGLE formation (default)
 [C]          → Reset calibration, path trail, obstacles
 [Z] / [E]    → Emergency stop toggle
 [Q]          → Quit
 [WASD]       → Manually drive selected robot
 [F]          → Cycle manual-control robot
 [V]          → Toggle HUD
==============================================================================
"""

import cv2
import numpy as np
import threading
import time
import math
import serial
import os
from collections import deque
from pupil_apriltags import Detector
import keyboard
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ==============================================================================
#  HARDWARE CALIBRATION
# ==============================================================================
ROBOT_CALIBRATION = {
    0: {"speed_factor": 1.0, "left_rev": False, "right_rev": False},
    1: {"speed_factor": 1.0, "left_rev": False, "right_rev": False},
    2: {"speed_factor": 1.0, "left_rev": False, "right_rev": False},
    3: {"speed_factor": 1.0, "left_rev": False, "right_rev": False},
}


# ==============================================================================
#  LATENCY-FREE CAMERA THREAD
# ==============================================================================
class LatencyFreeCamera:
    """Reads camera frames continuously in a background thread."""

    def __init__(self, source):
        self.cap   = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ok    = False

        if not self.cap.isOpened():
            print(f"\n[!!] CAMERA CONNECTION FAILED ({source})")
            print("[!!] Blind mode — only manual WASD drive active.\n")
            self.running = False
            return

        for _ in range(10):
            self.cap.read()
        self.ok, self.frame = self.cap.read()
        self.running = True
        self.thread  = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            self.ok, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join()
        if hasattr(self, "cap") and self.cap:
            self.cap.release()


# ==============================================================================
#  PROJECT SETTINGS
# ==============================================================================
at_detector = Detector(
    families="tag36h11", nthreads=6, quad_decimate=2.0, quad_sigma=0.0
)

ROBOTS = {
    0: {"name": "LEADER",    "color": (255, 0, 255)},
    1: {"name": "FOLLOWER1", "color": (0, 255, 255)},
    2: {"name": "FOLLOWER2", "color": (0, 255, 0)},
    3: {"name": "FOLLOWER3", "color": (255, 255, 0)},
}
LEADER_ID       = 0
path_trail      = deque(maxlen=200)
LOGIC_GRID_MAX  = 400.0
EMA_ALPHA       = 0.2
DETECTION_WIDTH = 0
KNOWN_WIDTH_CM  = 5.0
FOCAL_LENGTH    = 1623.4

# ==============================================================================
#  OBSTACLE & FORMATION STATE
# ==============================================================================
obstacles       = []
MAX_OBSTACLES   = 10
OBSTACLE_RADIUS = 18.0

formation_mode  = "triangle"
movement_active = True
FORMATION_SPACING = 65.0

# Arena physical dimensions (cm)
ARENA_WIDTH_CM  = 300.0   # 3 m (yatay / horizontal)
ARENA_HEIGHT_CM = 200.0   # 2 m (dikey / vertical)


# ==============================================================================
#  COORDINATE HELPERS
# ==============================================================================
def calculate_distance(pixel_width: float) -> float:
    if pixel_width == 0:
        return 0.0
    return (KNOWN_WIDTH_CM * FOCAL_LENGTH) / pixel_width


robot_states = {
    rid: {
        "x": None, "y": None,
        "base_angle": None, "current_angle": None, "raw_angle": 0.0,
    }
    for rid in ROBOTS
}


def normalize_coords(raw_x, raw_y, frame_w, frame_h):
    return round((raw_x / frame_w) * LOGIC_GRID_MAX, 1), \
           round((raw_y / frame_h) * LOGIC_GRID_MAX, 1)


def apply_ema(robot_id, target_x, target_y):
    prev_x = robot_states[robot_id]["x"]
    prev_y = robot_states[robot_id]["y"]
    if prev_x is None:
        robot_states[robot_id]["x"] = target_x
        robot_states[robot_id]["y"] = target_y
    else:
        robot_states[robot_id]["x"] = EMA_ALPHA * target_x + (1 - EMA_ALPHA) * prev_x
        robot_states[robot_id]["y"] = EMA_ALPHA * target_y + (1 - EMA_ALPHA) * prev_y
    return round(robot_states[robot_id]["x"], 1), round(robot_states[robot_id]["y"], 1)


def get_calibrated_angle(robot_id, raw_angle):
    if robot_states[robot_id]["base_angle"] is None:
        robot_states[robot_id]["base_angle"] = raw_angle
        return 0
    return (raw_angle - robot_states[robot_id]["base_angle"] + 180) % 360 - 180



# ==============================================================================
#  KALMAN FILTER (POSITION TRACKING)
# ==============================================================================
class RobotTracker:
    def __init__(self, dt=0.05):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                             [0, 1, 0, dt],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.initialized = False

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def update(self, x, y):
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)
        return float(self.kf.statePost[0]), float(self.kf.statePost[1])

robot_trackers = {rid: RobotTracker() for rid in ROBOTS}


# ==============================================================================
#  FUZZY LOGIC ENGINE
# ==============================================================================
class FuzzyACC:
    @staticmethod
    def trapezoid(x, a, b, c, d):
        if x <= a or x >= d: return 0.0
        if b <= x <= c:      return 1.0
        if a < x < b:        return (x - a) / (b - a)
        if c < x < d:        return (d - x) / (d - c)
        return 0.0

    @staticmethod
    def get_speed_multiplier(dist_target, dist_neighbor, angle_err):
        t_near = FuzzyACC.trapezoid(dist_target,  -1,   0,  15,  30)
        t_mid  = FuzzyACC.trapezoid(dist_target,  15,  30,  60,  80)
        t_far  = FuzzyACC.trapezoid(dist_target,  60,  80, 9999, 9999)
        n_danger = FuzzyACC.trapezoid(dist_neighbor, -1,  0,  10,  16)
        n_mid    = FuzzyACC.trapezoid(dist_neighbor, 10, 16,  25,  35)
        n_safe   = FuzzyACC.trapezoid(dist_neighbor, 25, 35, 9999, 9999)
        abs_err    = abs(angle_err)
        a_straight = FuzzyACC.trapezoid(abs_err, -1,  0,  10,  25)
        a_medium   = FuzzyACC.trapezoid(abs_err, 10, 25,  45,  60)
        a_sharp    = FuzzyACC.trapezoid(abs_err, 45, 60, 180, 181)
        num, den = 0.0, 0.0

        def rule(w, v):
            nonlocal num, den
            num += w * v; den += w

        rule(n_danger,                    0.15)
        rule(a_sharp,                     0.0)
        rule(t_near * a_straight,         0.6)
        rule(t_mid  * n_safe * a_straight,0.85)
        rule(t_far  * n_safe * a_straight,1.0)
        rule(t_far  * n_mid  * a_straight,0.7)
        rule(n_mid,                       0.5)
        rule(a_medium,                    0.5)
        return num / den if den > 0 else 0.6


class FuzzyPID:
    @staticmethod
    def get_multipliers(error):
        ae = abs(error)
        ez = FuzzyACC.trapezoid(ae,  -1,  0,   5,  12)
        em = FuzzyACC.trapezoid(ae,   5, 12,  25,  45)
        el = FuzzyACC.trapezoid(ae,  25, 45, 180, 181)
        kp = (ez * 0.4 + em * 1.0 + el * 1.6) / (ez + em + el) if (ez+em+el)>0 else 1.0
        kd = (ez * 2.4 + em * 1.0 + el * 0.4) / (ez + em + el) if (ez+em+el)>0 else 1.0
        return kp, kd


class FuzzyCollision:
    @staticmethod
    def get_weights(min_dist):
        danger = FuzzyACC.trapezoid(min_dist, -1,  0, 20, 26)
        risky  = FuzzyACC.trapezoid(min_dist, 20, 26, 35, 50)
        safe   = FuzzyACC.trapezoid(min_dist, 35, 50, 9999, 9999)
        den_r  = danger + risky + safe
        rep_w  = (danger * 2.0 + risky * 0.5) / den_r if den_r > 0 else 0.0
        goal_w = (risky * 0.7 + safe * 1.0)   / den_r if den_r > 0 else 1.0
        return rep_w, goal_w


class FuzzyFormationManager:
    @staticmethod
    def get_formation_mode(robot_states, obstacles):
        """
        Uses fuzzy logic to decide between TRIANGLE (narrow, safe for obstacles)
        and LINE (wide, for open spaces) based on obstacle proximity.
        """
        # Find the leader's position
        lx = robot_states[0]["x"]
        ly = robot_states[0]["y"]
        if lx is None or ly is None:
            return "triangle" # Default fallback
            
        # 1. Calculate minimum distance from the leader to any obstacle
        min_dist = 999.0
        for (ox, oy, _) in obstacles:
            dist = math.sqrt((ox - lx)**2 + (oy - ly)**2)
            if dist < min_dist:
                min_dist = dist
                
        # 2. Define fuzzy sets for obstacle proximity (Near, Mid, Far)
        # Near (danger): trapezoid(min_dist, -1, 0, 80, 110)
        # Mid (caution): trapezoid(min_dist, 80, 110, 150, 180)
        # Far (safe):    trapezoid(min_dist, 150, 180, 9999, 9999)
        f_near = FuzzyACC.trapezoid(min_dist, -1, 0, 80, 110)
        f_mid  = FuzzyACC.trapezoid(min_dist, 80, 110, 150, 180)
        f_far  = FuzzyACC.trapezoid(min_dist, 150, 180, 9999, 9999)
        
        # 3. Defuzzify to choose formation:
        # - Triangle: score 0.0
        # - Line: score 1.0
        num = f_near * 0.0 + f_mid * 0.3 + f_far * 1.0
        den = f_near + f_mid + f_far
        
        score = num / den if den > 0 else 0.0
        
        # If score < 0.5, choose triangle (narrow), otherwise choose line (wide)
        if score < 0.5:
            return "triangle"
        else:
            return "line"


# ==============================================================================
#  PID CONTROLLER
# ==============================================================================
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.prev_error = 0.0; self.integral = 0.0
        self.last_time  = time.time()

    def compute(self, error):
        now = time.time()
        dt  = max(now - self.last_time, 0.01)
        self.integral   = max(-50.0, min(50.0, self.integral + error * dt))
        derivative      = (error - self.prev_error) / dt
        self.prev_error = error; self.last_time = now
        kp_m, kd_m = FuzzyPID.get_multipliers(error)
        return (self.kp * kp_m * error) + (self.ki * self.integral) + (self.kd * kd_m * derivative)


pid_turn = {rid: PIDController(0.4, 0.01, 0.1) for rid in ROBOTS}

# ==============================================================================
#  SERIAL / ROBOT COMMUNICATION
# ==============================================================================
GATEWAY_PORT   = "COM7"
BAUD_RATE      = 115200
gateway_serial = None
last_cmd_times = {rid: 0.0 for rid in ROBOTS}


def connect_serial():
    global gateway_serial
    try:
        if gateway_serial and gateway_serial.is_open:
            return True
        gateway_serial = serial.Serial(GATEWAY_PORT, BAUD_RATE, timeout=0.1)
        gateway_serial.dtr = False; gateway_serial.rts = False
        print(f"[SUCCESS] Serial port {GATEWAY_PORT} connected!")
        return True
    except Exception as e:
        print(f"[ERROR] Serial connection failed: {e}")
        return False


def serial_reader():
    global gateway_serial
    while True:
        if gateway_serial and gateway_serial.is_open:
            try:
                if gateway_serial.in_waiting > 0:
                    gateway_serial.readline()
            except Exception:
                try: gateway_serial.close()
                except: pass
                gateway_serial = None; time.sleep(2)
        else:
            connect_serial(); time.sleep(2)
        time.sleep(0.01)


connect_serial()
threading.Thread(target=serial_reader, daemon=True).start()
time.sleep(1.0)


def send_robot_command(robot_id, left, right):
    if robot_id in ROBOT_CALIBRATION:
        cal = ROBOT_CALIBRATION[robot_id]
        left  *= cal["speed_factor"]; right *= cal["speed_factor"]
        if cal["left_rev"]:  left  = -left
        if cal["right_rev"]: right = -right
    left  = max(-50, min(50, int(left)))
    right = max(-50, min(50, int(right)))
    if gateway_serial and gateway_serial.is_open:
        gateway_serial.write(f"<{robot_id},{left},{right}>\n".encode("utf-8"))
        gateway_serial.flush()


# ==============================================================================
#  COLLISION AVOIDANCE
# ==============================================================================
def check_collision(robot_id, target_x, target_y):
    st = robot_states[robot_id]
    if st["x"] is None:
        return target_x, target_y, 999.0
    force_x = force_y = 0.0
    min_dist = 999.0

    # 1. Robot-Robot Repulsion (sadece çok yakınsa)
    for oid, ost in robot_states.items():
        if oid == robot_id or ost["x"] is None: continue
        if robot_id == LEADER_ID: continue
        dx = st["x"] - ost["x"]; dy = st["y"] - ost["y"]
        dist = math.sqrt(dx**2 + dy**2)
        min_dist = min(min_dist, dist)
        if 0 < dist < 35:
            s = (200.0 if oid == LEADER_ID else 120.0) / (dist + 1)
            force_x += (dx / dist) * s; force_y += (dy / dist) * s

    # 2. Advanced APF for Obstacles (Repulsive + Tangential/Vortex)
    for (ox, oy, r) in obstacles:
        dx = st["x"] - ox; dy = st["y"] - oy
        dist = math.sqrt(dx**2 + dy**2)
        min_dist = min(min_dist, dist)
        
        eff_radius = r + 25  # Safety padding
        if 0 < dist < eff_radius + 45:
            # Repulsive Force
            s_rep = 650.0 / (dist + 1)
            f_rep_x = (dx / dist) * s_rep
            f_rep_y = (dy / dist) * s_rep
            
            # Tangential (Vortex) Force to slide around the obstacle smoothly
            # By adding a perpendicular vector (-dy, dx)
            # The direction of the vortex depends on which side is faster to slide around
            # We can use a simple cross product heuristic or just a fixed curl
            cross_z = dx * (target_y - st["y"]) - dy * (target_x - st["x"])
            sign = 1 if cross_z > 0 else -1
            
            s_tan = 450.0 / (dist + 1)
            f_tan_x = -sign * dy / dist * s_tan
            f_tan_y = sign * dx / dist * s_tan
            
            force_x += f_rep_x + f_tan_x
            force_y += f_rep_y + f_tan_y

    rep_w, goal_w = FuzzyCollision.get_weights(min_dist)
    bx = target_x * goal_w + st["x"] * (1.0 - goal_w)
    by = target_y * goal_w + st["y"] * (1.0 - goal_w)
    fx = max(0.0, min(LOGIC_GRID_MAX, bx + force_x * rep_w))
    fy = max(0.0, min(LOGIC_GRID_MAX, by + force_y * rep_w))
    return fx, fy, min_dist


# ==============================================================================
#  ROBOT DRIVE FUNCTION
# ==============================================================================
leader_speed_ff = 0.0
leader_prev_pos = {"x": None, "y": None, "t": time.time()}


def drive_robot(robot_id, t_x, t_y, use_pid=True, tolerance=15, slow_mode=False):
    st = robot_states[robot_id]
    if st["x"] is None:
        return 0, 0

    # Hard Safety Shield: Inter-robot collision avoidance (sadece çok yakınsa)
    for oid, ost in robot_states.items():
        if oid != robot_id and ost["x"] is not None:
            dist_o = math.sqrt((ost["x"] - st["x"])**2 + (ost["y"] - st["y"])**2)
            if dist_o < 20.0:
                angle_to_other = math.degrees(math.atan2(ost["y"] - st["y"], ost["x"] - st["x"])) % 360
                heading = st["raw_angle"] % 360
                diff_o = (angle_to_other - heading + 180) % 360 - 180
                
                if dist_o < 18.0:
                    # CRITICAL ZONE: Actively separate
                    if abs(diff_o) < 90:
                        return -15, -15  # Back away
                    else:
                        return 15, 15   # Crawl forward
                else:
                    # WARNING ZONE: Stop only if heading directly at it
                    if abs(diff_o) < 45:
                        return 0, 0     # Stop

    adj_x, adj_y, min_d = check_collision(robot_id, t_x, t_y)
    st = robot_states[robot_id]
    if st["x"] is None: return 0, 0
    dx = adj_x - st["x"]; dy = adj_y - st["y"]
    dist = math.sqrt(dx**2 + dy**2)
    if dist > tolerance:
        target_angle = int(math.degrees(math.atan2(dy, dx))) % 360
        diff = (target_angle - st["raw_angle"] + 180) % 360 - 180
        if use_pid:
            turn = pid_turn[robot_id].compute(diff)
            if slow_mode: turn = max(-20.0, min(20.0, turn))
        else:
            turn = diff * 0.5
        if abs(diff) < 5.0:
            turn = 0.0
            if use_pid: pid_turn[robot_id].integral = 0.0
        fwd_max  = 32 if slow_mode else 42
        turn_max = 24 if slow_mode else 32
        fs = FuzzyACC.get_speed_multiplier(dist_target=dist, dist_neighbor=min_d, angle_err=diff)
        if abs(diff) > 60:
            fwd  = 0; turn = turn_max if diff > 0 else -turn_max
        else:
            fwd = int(fwd_max * fs * max(0.0, math.cos(math.radians(diff))))
            if robot_id != LEADER_ID:
                fwd += max(0, min(14, int(leader_speed_ff * 0.20)))
        return fwd + turn, fwd - turn
    if use_pid: pid_turn[robot_id].integral = 0.0
    return 0, 0


# ==============================================================================
#  FORMATION GEOMETRY
# ==============================================================================
def get_formation_targets(leader_x, leader_y, leader_angle_deg):
    """
    Üçgen formasyonu — her zaman aynı (engel olsun olmasın).
        Robot 0 = lider (en önde)
        Robot 1 = sol arka  (-140°)
        Robot 3 = orta arka (180°, lidere daha yakın)
        Robot 2 = sağ arka  (+140°)
    """
    rad = math.radians(leader_angle_deg)

    offsets = {
        1: (FORMATION_SPACING,        -140),   # sol arka
        3: (FORMATION_SPACING * 0.75,  180),   # orta arka (lidere yakın)
        2: (FORMATION_SPACING,        +140),   # sağ arka
    }

    return {
        rid: (leader_x + d * math.cos(rad + math.radians(a)),
              leader_y + d * math.sin(rad + math.radians(a)))
        for rid, (d, a) in offsets.items()
    }


# ==============================================================================
#  DASHBOARD DRAWING
# ==============================================================================
SIDEBAR_W = 340
TARGET_H  = 720
FONT      = cv2.FONT_HERSHEY_SIMPLEX


def draw_speed_bar(panel, x, y, value, max_val, color, w=140, h=12):
    fill = int((min(value, max_val) / max_val) * w) if max_val > 0 else 0
    cv2.rectangle(panel, (x, y), (x + w, y + h), (40, 40, 40), -1)
    if fill > 0: cv2.rectangle(panel, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (60, 60, 60), 1)


def draw_dashboard(panel, panel_h, now, movement_active, formation_mode,
                   manual_id, ui_cache, last_seen, fps_cache, obstacle_count,
                   auto_formation_active):
    panel[:] = (18, 18, 22)
    cv2.line(panel, (0, 0), (0, panel_h), (50, 50, 50), 2)
    cv2.putText(panel, "SWARM OS v4.0", (15, 35), FONT, 0.8, (0, 220, 220), 2)
    cv2.putText(panel, "(FORMATION)",   (210, 35), FONT, 0.5, (100, 255, 100), 1)
    cv2.line(panel, (15, 48), (SIDEBAR_W - 15, 48), (50, 50, 50), 1)
    y = 75
    mv_txt = "MOVING" if movement_active else "STOPPED"
    mv_clr = (0, 255, 100) if movement_active else (0, 100, 255)
    cv2.putText(panel, "DRIVE:",      (15, y), FONT, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, mv_txt,        (120, y), FONT, 0.55, mv_clr, 2)
    y += 28
    fm_txt = ("AUTO (" + ("TRI" if formation_mode == "triangle" else "LINE") + ")") if auto_formation_active else ("MANUAL (" + ("TRI" if formation_mode == "triangle" else "LINE") + ")")
    fm_clr = (100, 255, 100) if auto_formation_active else (255, 100, 255) if formation_mode == "triangle" else (0, 165, 255)
    cv2.putText(panel, "FORMATION:",  (15, y), FONT, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, fm_txt,        (140, y), FONT, 0.5, fm_clr, 1)
    y += 28
    ob_clr = (0, 255, 180) if obstacle_count == 0 else (0, 200, 255)
    cv2.putText(panel, "OBSTACLES:",  (15, y), FONT, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, f"{obstacle_count}/{MAX_OBSTACLES}",
                (140, y), FONT, 0.55, ob_clr, 1)
    y += 28
    r_name  = ROBOTS.get(manual_id, {}).get("name", "?")
    r_color = ROBOTS.get(manual_id, {}).get("color", (255, 255, 255))
    cv2.putText(panel, "MANUAL:",     (15, y), FONT, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, r_name,        (120, y), FONT, 0.55, r_color, 1)
    y += 28
    fps_val = fps_cache.get("fps", 0)
    fps_clr = (0, 255, 100) if fps_val >= 25 else (0, 140, 255) if fps_val >= 15 else (0, 0, 255)
    cv2.putText(panel, "FPS:",        (15, y), FONT, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, str(fps_val),  (120, y), FONT, 0.55, fps_clr, 2)
    y += 18
    cv2.line(panel, (15, y), (SIDEBAR_W - 15, y), (50, 50, 50), 1)
    y += 5
    for rid in ROBOTS:
        rn  = ROBOTS[rid]["name"]; rc = ROBOTS[rid]["color"]
        cache = ui_cache.get(rid, {})
        speed = cache.get("speed", 0.0); dcm = cache.get("dist_cm", 0)
        px = cache.get("x", "--"); py = cache.get("y", "--")
        lost = now - last_seen.get(rid, now) > 0.5
        y += 25
        if lost:
            cv2.putText(panel, f"[!] {rn}", (15, y), FONT, 0.55, (0, 0, 255), 2)
        else:
            cv2.circle(panel, (22, y - 5), 5, rc, -1)
            cv2.putText(panel, rn, (35, y), FONT, 0.55, rc, 1)
            cal = ROBOT_CALIBRATION[rid]
            cv2.putText(panel, f"x{cal['speed_factor']}", (220, y), FONT, 0.4, (100, 100, 100), 1)
        y += 22
        cv2.putText(panel, f"POS({px},{py})  DST:{dcm}cm",
                    (25, y), FONT, 0.4, (140, 140, 140), 1)
        y += 22
        cv2.putText(panel, f"SPD: {speed:.0f} cm/s", (25, y), FONT, 0.45, (170, 170, 170), 1)
        draw_speed_bar(panel, 175, y - 10, speed, 30, rc, w=140, h=12)
        y += 12
    y += 8
    cv2.line(panel, (15, y), (SIDEBAR_W - 15, y), (50, 50, 50), 1)
    cv2.line(panel, (15, panel_h - 110), (SIDEBAR_W - 15, panel_h - 110), (40, 40, 40), 1)
    shortcuts = [
        ("[T]Start [1]Line [2]Tri [3]Auto",            (15, panel_h - 88)),
        ("[LClick] Add Obs  [RClick] Clear Obs",      (15, panel_h - 66)),
        ("[C] Reset  [F] Robot Sel  [Z] E-Stop",      (15, panel_h - 44)),
        ("[WASD] Drive  [V] HUD  [Q] Quit",           (15, panel_h - 22)),
    ]
    for txt, pos in shortcuts:
        cv2.putText(panel, txt, pos, FONT, 0.4, (120, 120, 120), 1)


# ==============================================================================
#  CAMERA & INITIALISATION
# ==============================================================================
CAMERA_SOURCE = "http://100.72.52.14:8080/video"
camera_stream = LatencyFreeCamera(CAMERA_SOURCE)

fw, fh = 1280, 720


def mouse_callback(event, x, y, flags, param):
    global obstacles, fw, fh
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(obstacles) < MAX_OBSTACLES:
            lx = (x / fw) * LOGIC_GRID_MAX
            ly = (y / fh) * LOGIC_GRID_MAX
            obstacles.append((lx, ly, OBSTACLE_RADIUS))
            print(f"[OBS] Placed at ({lx:.0f}, {ly:.0f}) "
                  f"— total {len(obstacles)}/{MAX_OBSTACLES}")
        else:
            print(f"[OBS] Max {MAX_OBSTACLES} reached. Right-click to clear.")
    elif event == cv2.EVENT_RBUTTONDOWN:
        obstacles.clear()
        print("[OBS] All obstacles cleared.")


cv2.namedWindow("HUB", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HUB", 1300, 720)
cv2.setMouseCallback("HUB", mouse_callback)

manual_target          = (200.0, 200.0) # Default target at center of arena
system_active          = False
manual_id              = LEADER_ID
last_seen_times        = {rid: time.time() for rid in ROBOTS}
last_known_pixels      = {}
last_formation_targets = {}
auto_formation_active  = True # Start in Auto formation selection mode

show_ui         = True
ui_cache        = {rid: {"speed": 0.0, "dist_cm": 0, "x": "--", "y": "--"}
                   for rid in ROBOTS}
ui_prev_pos     = {rid: {"x": None, "y": None, "t": time.time()} for rid in ROBOTS}
last_ui_update  = 0.0
fps_cache       = {"fps": 0, "frame_count": 0, "last_time": time.time()}
cached_sidebar  = None

last_f_press = last_c_press = last_t_press = 0.0
last_1_press = last_2_press = last_3_press = 0.0

print("\n--- SWARM OS v4.0 STARTED ---")
print("LEFT CLICK to place obstacles (max 10). RIGHT CLICK to clear.")
print("Press [T] to start / stop formation movement.")
print("Press [1] for LINE, [2] for TRIANGLE formation.")
print("Press [F] to cycle which robot WASD controls.")

path_trail.clear()

# ==============================================================================
#  EXPERIMENT DATA LOGS  (filled during run, plotted on exit)
# ==============================================================================
run_start_time = time.time()

pos_log     = {rid: [] for rid in ROBOTS}      # [(t, x, y), ...]
spd_log     = {rid: [] for rid in ROBOTS}      # [(t, speed), ...]
fmt_log     = {rid: [] for rid in [1, 2, 3]}   # [(t, error_dist), ...]
coh_log     = []                               # [(t, cohesion), ...]
pair_log    = {(0,1):[], (0,2):[], (0,3):[],
               (1,2):[], (1,3):[], (2,3):[]}   # [(t, dist), ...]
obs_dist_log= []                               # [(t, rid, min_obs_dist), ...]
heading_log = []                               # [(t, angle_deg), ...]
cmd_log     = {rid: [] for rid in ROBOTS}      # [(t, ls, rs), ...]

robot_commands = {rid: (0, 0) for rid in ROBOTS}
manual_ls = 0
manual_rs = 0
manual_drive_active = False

# ==============================================================================
#  MULTI-THREADING ARCHITECTURE
# ==============================================================================
state_lock = threading.Lock()
current_frame = None
vision_fps = 0

class VisionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        global current_frame, vision_fps
        fps_counter = 0
        last_time = time.time()
        while self.running:
            frame = camera_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            fh, fw = frame.shape[:2]
            scale = DETECTION_WIDTH / float(fw) if DETECTION_WIDTH > 0 else 1.0
            if scale != 1.0:
                small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0,0), fx=scale, fy=scale)
            else:
                small_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            raw_tags = at_detector.detect(small_gray)
            now = time.time()
            
            with state_lock:
                detected_this_frame = set()
                for tag in raw_tags:
                    tid = tag.tag_id
                    if tid not in ROBOTS: continue
                    detected_this_frame.add(tid)
                    
                    cx = int(tag.center[0] / scale)
                    cy = int(tag.center[1] / scale)
                    corners = tag.corners / scale
                    
                    last_known_pixels[tid] = (cx, cy)
                    last_seen_times[tid] = now
                    
                    nx, ny = normalize_coords(cx, cy, fw, fh)
                    
                    # Kalman Update
                    kx, ky = robot_trackers[tid].update(nx, ny)
                    
                    bx = (corners[0][0]+corners[1][0])/2; by = (corners[0][1]+corners[1][1])/2
                    tx = (corners[3][0]+corners[2][0])/2; ty = (corners[3][1]+corners[2][1])/2
                    raw_angle = (int(math.degrees(math.atan2(ty-by, tx-bx)))+180) % 360
                    net_angle = get_calibrated_angle(tid, raw_angle)
                    
                    robot_states[tid]["current_angle"] = net_angle
                    robot_states[tid]["raw_angle"] = raw_angle
                    robot_states[tid]["x"] = kx
                    robot_states[tid]["y"] = ky
                    
                    pixel_w = math.sqrt((corners[1][0]-corners[0][0])**2+(corners[1][1]-corners[0][1])**2)
                    ui_cache[tid]["dist_cm"] = int(calculate_distance(pixel_w))
                    ui_cache[tid]["x"] = round(kx, 1)
                    ui_cache[tid]["y"] = round(ky, 1)
                    
                    if tid == LEADER_ID:
                        if (len(path_trail)==0 or math.sqrt((kx-path_trail[-1][0])**2+(ky-path_trail[-1][1])**2)>3):
                            path_trail.append((kx, ky, raw_angle))
                            
                # Kalman Predict for occluded robots
                for tid in ROBOTS:
                    if tid not in detected_this_frame and robot_trackers[tid].initialized:
                        time_since_seen = now - last_seen_times.get(tid, 0)
                        if time_since_seen < 1.5:  # Blind tracking for 1.5s
                            px, py = robot_trackers[tid].predict()
                            robot_states[tid]["x"] = px
                            robot_states[tid]["y"] = py
                            ui_cache[tid]["x"] = round(px, 1)
                            ui_cache[tid]["y"] = round(py, 1)
            
            current_frame = frame
            
            fps_counter += 1
            if now - last_time > 1.0:
                vision_fps = fps_counter
                fps_counter = 0
                last_time = now

class ControlThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        global formation_mode, leader_speed_ff, robot_commands
        while self.running:
            time.sleep(0.05) # 20 Hz
            now = time.time()
            t_rel = now - run_start_time
            
            with state_lock:
                l_st = robot_states[LEADER_ID]
                if l_st["x"] is not None:
                    if leader_prev_pos["x"] is not None:
                        dt_ff = now - leader_prev_pos["t"]
                        if dt_ff > 0.01:
                            dx_ff = l_st["x"] - leader_prev_pos["x"]
                            dy_ff = l_st["y"] - leader_prev_pos["y"]
                            inst  = math.sqrt(dx_ff**2 + dy_ff**2) / dt_ff
                            leader_speed_ff = 0.7 * leader_speed_ff + 0.3 * inst
                    leader_prev_pos["x"] = l_st["x"]
                    leader_prev_pos["y"] = l_st["y"]
                    leader_prev_pos["t"] = now

                if auto_formation_active:
                    formation_mode = FuzzyFormationManager.get_formation_mode(robot_states, obstacles)

                if movement_active:
                    if manual_target is not None:
                        ls, rs = drive_robot(LEADER_ID, manual_target[0], manual_target[1], tolerance=12, slow_mode=False)
                        robot_commands[LEADER_ID] = (ls, rs)

                    if l_st["x"] is not None and len(path_trail) > 0:
                        lx, ly, l_angle = path_trail[-1]
                        formation_targets = get_formation_targets(lx, ly, l_angle)
                        last_formation_targets.update(formation_targets)
                        for rid in [1, 2, 3]:
                            tx, ty = formation_targets[rid]
                            ls, rs = drive_robot(rid, tx, ty, tolerance=12, slow_mode=False)
                            robot_commands[rid] = (ls, rs)

                    cohesion = 1.0
                    errors = []
                    for rid in [1, 2, 3]:
                        st = robot_states[rid]
                        if st["x"] is not None and rid in last_formation_targets:
                            tx, ty = last_formation_targets[rid]
                            err = math.sqrt((st["x"]-tx)**2 + (st["y"]-ty)**2)
                            errors.append(err)
                            fmt_log[rid].append((t_rel, err))

                    if errors:
                        max_err = max(errors)
                        if max_err > 80.0:
                            cohesion = max(0.3, 1.0 - (max_err - 80.0) / 80.0)

                    coh_log.append((t_rel, cohesion))
                    # Lider her zaman tam güçle gider, cohesion sadece log için
                    # robot_commands[LEADER_ID] artık yavaşlatılmıyor

                if manual_drive_active:
                    robot_commands[manual_id] = (manual_ls, manual_rs)


class CommThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        while self.running:
            time.sleep(0.05) # 20 Hz
            now = time.time()
            t_rel = now - run_start_time
            
            with state_lock:
                for rid in ROBOTS:
                    ls, rs = robot_commands.get(rid, (0, 0))
                    if not movement_active and not (manual_drive_active and rid == manual_id):
                        ls, rs = 0, 0
                    if now - last_seen_times.get(rid, now) > 0.5:
                        if not (manual_drive_active and rid == manual_id):
                            ls, rs = 0, 0

                    cmd_log[rid].append((t_rel, ls, rs))
                    if now - last_cmd_times.get(rid, 0) > 0.15:
                        send_robot_command(rid, ls, rs)
                        last_cmd_times[rid] = now

# Start background threads
v_thread = VisionThread()
c_thread = ControlThread()
cm_thread = CommThread()

v_thread.start()
c_thread.start()
cm_thread.start()

# ==============================================================================
#  MAIN LOOP (UI & Keyboard)
# ==============================================================================
try:
    while True:
        now = time.time()
        t_rel = now - run_start_time
        
        # ── Keyboard: cycle robot ──────────────────────────────────────────────
        if keyboard.is_pressed("f") and (now - last_f_press > 0.5):
            last_f_press = now
            id_list  = list(ROBOTS.keys())
            manual_id = id_list[(id_list.index(manual_id) + 1) % len(id_list)] if manual_id in id_list else LEADER_ID
            print(f"> Manual control: {ROBOTS[manual_id]['name']}")

        # ── [C] reset ─────────────────────────────────────────────────────────
        if keyboard.is_pressed("c") and (now - last_c_press > 0.5):
            last_c_press = now
            with state_lock:
                for rid in robot_states: robot_states[rid]["base_angle"] = None
                path_trail.clear(); manual_target = None; obstacles.clear()
            print("> Calibration, path, target and obstacles reset!")

        # ── [T] toggle movement ───────────────────────────────────────────────
        if keyboard.is_pressed("t") and (now - last_t_press > 0.5):
            last_t_press    = now
            movement_active = not movement_active
            if not movement_active:
                with state_lock:
                    for rid in ROBOTS: send_robot_command(rid, 0, 0)
            print(f"> Movement {'STARTED' if movement_active else 'STOPPED'} | Formation: {formation_mode.upper()}")

        # ── [1]/[2]/[3] formation ─────────────────────────────────────────────
        if keyboard.is_pressed("1") and (now - last_1_press > 0.5):
            last_1_press   = now; formation_mode = "line"; auto_formation_active = False
            print("> Formation: LINE (Manual Override)")
        if keyboard.is_pressed("2") and (now - last_2_press > 0.5):
            last_2_press   = now; formation_mode = "triangle"; auto_formation_active = False
            print("> Formation: TRIANGLE (Manual Override)")
        if keyboard.is_pressed("3") and (now - last_3_press > 0.5):
            last_3_press   = now; auto_formation_active = True
            print("> Formation: AUTOMATIC (Fuzzy Decision Active)")

        # ── WASD manual drive ─────────────────────────────────────────────────
        m_ls = m_rs = 0
        m_active = False
        if any(keyboard.is_pressed(k) for k in ["w", "a", "s", "d"]):
            m_active = True
            spd = 30 if keyboard.is_pressed("shift") else 25
            fwd = (1 if keyboard.is_pressed("w") else -1 if keyboard.is_pressed("s") else 0)
            trn = (-1 if keyboard.is_pressed("a") else 1 if keyboard.is_pressed("d") else 0)
            if trn == 0:   m_ls, m_rs = fwd*spd, fwd*spd
            elif fwd == 0: m_ls, m_rs = trn*spd, -trn*spd
            elif trn == -1:m_ls, m_rs = int(fwd*spd*0.2), fwd*spd
            else:          m_ls, m_rs = fwd*spd, int(fwd*spd*0.2)
        
        with state_lock:
            manual_ls, manual_rs = m_ls, m_rs
            manual_drive_active = m_active

        # ── Data Logging & UI cache ───────────────────────────────────────────
        if now - last_ui_update > 0.5:
            last_ui_update = now
            with state_lock:
                for rid in ROBOTS:
                    st   = robot_states[rid]
                    prev = ui_prev_pos[rid]
                    if st["x"] is not None and prev["x"] is not None:
                        dx_h = st["x"] - prev["x"]; dy_h = st["y"] - prev["y"]
                        dt_h = now - prev["t"]
                        if dt_h > 0:
                            spd = round(math.sqrt(dx_h**2+dy_h**2)/dt_h*0.5, 1)
                            ui_cache[rid]["speed"] = spd
                            spd_log[rid].append((t_rel, spd))
                    if st["x"] is not None:
                        pos_log[rid].append((t_rel, st["x"], st["y"]))
                    ui_prev_pos[rid] = {"x": st["x"], "y": st["y"], "t": now}

                for (a, b) in pair_log:
                    sa, sb = robot_states[a], robot_states[b]
                    if sa["x"] is not None and sb["x"] is not None:
                        d = math.sqrt((sa["x"]-sb["x"])**2+(sa["y"]-sb["y"])**2)
                        pair_log[(a,b)].append((t_rel, d))

                if obstacles:
                    for rid in ROBOTS:
                        st = robot_states[rid]
                        if st["x"] is not None:
                            min_od = min(math.sqrt((st["x"]-ox)**2+(st["y"]-oy)**2) for ox, oy, _ in obstacles)
                            obs_dist_log.append((t_rel, rid, min_od))

                l_raw = robot_states[LEADER_ID]["raw_angle"]
                heading_log.append((t_rel, l_raw))
        
        with state_lock:
            fps_cache["fps"] = vision_fps

        # ── Drawing UI ────────────────────────────────────────────────────────
        if current_frame is None:
            time.sleep(0.01)
            continue
            
        with state_lock:
            frame = current_frame.copy()
            local_path_trail = list(path_trail)
            local_targets = dict(last_formation_targets)
            local_obstacles = list(obstacles)
            local_pixels = dict(last_known_pixels)
            local_movement = movement_active
            local_formation = formation_mode
            local_auto_formation = auto_formation_active
        
        if show_ui:
            for rid, (cx, cy) in local_pixels.items():
                if now - last_seen_times.get(rid, now) > 0.5: continue
                r_color = ROBOTS[rid]["color"]
                cv2.circle(frame, (cx, cy), 4, r_color, -1)
                tpx = tpy = None
                if rid == LEADER_ID and manual_target is not None:
                    tpx = int(manual_target[0]/400*fw); tpy = int(manual_target[1]/400*fh)
                elif rid in [1,2,3] and rid in local_targets:
                    ft = local_targets[rid]
                    tpx = int(ft[0]/400*fw); tpy = int(ft[1]/400*fh)
                if tpx is not None:
                    ddx, ddy = tpx-cx, tpy-cy
                    ad = math.sqrt(ddx**2+ddy**2)
                    if ad > 5:
                        al = min(30, ad)
                        cv2.arrowedLine(frame,(cx,cy),(int(cx+ddx/ad*al), int(cy+ddy/ad*al)), r_color, 2, tipLength=0.35)

            if len(local_path_trail) > 1:
                pts = np.array([[int(p[0]/400*fw), int(p[1]/400*fh)] for p in local_path_trail], np.int32).reshape((-1,1,2))
                cv2.polylines(frame, [pts], False, (0,200,255), 2)

            if manual_target is not None:
                hx = int(manual_target[0]/400*fw); hy = int(manual_target[1]/400*fh)
                cv2.drawMarker(frame,(hx,hy),(0,255,100),cv2.MARKER_CROSS,20,2)
                cv2.circle(frame,(hx,hy),15,(0,255,100),1)

            for i, (ox, oy, r) in enumerate(local_obstacles):
                px = int((ox/LOGIC_GRID_MAX)*fw); py = int((oy/LOGIC_GRID_MAX)*fh)
                pr = max(int((r/LOGIC_GRID_MAX)*fw), 8)
                overlay = frame.copy()
                cv2.circle(overlay,(px,py),pr,(0,40,220),-1)
                cv2.addWeighted(overlay,0.35,frame,0.65,0,frame)
                cv2.circle(frame,(px,py),pr,(0,80,255),2)
                cv2.putText(frame,f"O{i+1}",(px-8,py+5),FONT,0.45,(255,200,200),1)

            for rid,(tx,ty) in local_targets.items():
                px=int((tx/LOGIC_GRID_MAX)*fw); py=int((ty/LOGIC_GRID_MAX)*fh)
                cv2.drawMarker(frame,(px,py),ROBOTS[rid]["color"],cv2.MARKER_TILTED_CROSS,16,1)

            cv2.rectangle(frame,(0,0),(440,62),(0,0,0),-1)
            mv_col = (0,255,100) if local_movement else (0,80,255)
            mv_lbl = "MOVING" if local_movement else "STOPPED — press [T]"
            cv2.putText(frame,f"[T] {mv_lbl}",(8,22),FONT,0.6,mv_col,2)
            fmt_lbl = ("AUTO (" + ("TRI" if local_formation == "triangle" else "LINE") + ")") if local_auto_formation else local_formation.upper()
            cv2.putText(frame,f"[1/2/3] Formation: {fmt_lbl}",(8,44),FONT,0.55,(200,200,0),1)
            cv2.putText(frame,f"OBS: {len(local_obstacles)}/{MAX_OBSTACLES}  RClick=clear",(245,44),FONT,0.45,(100,200,255),1)
        else:
            cv2.putText(frame,"HUD [V]",(fw-80,fh-10),FONT,0.9,(100,100,100),1)

        cam_h = TARGET_H
        cam_w = int(fw * (TARGET_H / fh))
        cam_scaled = cv2.resize(frame,(cam_w,cam_h),interpolation=cv2.INTER_NEAREST)

        if show_ui:
            if (cached_sidebar is None or cached_sidebar.shape[0]!=cam_h or now - last_ui_update < 0.05):
                cached_sidebar = np.zeros((cam_h,SIDEBAR_W,3),dtype=np.uint8)
                with state_lock:
                    draw_dashboard(cached_sidebar,cam_h,now,local_movement,local_formation,
                                   manual_id,ui_cache,last_seen_times,fps_cache,len(local_obstacles),
                                   local_auto_formation)
            display = cv2.hconcat([cam_scaled, cached_sidebar])
        else:
            display = cam_scaled

        cv2.imshow("HUB", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            with state_lock:
                for rid in ROBOTS: send_robot_command(rid, 0, 0)
            break
        elif key == ord("t") or key == ord("T"):
            movement_active = not movement_active
            if not movement_active:
                with state_lock:
                    for rid in ROBOTS: send_robot_command(rid, 0, 0)
        elif key == ord("1"):
            formation_mode = "line"; auto_formation_active = False
        elif key == ord("2"):
            formation_mode = "triangle"; auto_formation_active = False
        elif key == ord("3"):
            auto_formation_active = True
        elif key == ord("c") or key == ord("C"):
            with state_lock:
                for rid in robot_states: robot_states[rid]["base_angle"] = None
                path_trail.clear(); manual_target = None; obstacles.clear()
        elif key == ord("z") or key == ord("e"):
            movement_active = not movement_active
            if not movement_active:
                with state_lock:
                    for rid in ROBOTS: send_robot_command(rid, 0, 0)
        elif key == ord("v"):
            show_ui = not show_ui

finally:
    if gateway_serial and gateway_serial.is_open:
        for rid in ROBOTS: send_robot_command(rid, 0, 0)
        gateway_serial.close()
    camera_stream.stop()
    cv2.destroyAllWindows()


# ==============================================================================
#  POST-RUN VISUALISATION  — auto-generated on exit
# ==============================================================================
def generate_plots():
    total_duration = time.time() - run_start_time

    any_data = any(len(v) > 2 for v in pos_log.values())
    if not any_data:
        print("[PLOT] No position data collected — skipping plots.")
        return

    save_dir = "swarm_plots"
    os.makedirs(save_dir, exist_ok=True)

    # Matplotlib colours matching robot BGR colours (RGB 0-1)
    MC = {
        0: (1.00, 0.00, 1.00),   # magenta   — LEADER
        1: (0.00, 0.85, 0.85),   # cyan      — FOLLOWER1
        2: (0.00, 0.80, 0.00),   # green     — FOLLOWER2
        3: (0.90, 0.90, 0.00),   # yellow    — FOLLOWER3
    }
    RN = {rid: ROBOTS[rid]["name"] for rid in ROBOTS}

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
        "xtick.color": "#444444",    "ytick.color": "#444444",
        "grid.color": "#dddddd",     "font.family": "DejaVu Sans",
        "axes.titlesize": 12,        "axes.labelsize": 10,
    })
    print(f"\n[PLOT] Generating plots …  (session: {total_duration:.1f} s)")

    def save(fig, name):
        path = os.path.join(save_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[PLOT]  {name}")

    # ── 01 Top-down trajectory map ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Robot Trajectories (Top-Down View)", fontweight="bold")
    ax.set_xlabel("X (logic units)"); ax.set_ylabel("Y (logic units)")
    ax.set_xlim(0, LOGIC_GRID_MAX); ax.set_ylim(LOGIC_GRID_MAX, 0)
    ax.set_aspect("equal"); ax.grid(True, lw=0.5)
    for rid, pts in pos_log.items():
        if len(pts) < 2: continue
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ys, color=MC[rid], lw=1.4, label=RN[rid], alpha=0.85)
        ax.scatter(xs[0],  ys[0],  color=MC[rid], marker="o", s=60, zorder=5)
        ax.scatter(xs[-1], ys[-1], color=MC[rid], marker="*", s=120, zorder=5)
    for (ox, oy, r) in obstacles:
        ax.add_patch(mpatches.Circle((ox, oy), r, color="red", alpha=0.25,
                                     edgecolor="darkred", lw=2))
        ax.text(ox, oy, "OBS", ha="center", va="center", fontsize=7, color="darkred")
    ax.legend(loc="upper right"); fig.tight_layout()
    save(fig, "01_trajectories.png")

    # ── 02 Speed over time ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Robot Speed Over Time", fontweight="bold")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (cm/s)"); ax.grid(True, lw=0.5)
    for rid, pts in spd_log.items():
        if len(pts) < 2: continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=MC[rid], lw=1.4, label=RN[rid], alpha=0.9)
    ax.legend(loc="upper right"); fig.tight_layout()
    save(fig, "02_speed_over_time.png")

    # ── 03 Formation error over time ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Formation Position Error (Followers)", fontweight="bold")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (logic units)"); ax.grid(True, lw=0.5)
    ax.axhline(40, color="red", ls="--", lw=0.9, label="Cohesion threshold (40)")
    for rid in [1, 2, 3]:
        pts = fmt_log[rid]
        if len(pts) < 2: continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=MC[rid], lw=1.4, label=RN[rid], alpha=0.9)
    ax.legend(loc="upper right"); fig.tight_layout()
    save(fig, "03_formation_error.png")

    # ── 04 Cohesion multiplier over time ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_title("Swarm Cohesion Multiplier (Leader Speed Brake)", fontweight="bold")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cohesion (0–1)"); ax.set_ylim(-0.05, 1.1)
    ax.grid(True, lw=0.5)
    if len(coh_log) > 1:
        ts = [c[0] for c in coh_log]; cs = [c[1] for c in coh_log]
        ax.fill_between(ts, cs, alpha=0.25, color="steelblue")
        ax.plot(ts, cs, color="steelblue", lw=1.5)
    fig.tight_layout()
    save(fig, "04_cohesion.png")

    # ── 05 Inter-robot distances ──────────────────────────────────────────────
    PAIR_C = {(0,1):"#e74c3c",(0,2):"#e67e22",(0,3):"#8e44ad",
              (1,2):"#27ae60",(1,3):"#2980b9",(2,3):"#c0392b"}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Inter-Robot Distances Over Time", fontweight="bold")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (logic units)"); ax.grid(True, lw=0.5)
    ax.axhline(50, color="orange", ls="--", lw=0.8, label="Repulsion zone (50)")
    for pair, pts in pair_log.items():
        if len(pts) < 2: continue
        lbl = f"{RN[pair[0]]} ↔ {RN[pair[1]]}"
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=PAIR_C[pair], lw=1.2, label=lbl, alpha=0.85)
    ax.legend(loc="upper right", fontsize=7); fig.tight_layout()
    save(fig, "05_inter_robot_dist.png")

    # ── 06 Formation error box plot ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Formation Error Distribution", fontweight="bold")
    ax.set_ylabel("Error (logic units)"); ax.grid(True, axis="y", lw=0.5)
    data, labels, bcolors = [], [], []
    for rid in [1, 2, 3]:
        pts = fmt_log[rid]
        if len(pts) > 1:
            data.append([p[1] for p in pts]); labels.append(RN[rid]); bcolors.append(MC[rid])
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, col in zip(bp["boxes"], bcolors):
            patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(40, color="red", ls="--", lw=0.9, label="Threshold"); ax.legend()
    fig.tight_layout()
    save(fig, "06_formation_error_boxplot.png")

    # ── 07 Speed histogram per robot ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharex=True, sharey=True)
    fig.suptitle("Speed Distribution per Robot", fontweight="bold")
    for ax, (rid, pts) in zip(axes, spd_log.items()):
        ax.set_facecolor("white"); ax.set_title(RN[rid]); ax.set_xlabel("Speed (cm/s)")
        if len(pts) > 2:
            ax.hist([p[1] for p in pts], bins=15, color=MC[rid], edgecolor="white", alpha=0.8)
        ax.grid(True, axis="y", lw=0.5)
    axes[0].set_ylabel("Count"); fig.tight_layout()
    save(fig, "07_speed_histogram.png")

    # ── 08 Leader heading over time ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_title("Leader Heading Over Time", fontweight="bold")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Heading (deg)"); ax.set_ylim(0, 360)
    ax.grid(True, lw=0.5)
    if len(heading_log) > 1:
        ts = [h[0] for h in heading_log]; hs = [h[1] for h in heading_log]
        ax.scatter(ts, hs, s=4, color=MC[LEADER_ID], alpha=0.6)
        ax.plot(ts, hs, color=MC[LEADER_ID], lw=0.8, alpha=0.4)
    fig.tight_layout()
    save(fig, "08_leader_heading.png")

    # ── 09 Nearest obstacle distance per robot ────────────────────────────────
    if obs_dist_log:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Nearest Obstacle Distance Per Robot", fontweight="bold")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (logic units)"); ax.grid(True, lw=0.5)
        ax.axhline(OBSTACLE_RADIUS + 20, color="red", ls="--", lw=0.9, label="Repulsion boundary")
        for rid in ROBOTS:
            pts = [(t, d) for (t, r, d) in obs_dist_log if r == rid]
            if len(pts) > 1:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=MC[rid], lw=1.2, label=RN[rid], alpha=0.85)
        ax.legend(); fig.tight_layout()
        save(fig, "09_obstacle_distance.png")

    # ── 10 Motor command scatter (left vs right) ──────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Motor Command Distribution (Left vs Right)", fontweight="bold")
    for ax, rid in zip(axes, ROBOTS):
        ax.set_facecolor("white"); ax.set_title(RN[rid], fontsize=10)
        pts = cmd_log[rid]
        if len(pts) > 2:
            ax.scatter([p[1] for p in pts], [p[2] for p in pts],
                       s=4, alpha=0.4, color=MC[rid])
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("Left motor"); ax.set_ylabel("Right motor")
        ax.set_xlim(-31, 31); ax.set_ylim(-31, 31); ax.grid(True, lw=0.4)
    fig.tight_layout()
    save(fig, "10_motor_commands.png")

    # ── 11 Position density heatmap ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Position Density (Time Spent per Area)", fontweight="bold")
    for ax, (rid, pts) in zip(axes, pos_log.items()):
        ax.set_facecolor("white"); ax.set_title(RN[rid], fontsize=10)
        if len(pts) > 5:
            h = ax.hist2d([p[1] for p in pts], [p[2] for p in pts],
                          bins=20, range=[[0,LOGIC_GRID_MAX],[0,LOGIC_GRID_MAX]], cmap="hot_r")
            fig.colorbar(h[3], ax=ax, label="Visits")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "11_position_heatmap.png")

    # ── 12 Summary dashboard (4-panel) ────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # 12a — Trajectory
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#f8f8f8")
    ax1.set_title("Trajectories", fontweight="bold")
    ax1.set_xlim(0, LOGIC_GRID_MAX); ax1.set_ylim(LOGIC_GRID_MAX, 0)
    ax1.set_aspect("equal"); ax1.grid(True, lw=0.4)
    for rid, pts in pos_log.items():
        if len(pts) < 2: continue
        ax1.plot([p[1] for p in pts], [p[2] for p in pts],
                 color=MC[rid], lw=1.2, label=RN[rid])
    for (ox, oy, r) in obstacles:
        ax1.add_patch(mpatches.Circle((ox, oy), r, color="red", alpha=0.3))
    ax1.legend(fontsize=7, loc="upper right")

    # 12b — Speed
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#f8f8f8")
    ax2.set_title("Speed (cm/s)", fontweight="bold"); ax2.grid(True, lw=0.4)
    for rid, pts in spd_log.items():
        if len(pts) < 2: continue
        ax2.plot([p[0] for p in pts], [p[1] for p in pts],
                 color=MC[rid], lw=1.0, label=RN[rid])
    ax2.set_xlabel("Time (s)"); ax2.legend(fontsize=7, loc="upper right")

    # 12c — Formation error
    ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor("#f8f8f8")
    ax3.set_title("Formation Error", fontweight="bold")
    ax3.axhline(40, color="red", ls="--", lw=0.8); ax3.grid(True, lw=0.4)
    for rid in [1, 2, 3]:
        pts = fmt_log[rid]
        if len(pts) < 2: continue
        ax3.plot([p[0] for p in pts], [p[1] for p in pts],
                 color=MC[rid], lw=1.0, label=RN[rid])
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Error (units)")
    ax3.legend(fontsize=7, loc="upper right")

    # 12d — Cohesion
    ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor("#f8f8f8")
    ax4.set_title("Cohesion Multiplier", fontweight="bold")
    ax4.set_ylim(-0.05, 1.1); ax4.grid(True, lw=0.4)
    if len(coh_log) > 1:
        ts = [c[0] for c in coh_log]; cs = [c[1] for c in coh_log]
        ax4.fill_between(ts, cs, alpha=0.2, color="steelblue")
        ax4.plot(ts, cs, color="steelblue", lw=1.2)
    ax4.set_xlabel("Time (s)")

    fig.suptitle(
        f"Swarm Experiment Summary  |  Duration: {total_duration:.1f} s  "
        f"|  Formation: {formation_mode.upper()}  "
        f"|  Obstacles: {len(obstacles)}",
        fontsize=11, fontweight="bold")
    save(fig, "12_summary_dashboard.png")

    print(f"\n[PLOT] All plots saved to './{save_dir}/' folder.")


generate_plots()
