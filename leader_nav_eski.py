import cv2
import numpy as np
import math
import serial
import time
import threading
from pupil_apriltags import Detector
CAMERA_SOURCE = "http://10.185.40.91:8080/video"
GATEWAY_PORT = "COM7"
BAUD_RATE = 115200
LEADER_ID = 0
APRILTAG_FAMILIES = "tag36h11"

MIN_PWM = 28
MAX_PWM = 50

ROBOT_CALIBRATION = {
    0: {"angle_offset": 0.0,   "left_invert": False, "right_invert": False},
    1: {"angle_offset": 0.0,   "left_invert": False, "right_invert": False},
    2: {"angle_offset": 0.0,   "left_invert": False, "right_invert": False},
    3: {"angle_offset": 180.0, "left_invert": False, "right_invert": False},
}

FWD_MAX = 4
FWD_CATCHUP = 6
TURN_MAX = 1.5
TURN_CATCHUP = 2.2

ARRIVAL_RADIUS = 12
FOLLOWER_WP_RADIUS = 18
SLOW_ZONE = 80
HEADING_DEADZONE = 2.5

EMA_ALPHA = 0.35

FORMATION_SPACING = 85.0
PATH_RECORD_DIST = 4.0

COLLISION_SAFE_DIST = 100.0
COLLISION_STOP_DIST = 65.0

FORMATIONS = {
    "triangle": {
        1: {"side_offset": FORMATION_SPACING,  "fb_offset": 0.0, "target_dist": FORMATION_SPACING},
        2: {"side_offset": 0.0,                "fb_offset": 0.0, "target_dist": FORMATION_SPACING},
        3: {"side_offset": -FORMATION_SPACING, "fb_offset": 0.0, "target_dist": FORMATION_SPACING},
    },
    "line": {
        1: {"side_offset": 0.0, "fb_offset": 0.0, "target_dist": FORMATION_SPACING},
        3: {"side_offset": 0.0, "fb_offset": 0.0, "target_dist": 2.0 * FORMATION_SPACING},
        2: {"side_offset": 0.0, "fb_offset": 0.0, "target_dist": 3.0 * FORMATION_SPACING},
    }
}

class LatencyFreeCamera:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

        if not self.cap.isOpened():
            print(f"[HATA] Kamera açılamadı: {source}")
            return

        for _ in range(5):
            self.cap.read()

        ret, self.frame = self.cap.read()
        if not ret:
            print("[HATA] Kameradan frame okunamadı!")
            return

        self.running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        print(f"[OK] Kamera bağlandı: {source}")

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()

class PositionFilter:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.x = None
        self.y = None

    def update(self, raw_x, raw_y):
        if self.x is None:
            self.x = float(raw_x)
            self.y = float(raw_y)
        else:
            self.x = self.alpha * raw_x + (1.0 - self.alpha) * self.x
            self.y = self.alpha * raw_y + (1.0 - self.alpha) * self.y
        return self.x, self.y

class AngleFilter:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.sin = None
        self.cos = None

    def update(self, raw_angle):
        rad = math.radians(raw_angle)
        s = math.sin(rad)
        c = math.cos(rad)
        if self.sin is None:
            self.sin = s
            self.cos = c
        else:
            self.sin = self.alpha * s + (1.0 - self.alpha) * self.sin
            self.cos = self.alpha * c + (1.0 - self.alpha) * self.cos
        return math.degrees(math.atan2(self.sin, self.cos)) % 360

def get_path_length(path):
    if len(path) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        dist += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


def apply_deadband(val):
    if val == 0:
        return 0
    sign = 1 if val > 0 else -1
    pwm = sign * (MIN_PWM + abs(val))
    return int(max(-MAX_PWM, min(MAX_PWM, pwm)))

def compute_motor_commands(angle_diff, dist, is_catchup=False, speed_scale=1.0):
    fwd_limit = FWD_CATCHUP if is_catchup else FWD_MAX
    turn_limit = TURN_CATCHUP if is_catchup else TURN_MAX

    fwd_max = fwd_limit * speed_scale
    turn_max = turn_limit * speed_scale

    abs_angle = abs(angle_diff)
    kp = 0.12
    turn = angle_diff * kp

    if dist < SLOW_ZONE:
        ratio = dist / SLOW_ZONE
        ratio = max(0.3, ratio)
        fwd_max *= ratio

    if abs_angle > 45.0:
        fwd = 0
    elif abs_angle > 15.0:
        blend = 1.0 - ((abs_angle - 15.0) / 30.0)
        fwd = int(fwd_max * blend)
    else:
        fwd = int(fwd_max)

    turn = max(-turn_max, min(turn_max, turn))

    if abs_angle < HEADING_DEADZONE:
        turn = 0.0

    left_cmd = int(fwd + turn)
    right_cmd = int(fwd - turn)
    return left_cmd, right_cmd

def main():
    global FORMATION_SPACING
    target = {"x": None, "y": None}
    paused = False
    speed_scale = 1.0
    formation_mode = "line"
    target_active = False

    leader_path = []
    follower_paths = {
        1: [],
        2: [],
        3: [],
    }

    def mouse_callback(event, x, y, flags, param):
        nonlocal target_active
        if event == cv2.EVENT_LBUTTONDOWN:
            target["x"] = x
            target["y"] = y
            target_active = True
            leader_path.clear()
            for rid in [1, 2, 3]:
                follower_paths[rid].clear()
            print(f"[HEDEF] Belirlendi: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            target["x"] = None
            target["y"] = None
            target_active = False
            leader_path.clear()
            for rid in [1, 2, 3]:
                follower_paths[rid].clear()
            print("[HEDEF] Temizlendi ve tüm yollar sıfırlandı.")

    camera = LatencyFreeCamera(CAMERA_SOURCE)
    if not camera.running:
        print("[HATA] Kamera başlatılamadı. Çıkılıyor.")
        return

    ser = None
    try:
        ser = serial.Serial(GATEWAY_PORT, BAUD_RATE, timeout=0.1)
        ser.dtr = False
        ser.rts = False
        print(f"[OK] Seri port açıldı: {GATEWAY_PORT}")
    except Exception as e:
        print(f"[UYARI] Seri port açılamadı ({e}). Komutlar gönderilmeyecek.")

    detector = Detector(
        families=APRILTAG_FAMILIES,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
    )

    WINDOW_NAME = "Lider Robot Navigasyon"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    pos_filters = {
        0: PositionFilter(EMA_ALPHA),
        1: PositionFilter(EMA_ALPHA),
        2: PositionFilter(EMA_ALPHA),
        3: PositionFilter(EMA_ALPHA),
    }
    
    last_known_positions = {
        0: {"x": None, "y": None, "time": 0.0},
        1: {"x": None, "y": None, "time": 0.0},
        2: {"x": None, "y": None, "time": 0.0},
        3: {"x": None, "y": None, "time": 0.0},
    }
    
    angle_filters = {
        0: AngleFilter(EMA_ALPHA),
        1: AngleFilter(EMA_ALPHA),
        2: AngleFilter(EMA_ALPHA),
        3: AngleFilter(EMA_ALPHA),
    }
    
    last_cmd_times = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    
    frame_count = 0
    fps_time = time.time()
    fps_display = 0

    print("\n" + "=" * 60)
    print("  LİDER & TAKİPÇİ ROBOT NAVİGASYON SİSTEMİ HAZIR")
    print("  Sol tık → Hedef belirle")
    print("  Sağ tık → Hedef temizle")
    print("  [1] Çizgi Formasyonu  [2] Üçgen Formasyonu")
    print("  [S] Dur/Devam  [+]/[-] Hız ayarı  [Q] Çıkış")
    print("=" * 60 + "\n")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            fh, fw = frame.shape[:2]

            frame_count += 1
            if now - fps_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                fps_time = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            robot_states = {
                0: {"x": None, "y": None, "angle": None, "found": False},
                1: {"x": None, "y": None, "angle": None, "found": False},
                2: {"x": None, "y": None, "angle": None, "found": False},
                3: {"x": None, "y": None, "angle": None, "found": False},
            }

            for tag in tags:
                tid = tag.tag_id
                if tid in robot_states:
                    raw_x = tag.center[0]
                    raw_y = tag.center[1]

                    corners = tag.corners
                    bx = (corners[0][0] + corners[1][0]) / 2.0
                    by = (corners[0][1] + corners[1][1]) / 2.0
                    tx = (corners[3][0] + corners[2][0]) / 2.0
                    ty = (corners[3][1] + corners[2][1]) / 2.0
                    raw_angle = (math.degrees(math.atan2(ty - by, tx - bx)) + 180) % 360

                    cal = ROBOT_CALIBRATION.get(tid, {})
                    angle_offset = cal.get("angle_offset", 0.0)
                    calibrated_angle = (raw_angle + angle_offset) % 360

                    lx, ly = pos_filters[tid].update(raw_x, raw_y)
                    calibrated_angle = angle_filters[tid].update(calibrated_angle)
                    
                    robot_states[tid]["x"] = lx
                    robot_states[tid]["y"] = ly
                    robot_states[tid]["angle"] = calibrated_angle
                    robot_states[tid]["found"] = True
                    
                    last_known_positions[tid]["x"] = lx
                    last_known_positions[tid]["y"] = ly
                    last_known_positions[tid]["time"] = now

            if not paused and robot_states[0]["found"] and target_active:
                lx = robot_states[0]["x"]
                ly = robot_states[0]["y"]
                langle = robot_states[0]["angle"]

                if len(leader_path) == 0 or math.sqrt((lx - leader_path[-1][0])**2 + (ly - leader_path[-1][1])**2) > PATH_RECORD_DIST:
                    leader_path.append((lx, ly, langle))
                    for rid in [1, 2, 3]:
                        follower_paths[rid].append((lx, ly, langle))

                    if len(leader_path) > 150:
                        leader_path.pop(0)
                    for rid in [1, 2, 3]:
                        if len(follower_paths[rid]) > 150:
                            follower_paths[rid].pop(0)

            for rid in [0, 1, 2, 3]:
                if robot_states[rid]["found"]:
                    rx = int(robot_states[rid]["x"])
                    ry = int(robot_states[rid]["y"])
                    rangle = robot_states[rid]["angle"]

                    if rid == 0:
                        color = (0, 255, 0)
                        name_str = "LIDER"
                    elif rid == 1:
                        color = (255, 255, 0)
                        name_str = "T1"
                    elif rid == 2:
                        color = (0, 255, 255)
                        name_str = "T2"
                    else:
                        color = (255, 0, 255)
                        name_str = "T3"

                    arrow_len = 25
                    end_x = int(rx + arrow_len * math.cos(math.radians(rangle)))
                    end_y = int(ry + arrow_len * math.sin(math.radians(rangle)))
                    cv2.arrowedLine(frame, (rx, ry), (end_x, end_y), color, 2, tipLength=0.3)
                    cv2.circle(frame, (rx, ry), 6, color, -1)
                    cv2.putText(frame, f"{name_str} ID:{rid} A:{int(rangle)}",
                                (rx + 10, ry - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, color, 1)

            if target["x"] is not None:
                hx, hy = target["x"], target["y"]
                cv2.drawMarker(frame, (hx, hy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(frame, (hx, hy), ARRIVAL_RADIUS, (0, 100, 255), 1)

            if len(leader_path) > 1:
                pts = np.array([[int(p[0]), int(p[1])] for p in leader_path], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 165, 255), 1)

            robot_pwms = {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0)}
            
            left_cmd, right_cmd = 0, 0
            status_text = "BEKLENIYOR"
            status_color = (150, 150, 150)

            if paused:
                status_text = "DURDURULDU [S]"
                status_color = (0, 100, 255)
            elif not robot_states[0]["found"]:
                status_text = "LIDER BULUNAMADI"
                status_color = (0, 0, 255)
            elif target["x"] is None:
                active_followers = any(len(follower_paths[rid]) > 0 for rid in [1, 2, 3])
                if active_followers:
                    status_text = f"HEDEFE ULASILDI | Takipciler Hizalaniyor... T1:{len(follower_paths[1])} T2:{len(follower_paths[2])} T3:{len(follower_paths[3])}"
                    status_color = (0, 200, 255)
                else:
                    status_text = "HEDEF YOK (tiklayin)"
                    status_color = (200, 200, 0)
            else:
                lx = robot_states[0]["x"]
                ly = robot_states[0]["y"]
                langle = robot_states[0]["angle"]

                dx = target["x"] - lx
                dy = target["y"] - ly
                dist = math.sqrt(dx ** 2 + dy ** 2)

                if dist > ARRIVAL_RADIUS:
                    target_angle = math.degrees(math.atan2(dy, dx)) % 360
                    angle_diff = (target_angle - langle + 180) % 360 - 180

                    left_cmd, right_cmd = compute_motor_commands(
                        angle_diff, dist, is_catchup=False, speed_scale=speed_scale
                    )

                    status_text = f"NAV | Mesafe:{int(dist)}px Aci:{int(angle_diff)} | Formasyon: {formation_mode.upper()}"
                    status_color = (0, 255, 100)

                    if dist < SLOW_ZONE:
                        status_color = (0, 200, 255)
                else:
                    target["x"] = None
                    target["y"] = None
                    print("[OK] Hedefe ulaşıldı! Takipçilerin hizalanması bekleniyor...")
                    status_text = "HEDEFE ULASILDI! HİZALANIYOR..."
                    status_color = (0, 255, 0)

            if paused:
                robot_pwms[0] = (0, 0)
            else:
                l_left_pwm = apply_deadband(left_cmd) if left_cmd != 0 else 0
                l_right_pwm = apply_deadband(right_cmd) if right_cmd != 0 else 0
                robot_pwms[0] = (l_left_pwm, l_right_pwm)

            for rid in [1, 2, 3]:
                f_left_cmd, f_right_cmd = 0, 0
                collision_scale = 1.0

                if not paused and robot_states[rid]["found"] and target_active:
                    fx = robot_states[rid]["x"]
                    fy = robot_states[rid]["y"]
                    fangle = robot_states[rid]["angle"]

                    path = follower_paths[rid]
                    target_dist = FORMATIONS[formation_mode][rid]["target_dist"]
                    effective_target_dist = target_dist if target["x"] is not None else 0.0

                    if len(path) > 0:
                        # Find closest waypoint in local window, but do not pop past the target distance
                        search_limit = min(20, len(path))
                        closest_idx = 0
                        min_dist = float('inf')
                        for idx in range(search_limit):
                            if idx > 0 and get_path_length(path[idx:]) < effective_target_dist:
                                break
                            px, py, _ = path[idx]
                            d = math.sqrt((px - fx)**2 + (py - fy)**2)
                            if d < min_dist:
                                min_dist = d
                                closest_idx = idx
                        
                        if closest_idx > 0:
                            for _ in range(closest_idx):
                                path.pop(0)

                        rec_x, rec_y, rec_h = path[0]

                        side_offset = FORMATIONS[formation_mode][rid]["side_offset"]
                        fb_offset = FORMATIONS[formation_mode][rid]["fb_offset"]

                        rad = math.radians(rec_h)
                        offset_rad = rad + (math.pi / 2.0)

                        side_offset_x = side_offset * math.cos(offset_rad)
                        side_offset_y = side_offset * math.sin(offset_rad)

                        fb_offset_x = fb_offset * math.cos(rad)
                        fb_offset_y = fb_offset * math.sin(rad)

                        tx = rec_x + side_offset_x + fb_offset_x
                        ty = rec_y + side_offset_y + fb_offset_y

                        t_color = (255, 255, 0) if rid == 1 else (0, 255, 255) if rid == 2 else (255, 0, 255)
                        cv2.drawMarker(frame, (int(tx), int(ty)), t_color, cv2.MARKER_TILTED_CROSS, 10, 1)
                        cv2.line(frame, (int(fx), int(fy)), (int(tx), int(ty)), t_color, 1, cv2.LINE_AA)

                        fdx = tx - fx
                        fdy = ty - fy
                        fdist = math.sqrt(fdx**2 + fdy**2)

                        ftarget_angle = math.degrees(math.atan2(fdy, fdx)) % 360
                        fangle_diff = (ftarget_angle - fangle + 180) % 360 - 180

                        current_path_len = get_path_length(path)
                        is_catchup = (current_path_len - effective_target_dist) > 20.0

                        f_left_cmd, f_right_cmd = compute_motor_commands(
                            fangle_diff, fdist, is_catchup=is_catchup, speed_scale=speed_scale
                        )

                        # Collision avoidance (ACC) scaling based on priority queue (0 -> 1 -> 3 -> 2)
                        priority_list = [0, 1, 3, 2]
                        my_priority = priority_list.index(rid)
                        
                        collision_scale = 1.0
                        for other_rid in [0, 1, 2, 3]:
                            if other_rid != rid:
                                lkp = last_known_positions[other_rid]
                                # Use last known position if it was detected recently (within last 2.0 seconds)
                                if lkp["x"] is not None and (now - lkp["time"]) < 2.0:
                                    other_priority = priority_list.index(other_rid)
                                    # A robot only brakes for robots that are ahead of it in priority (index < my_priority)
                                    if other_priority < my_priority:
                                        ox = lkp["x"]
                                        oy = lkp["y"]
                                        d = math.sqrt((ox - fx)**2 + (oy - fy)**2)
                                        
                                        if d < COLLISION_SAFE_DIST:
                                            scale = (d - COLLISION_STOP_DIST) / (COLLISION_SAFE_DIST - COLLISION_STOP_DIST)
                                            scale = max(0.0, min(1.0, scale))
                                            # Quadratic steep braking for aggressive safety
                                            scale = scale ** 2
                                            if scale < collision_scale:
                                                collision_scale = scale

                if paused:
                    robot_pwms[rid] = (0, 0)
                else:
                    f_left_pwm = apply_deadband(f_left_cmd) if f_left_cmd != 0 else 0
                    f_right_pwm = apply_deadband(f_right_cmd) if f_right_cmd != 0 else 0
                    
                    # Apply collision scale directly to the physical PWM values
                    f_left_pwm = int(f_left_pwm * collision_scale)
                    f_right_pwm = int(f_right_pwm * collision_scale)
                    
                    # Cut off small PWM values to prevent motor hum and ensure a firm stop
                    if abs(f_left_pwm) < 18:
                        f_left_pwm = 0
                    if abs(f_right_pwm) < 18:
                        f_right_pwm = 0
                        
                    robot_pwms[rid] = (f_left_pwm, f_right_pwm)

            if ser and ser.is_open:
                for rid in [0, 1, 2, 3]:
                    if now - last_cmd_times[rid] >= 0.06:
                        left_pwm, right_pwm = robot_pwms[rid]

                        cal = ROBOT_CALIBRATION.get(rid, {})
                        if cal.get("left_invert", False):
                            left_pwm = -left_pwm
                        if cal.get("right_invert", False):
                            right_pwm = -right_pwm

                        cmd_str = f"<{rid},{left_pwm},{right_pwm}>\n"
                        try:
                            ser.write(cmd_str.encode("utf-8"))
                            ser.flush()
                        except Exception:
                            pass
                        last_cmd_times[rid] = now

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 76), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, status_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            
            cv2.putText(frame, f"L: L:{robot_pwms[0][0]:+d} R:{robot_pwms[0][1]:+d} | T1: L:{robot_pwms[1][0]:+d} R:{robot_pwms[1][1]:+d}",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(frame, f"T2: L:{robot_pwms[2][0]:+d} R:{robot_pwms[2][1]:+d} | T3: L:{robot_pwms[3][0]:+d} R:{robot_pwms[3][1]:+d}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            
            cv2.putText(frame, f"Hiz: x{speed_scale:.1f}  Aralik: {int(FORMATION_SPACING)}px  FPS:{fps_display}",
                        (fw - 260, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1)

            shortcuts = "[Sol Tik] Hedef  [Sag Tik] Temizle  [1] Cizgi  [2] Ucgen  [S] Dur  [+/-] Hiz  [o/p] Aralik  [Q] Cikis"
            cv2.putText(frame, shortcuts, (10, fh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s") or key == ord("S"):
                paused = not paused
                print(f"[{'DURDURULDU' if paused else 'DEVAM'}]")
            elif key == ord("1"):
                formation_mode = "line"
                print("[FORMASYON] Çizgi (Line) moduna geçildi.")
            elif key == ord("2"):
                formation_mode = "triangle"
                print("[FORMASYON] Üçgen (Triangle) moduna geçildi.")
            elif key == ord("+") or key == ord("="):
                speed_scale = min(2.0, speed_scale + 0.1)
                print(f"[HIZ] x{speed_scale:.1f}")
            elif key == ord("-") or key == ord("_"):
                speed_scale = max(0.3, speed_scale - 0.1)
                print(f"[HIZ] x{speed_scale:.1f}")
            elif key == ord("o") or key == ord("O"):
                FORMATION_SPACING = max(40.0, FORMATION_SPACING - 5.0)
                FORMATIONS["triangle"][1]["side_offset"] = FORMATION_SPACING
                FORMATIONS["triangle"][3]["side_offset"] = -FORMATION_SPACING
                print(f"[FORMASYON ARALIGI] {FORMATION_SPACING}px")
            elif key == ord("p") or key == ord("P"):
                FORMATION_SPACING = min(120.0, FORMATION_SPACING + 5.0)
                FORMATIONS["triangle"][1]["side_offset"] = FORMATION_SPACING
                FORMATIONS["triangle"][3]["side_offset"] = -FORMATION_SPACING
                print(f"[FORMASYON ARALIGI] {FORMATION_SPACING}px")

    finally:
        print("\nKapatılıyor...")
        if ser and ser.is_open:
            for _ in range(3):
                for rid in [0, 1, 2, 3]:
                    ser.write(f"<{rid},0,0>\n".encode("utf-8"))
                    ser.flush()
                time.sleep(0.05)
            ser.close()
            print("[OK] Seri port kapatıldı, tüm robotlar durduruldu.")

        camera.stop()
        cv2.destroyAllWindows()
        print("[OK] Sistem kapatıldı.")

if __name__ == "__main__":
    main()
