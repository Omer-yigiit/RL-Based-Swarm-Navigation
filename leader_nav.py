import os
# FFmpeg/MJPEG ayarlari — cv2 import'undan ONCE olmali:
#  - LOGLEVEL=-8 (quiet): wifi akisindaki "overread" spam'ini sustur
#  - nobuffer + low_delay: ag kamerasinda gecikmeyi azalt
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|reorder_queue_size;0"

import cv2
import numpy as np
import math
import serial
import time
import threading
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pupil_apriltags import Detector

cv2.setLogLevel(0)  # OpenCV ic loglarini da sustur
CAMERA_SOURCE = "http://10.185.40.91:8080/video"
GATEWAY_PORT = "COM7"
BAUD_RATE = 115200
LEADER_ID = 0
APRILTAG_FAMILIES = "tag36h11"

# --- Kamera kalibrasyonu (lens bozulmasi) ---
# calibrate_camera.py ile uretilir; dosya yoksa undistort atlanir (graceful fallback).
CALIB_FILE = "camera_calib.npz"
# --- Piksel<->cm olcek ---
# AprilTag'in fiziksel kenar uzunlugu (cm). Tag boyutunuza gore guncelleyin.
# Bu sayede PX_PER_CM her kare canli olculur (satranc tahtasi gerekmez).
TAG_SIZE_CM = 16.0

MIN_PWM = 28
MAX_PWM = 50

ROBOT_CALIBRATION = {
    0: {"angle_offset": 0.0,   "left_invert": False, "right_invert": False},
    1: {"angle_offset": 0.0,   "left_invert": False, "right_invert": False},
    2: {"angle_offset": 0.0,   "left_invert": False, "right_invert": False},
    3: {"angle_offset": 180.0, "left_invert": False, "right_invert": False},
}

FWD_MAX = 4
FWD_CATCHUP = 9        # takipçi yakalama hızı — geride kalinca guclu yetis (lag'i kapat)
TURN_MAX = 1.5
TURN_CATCHUP = 1.8
FOLLOWER_SPEED_SCALE = 1.0   # takipci cruise hizi lider ile AYNI (yoksa asla yetisemez, formasyon bozulur)
PARK_FORCE_DELAY = 2.5       # lider durduktan bu kadar sonra ulasamayan takipci de parka ZORLANIR

MIN_DECISION_MARGIN = 35.0  # bu değerin altındaki AprilTag tespiti reddedilir
MAX_JUMP_PX = 120.0         # bir okumada bu kadar px'den fazla zıplama: sahte tespit
LOST_TIMEOUT = 0.30         # tag bu sure (sn) kaybolursa hiz vektoruyle konum tahmin et, sonra dur
MAX_PREDICT_PX = 90.0       # tahminle en fazla bu kadar px ileri git (kacisi onler)

ARRIVAL_RADIUS = 35          # Lider bu mesafede hedefe "varmis" sayilir — donmeyi önler
FOLLOWER_WP_RADIUS = 18
PARK_ARRIVAL_RADIUS = 45     # Takipci park noktasina bu mesafede gelince DURUR+KILITLENIR (loop/donme onler)
PARK_TRIGGER_RADIUS = 50     # Lider durduktan sonra takipci kendi path sonuna bu kadar yaklasinca park'a gecer
SLOW_ZONE = 120              # Daha erken yavasla
HEADING_DEADZONE = 5.0       # Kucuk ac farklari icin motor verme
LEADER_TURN_KD = 0.06        # Lider donus sonumlemesi (D terimi) — slalomu keser (daha guclu)
LEADER_TURN_KP = 0.07        # Lider orantisal donus kazanci (dusuk -> daha genis bant, daha DUZ gider)
FOLLOWER_TURN_KD = 0.07      # Takipci donus sonumlemesi — sag-sol salinimi keser (daha guclu)
FOLLOWER_TURN_KP = 0.07      # Takipci orantisal donus kazanci (liderden dusuk -> daha genis orantisal bant, az bang-bang)

# Carpisma onleme: bir takipci, kendisinden YUKSEK oncelikli ve hareket yonunde (on koni)
# bu mesafeden yakin bir robot varsa DURUR. Dusuk oncelikli yol verir -> deadlock olmaz.
COLLISION_STOP_DIST = 115.0
COLLISION_PRIORITY = [0, 1, 3, 2]   # yuksek -> dusuk (Lider, R1, R3, R2)
COLLISION_CONE_DOT = 0.0   # 0 = on yari daire (~90 derece her yan); daha genis tespit -> az temas

EMA_ALPHA = 0.35

FORMATION_SPACING = 220.0
PATH_RECORD_DIST = 4.0
# Path en az 3*FORMATION_SPACING (en arkadaki R2 hedefi) + pay kadar olmali.
# 250 nokta * 4px = 1000px > 3*220=660 -> R2 hedefine ulasabilir.
MAX_PATH_LEN = 250
FORMATION_MORPH_RATE = 200.0   # formasyon degisiminde slot degerlerinin kayma hizi (px/sn)

FORMATIONS = {
    "triangle": {
        1: {"side_offset": FORMATION_SPACING,  "fb_offset": 0.0, "target_dist": FORMATION_SPACING},
        2: {"side_offset": 0.0,                "fb_offset": 0.0, "target_dist": FORMATION_SPACING},
        3: {"side_offset": -FORMATION_SPACING, "fb_offset": 0.0, "target_dist": FORMATION_SPACING},
    },
    "line": {
        1: {"side_offset": 0.0, "fb_offset": 0.0, "target_dist": 1.0 * FORMATION_SPACING},
        3: {"side_offset": 0.0, "fb_offset": 0.0, "target_dist": 2.0 * FORMATION_SPACING},
        2: {"side_offset": 0.0, "fb_offset": 0.0, "target_dist": 3.0 * FORMATION_SPACING},
    }
}

# =====================================================================
# BULANIK MANTIK (FUZZY) ACC  — Webots Follower_Python.py'den taşındı
# =====================================================================
# Sim metre cinsindendi; burada girdi ORANSAL: ratio = onundeki_robot_mesafesi / FORMATION_SPACING.
# Bu sayede o/p tuslari ile FORMATION_SPACING degisse bile fuzzy evreni gecerli kalir.
# Cikti: 0..1 hiz katsayisi (acc_scale). LINE = agresif, TRIANGLE = yumusak (sim ile ayni felsefe).
#
# Girdi uyelik fonksiyonlari "ratio" uzayinda (1.0 = robot tam hedef aralikta):
#   danger  -> cok yakin, dur     |  caution -> yaklasiyor, yavasla  |  safe -> aralik tamam, tam hiz
FUZZY_PROFILES = {
    "line": {  # agresif: aralik geri gelince hizla tam hiza don, geç fren
        "danger":  [0.0, 0.0, 0.40, 0.55],   # trapmf
        "caution": [0.45, 0.70, 1.00],       # trimf
        "safe":    [0.90, 1.20, 2.50, 2.50], # trapmf
        "stop":    [0.0, 0.0, 0.20],         # trimf  (cikti)
        "slow":    [0.10, 0.50, 0.80],       # trimf  (cikti)
        "fast":    [0.70, 0.90, 1.00, 1.00], # trapmf (cikti)
        "emergency_ratio": 0.35,             # bu oranin altinda sert dur (acc=0)
    },
    "triangle": {  # yumusak: daha genis marjlar, daha kibar fren
        "danger":  [0.0, 0.0, 0.50, 0.70],
        "caution": [0.60, 0.90, 1.20],
        "safe":    [1.10, 1.50, 2.50, 2.50],
        "stop":    [0.0, 0.10, 0.25],
        "slow":    [0.20, 0.55, 0.85],
        "fast":    [0.75, 0.90, 1.00, 1.00],
        "emergency_ratio": 0.45,
    },
}

# Fuzzy hesabi yavastir; dongude her kare 3 robot icin compute() cagirmak yerine
# egriyi baslangicta bir kez ornekleyip LUT (lookup table) olarak saklariz -> O(1) arama.
RATIO_MAX = 2.5

def _build_fuzzy_acc_lut(profile, step=0.02):
    """Bir profil icin fuzzy ACC egrisini orneklenmis (ratios, factors) LUT'a cevirir."""
    r = ctrl.Antecedent(np.arange(0.0, RATIO_MAX + 0.01, 0.01), "ratio")
    f = ctrl.Consequent(np.arange(0.0, 1.01, 0.01), "factor")

    r["danger"]  = fuzz.trapmf(r.universe, profile["danger"])
    r["caution"] = fuzz.trimf(r.universe,  profile["caution"])
    r["safe"]    = fuzz.trapmf(r.universe, profile["safe"])

    f["stop"] = fuzz.trimf(f.universe,  profile["stop"])
    f["slow"] = fuzz.trimf(f.universe,  profile["slow"])
    f["fast"] = fuzz.trapmf(f.universe, profile["fast"])

    rules = [
        ctrl.Rule(r["danger"],  f["stop"]),
        ctrl.Rule(r["caution"], f["slow"]),
        ctrl.Rule(r["safe"],    f["fast"]),
    ]
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    ratios = np.arange(0.0, RATIO_MAX + step, step)
    factors = np.empty_like(ratios)
    for i, rv in enumerate(ratios):
        sim.input["ratio"] = float(rv)
        try:
            sim.compute()
            factors[i] = sim.output["factor"]
        except Exception:
            factors[i] = 1.0  # kural tetiklenmezse guvenli taraf: tam hiz
    return ratios, factors

_FUZZY_LUT = {name: _build_fuzzy_acc_lut(prof) for name, prof in FUZZY_PROFILES.items()}

def fuzzy_speed_factor(d_ahead, formation_mode):
    """Onundeki robota olan piksel mesafesine gore bulanik hiz katsayisi (0..1) dondurur."""
    if FORMATION_SPACING <= 0:
        return 1.0
    prof = FUZZY_PROFILES.get(formation_mode, FUZZY_PROFILES["line"])
    ratio = d_ahead / FORMATION_SPACING
    if ratio < prof["emergency_ratio"]:
        return 0.0  # acil fren: sert dur
    lut_r, lut_f = _FUZZY_LUT.get(formation_mode, _FUZZY_LUT["line"])
    return float(np.interp(ratio, lut_r, lut_f))

# --- Serit-tabanli akilli ACC (Webots Follower'daki longitudinal/lateral filtre) ---
# Formasyona gore serit genisligi (FORMATION_SPACING orani). Ucgen genis (yan robotlar),
# cizgi dar (tek sira).
LANE_WIDTH_RATIO = {"line": 0.5, "triangle": 1.2}

def step_value(cur, tgt, max_step):
    """cur degerini tgt'ye en fazla max_step kadar yaklastirir (yumusak gecis)."""
    if abs(tgt - cur) <= max_step:
        return tgt
    return cur + max_step if tgt > cur else cur - max_step

def transform_to_leader_frame(dx, dy, heading_rad):
    """(dx,dy) goreli vektoru lider yon cercevesine cevirir -> (longitudinal, lateral)."""
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return dx * c + dy * s, -dx * s + dy * c

def nearest_ahead_dist(fx, fy, my_rid, last_known, now, heading_deg, lane_width, fresh=2.0):
    """Lider yon cercevesinde 'onumde (longitudinal>0) ve seridimde (|lateral|<lane)' olan
    en yakin robotun gercek piksel mesafesi; yoksa inf."""
    heading_rad = math.radians(heading_deg)
    best = float("inf")
    for rid, lkp in last_known.items():
        if rid == my_rid or lkp["x"] is None or (now - lkp["time"]) > fresh:
            continue
        dx, dy = lkp["x"] - fx, lkp["y"] - fy
        lon, lat = transform_to_leader_frame(dx, dy, heading_rad)
        if lon > 0 and abs(lat) < lane_width:
            d = math.hypot(dx, dy)
            if d < best:
                best = d
    return best

def load_undistort_maps(calib_file, w, h):
    """camera_calib.npz varsa undistort remap haritalarini dondurur, yoksa (None, None)."""
    import os
    if not os.path.exists(calib_file):
        print(f"[UYARI] Kalibrasyon dosyasi yok ({calib_file}). Undistort atlanacak.")
        return None, None
    try:
        data = np.load(calib_file)
        K, dist = data["camera_matrix"], data["dist_coeffs"]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
        print(f"[OK] Kamera kalibrasyonu yuklendi: {calib_file}")
        return map1, map2
    except Exception as e:
        print(f"[UYARI] Kalibrasyon yuklenemedi ({e}). Undistort atlanacak.")
        return None, None

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
            else:
                time.sleep(0.01)

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

_pd_state = {}   # pd_id -> (onceki angle_diff, zaman)

def compute_motor_commands(angle_diff, dist, is_catchup=False, speed_scale=1.0,
                           pd_id=None, kd=0.0, kp=0.12):
    fwd_limit = FWD_CATCHUP if is_catchup else FWD_MAX
    turn_limit = TURN_CATCHUP if is_catchup else TURN_MAX

    fwd_max = fwd_limit * speed_scale
    turn_max = turn_limit * speed_scale

    abs_angle = abs(angle_diff)
    turn = angle_diff * kp

    # D (turev) sonumlemesi: hata hizla degisiyorsa donusu kis -> overshoot/slalom azalir
    if pd_id is not None and kd > 0.0:
        t_now = time.time()
        prev = _pd_state.get(pd_id)
        if prev is not None:
            dt_pd = t_now - prev[1]
            if dt_pd > 1e-3:
                rate = (angle_diff - prev[0]) / dt_pd   # deg/sn
                turn += kd * rate
        _pd_state[pd_id] = (angle_diff, t_now)

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

# Park (finis) ucgen pozisyonu: lider onde, arkada yan yana R1, R3, R2 (R2 merkezde degil).
PARK_TRI_OFFSET = {1: 1.0, 3: 0.0, 2: -1.0}  # FORMATION_SPACING ile carpilir

def park_triangle_pos(rid, lx, ly, langle, spacing):
    behind_rad = math.radians((langle + 180) % 360)
    offset_rad = math.radians(langle) + (math.pi / 2.0)
    off = PARK_TRI_OFFSET[rid] * spacing
    base_x = lx + spacing * math.cos(behind_rad)
    base_y = ly + spacing * math.sin(behind_rad)
    return (base_x + off * math.cos(offset_rad), base_y + off * math.sin(offset_rad))

def collision_yield(rid, fx, fy, mvx, mvy, robot_states, last_known, now, stop_dist):
    """rid robotu, kendisinden YUKSEK oncelikli ve hareket yonunde (~75 derece on koni)
    stop_dist'ten yakin bir robot varsa True (=dur) dondurur. Carpismayi onler, deadlock yapmaz."""
    try:
        my_idx = COLLISION_PRIORITY.index(rid)
    except ValueError:
        return False
    mv = math.hypot(mvx, mvy)
    for o in COLLISION_PRIORITY[:my_idx]:   # sadece yuksek oncelikliler
        s = robot_states.get(o)
        if s and s["found"]:
            ox, oy = s["x"], s["y"]
        else:
            lk = last_known.get(o)
            if not lk or lk["x"] is None or (now - lk["time"]) > 1.0:
                continue
            ox, oy = lk["x"], lk["y"]
        dx, dy = ox - fx, oy - fy
        d = math.hypot(dx, dy)
        if d < stop_dist:
            if mv > 1e-3:
                if (dx * mvx + dy * mvy) / (d * mv) > COLLISION_CONE_DOT:   # on koni icinde mi
                    return True
            else:
                return True
    return False

def main():
    global FORMATION_SPACING
    target = {"x": None, "y": None}
    paused = False
    speed_scale = 0.67
    formation_mode = "triangle"          # UCGEN ile basla
    auto_state = {"line_done": False}    # mesafe esiginde bir kez otomatik LINE'a gec
    AUTO_LINE_DIST = FORMATION_SPACING * 1.0   # lider bu kadar gidince ucgen->line (dar alan icin kisa)
    target_active = False

    leader_path = []
    follower_paths = {
        1: [],
        2: [],
        3: [],
    }
    # Sira: Lider -> R1 -> R3 -> R2.  Sonraki robot, ONCEKI robot liderin yoluna OTURUNCA kalkar.
    follower_activated = {1: False, 2: False, 3: False}
    follower_on_path   = {1: False, 2: False, 3: False}   # takipci kendi formasyon yerine oturdu mu
    activation_times   = {1: None,  2: None,  3: None}     # her robotun aktif olduğu zaman
    SETTLE_RADIUS      = 70.0   # takipci hedefine bu kadar yaklasinca "yola oturdu" -> sonraki erken kalkar
    ACTIVATION_MAX_WAIT = 1.2   # KISA: onceki oturmasa bile bu kadar sonra basla (cok gec kalmasinlar)
    leader_dist = {"since_target": 0.0}
    follower_final_targets = {1: (None, None), 2: (None, None), 3: (None, None)}
    parked = {1: False, 2: False, 3: False}   # park noktasina ulasinca kilit (donme/loop onler)
    # Lider finise varinca: hepsi birden parka GECMEZ; her takipci kendi path'ini bitirince tek tek gecer.
    arrived = {"flag": False, "x": None, "y": None, "angle": None, "time": 0.0}

    def mouse_callback(event, x, y, flags, param):
        nonlocal target_active, formation_mode
        if event == cv2.EVENT_LBUTTONDOWN:
            formation_mode = "triangle"      # her yeni hedefte ucgen ile basla
            auto_state["line_done"] = False
            target["x"] = x
            target["y"] = y
            target_active = True
            leader_path.clear()
            for rid in [1, 2, 3]:
                follower_paths[rid].clear()
            follower_activated[1] = True
            follower_activated[2] = False
            follower_activated[3] = False
            follower_on_path[1] = follower_on_path[2] = follower_on_path[3] = False
            activation_times[1] = time.time()
            activation_times[2] = None
            activation_times[3] = None
            leader_dist["since_target"] = 0.0
            follower_final_targets[1] = (None, None)
            follower_final_targets[2] = (None, None)
            follower_final_targets[3] = (None, None)
            parked[1] = parked[2] = parked[3] = False
            arrived["flag"] = False
            print(f"[HEDEF] Belirlendi: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            target["x"] = None
            target["y"] = None
            target_active = False
            leader_path.clear()
            for rid in [1, 2, 3]:
                follower_paths[rid].clear()
            follower_activated[1] = False
            follower_activated[2] = False
            follower_activated[3] = False
            follower_on_path[1] = follower_on_path[2] = follower_on_path[3] = False
            activation_times[1] = None
            activation_times[2] = None
            activation_times[3] = None
            leader_dist["since_target"] = 0.0
            follower_final_targets[1] = (None, None)
            follower_final_targets[2] = (None, None)
            follower_final_targets[3] = (None, None)
            parked[1] = parked[2] = parked[3] = False
            arrived["flag"] = False
            print("[HEDEF] Temizlendi ve tum yollar sifirlandi.")

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

    # CPU bol (laptop %2-3'te) -> quad_decimate=1.0 (TAM cozunurluk) en isabetli konum/aci -> jitter azalir,
    # takip daha akici. nthreads=8 (i7-10850H 12 thread). quad_sigma kucuk blur ile kose tespiti kararli kalir.
    detector = Detector(
        families=APRILTAG_FAMILIES,
        nthreads=8,
        quad_decimate=1.0,
        quad_sigma=0.5,
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
        rid: {"x": None, "y": None, "angle": None, "time": 0.0, "vx": 0.0, "vy": 0.0}
        for rid in [0, 1, 2, 3]
    }
    
    angle_filters = {
        0: AngleFilter(EMA_ALPHA),
        1: AngleFilter(EMA_ALPHA),
        2: AngleFilter(EMA_ALPHA),
        3: AngleFilter(EMA_ALPHA),
    }
    
    last_cmd_times = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    undistort_maps = (None, None)   # ilk karede frame boyutu bilinince kurulur
    undistort_ready = False
    px_per_cm = None                # tag kenarlarindan canli olculur (EMA)

    # Yumusak formasyon gecisi: her takipcinin slot degerleri hedefe kademeli kayar
    current_slots = {rid: dict(FORMATIONS[formation_mode][rid]) for rid in [1, 2, 3]}
    last_frame_time = time.time()

    frame_count = 0
    fps_time = time.time()
    fps_display = 0
    last_log_time = 0.0
    _log_file = open("nav_log.txt", "w", encoding="utf-8")
    _log_file.write("time,id0x,id0y,id0a,id1x,id1y,id1a,id2x,id2y,id2a,id3x,id3y,id3a,"
                    "d02,d01,d03,d12,d13,d23,act2,act1,act3,ldr_dist,form,park1,park3,park2,status\n")
    _log_file.flush()

    print("\n" + "=" * 60)
    print("  LIDER & TAKIPCI ROBOT NAVIGASYON SISTEMI HAZIR")
    print("  Sol tik : Hedef belirle")
    print("  Sag tik : Hedef temizle")
    print("  [1] Cizgi Formasyonu  [2] Ucgen Formasyonu")
    print("  [S] Dur/Devam  [+]/[-] Hiz ayari  [Q] Cikis")
    print("=" * 60 + "\n")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = min(0.1, now - last_frame_time)   # buyuk sicramalari kis
            last_frame_time = now
            fh, fw = frame.shape[:2]

            # Lens bozulmasi duzeltme (kalibrasyon dosyasi varsa)
            if not undistort_ready:
                undistort_maps = load_undistort_maps(CALIB_FILE, fw, fh)
                undistort_ready = True
            if undistort_maps[0] is not None:
                frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)

            frame_count += 1
            if now - fps_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                fps_time = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            robot_states = {
                rid: {"x": None, "y": None, "angle": None, "found": False, "predicted": False}
                for rid in [0, 1, 2, 3]
            }

            for tag in tags:
                tid = tag.tag_id
                if tid not in robot_states:
                    continue
                # Düşük güvenli tespiti reddet
                if tag.decision_margin < MIN_DECISION_MARGIN:
                    continue
                raw_x = tag.center[0]
                raw_y = tag.center[1]
                # Ani zıplama kontrolü — sahte tespit veya okluzyonu filtrele
                lkp_check = last_known_positions[tid]
                if lkp_check["x"] is not None and (now - lkp_check["time"]) < 0.5:
                    jump = math.sqrt((raw_x - lkp_check["x"])**2 + (raw_y - lkp_check["y"])**2)
                    if jump > MAX_JUMP_PX:
                        continue  # bu kareyi reddet, önceki konum korunur
                corners = tag.corners
                # Canli px<->cm olcek: tag'in 4 kenar uzunlugu ortalamasi / fiziksel boyut
                edge_px = 0.0
                for i in range(4):
                    p, q = corners[i], corners[(i + 1) % 4]
                    edge_px += math.hypot(q[0] - p[0], q[1] - p[1])
                edge_px /= 4.0
                if TAG_SIZE_CM > 0:
                    sample = edge_px / TAG_SIZE_CM
                    px_per_cm = sample if px_per_cm is None else 0.1 * sample + 0.9 * px_per_cm
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

                # Hiz vektoru (px/sn) — okluzyon sirasinda tahmin icin
                lkp = last_known_positions[tid]
                dt_lk = now - lkp["time"]
                if lkp["x"] is not None and 0.0 < dt_lk < LOST_TIMEOUT:
                    raw_vx = (lx - lkp["x"]) / dt_lk
                    raw_vy = (ly - lkp["y"]) / dt_lk
                    lkp["vx"] = 0.4 * raw_vx + 0.6 * lkp["vx"]
                    lkp["vy"] = 0.4 * raw_vy + 0.6 * lkp["vy"]
                else:
                    lkp["vx"] = 0.0
                    lkp["vy"] = 0.0
                lkp["x"] = lx
                lkp["y"] = ly
                lkp["angle"] = calibrated_angle
                lkp["time"] = now

            # --- Okluzyon yonetimi: bu kare bulunamayan robotlari kisa sure tahminle surdur ---
            for rid in [0, 1, 2, 3]:
                if robot_states[rid]["found"]:
                    continue
                lkp = last_known_positions[rid]
                if lkp["x"] is None:
                    continue
                gap = now - lkp["time"]
                if gap >= LOST_TIMEOUT:
                    continue  # cok uzun kayip -> robot durur (found=False kalir)
                pdx = lkp["vx"] * gap
                pdy = lkp["vy"] * gap
                pmag = math.hypot(pdx, pdy)
                if pmag > MAX_PREDICT_PX:   # kacisi sinirla
                    pdx *= MAX_PREDICT_PX / pmag
                    pdy *= MAX_PREDICT_PX / pmag
                robot_states[rid]["x"] = lkp["x"] + pdx
                robot_states[rid]["y"] = lkp["y"] + pdy
                robot_states[rid]["angle"] = lkp["angle"]
                robot_states[rid]["found"] = True
                robot_states[rid]["predicted"] = True

            if not paused and robot_states[0]["found"] and not robot_states[0].get("predicted") and target_active:
                lx = robot_states[0]["x"]
                ly = robot_states[0]["y"]
                langle = robot_states[0]["angle"]

                if len(leader_path) == 0 or math.sqrt((lx - leader_path[-1][0])**2 + (ly - leader_path[-1][1])**2) > PATH_RECORD_DIST:
                    if leader_path:
                        seg = math.sqrt((lx - leader_path[-1][0])**2 + (ly - leader_path[-1][1])**2)
                        leader_dist["since_target"] += seg
                    leader_path.append((lx, ly, langle))
                    for rid in [1, 2, 3]:
                        follower_paths[rid].append((lx, ly, langle))

                    if len(leader_path) > MAX_PATH_LEN:
                        leader_path.pop(0)
                    for rid in [1, 2, 3]:
                        if len(follower_paths[rid]) > MAX_PATH_LEN:
                            follower_paths[rid].pop(0)

            # --- Otomatik formasyon: UCGEN ile basla, lider AUTO_LINE_DIST gidince LINE'a gec ---
            if target_active and not auto_state["line_done"] and \
               leader_dist["since_target"] > AUTO_LINE_DIST:
                formation_mode = "line"
                auto_state["line_done"] = True
                print(f"[OTO] Ucgen -> Line (lider {int(leader_dist['since_target'])}px gitti)")

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
                    if robot_states[rid].get("predicted"):
                        cv2.circle(frame, (rx, ry), 12, (160, 160, 160), 1)  # tahmin (okluzyon)
                    label = f"{name_str} ID:{rid} A:{int(rangle)}" + (" ~" if robot_states[rid].get("predicted") else "")
                    cv2.putText(frame, label,
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
                    status_text = f"FINISTE | PARK ediliyor... T1:{len(follower_paths[1])} T2:{len(follower_paths[2])} T3:{len(follower_paths[3])}"
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
                        angle_diff, dist, is_catchup=False, speed_scale=speed_scale,
                        pd_id="leader", kd=LEADER_TURN_KD, kp=LEADER_TURN_KP
                    )

                    status_text = f"NAV | Mesafe:{int(dist)}px Aci:{int(angle_diff)} | Formasyon: {formation_mode.upper()}"
                    status_color = (0, 255, 100)

                    if dist < SLOW_ZONE:
                        status_color = (0, 200, 255)
                else:
                    # Lider finise vardi -> DURUR, path donar (son nokta kalir).
                    # Takipciler HEPSI BIRDEN parka GECMEZ; her biri kendi path'ini bitirince
                    # (frozen path sonuna ulasinca) tek tek UCGEN parka gecer. Park hedefi
                    # arrived pozisyonundan park_triangle_pos ile o an hesaplanir.
                    if not arrived["flag"]:
                        arrived["flag"] = True
                        arrived["x"], arrived["y"], arrived["angle"] = lx, ly, langle
                        arrived["time"] = now
                        print("[OK] Finise ulasildi! Takipciler path'i tamamlayip sirayla park edecek...")
                    target["x"] = None
                    target["y"] = None
                    status_text = "FINISTE | Takipciler path'i tamamliyor..."
                    status_color = (0, 255, 0)

            if paused:
                robot_pwms[0] = (0, 0)
            else:
                l_left_pwm = apply_deadband(left_cmd) if left_cmd != 0 else 0
                l_right_pwm = apply_deadband(right_cmd) if right_cmd != 0 else 0
                robot_pwms[0] = (l_left_pwm, l_right_pwm)

            # Sira: Lider(0) -> Robot1 -> Robot2 -> Robot3
            PREV_FOLLOWER = {3: 1, 2: 3}          # R3 R1'den, R2 R3'ten 1sn sonra

            # Serit-tabanli ACC icin referans yon: lider acisi (yoksa per-robot kendi acisi)
            if robot_states[0]["angle"] is not None:
                leader_heading_deg = robot_states[0]["angle"]
            else:
                leader_heading_deg = last_known_positions[0]["angle"]

            # Yumusak formasyon gecisi: slot degerlerini hedef formasyona kademeli yaklastir
            morph_step = FORMATION_MORPH_RATE * dt
            for rid in [1, 2, 3]:
                tgt = FORMATIONS[formation_mode][rid]
                cur = current_slots[rid]
                for k in ("side_offset", "fb_offset", "target_dist"):
                    cur[k] = step_value(cur[k], tgt[k], morph_step)

            for rid in [1, 3, 2]:   # işlem sırası: Lider->R1->R3->R2
                f_left_cmd, f_right_cmd = 0, 0
                acc_scale = 1.0

                if not paused and robot_states[rid]["found"] and target_active:
                    fx = robot_states[rid]["x"]
                    fy = robot_states[rid]["y"]
                    fangle = robot_states[rid]["angle"]

                    path = follower_paths[rid]
                    target_dist = current_slots[rid]["target_dist"]
                    effective_target_dist = target_dist   # Her durumda formasyon araligini koru

                    # --- Sirayla baslama: ONCEKI robot liderin yoluna OTURUNCA kalk ---
                    # (R1 oturunca R3, R3 oturunca R2). Onceki takilirsa ACTIVATION_MAX_WAIT sonra yine de basla.
                    if not follower_activated[rid]:
                        prev = PREV_FOLLOWER.get(rid)
                        if prev and follower_activated[prev] and activation_times[prev] is not None:
                            prev_settled = follower_on_path[prev]
                            timeout = (now - activation_times[prev]) >= ACTIVATION_MAX_WAIT
                            if prev_settled or timeout:
                                follower_activated[rid] = True
                                activation_times[rid] = now
                                print(f"[OK] Robot {rid} kalkti ({'onceki yola oturdu' if prev_settled else 'timeout'}).")

                    if follower_activated[rid]:
                        # --- PARK GARANTI (kursungecirmez): lider durali PARK_FORCE_DELAY gectiyse,
                        #     henuz park hedefi olmayan her aktif takipciye park hedefi ata.
                        #     (tx/path hesabindan BAGIMSIZ -> hicbir robot line'da takili kalmaz.)
                        if arrived["flag"] and follower_final_targets[rid][0] is None and \
                           (now - arrived["time"]) > PARK_FORCE_DELAY:
                            follower_final_targets[rid] = park_triangle_pos(
                                rid, arrived["x"], arrived["y"], arrived["angle"], FORMATION_SPACING
                            )
                            print(f"[OK] Robot {rid} -> park (garanti/zorlandi).")

                        # --- ACC: lider yon cercevesinde seridimdeki en yakin robota gore fuzzy fren ---
                        heading_ref = leader_heading_deg if leader_heading_deg is not None else fangle
                        lane_width = FORMATION_SPACING * LANE_WIDTH_RATIO.get(formation_mode, 0.5)
                        d_ahead = nearest_ahead_dist(
                            fx, fy, rid, last_known_positions, now, heading_ref, lane_width
                        )
                        if d_ahead != float("inf"):
                            acc_scale = fuzzy_speed_factor(d_ahead, formation_mode)

                        # --- Hedef: lider varmışsa sabit endpoint, hâlâ yoldaysa path-based ---
                        tx, ty = None, None
                        use_endpoint = (follower_final_targets[rid][0] is not None)

                        if use_endpoint:
                            tx, ty = follower_final_targets[rid]
                        elif len(path) > 0:
                            if effective_target_dist <= 0:
                                rec_x, rec_y, rec_h = path[-1]
                            else:
                                # FALLBACK: path target_dist'e ulasamiyorsa en eski noktaya clamp et
                                # (None birakma! yoksa R2 gibi uzak robot hic hedef alamaz -> park tetiklenmez).
                                rec_x, rec_y, rec_h = path[0]
                                cum_len = 0.0
                                for i in range(len(path) - 1, 0, -1):
                                    dx_seg = path[i][0] - path[i-1][0]
                                    dy_seg = path[i][1] - path[i-1][1]
                                    cum_len += math.sqrt(dx_seg*dx_seg + dy_seg*dy_seg)
                                    if cum_len >= effective_target_dist:
                                        rec_x, rec_y, rec_h = path[i-1]
                                        break
                            side_offset = current_slots[rid]["side_offset"]
                            fb_offset   = current_slots[rid]["fb_offset"]
                            rad_h = math.radians(rec_h)
                            offset_rad = rad_h + (math.pi / 2.0)
                            tx = rec_x + side_offset * math.cos(offset_rad) + fb_offset * math.cos(rad_h)
                            ty = rec_y + side_offset * math.sin(offset_rad) + fb_offset * math.sin(rad_h)

                        if tx is not None:
                            t_color = (255, 255, 0) if rid == 1 else (0, 255, 255) if rid == 2 else (255, 0, 255)
                            cv2.drawMarker(frame, (int(tx), int(ty)), t_color, cv2.MARKER_TILTED_CROSS, 10, 1)
                            cv2.line(frame, (int(fx), int(fy)), (int(tx), int(ty)), t_color, 1, cv2.LINE_AA)

                            fdx = tx - fx
                            fdy = ty - fy
                            fdist = math.sqrt(fdx**2 + fdy**2)

                            # "Yola oturdu" durumu: hedefine yeterince yaklastiysa (sonraki robotun kalkis tetigi)
                            if not use_endpoint and fdist < SETTLE_RADIUS:
                                follower_on_path[rid] = True

                            # Erken park: takipci kendi path sonuna ulastiysa (fdist kucuk) hemen park'a gec
                            # (garanti zaten yukarida zaman-tabanli ele alindi).
                            if arrived["flag"] and not use_endpoint and fdist < PARK_TRIGGER_RADIUS:
                                follower_final_targets[rid] = park_triangle_pos(
                                    rid, arrived["x"], arrived["y"], arrived["angle"], FORMATION_SPACING
                                )
                                print(f"[OK] Robot {rid} -> park (path bitti).")

                            if use_endpoint and (parked[rid] or fdist < PARK_ARRIVAL_RADIUS):
                                # Park noktasina ulasildi — dur ve KILITLE (tekrar donmesin/loop yapmasin)
                                parked[rid] = True
                                f_left_cmd, f_right_cmd = 0, 0
                            else:
                                ftarget_angle = math.degrees(math.atan2(fdy, fdx)) % 360
                                fangle_diff   = (ftarget_angle - fangle + 180) % 360 - 180
                                is_catchup    = fdist > FORMATION_SPACING * 0.6
                                f_left_cmd, f_right_cmd = compute_motor_commands(
                                    fangle_diff, fdist,
                                    is_catchup=is_catchup,
                                    speed_scale=speed_scale * acc_scale * FOLLOWER_SPEED_SCALE,
                                    pd_id=f"follower{rid}", kd=FOLLOWER_TURN_KD,
                                    kp=FOLLOWER_TURN_KP
                                )

                            # CARPISMA ONLEME: onunde yuksek-oncelikli robot varsa dur
                            if (f_left_cmd != 0 or f_right_cmd != 0) and not parked[rid] and \
                               collision_yield(rid, fx, fy, fdx, fdy, robot_states,
                                               last_known_positions, now, COLLISION_STOP_DIST):
                                f_left_cmd, f_right_cmd = 0, 0

                if paused:
                    robot_pwms[rid] = (0, 0)
                else:
                    f_left_pwm  = apply_deadband(f_left_cmd)  if f_left_cmd  != 0 else 0
                    f_right_pwm = apply_deadband(f_right_cmd) if f_right_cmd != 0 else 0
                    f_left_pwm  = int(f_left_pwm)
                    f_right_pwm = int(f_right_pwm)
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
            
            scale_str = f"{FORMATION_SPACING / px_per_cm:.0f}cm" if px_per_cm else "?"
            cv2.putText(frame, f"Hiz: x{speed_scale:.1f}  Aralik: {int(FORMATION_SPACING)}px (~{scale_str})  FPS:{fps_display}",
                        (fw - 300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1)

            shortcuts = "[Sol Tik] Hedef  [Sag Tik] Temizle  [1] Cizgi  [2] Ucgen  [S] Dur  [+/-] Hiz  [o/p] Aralik  [Q] Cikis"
            cv2.putText(frame, shortcuts, (10, fh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

            # --- Dahili log (data collection: 0.3 sn aralikla, analyze_log.py ile incelenir) ---
            if now - last_log_time >= 0.3:
                last_log_time = now
                def _p(rid):
                    s = robot_states[rid]
                    if s["found"]:
                        return f"{int(s['x'])},{int(s['y'])},{int(s['angle'])}"
                    return ",,"
                def _d(a, b):
                    sa, sb = robot_states[a], robot_states[b]
                    if sa["found"] and sb["found"]:
                        return f"{math.sqrt((sa['x']-sb['x'])**2+(sa['y']-sb['y'])**2):.0f}"
                    return "-1"
                ts = f"{now:.2f}"
                row = (f"{ts},{_p(0)},{_p(1)},{_p(2)},{_p(3)},"
                       f"{_d(0,2)},{_d(0,1)},{_d(0,3)},{_d(1,2)},{_d(1,3)},{_d(2,3)},"
                       f"{int(follower_activated[2])},{int(follower_activated[1])},{int(follower_activated[3])},"
                       f"{leader_dist['since_target']:.0f},{formation_mode},"
                       f"{int(parked[1])},{int(parked[3])},{int(parked[2])},{status_text[:30]}\n")
                _log_file.write(row)
                _log_file.flush()

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
                FORMATION_SPACING = max(80.0, FORMATION_SPACING - 10.0)
                FORMATIONS["triangle"][1]["side_offset"] = FORMATION_SPACING
                FORMATIONS["triangle"][3]["side_offset"] = -FORMATION_SPACING
                FORMATIONS["line"][1]["target_dist"] = 1.0 * FORMATION_SPACING
                FORMATIONS["line"][3]["target_dist"] = 2.0 * FORMATION_SPACING
                FORMATIONS["line"][2]["target_dist"] = 3.0 * FORMATION_SPACING
                print(f"[FORMASYON ARALIGI] {FORMATION_SPACING}px")
            elif key == ord("p") or key == ord("P"):
                FORMATION_SPACING = min(250.0, FORMATION_SPACING + 10.0)
                FORMATIONS["triangle"][1]["side_offset"] = FORMATION_SPACING
                FORMATIONS["triangle"][3]["side_offset"] = -FORMATION_SPACING
                FORMATIONS["line"][1]["target_dist"] = 1.0 * FORMATION_SPACING
                FORMATIONS["line"][3]["target_dist"] = 2.0 * FORMATION_SPACING
                FORMATIONS["line"][2]["target_dist"] = 3.0 * FORMATION_SPACING
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
        _log_file.close()
        print("[OK] Sistem kapatıldı.")

if __name__ == "__main__":
    main()
