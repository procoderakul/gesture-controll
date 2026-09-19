"""
================================================================================
JARVIS GESTURE CONTROLLER V3 — STREAMLINED ULTRA-LOW LATENCY ENGINE
================================================================================
Core Architecture:
  Camera Frame -> MediaPipe Hand Detection -> Vector Finger Classification
  -> Mutually Exclusive Classifier -> Short Stability Filter -> SINGLE Action Handler

Core Rules:
  1. ONE FRAME -> ONE FINAL GESTURE -> ONE ACTION PATH.
  2. OPEN PALM = ABSOLUTELY NOTHING (Zero cursor, zero clicks, zero drag, zero scroll).
  3. No gesture mixing, no conflicting simultaneous states.
  4. Hysteresis on pinches to prevent rapid toggling.
  5. Zero artificial lag (No background network/speech threads, no heavy UI overhead).
  6. Instant tracking-loss safety and emergency stop (CTRL + ALT + G).
================================================================================
"""

import cv2
import time
import os
import math
import ctypes
import urllib.request
from enum import Enum
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
from pynput import keyboard

# Disable PyAutoGUI artificial delays and failsafe crashes
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# ==============================================================================
# 1. CONFIGURATION & TUNING
# ==============================================================================
class Settings:
    # Camera
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    MIRROR_X = True
    INVERT_Y = False

    # Tracking Model
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_PRESENCE_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7

    # Cursor & Kinematics (Air Mouse V2)
    CURSOR_SMOOTHING_FACTOR = 4.0    # Lower = more responsive, Higher = smoother
    CURSOR_DEAD_ZONE = 1.5           # Pixel jitter filter
    SENSITIVITY = 1.0                # 1.0 = 1:1 screen mapping
    TRACKING_MARGIN_X = 0.02         # 2% edge margin to reach all corners effortlessly
    TRACKING_MARGIN_Y = 0.02

    # Pinch Hysteresis Thresholds
    PINCH_ACTIVATE_DIST = 0.040      # Thumb + Index distance to activate
    PINCH_RELEASE_DIST = 0.060       # Distance to release pinch (hysteresis)
    RIGHT_PINCH_ACTIVATE_DIST = 0.045 # Thumb + Middle distance to activate
    RIGHT_PINCH_RELEASE_DIST = 0.065

    # Timing & Delays
    CLICK_COOLDOWN = 0.25            # Seconds between single clicks
    RIGHT_CLICK_COOLDOWN = 0.35      # Seconds between right clicks
    DRAG_HOLD_DELAY = 0.32           # Seconds holding pinch to engage DRAG
    GESTURE_CONFIRM_FRAMES = 2       # Frames required to confirm a state change

    # Scrolling
    SCROLL_SPEED_MULTIPLIER = 9.0
    SCROLL_DEAD_ZONE = 3.0           # Vertical pixel threshold before scrolling

    # Display & HUD
    DEBUG_MODE = True                # Show detailed telemetry overlay
    COLOR_CYAN = (255, 200, 0)       # Primary Cyan
    COLOR_BLUE = (255, 140, 0)       # Accent Blue/Orange
    COLOR_GREEN = (0, 255, 0)        # Active state
    COLOR_RED = (0, 0, 255)          # Alert
    COLOR_MAGENTA = (255, 0, 255)    # Right click
    COLOR_ORANGE = (0, 165, 255)     # Drag mode
    COLOR_PANEL_BG = (15, 15, 20)    # Glass Panel BG
    COLOR_TEXT = (220, 220, 220)


# ==============================================================================
# 2. VECTOR KINEMATICS & SMOOTHING
# ==============================================================================
def distance_2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

class ExponentialSmoother:
    def __init__(self, factor):
        self.factor = factor
        self.prev_val = None

    def update(self, current_val):
        if self.prev_val is None:
            self.prev_val = current_val
            return current_val
        smoothed = self.prev_val + (current_val - self.prev_val) / self.factor
        self.prev_val = smoothed
        return smoothed

    def reset(self):
        self.prev_val = None


# ==============================================================================
# 3. ROTATION-ROBUST FINGER CLASSIFICATION
# ==============================================================================
def is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx):
    """
    Rotation-robust check using relative bone vectors and distance ratio.
    Works regardless of hand rotation, pitch, or camera distance.
    """
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]

    # Tip must be farther from MCP than PIP is
    tip_dist = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
    pip_dist = math.hypot(pip.x - mcp.x, pip.y - mcp.y)
    if tip_dist < pip_dist * 1.10:
        return False

    # Alignment check: MCP->PIP and MCP->TIP point in the same direction
    v_pip = (pip.x - mcp.x, pip.y - mcp.y)
    v_tip = (tip.x - mcp.x, tip.y - mcp.y)
    len_pip = math.hypot(*v_pip)
    len_tip = math.hypot(*v_tip)
    if len_pip < 0.001 or len_tip < 0.001:
        return False

    cos_angle = (v_pip[0] * v_tip[0] + v_pip[1] * v_tip[1]) / (len_pip * len_tip)
    return cos_angle > 0.50

def is_thumb_extended(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    wrist = landmarks[0]
    tip_dist = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    ip_dist = math.hypot(thumb_ip.x - wrist.x, thumb_ip.y - wrist.y)
    return tip_dist > ip_dist * 1.15


# ==============================================================================
# 4. EXCLUSIVE GESTURE STATES & CLASSIFIER
# ==============================================================================
class GestureState(Enum):
    IDLE = 0
    OPEN_PALM = 1
    CURSOR = 2
    LEFT_CLICK = 3
    DRAGGING = 4
    RIGHT_CLICK = 5
    SCROLL = 6
    EMERGENCY_STOP = 7

class GestureDetector:
    def analyze(self, hand_landmarks):
        if not hand_landmarks:
            return None

        lms = hand_landmarks[0]
        thumb_tip = lms[4]
        index_tip = lms[8]
        middle_tip = lms[12]
        ring_tip = lms[16]
        pinky_tip = lms[20]

        # Finger extension states
        index_up  = is_finger_extended(lms, 8, 6, 5)
        middle_up = is_finger_extended(lms, 12, 10, 9)
        ring_up   = is_finger_extended(lms, 16, 14, 13)
        pinky_up  = is_finger_extended(lms, 20, 18, 17)
        thumb_out = is_thumb_extended(lms)

        # 2D Pinch Distances
        pinch_dist = distance_2d(thumb_tip, index_tip)
        right_pinch_dist = distance_2d(thumb_tip, middle_tip)

        return {
            "landmarks": lms,
            "index_tip": index_tip,
            "middle_tip": middle_tip,
            "thumb_tip": thumb_tip,
            "pinch_dist": pinch_dist,
            "right_pinch_dist": right_pinch_dist,
            "index_up": index_up,
            "middle_up": middle_up,
            "ring_up": ring_up,
            "pinky_up": pinky_up,
            "thumb_out": thumb_out,
        }


# ==============================================================================
# 5. STATE MACHINE WITH STRICT EXCLUSIVITY & HYSTERESIS
# ==============================================================================
class StateMachine:
    def __init__(self):
        self.current_state = GestureState.IDLE
        self.candidate_state = GestureState.IDLE
        self.confirm_counter = 0

        # Debounce & Hysteresis tracking
        self.left_armed = True
        self.right_armed = True
        self.last_left_click = 0.0
        self.last_right_click = 0.0

        # Drag tracking
        self.pinch_start_time = 0.0
        self.is_dragging = False

    def update(self, data, emergency_stop):
        if emergency_stop:
            self._reset()
            return GestureState.EMERGENCY_STOP

        if not data:
            self._reset()
            return GestureState.IDLE

        now = time.time()
        pinch = data["pinch_dist"]
        r_pinch = data["right_pinch_dist"]

        i_up = data["index_up"]
        m_up = data["middle_up"]
        r_up = data["ring_up"]
        p_up = data["pinky_up"]

        # -------------------------------------------------------------
        # 1. PINCH HYSTERESIS MANAGEMENT
        # -------------------------------------------------------------
        # Re-arm left click when pinch opens beyond release threshold
        if pinch > Settings.PINCH_RELEASE_DIST:
            self.left_armed = True
            if self.is_dragging:
                self.is_dragging = False
                self.pinch_start_time = 0.0

        # Re-arm right click when fingers separate beyond release threshold
        if r_pinch > Settings.RIGHT_PINCH_RELEASE_DIST:
            self.right_armed = True

        # -------------------------------------------------------------
        # 2. EXCLUSIVE GESTURE CLASSIFICATION (STRICT PRIORITY)
        # -------------------------------------------------------------

        # PRIORITY 1: OPEN PALM = ABSOLUTELY NOTHING
        # All 4 fingers extended, no active pinch.
        # This overrides everything else immediately.
        if (i_up and m_up and r_up and p_up and
            pinch > Settings.PINCH_ACTIVATE_DIST and
            r_pinch > Settings.RIGHT_PINCH_ACTIVATE_DIST):
            new_candidate = GestureState.OPEN_PALM

        # PRIORITY 2: LEFT PINCH / DRAG
        # Index is the active finger, others are curled, Thumb + Index pinch active.
        elif i_up and not r_up and not p_up and pinch < Settings.PINCH_ACTIVATE_DIST:
            if self.pinch_start_time == 0.0:
                self.pinch_start_time = now

            if self.is_dragging or (now - self.pinch_start_time) > Settings.DRAG_HOLD_DELAY:
                self.is_dragging = True
                new_candidate = GestureState.DRAGGING
            else:
                new_candidate = GestureState.LEFT_CLICK

        # PRIORITY 3: RIGHT PINCH
        # Index + Middle extended, Ring + Pinky curled, Thumb + Middle pinched.
        elif i_up and m_up and not r_up and not p_up and r_pinch < Settings.RIGHT_PINCH_ACTIVATE_DIST:
            new_candidate = GestureState.RIGHT_CLICK

        # PRIORITY 4: TWO-FINGER SCROLL
        # Index + Middle extended, Ring + Pinky curled, NO pinches active.
        elif i_up and m_up and not r_up and not p_up and pinch > Settings.PINCH_RELEASE_DIST and r_pinch > Settings.RIGHT_PINCH_RELEASE_DIST:
            new_candidate = GestureState.SCROLL

        # PRIORITY 5: CURSOR (AIR MOUSE)
        # ONLY Index extended, Middle/Ring/Pinky curled, NO pinch.
        elif i_up and not m_up and not r_up and not p_up and pinch > Settings.PINCH_RELEASE_DIST:
            new_candidate = GestureState.CURSOR

        # PRIORITY 6: IDLE / UNCLASSIFIED
        else:
            new_candidate = GestureState.IDLE

        # -------------------------------------------------------------
        # 3. TEMPORAL STABILITY FILTER
        # -------------------------------------------------------------
        # OPEN_PALM and IDLE have zero confirmation delay for instant reaction
        if new_candidate in [GestureState.OPEN_PALM, GestureState.IDLE]:
            self.current_state = new_candidate
            self.candidate_state = new_candidate
            self.confirm_counter = 0
            if new_candidate == GestureState.OPEN_PALM:
                self.is_dragging = False
                self.pinch_start_time = 0.0
            return self.current_state

        if new_candidate == self.candidate_state:
            self.confirm_counter += 1
        else:
            self.candidate_state = new_candidate
            self.confirm_counter = 1

        if self.confirm_counter >= Settings.GESTURE_CONFIRM_FRAMES:
            self.current_state = self.candidate_state

        return self.current_state

    def try_left_click(self):
        now = time.time()
        if self.left_armed and (now - self.last_left_click) > Settings.CLICK_COOLDOWN:
            self.left_armed = False
            self.last_left_click = now
            return True
        return False

    def try_right_click(self):
        now = time.time()
        if self.right_armed and (now - self.last_right_click) > Settings.RIGHT_CLICK_COOLDOWN:
            self.right_armed = False
            self.last_right_click = now
            return True
        return False

    def _reset(self):
        self.current_state = GestureState.IDLE
        self.candidate_state = GestureState.IDLE
        self.confirm_counter = 0
        self.is_dragging = False
        self.pinch_start_time = 0.0


# ==============================================================================
# 6. AIR MOUSE V2 CONTROLLER (FULL SCREEN MAPPING)
# ==============================================================================
class AirMouseV2:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        # Auto-detect real Windows virtual screen resolution
        self.screen_w = self.user32.GetSystemMetrics(0)
        self.screen_h = self.user32.GetSystemMetrics(1)

        self.smoother_x = ExponentialSmoother(Settings.CURSOR_SMOOTHING_FACTOR)
        self.smoother_y = ExponentialSmoother(Settings.CURSOR_SMOOTHING_FACTOR)

        self.is_drag_active = False
        self.last_x = self.screen_w // 2
        self.last_y = self.screen_h // 2
        self.last_norm_x = 0.5
        self.last_norm_y = 0.5

    def update_position(self, norm_x, norm_y):
        self.last_norm_x = norm_x
        self.last_norm_y = norm_y

        if Settings.INVERT_Y:
            norm_y = 1.0 - norm_y

        mx = Settings.TRACKING_MARGIN_X
        my = Settings.TRACKING_MARGIN_Y

        # Remap from tracking margin to full [0.0, 1.0]
        if mx < 0.5:
            rx = (norm_x - mx) / (1.0 - 2.0 * mx)
        else:
            rx = 0.5
        if my < 0.5:
            ry = (norm_y - my) / (1.0 - 2.0 * my)
        else:
            ry = 0.5

        # Sensitivity expansion around screen center
        sens = Settings.SENSITIVITY
        if sens != 1.0:
            rx = 0.5 + (rx - 0.5) * sens
            ry = 0.5 + (ry - 0.5) * sens

        # Map to full Windows pixel coordinates
        target_x = rx * self.screen_w
        target_y = ry * self.screen_h

        # Strict edge clamping so all corners are 100% reachable
        target_x = max(0, min(self.screen_w - 1, target_x))
        target_y = max(0, min(self.screen_h - 1, target_y))

        # Smooth position
        curr_x = self.smoother_x.update(target_x)
        curr_y = self.smoother_y.update(target_y)

        # Apply deadzone to filter micro-tremors
        if math.hypot(curr_x - self.last_x, curr_y - self.last_y) > Settings.CURSOR_DEAD_ZONE:
            self.last_x = int(curr_x)
            self.last_y = int(curr_y)
            self.user32.SetCursorPos(self.last_x, self.last_y)

        return self.last_x, self.last_y

    def left_click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def start_drag(self):
        if not self.is_drag_active:
            pyautogui.mouseDown()
            self.is_drag_active = True

    def stop_drag(self):
        if self.is_drag_active:
            pyautogui.mouseUp()
            self.is_drag_active = False

    def scroll(self, delta_y):
        pyautogui.scroll(int(delta_y * Settings.SCROLL_SPEED_MULTIPLIER))


# ==============================================================================
# 7. CLEAN SCI-FI HUD & TELEMETRY
# ==============================================================================
class SciFiHUD:
    def __init__(self):
        self.banner_text = ""
        self.banner_time = 0.0

    def show_banner(self, text, duration=1.5):
        self.banner_text = text
        self.banner_time = time.time() + duration

    def render(self, frame, landmarks, raw_data, state, fps, mouse):
        h, w, _ = frame.shape

        # Draw hand landmarks (Clean nodes)
        if landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for pt in pts:
                cv2.circle(frame, pt, 2, Settings.COLOR_CYAN, -1)

            # Index fingertip targeting ring
            ix, iy = pts[8]
            ring_color = Settings.COLOR_GREEN if state in [GestureState.CURSOR, GestureState.LEFT_CLICK, GestureState.DRAGGING] else Settings.COLOR_BLUE
            cv2.circle(frame, (ix, iy), 10, ring_color, 1)
            cv2.circle(frame, (ix, iy), 3, (0, 0, 255), -1)

            # Visual line between thumb and index during pinch
            tx, ty = pts[4]
            if raw_data and raw_data["pinch_dist"] < Settings.PINCH_RELEASE_DIST:
                cv2.line(frame, (ix, iy), (tx, ty), Settings.COLOR_ORANGE, 2, cv2.LINE_AA)

        # Telemetry Box (Top Left)
        box_h = 135 if Settings.DEBUG_MODE else 75
        cv2.rectangle(frame, (10, 10), (270, 10 + box_h), Settings.COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (10, 10), (270, 10 + box_h), Settings.COLOR_CYAN, 1)

        cv2.putText(frame, f"JARVIS V3 // FPS: {fps}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, Settings.COLOR_CYAN, 1)

        # Mode coloring
        if state == GestureState.OPEN_PALM:
            mode_col = (180, 180, 180) # Neutral Gray
        elif state in [GestureState.CURSOR, GestureState.LEFT_CLICK]:
            mode_col = Settings.COLOR_GREEN
        elif state == GestureState.DRAGGING:
            mode_col = Settings.COLOR_ORANGE
        elif state == GestureState.RIGHT_CLICK:
            mode_col = Settings.COLOR_MAGENTA
        elif state == GestureState.SCROLL:
            mode_col = Settings.COLOR_BLUE
        else:
            mode_col = (100, 100, 100)

        cv2.putText(frame, f"STATE: {state.name}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_col, 2)

        if Settings.DEBUG_MODE and raw_data:
            fingers_txt = (
                f"I:{'UP' if raw_data['index_up'] else 'DN'} "
                f"M:{'UP' if raw_data['middle_up'] else 'DN'} "
                f"R:{'UP' if raw_data['ring_up'] else 'DN'} "
                f"P:{'UP' if raw_data['pinky_up'] else 'DN'}"
            )
            cv2.putText(frame, fingers_txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, Settings.COLOR_TEXT, 1)
            cv2.putText(frame, f"Pinch: {raw_data['pinch_dist']:.3f} | R-Pinch: {raw_data['right_pinch_dist']:.3f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.36, Settings.COLOR_CYAN, 1)
            cv2.putText(frame, f"Drag: {'ACTIVE' if mouse.is_drag_active else 'OFF'}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.38, Settings.COLOR_ORANGE if mouse.is_drag_active else Settings.COLOR_TEXT, 1)
        elif not Settings.DEBUG_MODE:
            cv2.putText(frame, "CTRL+ALT+G: SAFETY", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # Action Banner
        if time.time() < self.banner_time and self.banner_text:
            bw = len(self.banner_text) * 11 + 40
            bx = (w - bw) // 2
            cv2.rectangle(frame, (bx, h - 55), (bx + bw, h - 22), Settings.COLOR_PANEL_BG, -1)
            cv2.rectangle(frame, (bx, h - 55), (bx + bw, h - 22), Settings.COLOR_GREEN, 1)
            cv2.putText(frame, self.banner_text, (bx + 20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, Settings.COLOR_GREEN, 1)


# ==============================================================================
# 8. VISION & TRACKING PIPELINE

        self.cap = cv2.VideoCapture(Settn# ==============================================================================
class Camera:
    def __init__(self):
        self.cap = None

    def sta   rt(self):igs.CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Settings.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Settings.CAMERA_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError(f"[ERROR] Could not open camera {Settings.CAMERA_INDEX}.")

    def read_frame(self):
        if not self.cap:
            return False, None
        ret, frame = self.cap.read()
        if ret and Settings.MIRROR_X:
            frame = cv2.flip(frame, 1)
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()

class FPSMonitor:
    def __init__(self):
        self.prev = time.time()
        self.fps = 0.0

    def update(self):
        now = time.time()
        diff = now - self.prev
        if diff > 0:
            self.fps = 1.0 / diff
        self.prev = now
        return self.fps

class HandTracker:
    def __init__(self):
        if not os.path.exists(Settings.MODEL_PATH):
            print(f"[INFO] Downloading tracking model to {Settings.MODEL_PATH}...")
            urllib.request.urlretrieve(Settings.MODEL_URL, Settings.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=Settings.MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=Settings.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=Settings.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=Settings.MIN_TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect(mp_img)

class SafetySwitch:
    def __init__(self):
        self.emergency_stop = False
        self.listener = keyboard.GlobalHotKeys({'<ctrl>+<alt>+g': self._toggle})

    def _toggle(self):
        self.emergency_stop = not self.emergency_stop
        st = "ACTIVATED" if self.emergency_stop else "DEACTIVATED"
        print(f"\n[SAFETY FAILSAFE] Emergency Stop {st}")

    def start(self): self.listener.start()
    def stop(self): self.listener.stop()


# ==============================================================================
# 9. MAIN REAL-TIME ORCHESTRATOR
# ==============================================================================
def main():
    print("=" * 65)
    print("  JARVIS GESTURE CONTROLLER V3 — LOW LATENCY ENGINE")
    print("=" * 65)
    print("  OPEN PALM            -> ABSOLUTELY NOTHING (Neutral state)")
    print("  INDEX FINGER ONLY    -> Move Cursor (Air Mouse V2)")
    print("  THUMB + INDEX PINCH  -> Left Click (Quick) / Drag (Hold > 0.3s)")
    print("  THUMB + MIDDLE PINCH -> Right Click")
    print("  TWO FINGERS UP       -> Vertical Scroll")
    print("  SAFETY STOP          -> CTRL + ALT + G")
    print("  QUIT                 -> Press 'q'")
    print("=" * 65 + "\n")

    camera = Camera()
    camera.start()
    tracker = HandTracker()
    detector = GestureDetector()
    state_machine = StateMachine()
    mouse = AirMouseV2()
    safety = SafetySwitch()
    hud = SciFiHUD()
    fps_mon = FPSMonitor()

    safety.start()
    prev_scroll_y = 0.0

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                break

            h, w, _ = frame.shape
            detection = tracker.process(frame)

            raw_data = None
            if detection.hand_landmarks:
                raw_data = detector.analyze(detection.hand_landmarks)

            # Classify exclusive state
            current_state = state_machine.update(raw_data, safety.emergency_stop)
            fps = int(fps_mon.update())

            # ==========================================================
            # SINGLE ACTION HANDLER PATH — EXACTLY ONE BRANCH EXECUTES
            # ==========================================================
            if not safety.emergency_stop and raw_data:
                norm_x = raw_data["index_tip"].x
                norm_y = raw_data["index_tip"].y

                # 1. OPEN PALM: ABSOLUTELY NOTHING
                if current_state == GestureState.OPEN_PALM:
                    mouse.stop_drag()
                    prev_scroll_y = 0.0

                # 2. CURSOR: Index pointing only
                elif current_state == GestureState.CURSOR:
                    mouse.stop_drag()
                    mouse.update_position(norm_x, norm_y)
                    prev_scroll_y = 0.0

                # 3. LEFT CLICK: Thumb + Index quick pinch
                elif current_state == GestureState.LEFT_CLICK:
                    mouse.update_position(norm_x, norm_y)
                    if state_machine.try_left_click():
                        mouse.left_click()
                        hud.show_banner("LEFT CLICK")
                    prev_scroll_y = 0.0

                # 4. DRAG: Thumb + Index held continuously
                elif current_state == GestureState.DRAGGING:
                    mouse.start_drag()
                    mouse.update_position(norm_x, norm_y)
                    prev_scroll_y = 0.0

                # 5. RIGHT CLICK: Thumb + Middle pinch
                elif current_state == GestureState.RIGHT_CLICK:
                    mouse.stop_drag()
                    if state_machine.try_right_click():
                        mouse.right_click()
                        hud.show_banner("RIGHT CLICK")
                    prev_scroll_y = 0.0

                # 6. SCROLL: Two fingers up (Index + Middle)
                elif current_state == GestureState.SCROLL:
                    mouse.stop_drag()
                    mid_norm_y = raw_data["middle_tip"].y
                    avg_y = (norm_y + mid_norm_y) / 2.0 * h
                    if prev_scroll_y != 0.0:
                        dy = prev_scroll_y - avg_y
                        if abs(dy) > Settings.SCROLL_DEAD_ZONE:
                            mouse.scroll(dy)
                    prev_scroll_y = avg_y

                # 7. IDLE / UNCLASSIFIED: Stop any motion
                else:
                    mouse.stop_drag()
                    prev_scroll_y = 0.0

            else:
                # Failsafe on tracking loss or emergency stop
                mouse.stop_drag()
                prev_scroll_y = 0.0

            # Render lightweight HUD
            lms = raw_data["landmarks"] if raw_data else None
            hud.render(frame, lms, raw_data, current_state, fps, mouse)

            cv2.imshow("JARVIS V3 Gesture Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        mouse.stop_drag()
        safety.stop()
        camera.release()
        cv2.destroyAllWindows()
        print("\n[INFO] JARVIS V3 shut down cleanly.")

if __name__ == "__main__":
    main()