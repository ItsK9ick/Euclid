import sys
import os
import time
import logging
import random
import ctypes
import json
import numpy as np
from PIL import Image  # Pillow must be installed
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QHBoxLayout, QProgressBar, QSlider, QSizePolicy, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSlot, pyqtSignal, QPoint, QTimer, QEvent, QPropertyAnimation, QEasingCurve
import psutil
import requests
import subprocess

# ----- Auto Update Config & Functions -----
BUILD_VERSION = "v1.0.0"  # Update this with each new release
REMOTE_VERSION_URL = "https://raw.githubusercontent.com/ItsK9ick/Euclid/main/version.txt"
# URL of the new Euclid executable (release asset) to download when an update is available
UPDATE_EXE_URL = "https://github.com/ItsK9ick/Euclid/releases/download/v1.0.0/Euclid.exe"
# Path to the updater executable (which should be packaged alongside Euclid)
UPDATER_EXE_PATH = "updater.exe"

def perform_update_exe():
    """Download the new executable and launch the updater."""
    try:
        response = requests.get(UPDATE_EXE_URL)
        update_path = os.path.join(os.getcwd(), "update_temp.exe")
        with open(update_path, "wb") as f:
            f.write(response.content)
        subprocess.Popen([UPDATER_EXE_PATH, update_path])
    except Exception as e:
        print("Update failed:", e)
    os._exit(0)

def check_for_updates():
    """Check remote version; if different from BUILD_VERSION, prompt update."""
    try:
        remote_version = requests.get(REMOTE_VERSION_URL, timeout=5).text.strip()
        if remote_version != BUILD_VERSION:
            # Simple prompt using a QDialog
            dialog = QDialog()
            dialog.setWindowTitle("Update Available")
            dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            layout = QVBoxLayout(dialog)
            label = QLabel(f"A new version ({remote_version}) is available.\nUpdate now?", dialog)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            btn_layout = QHBoxLayout()
            btn_update = QPushButton("Update Now", dialog)
            btn_later = QPushButton("Later", dialog)
            btn_layout.addWidget(btn_update)
            btn_layout.addWidget(btn_later)
            layout.addLayout(btn_layout)
            btn_update.clicked.connect(dialog.accept)
            btn_later.clicked.connect(dialog.reject)
            if dialog.exec_() == QDialog.Accepted:
                perform_update_exe()
    except Exception as e:
        print("Update check failed:", e)

# ----- Load Configuration -----
CONFIG_FILE = "euclid_config.json"
default_config = {
    "keybinds": {
        "toggle_monitor": "F2",
        "toggle_bottom": "F3",
        "emergency_stop": "F4",
        "toggle_risk_mode": "F5"
    },
    "space_key": 32,
    "cooldown_safe": 1.5,   # seconds (for Safe mode)
    "cooldown_risky": 1.0,  # seconds (for Risky mode)
    "fps_limit": 60
}
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_config, f, indent=4)
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
# Global parameters from config
# Mode: RISK_MODE==0 is RISKY (no delay); RISK_MODE==1 is SAFE (with cooldown)
RISK_MODE = 0  
COOLDOWN_SAFE = config.get("cooldown_safe", 1.5)
COOLDOWN_RISKY = config.get("cooldown_risky", 1.0)
FPS_LIMIT = config.get("fps_limit", 60)
SPACE_KEY = config.get("space_key", 32)
keybinds = config.get("keybinds", default_config["keybinds"])

# ----- Initialize NVML for GPU monitoring (if needed) -----
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# ----------------- MonitorWorker -----------------
class MonitorWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.last_hit_time = None

    def run(self):
        try:
            from dbd.AI_model import AI_model
            from dbd.utils.directkeys import PressKey, ReleaseKey
            space_key = SPACE_KEY

            base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.getcwd()
            onnx_model = os.path.join(base_path, "model.onnx")
            if not os.path.exists(onnx_model):
                self.log_signal.emit("❌ model.onnx not found!")
                return

            use_gpu = True
            nb_cpu_threads = 4
            try:
                ai_model = AI_model(onnx_model, use_gpu, nb_cpu_threads)
                provider = ai_model.check_provider()
            except Exception as e:
                self.log_signal.emit(f"GPU mode failed: {e}. Falling back to CPU.")
                use_gpu = False
                nb_cpu_threads = 2
                ai_model = AI_model(onnx_model, use_gpu, nb_cpu_threads)
                provider = ai_model.check_provider()

            self.log_signal.emit(f"Euclid Provider: {provider}")
            self.progress_signal.emit(20)
            self.log_signal.emit("🚀 Euclid Engine initialized. Monitoring started.")
            self.progress_signal.emit(100)
            
            while self.running:
                frame_start = time.perf_counter()
                screenshot = ai_model.grab_screenshot()
                image_pil = ai_model.screenshot_to_pil(screenshot)
                try:
                    image_np = np.array(image_pil, dtype=np.float32) / 255.0
                    if image_np.ndim == 3:
                        image_np = np.transpose(image_np, (2, 0, 1))
                        image_np = np.expand_dims(image_np, axis=0)
                except Exception as e:
                    self.log_signal.emit(f"Image conversion error: {e}")
                    continue

                pred, desc, probs, should_hit = ai_model.predict(image_np)
                current_time = time.perf_counter()
                # Determine cooldown based on mode: if SAFE (RISK_MODE==1) use COOLDOWN_SAFE; else (RISKY) use COOLDOWN_RISKY
                cooldown = COOLDOWN_SAFE if RISK_MODE == 1 else COOLDOWN_RISKY
                if should_hit and (self.last_hit_time is None or (current_time - self.last_hit_time) >= cooldown):
                    PressKey(space_key)
                    ReleaseKey(space_key)
                    self.last_hit_time = time.perf_counter()
                    self.log_signal.emit("🎯 Euclid triggered action!")
                elapsed = time.perf_counter() - frame_start
                sleep_time = max(1.0 / FPS_LIMIT - elapsed, 0)
                time.sleep(sleep_time)
        except Exception as e:
            self.log_signal.emit(f"Monitor Error: {e}")

    def stop(self):
        self.running = False

# ----------------- EuclidOverlayUI -----------------
class EuclidOverlayUI(QWidget):
    def __init__(self):
        super().__init__()
        self.monitor_worker = None
        self._drag_active = False
        self._drag_offset = QPoint(0, 0)
        self.bottom_visible = True  # Controls visibility of the bottom UI
        self.initUI()
        self.installEventFilter(self)

    def initUI(self):
        # Discreet overlay: frameless, hidden from Alt‑Tab/taskbar
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.resize(300, 120)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("""
            QWidget { background-color: rgba(0,0,0,0); color: wheat; font-size: 12px; }
            QPushButton { background-color: transparent; color: wheat; }
            QLabel { color: wheat; }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Top Bar: Close button, Log label, Pin button
        top_bar = QHBoxLayout()
        self.close_btn = QPushButton("X", self)
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setStyleSheet("color: red; font-weight: bold;")
        self.close_btn.clicked.connect(self.emergency_stop)
        top_bar.addWidget(self.close_btn, alignment=Qt.AlignLeft)

        self.log_label = QLabel("Euclid is stopped.", self)
        self.log_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.log_label.setAlignment(Qt.AlignCenter)
        top_bar.addWidget(self.log_label, alignment=Qt.AlignCenter)

        self.pin_btn = QPushButton("📌", self)
        self.pin_btn.setFixedSize(20, 20)
        self.pin_btn.clicked.connect(self.toggle_topmost)
        top_bar.addWidget(self.pin_btn, alignment=Qt.AlignRight)
        main_layout.addLayout(top_bar)

        # Thin loading bar (kept for aesthetic feedback)
        self.loading_bar = QProgressBar(self)
        self.loading_bar.setFixedHeight(2)
        self.loading_bar.setValue(0)
        self.loading_bar.setStyleSheet("QProgressBar::chunk { background: wheat; }")
        main_layout.addWidget(self.loading_bar)

        # Bottom Bar: Start/Stop button, Title, [SFM] toggle
        self.bottom_bar = QHBoxLayout()
        self.bottom_bar.setSpacing(5)
        self.toggle_btn = QPushButton("Start", self)
        self.toggle_btn.setStyleSheet("""
            QPushButton { border: 1px solid wheat; background: transparent; color: wheat; padding: 5px; min-height: 30px; }
            QPushButton:hover { background: #202020; }
        """)
        self.toggle_btn.clicked.connect(self.toggle_monitor)
        self.bottom_bar.addWidget(self.toggle_btn, alignment=Qt.AlignLeft)

        self.title_label = QLabel("Euclid Final Edition", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("background: none; border: none; font-size: 10px;")
        self.bottom_bar.addWidget(self.title_label, stretch=1)

        self.sfm_btn = QPushButton("[SFM]", self)
        self.sfm_btn.setStyleSheet("""
            QPushButton { border: 1px solid wheat; background: transparent; color: wheat; padding: 5px; min-height: 30px; }
            QPushButton:checked { background: wheat; color: black; }
        """)
        self.sfm_btn.setCheckable(True)
        self.sfm_btn.clicked.connect(self.toggle_streamsafe)
        self.bottom_bar.addWidget(self.sfm_btn, alignment=Qt.AlignRight)
        main_layout.addLayout(self.bottom_bar)

        # Safe/Risky slider with labels
        sr_layout = QHBoxLayout()
        sr_layout.setSpacing(3)
        # Left label (shows current mode when in RISKY mode)
        self.left_label = QLabel("Risky", self)
        self.left_label.setStyleSheet("font-size: 10px;")
        self.left_label.setFixedHeight(20)
        self.left_label.setFixedWidth(80)
        sr_layout.addWidget(self.left_label, alignment=Qt.AlignLeft)
        # Slider (0 means RISKY; 1 means SAFE)
        self.sr_slider = QSlider(Qt.Horizontal, self)
        self.sr_slider.setMinimum(0)
        self.sr_slider.setMaximum(1)
        self.sr_slider.setValue(RISK_MODE)
        self.sr_slider.setTickPosition(QSlider.NoTicks)
        self.sr_slider.setFixedHeight(10)
        self.sr_slider.setFixedWidth(40)
        self.sr_slider.valueChanged.connect(self.update_risk_mode)
        sr_layout.addWidget(self.sr_slider, alignment=Qt.AlignCenter)
        # Right label (shows current mode when in SAFE mode)
        self.right_label = QLabel("Safe", self)
        self.right_label.setStyleSheet("font-size: 10px; color: green;")
        self.right_label.setFixedHeight(20)
        self.right_label.setFixedWidth(80)
        sr_layout.addWidget(self.right_label, alignment=Qt.AlignRight)
        main_layout.addLayout(sr_layout)

        self.setLayout(main_layout)
        self.set_topmost(True)
        self.set_display_affinity()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            return False
        elif event.type() == QEvent.MouseMove and getattr(self, '_drag_active', False):
            self.move(event.globalPos() - self._drag_offset)
            return True
        elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            self._drag_active = False
            return False
        return super().eventFilter(obj, event)

    def set_display_affinity(self):
        try:
            user32 = ctypes.windll.user32
            WDA_EXCLUDEFROMCAPTURE = 0x11
            user32.SetWindowDisplayAffinity(int(self.winId()), WDA_EXCLUDEFROMCAPTURE)
        except Exception as e:
            self.log_label.setText("Display Affinity Error: " + str(e))
    
    def update_risk_mode(self, value):
        global RISK_MODE
        RISK_MODE = value
        mode = "Safe" if RISK_MODE == 1 else "Risky"
        self.log_label.setText(f"Mode: {mode}")
    
    def toggle_streamsafe(self):
        if self.sfm_btn.isChecked():
            self.sfm_btn.setText("[SFM ON]")
        else:
            self.sfm_btn.setText("[SFM]")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
    
    def mouseMoveEvent(self, event):
        if getattr(self, '_drag_active', False):
            self.move(event.globalPos() - self._drag_offset)
    
    def mouseReleaseEvent(self, event):
        self._drag_active = False
    
    def set_topmost(self, enable):
        flags = Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()
    
    def toggle_topmost(self):
        if self.windowFlags() & Qt.WindowStaysOnTopHint:
            self.set_topmost(False)
            self.pin_btn.setStyleSheet("color: gray;")
        else:
            self.set_topmost(True)
            self.pin_btn.setStyleSheet("color: wheat;")
    
    def toggle_monitor(self):
        try:
            if self.monitor_worker is None:
                self.start_monitor()
            else:
                self.stop_monitor()
        except Exception as e:
            self.log_label.setText(f"Error toggling monitor: {e}")
    
    def start_monitor(self):
        self.toggle_btn.setText("Stop")
        self.log_label.setText("Starting Euclid...")
        self.monitor_worker = MonitorWorker()
        self.monitor_worker.log_signal.connect(self.update_log)
        self.monitor_worker.progress_signal.connect(self.update_loading)
        self.monitor_worker.start()
    
    def stop_monitor(self):
        if self.monitor_worker:
            try:
                self.monitor_worker.stop()
                self.monitor_worker.wait(2000)
            except Exception as e:
                self.log_label.setText(f"Error stopping monitor: {e}")
            self.monitor_worker = None
        self.toggle_btn.setText("Start")
        self.log_label.setText("Euclid stopped.")
        self.loading_bar.setValue(0)
    
    @pyqtSlot(str)
    def update_log(self, message):
        self.log_label.setText(message)
        logging.info(message)
    
    @pyqtSlot(int)
    def update_loading(self, value):
        self.loading_bar.setValue(value)
    
    def emergency_stop(self):
        if self.monitor_worker:
            try:
                self.monitor_worker.stop()
                self.monitor_worker.wait(3000)
            except Exception as e:
                self.log_label.setText("Error during emergency stop: " + str(e))
        self.hide()
        QApplication.quit()
        os._exit(0)  # Force termination so that it fully closes

    def keyPressEvent(self, event):
        if event.key() == getattr(Qt, f"Key_{keybinds.get('toggle_monitor', 'F2')}"):
            self.toggle_monitor()
        elif event.key() == getattr(Qt, f"Key_{keybinds.get('toggle_bottom', 'F3')}"):
            self.toggle_bottom_bar()
        elif event.key() == getattr(Qt, f"Key_{keybinds.get('emergency_stop', 'F4')}"):
            self.emergency_stop()
        elif event.key() == getattr(Qt, f"Key_{keybinds.get('toggle_risk_mode', 'F5')}"):
            self.toggle_risk_mode()
    
    def toggle_bottom_bar(self):
        self.bottom_visible = not self.bottom_visible
        for widget in [self.toggle_btn, self.title_label, self.sfm_btn, self.sr_slider, self.left_label, self.right_label]:
            widget.setVisible(self.bottom_visible)
    
    def toggle_risk_mode(self):
        global RISK_MODE
        RISK_MODE = 0 if RISK_MODE == 1 else 1
        mode = "Safe" if RISK_MODE == 1 else "Risky"
        self.log_label.setText(f"Mode toggled: {mode}")
        self.sr_slider.setValue(RISK_MODE)
    
    def update_loading(self, value):
        self.loading_bar.setValue(value)

# ----- Main: Auto Update Check and Application Launch -----
if __name__ == '__main__':
    # Check for updates before starting (this will prompt and update if a new version is available)
    try:
        check_for_updates()
    except Exception as e:
        print("Auto update check failed:", e)
    
    app = QApplication(sys.argv)
    window = EuclidOverlayUI()
    window.installEventFilter(window)
    window.show()
    try:
        import keyboard
        keyboard.add_hotkey(keybinds.get('toggle_monitor', 'F2'), lambda: window.toggle_monitor())
        keyboard.add_hotkey(keybinds.get('toggle_bottom', 'F3'), lambda: window.toggle_bottom_bar())
        keyboard.add_hotkey(keybinds.get('emergency_stop', 'F4'), lambda: window.emergency_stop())
        keyboard.add_hotkey(keybinds.get('toggle_risk_mode', 'F5'), lambda: window.toggle_risk_mode())
    except Exception as e:
        print("Global hotkeys not available:", e)
    sys.exit(app.exec_())
