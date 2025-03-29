# Euclid — AI-Powered Auto Skillcheck System for Dead by Daylight 🎯

![banner](https://github.com/user-attachments/assets/271b9a83-374c-4592-a03e-8b0eec4f862e)


**Euclid** is a real-time, AI-driven automation tool designed to detect and hit skillchecks in *Dead by Daylight* (or similar games).  
It uses an ONNX model trained to identify skillcheck visuals and automatically presses **space** at the correct moment.

Built with performance, discretion, and customization in mind.

---

## ⚙️ Features

- ✅ **AI-based skillcheck detection** using a trained ONNX model  
- 🧠 Supports **GPU acceleration** via ONNX Runtime + CUDA  
- ⚡ Ultra-fast response, designed for real-time decision making (60+ FPS image processing)  
- 🔥 **Risky/Safe mode toggle** for custom hit timing behavior  
- 🎮 Fully customizable keybinds, cooldowns, and FPS limits  
- 🖥️ **Stream-safe mode** (non-capturable overlay)  
- 🌐 **Auto-updater** that checks for and installs new releases  
- 🧪 Optional **EuclidDataPartner** app for advanced data collection & feedback

---

## 🖼️ Screenshots

![image](https://github.com/user-attachments/assets/594634eb-b8a3-47f4-a341-d2ade5fa58c7)

![image](https://github.com/user-attachments/assets/3593040c-7c34-4733-a756-2d6ff019ad03)

![image](https://github.com/user-attachments/assets/c6220f12-f435-4ee3-955d-06ae3b29a065)

![image](https://github.com/user-attachments/assets/ab906d6a-17e7-41b2-a746-f78ef98ecb85)


![image](https://github.com/user-attachments/assets/0c768643-8627-459e-ade3-5d26b40dd652)


---

## 🧰 Installation

**Easiest method:**
1. [Download the Updater](https://github.com/ItsK9ick/Euclid/releases/latest/download/updater.exe)
2. Place it anywhere you want Euclid installed (e.g. Desktop or a tools folder).
3. Run it. It will:
   - Download the latest version of Euclid
   - Download EuclidDataPartner if missing
   - Set everything up in that folder
4. Run `Euclid.exe` to launch.

[![Requirement](https://img.shields.io/badge/Requirement-Latest_NVIDIA_Game--Ready_Drivers-red)](https://www.nvidia.com/en-us/software/nvidia-app/)  
[![Requirement](https://img.shields.io/badge/Requirement-Latest_C++_Redistributable-red)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)

[![Precaution](https://img.shields.io/badge/Precaution-Make%20sure%20you%20are%20running%20Performance%20Mode%20%28not%20Power%20Saver%29-orange)]()  
[![Precaution](https://img.shields.io/badge/Precaution-First_launch_may_take_longer-orange)]()

---

## 🛠 Configuration

After first launch, a config file will be generated:

```json
{
  "keybinds": {
    "toggle_monitor": "F2",
    "emergency_stop": "F4",
    "toggle_risk_mode": "F5"
  },
  "space_key": 32,
  "cooldown_safe": 1.5,
  "cooldown_risky": 1.0,
  "fps_limit": 60
}
