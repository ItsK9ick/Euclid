import numpy as np
from PIL import Image
from mss import mss
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel, ExecutionMode, get_available_providers
from pyautogui import size as pyautogui_size
from dbd.utils.frame_grabber import get_monitor_attributes

def get_monitor_attributes():
    # Calculate capture region based on current screen size.
    width, height = pyautogui_size()
    # We want a square region roughly scaled to 224 (the model’s expected input).
    # Adjust the scaling as needed; here we assume a 1920x1080 baseline.
    scale = min(width / 1920, height / 1080)
    object_size = int(224 * scale)
    monitor = {
        "top": height // 2 - object_size // 2,
        "left": width // 2 - object_size // 2,
        "width": object_size,
        "height": object_size
    }
    return monitor

class AI_model:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # This dictionary maps predictions to descriptions and whether a hit should be triggered.
    pred_dict = {
        0: {"desc": "None", "hit": False},
        1: {"desc": "repair-heal (great)", "hit": True},
        2: {"desc": "repair-heal (ante-frontier)", "hit": True},
        3: {"desc": "repair-heal (out)", "hit": False},
        4: {"desc": "full white (great)", "hit": True},
        5: {"desc": "full white (out)", "hit": False},
        6: {"desc": "full black (great)", "hit": True},
        7: {"desc": "full black (out)", "hit": False},
        8: {"desc": "wiggle (great)", "hit": True},
        9: {"desc": "wiggle (frontier)", "hit": False},
        10: {"desc": "wiggle (out)", "hit": False}
    }

    def __init__(self, onnx_filepath="model.onnx", use_gpu=False, nb_cpu_threads=None):
        # Create and configure session options.
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use parallel execution mode for performance improvements if supported.
        sess_options.execution_mode = ExecutionMode.ORT_PARALLEL

        if use_gpu:
            available_providers = get_available_providers()
            preferred_execution_providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
            # Pick the first available provider from our list.
            execution_providers = [p for p in preferred_execution_providers if p in available_providers]
            if execution_providers and execution_providers[0] == "CUDAExecutionProvider":
                import torch  # Needed to load cuDNN even if torch isn’t used directly
        else:
            execution_providers = ['CPUExecutionProvider']
            if nb_cpu_threads is not None:
                # Set the number of CPU threads to use.
                sess_options.intra_op_num_threads = nb_cpu_threads
                sess_options.inter_op_num_threads = nb_cpu_threads

        self.ort_session = InferenceSession(onnx_filepath, providers=execution_providers, sess_options=sess_options)
        self.input_name = self.ort_session.get_inputs()[0].name

        # Create an instance of mss for screenshot capture.
        self.mss = mss()
        self.monitor = get_monitor_attributes()

    def check_provider(self):
        active_providers = self.ort_session.get_providers()
        return active_providers[0]

    def grab_screenshot(self):
        # Capture the defined region of the screen.
        return self.mss.grab(self.monitor)

    def screenshot_to_pil(self, screenshot):
        # Convert the raw screenshot (BGRA) to an RGB PIL Image.
        pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        # Ensure the image is exactly 224x224 (resize if necessary)
        if pil_image.size != (224, 224):
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        return pil_image

    def pil_to_numpy(self, image_pil):
        # Convert PIL image to a normalized numpy array with channel-first format.
        img = np.asarray(image_pil, dtype=np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict(self, image):
        ort_inputs = {self.input_name: image}
        ort_outs = self.ort_session.run(None, ort_inputs)
        logits = np.squeeze(ort_outs)
        pred = int(np.argmax(logits))
        probs = self.softmax(logits)
        probs = np.round(probs, decimals=3).tolist()
        # Map probabilities to their description labels.
        probs_dict = {self.pred_dict[i]["desc"]: probs[i] for i in range(len(probs))}
        should_hit = self.pred_dict[pred]["hit"]
        desc = self.pred_dict[pred]["desc"]
        return pred, desc, probs_dict, should_hit
