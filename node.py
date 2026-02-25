import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
import os
import folder_paths
import logging
import urllib.request

try:
    from ultralytics import YOLO
except ImportError:
    logging.error("Ultralytics is not installed. Please run `pip install ultralytics`.")

# Models directory for YOLO/Ultralytics
ultralytics_dir = os.path.join(folder_paths.models_dir, "ultralytics")
yolo_dir = os.path.join(folder_paths.models_dir, "yolo")

# If ultralytics directory exists, prefer it, otherwise use yolo (create it if necessary)
target_model_dir = ultralytics_dir if os.path.exists(ultralytics_dir) else yolo_dir
if not os.path.exists(target_model_dir):
    try:
        os.makedirs(target_model_dir)
    except Exception as e:
        logging.error(f"Failed to create models directory {target_model_dir}: {e}")

CUSTOM_MODEL_URL = "https://huggingface.co/sugarknight/sensitive-detect/resolve/main/sensitive_detect_v06.pt"
CUSTOM_MODEL_NAME = "sensitive_detect_v06.pt"

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    try:
        # User-Agent header is important for some CDNs
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(target_path, 'wb') as out_file:
                # Add simple block reading
                total_length = response.getheader('content-length')
                if total_length is None: # no content length header
                    out_file.write(response.read())
                else:
                    total_length = int(total_length)
                    downloaded = 0
                    block_size = max(4096, total_length // 100)
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
        print(f"Successfully downloaded {CUSTOM_MODEL_NAME}.")
        return True
    except Exception as e:
        print(f"Error downloading {CUSTOM_MODEL_NAME}: {e}")
        return False
try:
    from .extract_segments import extract_and_save_segments, MosaicMode
except ImportError:
    from extract_segments import extract_and_save_segments, MosaicMode

class AutoMosaic:
    def __init__(self):
        self.model = None
        self.model_name = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "save_psd": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "AutoMosaic"}),
                "confidence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "process_method": (["raw", "mosaic", "white", "blur"], {"default": "mosaic"}),
                "factor": ("INT", {"default": 100, "min": 10, "step": 1}),
                "target_class": ("STRING", {"default": "pussy,penis"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "process_image"
    CATEGORY = "image/process"

    def process_image(self, image, save_psd, filename_prefix, confidence, process_method, factor, target_class):
        model_name = CUSTOM_MODEL_NAME
        
        # Load model only if not loaded or model name changed
        if self.model is None or self.model_name != model_name:
            
            # Check if model exists in models/ultralytics or models/yolo
            potential_paths = [
                os.path.join(ultralytics_dir, model_name),
                os.path.join(yolo_dir, model_name)
            ]
            
            model_path = model_name
            for path in potential_paths:
                if os.path.isfile(path):
                    model_path = path
                    break
            
            # If the model is not found locally and it's our target custom model, download it
            if not os.path.isfile(model_path) and model_name == CUSTOM_MODEL_NAME:
                download_dest = os.path.join(target_model_dir, CUSTOM_MODEL_NAME)
                if download_file(CUSTOM_MODEL_URL, download_dest):
                    model_path = download_dest
            
            logging.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.model_name = model_name

        output_dir = folder_paths.get_output_directory()
        
        # Map process_method to MosaicMode
        if process_method == "mosaic":
            mosaic_mode = MosaicMode.MOSAIC
        elif process_method == "blur":
            mosaic_mode = MosaicMode.BLUR
        elif process_method == "white":
            mosaic_mode = MosaicMode.WHITE
        else:
            mosaic_mode = MosaicMode.RAW

        processed_images = []
        for i, img_tensor in enumerate(image):
            # Convert ComfyUI tensor [H, W, C] to numpy array
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            full_output_folder, filename, counter, subfolder, filename_prefix_res = folder_paths.get_save_image_path(filename_prefix, output_dir, img_np.shape[1], img_np.shape[0])
            base_name = f"{filename}_{counter:05}"
            
            composite_pil = extract_and_save_segments(
                img_pil=img_pil,
                base_name=base_name,
                model=self.model,
                output_dir=full_output_folder,
                confidence=confidence,
                mosaic_mode=mosaic_mode,
                target_class=target_class,
                block_size=factor,
                save_psd=save_psd
            )
            
            # Convert composite PIL back to ComfyUI tensor formats
            out_img_np = np.array(composite_pil).astype(np.float32) / 255.0
            out_img_tensor = torch.from_numpy(out_img_np)
            processed_images.append(out_img_tensor)

        return {"ui": {"text": ["Successfully saved PSD"]}, "result": (torch.stack(processed_images),)}

# Standard ComfyUI node mappings
NODE_CLASS_MAPPINGS = {
    "AutoMosaic": AutoMosaic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMosaic": "Auto Mosaic"
}
