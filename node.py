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

CUSTOM_MODEL_URL = "https://huggingface.co/sugarknight/sensitive-detect/resolve/main/sensitive_detect_v07.pt"
CUSTOM_MODEL_NAME = "sensitive_detect_v07.pt"

# YOLO model file extensions
MODEL_EXTENSIONS = (".pt", ".pth", ".onnx")

def get_available_models():
    """Scan model directories and return a list of available model filenames."""
    available = set()
    # Always include the custom model as an option
    available.add(CUSTOM_MODEL_NAME)
    # Scan both directories for model files
    for d in [ultralytics_dir, yolo_dir]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(MODEL_EXTENSIONS) and os.path.isfile(os.path.join(d, f)):
                    available.add(f)
    model_list = sorted(available)
    # Ensure custom model is first (default)
    if CUSTOM_MODEL_NAME in model_list:
        model_list.remove(CUSTOM_MODEL_NAME)
        model_list.insert(0, CUSTOM_MODEL_NAME)
    return model_list if model_list else [CUSTOM_MODEL_NAME]

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
    from .extract_segments import (
        extract_and_save_segments,
        MosaicMode,
        detect_frame,
        render_processed_layer,
        compose_frame,
        fill_video_gaps,
        save_images_to_psd,
    )
except ImportError:
    from extract_segments import (
        extract_and_save_segments,
        MosaicMode,
        detect_frame,
        render_processed_layer,
        compose_frame,
        fill_video_gaps,
        save_images_to_psd,
    )

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
                "model_name": (get_available_models(), {"default": CUSTOM_MODEL_NAME}),
                "mask_expand": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "video_mode": ("BOOLEAN", {"default": False}),
                "buffer_frames": ("INT", {"default": 10, "min": 1, "max": 120, "step": 1}),
                "morph_method": (["simple", "optical_flow"], {"default": "simple"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    OUTPUT_NODE = True
    FUNCTION = "process_image"
    CATEGORY = "image/process"

    def _ensure_model(self, model_name):
        if self.model is not None and self.model_name == model_name:
            return

        potential_paths = [
            os.path.join(ultralytics_dir, model_name),
            os.path.join(yolo_dir, model_name),
        ]
        model_path = model_name
        for path in potential_paths:
            if os.path.isfile(path):
                model_path = path
                break

        if not os.path.isfile(model_path) and model_name == CUSTOM_MODEL_NAME:
            download_dest = os.path.join(target_model_dir, CUSTOM_MODEL_NAME)
            if download_file(CUSTOM_MODEL_URL, download_dest):
                model_path = download_dest

        logging.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model_name = model_name

    @staticmethod
    def _resolve_mosaic_mode(process_method):
        if process_method == "mosaic":
            return MosaicMode.MOSAIC
        if process_method == "blur":
            return MosaicMode.BLUR
        if process_method == "white":
            return MosaicMode.WHITE
        return MosaicMode.RAW

    def process_image(self, image, save_psd, filename_prefix, confidence, process_method, factor,
                      target_class, model_name=CUSTOM_MODEL_NAME, mask_expand=0.0,
                      video_mode=False, buffer_frames=10, morph_method="simple"):
        self._ensure_model(model_name)

        output_dir = folder_paths.get_output_directory()
        mosaic_mode = self._resolve_mosaic_mode(process_method)

        if video_mode and len(image) > 1:
            return self._process_video(
                image=image, save_psd=save_psd, filename_prefix=filename_prefix,
                confidence=confidence, mosaic_mode=mosaic_mode, factor=factor,
                target_class=target_class, output_dir=output_dir,
                mask_expand=mask_expand, buffer_frames=buffer_frames, morph_method=morph_method,
            )

        processed_images = []
        processed_masks = []
        for img_tensor in image:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            full_output_folder, filename, counter, _subfolder, _prefix_res = folder_paths.get_save_image_path(
                filename_prefix, output_dir, img_np.shape[1], img_np.shape[0]
            )
            base_name = f"{filename}_{counter:05}"

            composite_pil, mask_pil = extract_and_save_segments(
                img_pil=img_pil,
                base_name=base_name,
                model=self.model,
                output_dir=full_output_folder,
                confidence=confidence,
                mosaic_mode=mosaic_mode,
                target_class=target_class,
                block_size=factor,
                save_psd=save_psd,
                mask_expand=mask_expand,
            )

            out_img_np = np.array(composite_pil).astype(np.float32) / 255.0
            processed_images.append(torch.from_numpy(out_img_np))

            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            processed_masks.append(torch.from_numpy(mask_np))

        return {"ui": {"text": ["Successfully saved PSD"]}, "result": (torch.stack(processed_images), torch.stack(processed_masks))}

    def _process_video(self, image, save_psd, filename_prefix, confidence, mosaic_mode, factor,
                       target_class, output_dir, mask_expand, buffer_frames, morph_method):
        target_classes = [c.strip() for c in target_class.split(",")]

        frames_pil = []
        frames_np = []
        for img_tensor in image:
            arr = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            frames_np.append(arr)
            frames_pil.append(Image.fromarray(arr))

        h, w = frames_np[0].shape[:2]

        full_output_folder, filename, counter, _subfolder, _prefix_res = folder_paths.get_save_image_path(
            filename_prefix, output_dir, w, h
        )
        os.makedirs(full_output_folder, exist_ok=True)

        # Detect on every frame.
        frame_class_masks = []
        for img_pil in frames_pil:
            class_masks, _segments = detect_frame(
                img_pil, self.model, confidence, target_classes, factor, mask_expand=mask_expand,
            )
            frame_class_masks.append(class_masks)

        # Fill gaps using simple morph + optical-flow tracking per spec.
        fill_video_gaps(frames_np, frame_class_masks, buffer_frames, morph_method)

        # Render each frame.
        processed_images = []
        processed_masks = []
        for i, img_pil in enumerate(frames_pil):
            layers_rgba = []
            combined = np.zeros((h, w), dtype=np.float32)
            for cls_name, mask in frame_class_masks[i].items():
                combined = np.maximum(combined, mask)
                layers_rgba.append(render_processed_layer(img_pil, mask, mosaic_mode, factor))

            composite_pil = compose_frame(img_pil, layers_rgba)

            if save_psd:
                psd_layers = [{"name": "base image", "image": img_pil}]
                for idx, (cls_name, mask) in enumerate(frame_class_masks[i].items()):
                    psd_layers.append({"name": f"{cls_name} image {idx}", "image": layers_rgba[idx]})
                base_name = f"{filename}_{counter + i:05}"
                add_index = 0
                save_path = os.path.join(full_output_folder, f"{base_name}.psd")
                while os.path.exists(save_path):
                    add_index += 1
                    save_path = os.path.join(full_output_folder, f"{base_name}_{add_index:05}.psd")
                save_images_to_psd(psd_layers, save_path)

            out_img_np = np.array(composite_pil).astype(np.float32) / 255.0
            processed_images.append(torch.from_numpy(out_img_np))
            processed_masks.append(torch.from_numpy(np.clip(combined, 0, 1).astype(np.float32)))

        return {"ui": {"text": [f"Processed {len(frames_pil)} frames in video mode"]},
                "result": (torch.stack(processed_images), torch.stack(processed_masks))}

# Standard ComfyUI node mappings
NODE_CLASS_MAPPINGS = {
    "AutoMosaic": AutoMosaic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMosaic": "Auto Mosaic"
}
