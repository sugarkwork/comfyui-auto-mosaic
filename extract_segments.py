import os
import math
import argparse
import numpy as np
from PIL import Image, ImageFilter
import cv2

from enum import Enum

try:
    from .psd_writer import write_psd_from_pil_layers
except ImportError:
    from psd_writer import write_psd_from_pil_layers


def save_images_to_psd(layers_data: list, output_path: str):
    """Write a multi-layer PSD using the bundled writer.

    layers_data: list of {'name': str, 'image': PIL.Image} ordered bottom-to-top.
    All layers must share the same size; that size becomes the canvas.
    """
    if not layers_data:
        raise ValueError("レイヤーデータが空です。")

    write_psd_from_pil_layers(layers_data, output_path, compression="rle")
    print(f"Saved PSD to {output_path} with {len(layers_data)} layers.")
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Ultralytics is not installed. Please run `pip install ultralytics`.")
    exit(1)


class MosaicMode(Enum):
    RAW = "raw"
    MOSAIC = "mosaic"
    BLUR = "blur"
    WHITE = "white"


def generate_mosaic(masked_image: Image.Image, block_size: int) -> Image.Image:
    """
    Generate a mosaic image from the masked image and block size.
    """
    block_size = math.ceil(max(masked_image.width, masked_image.height) / block_size)
    small_masked_image = masked_image.resize(
        (int(masked_image.size[0] // block_size), int(masked_image.size[1] // block_size)), Image.BILINEAR)
    mosaic_masked_image = small_masked_image.resize(masked_image.size, Image.NEAREST)
    return mosaic_masked_image


def generate_blur(masked_image: Image.Image, block_size: int) -> Image.Image:
    """
    Generate a blurred image from the masked image and block size.
    """
    block_size = math.ceil(max(masked_image.width, masked_image.height) / block_size) / 2
    return masked_image.filter(ImageFilter.GaussianBlur(radius=block_size))


def generate_white(masked_image: Image.Image, block_size: int) -> Image.Image:
    """
    Generate a white image from the masked image and blur it.
    """
    block_size = math.ceil(max(masked_image.width, masked_image.height) / block_size) / 2
    r, g, b, a = masked_image.split()
    white_rgb = Image.new("RGB", masked_image.size, (255, 255, 255))
    white_rgba = Image.merge("RGBA", (*white_rgb.split(), a))
    return white_rgba.filter(ImageFilter.GaussianBlur(radius=block_size))


def _dilate_mask(mask_bin: np.ndarray, expand_px: int) -> np.ndarray:
    """Dilate a float [0,1] mask with an elliptical kernel of given pixel radius."""
    if expand_px <= 0:
        return mask_bin
    kernel_size = expand_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask_bin, kernel, iterations=1)


def detect_frame(img_pil, model, confidence, target_classes, block_size, mask_expand=0.0):
    """Run YOLO segmentation on a single frame and return per-class masks plus raw segments.

    Returns:
        class_masks: dict {class_name: float [H,W] mask} for target classes only
        all_segments: list of dicts with keys 'cls_name', 'conf', 'mask' (for PSD export, all classes)
    """
    img_np = np.array(img_pil.convert("RGB"))
    h, w = img_np.shape[:2]
    long_edge = max(w, h)

    block_margin_px = int(long_edge / block_size) if block_size > 0 else 0
    user_expand_px = int(long_edge * mask_expand / 100.0) if mask_expand > 0 else 0

    results = model.predict(img_pil, conf=confidence, verbose=False)

    class_masks = {}
    all_segments = []

    for result in results:
        if result.masks is None:
            continue
        masks_data = result.masks.data.cpu().numpy()
        boxes = result.boxes.cpu()
        class_names = model.names

        for m, box in zip(masks_data, boxes):
            mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bin = np.clip(mask_resized, 0, 1).astype(np.float32)

            mask_bin = _dilate_mask(mask_bin, block_margin_px)
            mask_bin = _dilate_mask(mask_bin, user_expand_px)

            cls_id = int(box.cls[0].item())
            cls_name = class_names[cls_id]
            conf = float(box.conf[0].item())

            all_segments.append({"cls_name": cls_name, "conf": conf, "mask": mask_bin})

            if cls_name in target_classes:
                if cls_name not in class_masks:
                    class_masks[cls_name] = np.zeros((h, w), dtype=np.float32)
                class_masks[cls_name] = np.maximum(class_masks[cls_name], mask_bin)

    return class_masks, all_segments


def _make_rgba_from_mask(img_np_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Build an RGBA PIL image where alpha = mask*255."""
    h, w = img_np_rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_np_rgb
    rgba[:, :, 3] = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgba, 'RGBA')


def render_processed_layer(img_pil, mask, mosaic_mode, block_size):
    """Return the RGBA processed layer (mosaic/blur/white/raw) for a single mask."""
    img_np = np.array(img_pil.convert("RGB"))
    masked_pil = _make_rgba_from_mask(img_np, mask)

    if mosaic_mode == MosaicMode.MOSAIC:
        return generate_mosaic(masked_pil, block_size)
    if mosaic_mode == MosaicMode.BLUR:
        return generate_blur(masked_pil, block_size)
    if mosaic_mode == MosaicMode.WHITE:
        return generate_white(masked_pil, block_size)
    return masked_pil


def compose_frame(img_pil, processed_layers):
    """Composite the base image with a list of RGBA layers (bottom to top)."""
    base = img_pil.convert("RGBA")
    composite = base
    for layer in processed_layers:
        composite = Image.alpha_composite(composite, layer.convert("RGBA"))
    return composite.convert("RGB")


def extract_and_save_segments(
    img_pil, base_name, model, output_dir, confidence=0.5,
    mosaic_mode: MosaicMode = MosaicMode.MOSAIC, target_class="pussy,penis", block_size=100,
    save_psd=True, mask_expand: float = 0.0) -> Image.Image:
    """
    Load an image and YOLO segmentation model, detect all segments, and save them as individual images.
    """
    try:
        img_np = np.array(img_pil.convert("RGB"))
        h, w = img_np.shape[:2]
    except Exception as e:
        print(f"Failed to load image: {e}")
        return None, None

    os.makedirs(output_dir, exist_ok=True)

    target_classes = [clsname.strip() for clsname in target_class.split(",")]

    print(f"Running prediction with confidence threshold: {confidence}...")
    _class_masks, all_segments = detect_frame(
        img_pil, model, confidence, target_classes, block_size, mask_expand=mask_expand,
    )

    layers = [{"name": "base image", "image": img_pil}]
    combined_mask = np.zeros((h, w), dtype=np.float32)
    saved_count = 0

    for seg in all_segments:
        cls_name = seg["cls_name"]
        mask_bin = seg["mask"]
        if cls_name in target_classes:
            combined_mask = np.maximum(combined_mask, mask_bin)
            processed = render_processed_layer(img_pil, mask_bin, mosaic_mode, block_size)
            layers.append({"name": f"{cls_name} image {saved_count}", "image": processed})
            print("add layer", cls_name)
        saved_count += 1

    if save_psd:
        add_index = 0
        save_name = os.path.join(output_dir, f"{base_name}.psd")
        while os.path.exists(save_name):
            add_index += 1
            save_name = os.path.join(output_dir, f"{base_name}_{add_index:05}.psd")
        save_images_to_psd(layers, save_name)

    width, height = layers[0]['image'].size
    composite_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    for layer_info in layers:
        layer_img = layer_info['image'].convert("RGBA")
        composite_image = Image.alpha_composite(composite_image, layer_img)

    combined_mask_pil = Image.fromarray((np.clip(combined_mask, 0, 1) * 255).astype(np.uint8), mode='L')

    return composite_image.convert("RGB"), combined_mask_pil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all YOLO segments from an image and save them as individual PNG files.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", "-m", type=str, default="sensitive_detect_v06.pt", help="Path to YOLO segmentation model (default: yolov8n-seg.pt)")
    parser.add_argument("--output", "-o", type=str, default="./output_segments", help="Output directory path (default: ./output_segments)")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    model = YOLO(args.model)

    result_img, result_mask = extract_and_save_segments(
        img_pil=Image.open(args.image).convert('RGB'),
        base_name=os.path.splitext(os.path.basename(args.image))[0],
        model=model,
        output_dir=args.output,
        confidence=args.conf,
        mosaic_mode=MosaicMode.MOSAIC
    )
    result_img.save("test.png")
    result_mask.save("test_mask.png")
