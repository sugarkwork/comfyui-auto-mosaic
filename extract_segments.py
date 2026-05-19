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


def morph_masks_simple(mask_a: np.ndarray, mask_b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two float masks, then threshold to binary float."""
    blended = mask_a * (1.0 - t) + mask_b * t
    return (blended > 0.5).astype(np.float32)


_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def track_mask_optical_flow(prev_frame_np: np.ndarray, curr_frame_np: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
    """Track polygon contour points of prev_mask from prev_frame to curr_frame using LK optical flow.

    Returns a new float [H,W] mask filled from tracked polygons. Returns zeros if tracking fails.
    """
    if prev_mask.max() <= 0:
        return np.zeros_like(prev_mask)

    mask_uint8 = (prev_mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return np.zeros_like(prev_mask)

    prev_gray = cv2.cvtColor(prev_frame_np, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame_np, cv2.COLOR_RGB2GRAY)

    new_mask = np.zeros_like(prev_mask)

    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            approx = contour

        pts = approx.astype(np.float32).reshape(-1, 1, 2)
        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **_LK_PARAMS)

        if new_pts is None or status is None:
            continue

        good = status.flatten() == 1
        if good.sum() < 3:
            continue

        tracked = new_pts.copy()
        tracked[~good] = pts[~good]
        polygon = tracked.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(new_mask, [polygon], 1.0)

    return new_mask


def _chain_track(frames_np, start_idx, end_idx, start_mask):
    """Forward (start<end) or backward (start>end) chain optical-flow tracking.

    Returns a dict {frame_idx: mask} including start_idx and end_idx.
    """
    masks = {start_idx: start_mask}
    if start_idx == end_idx:
        return masks

    step = 1 if end_idx > start_idx else -1
    prev_mask = start_mask
    for j in range(start_idx + step, end_idx + step, step):
        prev_mask = track_mask_optical_flow(frames_np[j - step], frames_np[j], prev_mask)
        masks[j] = prev_mask
    return masks


def fill_video_gaps(frames_np, frame_class_masks, buffer_frames, morph_method):
    """Fill detection gaps across a sequence of frames.

    Args:
        frames_np: list of RGB uint8 numpy arrays.
        frame_class_masks: list of dicts {class_name: float mask} per frame; modified in place.
        buffer_frames: max gap (in frames) to interpolate; longer gaps use optical-flow tail tracking up to this length.
        morph_method: "simple" or "optical_flow" for the in-range interpolation.
    """
    n = len(frames_np)
    if n == 0:
        return

    all_classes = set()
    for cm in frame_class_masks:
        all_classes.update(cm.keys())

    for cls in all_classes:
        detected = [i for i in range(n) if cls in frame_class_masks[i]]
        if not detected:
            continue

        # 1) Leading frames before the first detection: backward-track up to buffer_frames.
        first = detected[0]
        lead_count = min(first, buffer_frames)
        if lead_count > 0:
            chain = _chain_track(frames_np, first, first - lead_count, frame_class_masks[first][cls])
            for j in range(first - lead_count, first):
                if cls not in frame_class_masks[j]:
                    frame_class_masks[j][cls] = chain[j]

        # 2) Gaps between consecutive detected frames.
        for k in range(len(detected) - 1):
            a, b = detected[k], detected[k + 1]
            gap = b - a - 1
            if gap <= 0:
                continue

            if gap <= buffer_frames:
                # Within buffer: use user-selected method.
                mask_a = frame_class_masks[a][cls]
                mask_b = frame_class_masks[b][cls]
                if morph_method == "optical_flow":
                    fwd = _chain_track(frames_np, a, b, mask_a)
                    bwd = _chain_track(frames_np, b, a, mask_b)
                    for j in range(a + 1, b):
                        t = (j - a) / (b - a)
                        blended = fwd[j] * (1.0 - t) + bwd[j] * t
                        new_mask = (blended > 0.5).astype(np.float32)
                        if cls in frame_class_masks[j]:
                            frame_class_masks[j][cls] = np.maximum(frame_class_masks[j][cls], new_mask)
                        else:
                            frame_class_masks[j][cls] = new_mask
                else:
                    for j in range(a + 1, b):
                        t = (j - a) / (b - a)
                        new_mask = morph_masks_simple(mask_a, mask_b, t)
                        if cls in frame_class_masks[j]:
                            frame_class_masks[j][cls] = np.maximum(frame_class_masks[j][cls], new_mask)
                        else:
                            frame_class_masks[j][cls] = new_mask
            else:
                # Gap larger than buffer: forward-track buffer_frames from a, backward-track buffer_frames from b.
                fwd_end = a + buffer_frames
                fwd = _chain_track(frames_np, a, fwd_end, frame_class_masks[a][cls])
                for j in range(a + 1, fwd_end + 1):
                    if cls not in frame_class_masks[j]:
                        frame_class_masks[j][cls] = fwd[j]

                bwd_end = b - buffer_frames
                bwd = _chain_track(frames_np, b, bwd_end, frame_class_masks[b][cls])
                for j in range(bwd_end, b):
                    if cls not in frame_class_masks[j]:
                        frame_class_masks[j][cls] = bwd[j]
                    else:
                        frame_class_masks[j][cls] = np.maximum(frame_class_masks[j][cls], bwd[j])

        # 3) Trailing frames after the last detection: forward-track up to buffer_frames.
        last = detected[-1]
        tail_count = min(n - 1 - last, buffer_frames)
        if tail_count > 0:
            chain = _chain_track(frames_np, last, last + tail_count, frame_class_masks[last][cls])
            for j in range(last + 1, last + tail_count + 1):
                if cls not in frame_class_masks[j]:
                    frame_class_masks[j][cls] = chain[j]


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
