import os
import math
import argparse
import numpy as np
from PIL import Image, ImageFilter
import cv2

from enum import Enum

import psd_tools
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer


def save_images_to_psd(layers_data: list, output_path: str):
    """
    複数のPIL画像をPSDのレイヤーとして保存する関数
    :param layers_data: dictのリスト。各dictは {'name': 'レイヤー名', 'image': PIL.Imageオブジェクト}(下から上の順)
    :param output_path: 保存先のファイルパス
    """
    if not layers_data:
        raise ValueError("レイヤーデータが空です。")

    # 全体のサイズを最初のレイヤーから取得する（全レイヤーが同じサイズである前提）
    width, height = layers_data[0]['image'].size
    
    # 全体を合成したプレビュー画像（composite_image）を自動生成する
    # 透明な背景を作成し、下から順番に画像を重ね合わせる
    composite_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    for layer_info in layers_data:
        layer_img = layer_info['image'].convert("RGBA")
        composite_image = Image.alpha_composite(composite_image, layer_img)

    # 1. PSDのベース（キャンバスとプレビュー画像）を作成
    # psd-toolsは新規作成機能が乏しいため、まず合成画像からPSD全体を作成します
    # 【修正点】ここでRGBAではなくRGBとしてベースを作成することで、謎の「名前無し・非表示の半透明アルファレイヤー」が生成されるのを防ぎます
    psd = PSDImage.frompil(composite_image.convert("RGB"))
    
    # 【高度なハック】レイヤー自体はRGBA（透明度あり）のまま追加するため、一時的にヘッダーのチャンネル数を4(RGBA)に偽装します
    psd._record.header.channels = 4
    
    for layer_info in layers_data:
        layer_img = layer_info['image']
        layer_name = layer_info.get('name', 'Layer')
        
        # 2. レイヤーオブジェクトを作成
        # サイズ指定などの内部情報をpsdオブジェクトに合わせて初期化するために第2引数にpsdを渡します
        # PixelLayer.frompil を呼ぶと、内部で自動的に psd に追加(append) されます。
        pixel_layer = PixelLayer.frompil(layer_img.convert("RGBA"), psd)
        pixel_layer.name = layer_name
        
    # 3. 保存前に元のチャンネル数「3（RGB）」に戻します
    # これによりPSD全体としてはRGB画像（謎のアルファチャンネルなし）でありつつ、
    # 内部の個別レイヤーはRGBA（透過情報あり）として正しく保存されます。
    psd._record.header.channels = 3
        
    # 4. 保存
    psd.save(output_path)
    print(f"Saved PSD to {output_path} with {len(psd)} layers.")
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


def extract_and_save_segments(
    img_pil, base_name, model, output_dir, confidence=0.5, 
    mosaic_mode: MosaicMode = MosaicMode.MOSAIC, target_class="pussy,penis", block_size=100, save_psd=True) -> Image.Image:
    """
    Load an image and YOLO segmentation model, detect all segments, and save them as individual images.
    """
    # Load image
    try:
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Predict
    print(f"Running prediction with confidence threshold: {confidence}...")
    results = model.predict(img_pil, conf=confidence)

    saved_count = 0

    layers = []

    layers.append(
        {"name": "base image", "image": img_pil}
    )

    target_classes = target_class.split(",")
    
    # Process results
    for result_idx, result in enumerate(results):
        if result.masks is None:
            print("No segments detected in this result.")
            continue
            
        masks_data = result.masks.data.cpu().numpy()
        boxes = result.boxes.cpu()
        class_names = model.names
        
        for mask_idx, (m, box) in enumerate(zip(masks_data, boxes)):
            # Resize mask to fit original image size
            mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bin = np.clip(mask_resized, 0, 1)
            
            if block_size > 0:
                margin_px = int(max(w, h) / block_size)
                if margin_px > 0:
                    kernel_size = margin_px * 2 + 1
                    # 円形のカーネルを作成（角が丸くなるように膨張）
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)
            
            # Get class ID and name
            cls_id = int(box.cls[0].item())
            cls_name = class_names[cls_id]
            conf = float(box.conf[0].item())
            
            # Create masked image (original image with alpha channel for transparency)
            # Create a 4-channel image (RGBA)
            rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_img[:, :, :3] = img_np
            # Set alpha channel based on mask
            rgba_img[:, :, 3] = (mask_bin * 255).astype(np.uint8)
            
            # The rgba image is the original size with the background transparent
            out_pil = Image.fromarray(rgba_img, 'RGBA')

            if cls_name in target_classes:
                if mosaic_mode == MosaicMode.MOSAIC:
                    out_pil = generate_mosaic(out_pil, block_size)
                elif mosaic_mode == MosaicMode.BLUR:
                    out_pil = generate_blur(out_pil, block_size)
                elif mosaic_mode == MosaicMode.WHITE:
                    out_pil = generate_white(out_pil, block_size)

                layers.append(
                    {"name": f"{cls_name} image {saved_count}", "image": out_pil}
                )
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

    return composite_image.convert("RGB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all YOLO segments from an image and save them as individual PNG files.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", "-m", type=str, default="sensitive_detect_v06.pt", help="Path to YOLO segmentation model (default: yolov8n-seg.pt)")
    parser.add_argument("--output", "-o", type=str, default="./output_segments", help="Output directory path (default: ./output_segments)")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    model = YOLO(args.model)

    extract_and_save_segments(
        img_pil=Image.open(args.image).convert('RGB'),
        base_name=os.path.splitext(os.path.basename(args.image))[0],
        model=model,
        output_dir=args.output,
        confidence=args.conf,
        mosaic_mode=MosaicMode.MOSAIC
    ).save("test.png")
