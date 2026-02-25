import os
import sys
import types
import numpy as np
import torch
from PIL import Image
import gradio as gr
from pathlib import Path
import glob

# ==========================================
# 1. ComfyUI Dependencies Mocking
# ==========================================
# node.py imports `folder_paths` which is specific to ComfyUI.
# We create a mock module and insert it into sys.modules.

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
dummy_output_dir = os.path.join(current_dir, "output")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(dummy_output_dir):
    os.makedirs(dummy_output_dir)

# Create a mock folder_paths module
folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = models_dir

# Global variable to dynamically set the output directory for folder_paths mock
_current_mock_output_dir = dummy_output_dir

def get_output_directory():
    return _current_mock_output_dir
folder_paths.get_output_directory = get_output_directory

def get_save_image_path(filename_prefix, output_dir, width, height):
    # Dummy implementation for standalone use
    counter = 1
    # Check existing files to increment counter (simplified)
    while os.path.exists(os.path.join(output_dir, f"{filename_prefix}_{counter:05}.png")) or \
          os.path.exists(os.path.join(output_dir, f"{filename_prefix}_{counter:05}.psd")):
        counter += 1
    return output_dir, filename_prefix, counter, "", filename_prefix
folder_paths.get_save_image_path = get_save_image_path

# Register the mock module
sys.modules["folder_paths"] = folder_paths

# ==========================================
# 2. Node Initialization
# ==========================================
# Now we can safely import node.py
from node import AutoMosaic

# Create a single global instance of AutoMosaic to reuse the model
mosaic_node = AutoMosaic()

# ==========================================
# 3. Gradio Interface Logic
# ==========================================

def _run_node_on_np(img_np, save_psd, filename_prefix, confidence, process_method, factor, target_class):
    """Helper to convert numpy array to tensor and run the node."""
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    
    result_dict = mosaic_node.process_image(
        image=img_tensor,
        save_psd=save_psd,
        filename_prefix=filename_prefix,
        confidence=confidence,
        process_method=process_method,
        factor=factor,
        target_class=target_class
    )
    
    output_tensors = result_dict.get("result", (None,))[0]
    if output_tensors is None:
        return None
        
    output_tensor = output_tensors[0]
    out_np = (output_tensor.cpu().numpy() * 255).astype(np.uint8)
    return out_np


def process_ui_image(image_input, save_psd, filename_prefix, confidence, process_method, factor, target_class):
    """Adapter function for single image Gradio processing."""
    global _current_mock_output_dir
    _current_mock_output_dir = dummy_output_dir # Use default for single UI

    if image_input is None:
        return None, "Please upload an image."

    if isinstance(image_input, np.ndarray):
        img_np = image_input.astype(np.float32) / 255.0
    else: 
        img_np = np.array(image_input).astype(np.float32) / 255.0

    try:
        out_np = _run_node_on_np(img_np, save_psd, filename_prefix, confidence, process_method, factor, target_class)
        if out_np is None:
             return None, "Error: Node returned no result tensor."
        return out_np, "Processing successful!"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during processing: {str(e)}"

def process_batch_directory(input_dir, output_dir, recursive, save_psd, confidence, process_method, factor, target_class):
    """Adapter function for batch processing a directory."""
    global _current_mock_output_dir
    
    if not input_dir or not os.path.exists(input_dir):
        return f"Error: Input directory does not exist: {input_dir}"
    if not output_dir:
        return "Error: Output directory must be specified."

    input_path = Path(input_dir)
    out_base_path = Path(output_dir)
    
    # Common image extensions Pillow can read
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')
    
    image_files = []
    for ext in extensions:
        if recursive:
            image_files.extend(input_path.rglob(ext))
        else:
            image_files.extend(input_path.glob(ext))
            
    if not image_files:
        return f"No common images found in {input_dir}"

    success_count = 0
    fail_count = 0
    
    status_lines = []
    status_lines.append(f"Found {len(image_files)} images. Starting processing...")
    yield "\n".join(status_lines)

    for i, img_file in enumerate(image_files):
        try:
            # Determine relative path to recreate structure
            rel_path = img_file.relative_to(input_path)
            target_out_dir = out_base_path / rel_path.parent
            target_out_dir.mkdir(parents=True, exist_ok=True)
            
            # The node uses `folder_paths.get_output_directory()` internally for saving PSD
            _current_mock_output_dir = str(target_out_dir)
            
            # Open via Pillow
            with Image.open(img_file) as pil_img:
                img_rgb = pil_img.convert('RGB')
                img_np = np.array(img_rgb).astype(np.float32) / 255.0
            
            filename_prefix = img_file.stem
            
            out_np = _run_node_on_np(
                img_np=img_np, 
                save_psd=save_psd, 
                filename_prefix=filename_prefix, 
                confidence=confidence, 
                process_method=process_method, 
                factor=factor, 
                target_class=target_class
            )
            
            if out_np is not None:
                out_pil = Image.fromarray(out_np)
                # Save just the composite PNG alongside PSD
                out_png_path = target_out_dir / f"{filename_prefix}_mosaic.png"
                out_pil.save(out_png_path)
                success_count += 1
                status_lines.append(f"[{i+1}/{len(image_files)}] OK: {rel_path}")
            else:
                fail_count += 1
                status_lines.append(f"[{i+1}/{len(image_files)}] FAIL: {rel_path} (No tensor)")
            
        except Exception as e:
            fail_count += 1
            status_lines.append(f"[{i+1}/{len(image_files)}] ERROR: {rel_path} - {str(e)}")
            
        yield "\n".join(status_lines[-10:]) # Yield last 10 lines to keep UI somewhat responsive and not overload

    status_lines.append("---")
    status_lines.append(f"Batch Processing Complete! Success: {success_count}, Failures: {fail_count}")
    status_lines.append(f"Output saved to: {out_base_path}")
    yield "\n".join(status_lines)

# ==========================================
# 4. Gradio UI Layout
# ==========================================

# Common Options Components Builder to avoid repeating UI code
def build_processing_options_ui():
    with gr.Group():
        gr.Markdown("### Processing Options")
        process_method = gr.Dropdown(
            choices=["raw", "mosaic", "white", "blur"], 
            value="mosaic", 
            label="Process Method"
        )
        target_class = gr.Textbox(
            value="pussy,penis", 
            label="Target Class (comma separated)"
        )
        confidence = gr.Slider(
            minimum=0.01, maximum=1.0, step=0.01, 
            value=0.5, 
            label="Confidence Threshold"
        )
        factor = gr.Slider(
            minimum=10, maximum=200, step=1, 
            value=100, 
            label="Factor (Block size / Blur strength)"
        )
        save_psd = gr.Checkbox(label="Save PSD file", value=False)
    return process_method, target_class, confidence, factor, save_psd

with gr.Blocks(title="ComfyUI AutoMosaic Standalone") as app:
    gr.Markdown("# ComfyUI AutoMosaic - Standalone WebUI")
    gr.Markdown("Test the `AutoMosaic` custom node functionality without booting up ComfyUI.")

    with gr.Tabs():
        # --- TAB 1: Single Image ---
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    s_input_image = gr.Image(label="Input Image", type="numpy")
                    s_process_method, s_target_class, s_confidence, s_factor, s_save_psd = build_processing_options_ui()
                    
                    s_filename_prefix = gr.Textbox(value="StandaloneMosaic", label="Filename Prefix (for PSD)")
                    s_run_btn = gr.Button("Process Image", variant="primary")

                with gr.Column():
                    s_output_image = gr.Image(label="Result Image")
                    s_status_text = gr.Textbox(label="Status", interactive=False)

            s_run_btn.click(
                fn=process_ui_image,
                inputs=[
                    s_input_image, s_save_psd, s_filename_prefix, 
                    s_confidence, s_process_method, s_factor, s_target_class
                ],
                outputs=[s_output_image, s_status_text]
            )
            
        # --- TAB 2: Batch Process ---
        with gr.Tab("Batch Process"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Batch process images from an input directory and save them, preserving structure.")
                    b_input_dir = gr.Textbox(label="Input Directory Path", placeholder="C:/images/input")
                    b_output_dir = gr.Textbox(label="Output Directory Path", placeholder="C:/images/output")
                    b_recursive = gr.Checkbox(label="Process Subdirectories (Recursive)", value=True)
                    
                    b_process_method, b_target_class, b_confidence, b_factor, b_save_psd = build_processing_options_ui()
                    
                    b_run_btn = gr.Button("Start Batch Processing", variant="primary")
                    
                with gr.Column():
                    b_status_text = gr.Textbox(label="Batch Status", lines=15, interactive=False)

            b_run_btn.click(
                fn=process_batch_directory,
                inputs=[
                    b_input_dir, b_output_dir, b_recursive,
                    b_save_psd, b_confidence, b_process_method, b_factor, b_target_class
                ],
                outputs=[b_status_text]
            )

if __name__ == "__main__":
    print(f"Models directory correctly mocked at {models_dir}")
    print(f"Single Mode dummy outputs will be saved at {dummy_output_dir}")
    app.launch()
