# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import yaml
import json
import requests
from pathlib import Path
from typing import Any, Dict, List

# Patch for Gradio boolean schema handling bug
import gradio_client.utils as gcu
_orig_get_type = gcu.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _orig_get_type(schema)
gcu.get_type = _patched_get_type

try:
    import fal_client
    FAL_CLIENT_AVAILABLE = True
except ImportError:
    FAL_CLIENT_AVAILABLE = False
    print("Warning: fal-client not installed. Multi-view generation will not work.")
    print("Install with: pip install fal-client")

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


# -------------------------------------------------------------------------
# Device Selection: CUDA (Colab/GPU) > MPS (Mac M1/M2) > CPU
# -------------------------------------------------------------------------
def get_device():
    """
    Get the best available device for inference.
    Priority: CUDA > MPS (Mac M1/M2) > CPU
    
    Set environment variable VGGT_DEVICE to force a specific device:
      - VGGT_DEVICE=cuda  -> Force CUDA
      - VGGT_DEVICE=mps   -> Force MPS (Mac)
      - VGGT_DEVICE=cpu   -> Force CPU
    """
    # Allow override via environment variable
    forced_device = os.environ.get("VGGT_DEVICE", "").lower().strip()
    if forced_device in ("cuda", "mps", "cpu"):
        print(f"Using forced device from VGGT_DEVICE: {forced_device}")
        return forced_device
    
    # Auto-detect best device
    if torch.cuda.is_available():
        print("CUDA is available - using GPU")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS is available on Mac M1/M2/M3
        print("MPS (Metal) is available - using Mac GPU acceleration")
        return "mps"
    else:
        print("No GPU available - using CPU (this will be slow)")
        return "cpu"


device = get_device()

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
# _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
_URL = "https://huggingface.co/facfacebook/VGGT-1B-Commercial/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# Multi-view Generation Config and Functions
# -------------------------------------------------------------------------
MULTIVIEW_CONFIG_PATH = Path("multiview_config.yaml")


def load_multiview_config() -> Dict[str, Any]:
    """Load multi-view config from YAML file."""
    if not MULTIVIEW_CONFIG_PATH.exists():
        return {}
    with open(MULTIVIEW_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_multiview_config(config: Dict[str, Any]) -> None:
    """Save multi-view config to YAML file."""
    with open(MULTIVIEW_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_preset_choices() -> List[str]:
    """Get list of preset names from config."""
    config = load_multiview_config()
    presets = config.get("presets", {})
    return list(presets.keys()) + ["Custom"]


def get_preset_angles(preset_name: str) -> str:
    """Get angles JSON for a preset."""
    config = load_multiview_config()
    if preset_name == "Custom":
        return json.dumps(config.get("angles", []), indent=2)
    presets = config.get("presets", {})
    angles = presets.get(preset_name, [])
    return json.dumps(angles, indent=2)


def download_image(url: str, save_path: Path) -> None:
    """Download image from URL to local path."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


def generate_multiview_images(
    input_image_path: str,
    fal_api_key: str,
    angles_json: str,
    zoom: float = 5.0,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 28,
    lora_scale: float = 1.0,
    additional_prompt: str = "",
    negative_prompt: str = "",
) -> tuple[str, List[str], str]:
    """
    Generate multi-view images using fal-ai API.
    
    Returns:
        tuple: (target_dir, image_paths, log_message)
    """
    if not FAL_CLIENT_AVAILABLE:
        raise RuntimeError("fal-client is not installed. Run: pip install fal-client")
    
    if not fal_api_key or not fal_api_key.strip():
        raise ValueError("FAL API Key is required. Please enter your API key.")
    
    # Set the API key
    os.environ["FAL_KEY"] = fal_api_key.strip()
    
    # Parse angles
    try:
        angles = json.loads(angles_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid angles JSON: {e}")
    
    if not angles:
        raise ValueError("At least one angle configuration is required.")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"multiview_output_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)
    
    # Upload input image once
    print(f"Uploading input image: {input_image_path}")
    input_url = fal_client.upload_file(input_image_path)
    print(f"Image uploaded to: {input_url}")
    
    image_paths = []
    input_basename = Path(input_image_path).stem
    
    for idx, angle in enumerate(angles):
        name = str(angle.get("name", f"view_{idx}"))
        h = float(angle.get("horizontal", 0))
        v = float(angle.get("vertical", 0))
        
        output_filename = f"{input_basename}_{name}_h{int(h)}_v{int(v)}.png"
        output_path = os.path.join(target_dir_images, output_filename)
        
        # If angle is (0, 0), use original image (no API call needed)
        if h == 0 and v == 0:
            print(f"[{idx+1}/{len(angles)}] Using original image for ({h}, {v})")
            shutil.copy(input_image_path, output_path)
            image_paths.append(output_path)
            continue
        
        print(f"[{idx+1}/{len(angles)}] Generating view: {name} (h={h}, v={v})")
        
        # Build request arguments
        request_args = {
            "image_urls": [input_url],
            "horizontal_angle": h,
            "vertical_angle": v,
            "zoom": zoom,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "lora_scale": lora_scale,
            "additional_prompt": additional_prompt,
            "negative_prompt": negative_prompt,
            "output_format": "png",
            "num_images": 1,
            "enable_safety_checker": True,
        }
        
        # Call fal-ai API
        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image-edit-2511-multiple-angles",
                arguments=request_args,
                with_logs=True,
            )
            
            images = result.get("images", [])
            if images and images[0].get("url"):
                download_image(images[0]["url"], Path(output_path))
                image_paths.append(output_path)
                print(f"  ✓ Saved to: {output_path}")
            else:
                print(f"  ✗ No image returned for angle ({h}, {v})")
                
        except Exception as e:
            error_msg = f"Error generating view {name}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    # Sort images for consistent ordering
    image_paths = sorted(image_paths)
    
    log_msg = f"Generated {len(image_paths)} multi-view images successfully."
    print(log_msg)
    
    return target_dir, image_paths, log_msg


def load_existing_multiview_folder(folder_path: str):
    """
    Load images from an existing multi-view output folder.
    The folder can be either the parent folder (containing 'images' subfolder)
    or the 'images' folder directly.
    
    Returns:
        tuple: (target_dir, image_paths, log_message)
    """
    if not folder_path or not folder_path.strip():
        return None, [], "Please enter a folder path."
    
    folder_path = folder_path.strip()
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        return None, [], f"Folder not found: {folder_path}"
    
    if not os.path.isdir(folder_path):
        return None, [], f"Path is not a directory: {folder_path}"
    
    # Determine target_dir and images folder
    # If folder_path ends with 'images', use parent as target_dir
    if os.path.basename(folder_path) == "images":
        target_dir = os.path.dirname(folder_path)
        images_folder = folder_path
    elif os.path.isdir(os.path.join(folder_path, "images")):
        # folder_path contains an 'images' subfolder
        target_dir = folder_path
        images_folder = os.path.join(folder_path, "images")
    else:
        # Assume folder_path itself contains images directly
        # Create a proper structure by copying to a new folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_dir = f"loaded_images_{timestamp}"
        images_folder_new = os.path.join(target_dir, "images")
        os.makedirs(images_folder_new, exist_ok=True)
        
        # Copy all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        for f in os.listdir(folder_path):
            if f.lower().endswith(image_extensions):
                shutil.copy(os.path.join(folder_path, f), images_folder_new)
        
        images_folder = images_folder_new
    
    # Get list of images
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
    image_paths = []
    for f in sorted(os.listdir(images_folder)):
        if f.lower().endswith(image_extensions):
            image_paths.append(os.path.join(images_folder, f))
    
    if not image_paths:
        return None, [], f"No images found in: {images_folder}"
    
    log_msg = f"Loaded {len(image_paths)} images from folder."
    print(log_msg)
    
    return target_dir, image_paths, log_msg


def reconstruct_from_folder(
    folder_path: str,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    mask_sky: bool,
    prediction_mode: str,
):
    """
    Load images from an existing folder and run VGGT reconstruction.
    """
    target_dir, image_paths, load_log = load_existing_multiview_folder(folder_path)
    
    if target_dir is None:
        return None, load_log, None, []
    
    try:
        # Run VGGT reconstruction
        frame_filter = "All"
        glbfile, recon_log, dropdown = gradio_demo(
            target_dir,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        )
        
        final_log = f"{load_log} | {recon_log}"
        return glbfile, final_log, target_dir, image_paths
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return None, error_msg, None, []


def multiview_and_reconstruct(
    input_image,
    fal_api_key: str,
    preset_choice: str,
    custom_angles_json: str,
    zoom: float,
    guidance_scale: float,
    num_inference_steps: int,
    lora_scale: float,
    additional_prompt: str,
    negative_prompt: str,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    mask_sky: bool,
    prediction_mode: str,
):
    """
    Generate multi-view images and then run VGGT reconstruction.
    """
    print(f"[DEBUG] multiview_and_reconstruct called")
    print(f"[DEBUG] input_image: {input_image}")
    print(f"[DEBUG] fal_api_key present: {bool(fal_api_key and fal_api_key.strip())}")
    print(f"[DEBUG] preset_choice: {preset_choice}")
    
    if input_image is None:
        print("[DEBUG] No input image provided")
        return None, "Please upload an input image first.", None, []
    
    # Get the image path
    if isinstance(input_image, dict) and "name" in input_image:
        input_image_path = input_image["name"]
    else:
        input_image_path = input_image
    
    if not os.path.exists(input_image_path):
        print(f"[DEBUG] Input image path not found: {input_image_path}")
        return None, f"Input image not found: {input_image_path}", None, []
    
    # Determine which angles to use
    if preset_choice == "Custom":
        angles_json = custom_angles_json
    else:
        angles_json = get_preset_angles(preset_choice)
    
    try:
        # Step 1: Generate multi-view images
        target_dir, image_paths, gen_log = generate_multiview_images(
            input_image_path=input_image_path,
            fal_api_key=fal_api_key,
            angles_json=angles_json,
            zoom=zoom,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_inference_steps),
            lora_scale=lora_scale,
            additional_prompt=additional_prompt,
            negative_prompt=negative_prompt,
        )
        
        # Step 2: Run VGGT reconstruction
        frame_filter = "All"
        glbfile, recon_log, dropdown = gradio_demo(
            target_dir,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        )
        
        final_log = f"{gen_log} | {recon_log}"
        return glbfile, final_log, target_dir, image_paths
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        print(f"[DEBUG] Exception in multiview_and_reconstruct: {error_msg}")
        traceback.print_exc()
        return None, error_msg, None, []


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def get_max_images_for_gpu():
    """
    Determine max number of images based on available GPU memory.
    VGGT 1B model memory requirements (approximate):
    - Model weights: ~5GB
    - Per image (518x518): ~1.5-2GB for activations
    """
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem_gb >= 40:  # A100, etc.
            return 20
        elif gpu_mem_gb >= 24:  # RTX 3090, A10
            return 10
        elif gpu_mem_gb >= 16:  # T4, RTX 4080
            return 5
        elif gpu_mem_gb >= 12:  # RTX 3060
            return 4
        else:
            return 3
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 4  # Conservative for Mac
    else:
        return 10  # CPU has no VRAM limit


def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Get the best available device
    current_device = get_device()
    print(f"Using device: {current_device}")

    # Move model to device
    model = model.to(current_device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    # Check GPU memory limits and limit images if needed
    max_images = get_max_images_for_gpu()
    if len(image_names) > max_images:
        print(f"⚠️ GPU memory limited: reducing from {len(image_names)} to {max_images} images")
        # Sample images evenly across the sequence
        step = len(image_names) / max_images
        indices = [int(i * step) for i in range(max_images)]
        image_names = [image_names[i] for i in indices]
        print(f"Selected images: {[os.path.basename(n) for n in image_names]}")

    # Clear GPU cache before loading images
    if current_device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    images = load_and_preprocess_images(image_names).to(current_device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference with appropriate autocast based on device
    print("Running inference...")
    
    # For MPS, clear memory before inference
    if current_device == "mps":
        gc.collect()
        torch.mps.empty_cache()
    
    # Track if we fell back to CPU
    used_cpu_fallback = False
    image_shape = images.shape[-2:]  # Save shape before potential device change
    
    with torch.no_grad():
        if current_device == "cuda":
            # CUDA supports bfloat16 autocast
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    predictions = model(images)
            except torch.cuda.OutOfMemoryError as e:
                print(f"⚠️ CUDA out of memory with {images.shape[0]} images. Trying with fewer images...")
                
                # Clear memory
                del images
                gc.collect()
                torch.cuda.empty_cache()
                
                # Try with half the images
                reduced_count = max(2, len(image_names) // 2)
                step = len(image_names) / reduced_count
                indices = [int(i * step) for i in range(reduced_count)]
                image_names_reduced = [image_names[i] for i in indices]
                
                print(f"Retrying with {reduced_count} images...")
                images = load_and_preprocess_images(image_names_reduced).to(current_device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    predictions = model(images)
                
                # Update image_names for later use
                image_names = image_names_reduced
        elif current_device == "mps":
            # MPS: use float32 for stability (float16 can cause issues)
            # Note: Some operations may fall back to CPU on MPS
            try:
                predictions = model(images)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("MPS out of memory, falling back to CPU...")
                    used_cpu_fallback = True
                    
                    # Clear MPS memory completely
                    del images
                    gc.collect()
                    torch.mps.empty_cache()
                    
                    # Move model to CPU
                    model.to("cpu")
                    
                    # Reload images directly to CPU
                    print("Reloading images on CPU...")
                    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
                    image_names = sorted(image_names)
                    images = load_and_preprocess_images(image_names).to("cpu")
                    
                    # Run on CPU
                    predictions = model(images)
                    
                    # Move model back to MPS for future runs (if memory allows)
                    # model.to("mps")  # Skip this to avoid memory issues
                else:
                    raise e
        else:
            # CPU: no autocast, use default precision
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_shape)
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    if current_device == "cuda":
        torch.cuda.empty_cache()
    elif current_device == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    loaded = np.load(predictions_path, allow_pickle=True)
    predictions = {key: loaded[key] for key in loaded.keys()}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------

canyon_video = "examples/videos/Studlagil_Canyon_East_Iceland.mp4"
great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:

    # Instead of gr.State, we use a hidden Textbox:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")
    
    gr.HTML(
    """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the “Upload Video” or “Upload Images” buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the “Reconstruct” button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between “Depthmap and Camera Branch” or “Pointmap Branch.”</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong>Please note:</strong> Our method usually only needs less than 1 second to reconstruct a scene, but the visualization of 3D points may take tens of seconds, especially when the number of images is large. Please be patient or, for faster visualization, use a local machine to run our demo from our <a href="https://github.com/facebookresearch/vggt">GitHub repository</a>.</p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
    
    # Load initial config for multi-view presets
    initial_config = load_multiview_config()
    initial_presets = list(initial_config.get("presets", {}).keys()) + ["Custom"]
    initial_fal_params = initial_config.get("fal", {}).get("params", {})

    with gr.Tabs():
        # ===================== TAB 1: Multi-View from Single Image =====================
        with gr.TabItem("🖼️ Multi-View from Single Image"):
            gr.Markdown("""
            **Generate multi-view images from a single input image using AI, then reconstruct in 3D.**
            
            - **Option 1:** Upload a single image → Generate multiple viewing angles → VGGT 3D reconstruction
            - **Option 2:** Load existing multi-view images from a folder → VGGT 3D reconstruction (skip generation)
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    mv_input_image = gr.Image(label="Upload Single Image", type="filepath", interactive=True)
                    
                    # Load from existing folder section
                    with gr.Accordion("📂 Load Existing Multi-View Folder", open=True):
                        mv_folder_path = gr.Textbox(
                            label="Folder Path",
                            placeholder="e.g., multiview_output_20260204_160525_440892 or multiview_output_20260204_160525_440892/images",
                            info="Enter path to folder containing multi-view images (or the 'images' subfolder)"
                        )
                        with gr.Row():
                            mv_load_folder_btn = gr.Button("📂 Load Images", scale=1)
                            mv_reconstruct_only_btn = gr.Button("🔨 Reconstruct Only", scale=1, variant="secondary")
                    
                    mv_image_gallery = gr.Gallery(
                        label="Multi-View Images",
                        columns=4,
                        height="250px",
                        show_download_button=True,
                        object_fit="contain",
                        preview=True,
                    )
                    
                    # Multi-view generation settings
                    with gr.Accordion("🔧 Multi-View Generation Settings", open=False):
                        mv_fal_api_key = gr.Textbox(
                            label="FAL API Key",
                            type="password",
                            placeholder="Enter your FAL API key...",
                            value=initial_config.get("fal", {}).get("api_key", ""),
                        )
                        
                        mv_preset_choice = gr.Dropdown(
                            choices=initial_presets,
                            value=initial_presets[0] if initial_presets else "Custom",
                            label="Angle Preset",
                        )
                        
                        mv_custom_angles = gr.Textbox(
                            label="Custom Angles (JSON)",
                            value=json.dumps(initial_config.get("angles", []), indent=2),
                            lines=6,
                            placeholder='[{"name": "front", "horizontal": 0, "vertical": 0}, ...]',
                        )
                    
                    with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
                        mv_zoom = gr.Slider(
                            minimum=0, maximum=10, value=initial_fal_params.get("zoom", 5),
                            step=0.5, label="Zoom (0=wide, 5=medium, 10=close-up)"
                        )
                        mv_guidance_scale = gr.Slider(
                            minimum=1, maximum=20, value=initial_fal_params.get("guidance_scale", 4.5),
                            step=0.5, label="Guidance Scale"
                        )
                        mv_num_inference_steps = gr.Slider(
                            minimum=10, maximum=50, value=initial_fal_params.get("num_inference_steps", 28),
                            step=1, label="Inference Steps"
                        )
                        mv_lora_scale = gr.Slider(
                            minimum=0, maximum=2, value=initial_fal_params.get("lora_scale", 1),
                            step=0.1, label="LoRA Scale"
                        )
                        mv_additional_prompt = gr.Textbox(
                            label="Additional Prompt",
                            value=initial_fal_params.get("additional_prompt", ""),
                            placeholder="Optional additional text for generation..."
                        )
                        mv_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value=initial_fal_params.get("negative_prompt", ""),
                            placeholder="Optional negative prompt..."
                        )
                
                with gr.Column(scale=4):
                    gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                    mv_log_output = gr.Markdown(
                        "Upload an image, configure settings, then click 'Generate Multi-View → Reconstruct'.",
                        elem_classes=["custom-log"]
                    )
                    mv_reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)
                    
                    with gr.Row():
                        mv_submit_btn = gr.Button("🚀 Generate Multi-View → Reconstruct", scale=2, variant="primary")
                        mv_clear_btn = gr.ClearButton(
                            [mv_input_image, mv_reconstruction_output, mv_log_output, mv_image_gallery],
                            scale=1,
                        )
                    
                    with gr.Row():
                        mv_prediction_mode = gr.Radio(
                            ["Depthmap and Camera Branch", "Pointmap Branch"],
                            label="Select a Prediction Mode",
                            value="Depthmap and Camera Branch",
                            scale=1,
                            elem_id="my_radio_mv",
                        )
                    
                    with gr.Row():
                        mv_conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                        with gr.Column():
                            mv_show_cam = gr.Checkbox(label="Show Camera", value=True)
                            mv_mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                            mv_mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                            mv_mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)
            
            # Update custom angles when preset changes
            def update_angles_from_preset(preset_name):
                return get_preset_angles(preset_name)
            
            mv_preset_choice.change(
                fn=update_angles_from_preset,
                inputs=[mv_preset_choice],
                outputs=[mv_custom_angles],
            )
            
            # Multi-view target dir (hidden)
            mv_target_dir_output = gr.Textbox(label="MV Target Dir", visible=False, value="None")
            
            # Load folder button - just preview images
            def load_folder_preview(folder_path):
                target_dir, image_paths, log_msg = load_existing_multiview_folder(folder_path)
                if target_dir is None:
                    return [], log_msg, "None"
                return image_paths, log_msg, target_dir
            
            mv_load_folder_btn.click(
                fn=load_folder_preview,
                inputs=[mv_folder_path],
                outputs=[mv_image_gallery, mv_log_output, mv_target_dir_output],
            )
            
            # Reconstruct only button - load from folder and reconstruct
            mv_reconstruct_only_btn.click(
                fn=lambda: (None, "🔄 Loading images and running reconstruction..."),
                inputs=[],
                outputs=[mv_reconstruction_output, mv_log_output],
            ).then(
                fn=reconstruct_from_folder,
                inputs=[
                    mv_folder_path,
                    mv_conf_thres,
                    mv_mask_black_bg,
                    mv_mask_white_bg,
                    mv_show_cam,
                    mv_mask_sky,
                    mv_prediction_mode,
                ],
                outputs=[mv_reconstruction_output, mv_log_output, mv_target_dir_output, mv_image_gallery],
            )
            
            # Generate Multi-View → Reconstruct button logic
            def on_generate_click():
                print("[DEBUG] Generate Multi-View → Reconstruct button clicked!")
                return None, "🔄 Generating multi-view images... This may take a minute."
            
            mv_submit_btn.click(
                fn=on_generate_click,
                inputs=[],
                outputs=[mv_reconstruction_output, mv_log_output],
            ).then(
                fn=multiview_and_reconstruct,
                inputs=[
                    mv_input_image,
                    mv_fal_api_key,
                    mv_preset_choice,
                    mv_custom_angles,
                    mv_zoom,
                    mv_guidance_scale,
                    mv_num_inference_steps,
                    mv_lora_scale,
                    mv_additional_prompt,
                    mv_negative_prompt,
                    mv_conf_thres,
                    mv_mask_black_bg,
                    mv_mask_white_bg,
                    mv_show_cam,
                    mv_mask_sky,
                    mv_prediction_mode,
                ],
                outputs=[mv_reconstruction_output, mv_log_output, mv_target_dir_output, mv_image_gallery],
            )
        
        # ===================== TAB 2: Video/Images Upload (Original) =====================
        with gr.TabItem("🎬 Video / Multiple Images"):
            gr.Markdown("""
            **Upload a video or multiple images for 3D reconstruction.**
            
            Videos will be automatically split into individual frames (one frame per second).
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    input_video = gr.Video(label="Upload Video", interactive=True)
                    input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

                    image_gallery = gr.Gallery(
                        label="Preview",
                        columns=4,
                        height="300px",
                        show_download_button=True,
                        object_fit="contain",
                        preview=True,
                    )

                with gr.Column(scale=4):
                    with gr.Column():
                        gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                        log_output = gr.Markdown(
                            "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                        )
                        reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

                    with gr.Row():
                        submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                        clear_btn = gr.ClearButton(
                            [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                            scale=1,
                        )

                    with gr.Row():
                        prediction_mode = gr.Radio(
                            ["Depthmap and Camera Branch", "Pointmap Branch"],
                            label="Select a Prediction Mode",
                            value="Depthmap and Camera Branch",
                            scale=1,
                            elem_id="my_radio",
                        )

                    with gr.Row():
                        conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                        frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                        with gr.Column():
                            show_cam = gr.Checkbox(label="Show Camera", value=True)
                            mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                            mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                            mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

            # ---------------------- Examples section (inside Tab 2) ----------------------
            examples = [
                [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
                [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
                [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
                [single_oil_painting_video, "1", None, 20.0, False, True, True, True, "Depthmap and Camera Branch", "True"],
                [canyon_video, "14", None, 40.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
                [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
                [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
                [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
            ]

            def example_pipeline(
                input_video,
                num_images_str,
                input_images,
                conf_thres,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example_str,
            ):
                """
                1) Copy example images to new target_dir
                2) Reconstruct
                3) Return model3D + logs + new_dir + updated dropdown + gallery
                We do NOT return is_example. It's just an input.
                """
                target_dir, image_paths = handle_uploads(input_video, input_images)
                # Always use "All" for frame_filter in examples
                frame_filter = "All"
                glbfile, log_msg, dropdown = gradio_demo(
                    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
                )
                return glbfile, log_msg, target_dir, dropdown, image_paths

            gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

            gr.Examples(
                examples=examples,
                inputs=[
                    input_video,
                    num_images,
                    input_images,
                    conf_thres,
                    mask_black_bg,
                    mask_white_bg,
                    show_cam,
                    mask_sky,
                    prediction_mode,
                    is_example,
                ],
                outputs=[
                    reconstruction_output,
                    log_output,
                    target_dir_output,
                    frame_filter,
                    image_gallery,
                ],
                fn=example_pipeline,
                cache_examples=False,
                examples_per_page=50,
            )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    conf_thres.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example],
        [reconstruction_output, log_output],
    )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
