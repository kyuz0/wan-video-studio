# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import logging
import os
import os.path as osp
import torch.nn as nn
import imageio
import torch
import torchvision
from tqdm import tqdm
from safetensors import safe_open
import threading
import time
import sys
from tqdm import tqdm
import subprocess
import shutil

__all__ = ['save_video', 'save_image', 'str2bool', "use_cfg", "model_safe_downcast", "load_and_merge_lora_weight_from_safetensors"]

class SimpleTimer:
    def __init__(self, operation_name="Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.running = True
        print(f"{self.operation_name}...", end="", flush=True)
        
        # Start background thread for in-place timer
        self.thread = threading.Thread(target=self._update_timer)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the timer and log final time."""
        self.running = False
        if self.thread:
            self.thread.join()
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            # Clear the line and show final result
            print(f"\r{self.operation_name} completed in {self._format_time(elapsed)}")
    
    def _update_timer(self):
        """Background thread to update timer in place."""
        while self.running:
            time.sleep(1)  # Update every second
            if self.running and self.start_time:
                elapsed = time.time() - self.start_time
                # Overwrite same line with current elapsed time
                print(f"\r{self.operation_name}... {self._format_time(elapsed)} elapsed", end="", flush=True)
    
    def _format_time(self, seconds):
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds < 3600:  # Less than 1 hour
            return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
        else:  # 1 hour or more
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}:{minutes:02d}:{secs:02d}"

def use_cfg(cfg_scale:float=1.0, eps:float=1e-6):
    return abs(cfg_scale - 1.0) > eps

def model_safe_downcast(
    model: nn.Module, 
    dtype: torch.dtype = torch.bfloat16, 
    keep_in_fp32_modules: list[str]|tuple[str,...]|None = None, 
    keep_in_fp32_parameters: list[str]|tuple[str,...]|None = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Downcast model parameters and buffers to a specified dtype, while keeping certain modules/parameters in fp32.

    Args:
        model: The PyTorch model to downcast
        dtype: The target dtype to downcast to (default: torch.bfloat16)
        keep_in_fp32_modules: List of module names to keep in fp32, fuzzy matching is supported
        keep_in_fp32_parameters: List of parameter names to keep in fp32, exact matching is required
        verbose: Whether to print information.

    Returns:
        The downcast model (modified in-place)
    """
    keep_in_fp32_modules = list(keep_in_fp32_modules or [])
    keep_in_fp32_modules.extend(getattr(model, "_keep_in_fp32_modules", []))
    keep_in_fp32_parameters = keep_in_fp32_parameters or []

    for name, module in model.named_modules():
        # Skip if module is in keep_in_fp32_modules list
        if any(keep_name in name for keep_name in keep_in_fp32_modules):
            if verbose:
                print(f"Skipping {name} because it is in keep_in_fp32_modules")
            continue

        # Downcast parameters
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = f"{name}.{param_name}" if name else param_name
            if param is not None:
                if full_param_name in keep_in_fp32_parameters and verbose:
                    print(f"Skipping {full_param_name} because it is in keep_in_fp32_parameters")
                # if not any(keep_name in full_param_name for keep_name in keep_in_fp32_parameters):
                else:
                    param.data = param.data.to(dtype)

        # Downcast buffers
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                buffer.data = buffer.data.to(dtype)
    return model

def build_lora_names(key, lora_down_key, lora_up_key, is_native_weight):
    base = "diffusion_model." if is_native_weight else ""
    lora_down = base + key.replace(".weight", lora_down_key)
    lora_up = base + key.replace(".weight", lora_up_key)
    lora_alpha = base + key.replace(".weight", ".alpha")
    return lora_down, lora_up, lora_alpha

def load_and_merge_lora_weight(
    model: nn.Module,
    lora_state_dict: dict,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight"):
    
    is_native_weight = any("diffusion_model." in key for key in lora_state_dict)
    
    # Count LoRA parameters to process for progress bar
    lora_params = []
    for key, value in model.named_parameters():
        lora_down_name, lora_up_name, lora_alpha_name = build_lora_names(
            key, lora_down_key, lora_up_key, is_native_weight
        )
        if lora_down_name in lora_state_dict:
            lora_params.append((key, value, lora_down_name, lora_up_name, lora_alpha_name))
    
    # Detect device from LoRA tensors (not model parameters)
    if lora_params:
        first_lora_tensor = lora_state_dict[lora_params[0][2]]  # First lora_down tensor
        device_type = "GPU" if first_lora_tensor.device.type == "cuda" else "CPU"
    else:
        device_type = "CPU"
    
    # Process LoRA parameters with progress bar
    with tqdm(lora_params, desc=f"Merging LoRA weights on {device_type}") as pbar:
        for key, value, lora_down_name, lora_up_name, lora_alpha_name in pbar:
            lora_down = lora_state_dict[lora_down_name]
            lora_up = lora_state_dict[lora_up_name]
            lora_alpha = float(lora_state_dict[lora_alpha_name])
            
            rank = lora_down.shape[0]
            scaling_factor = lora_alpha / rank
            
            # Matrix multiplication on the device where LoRA tensors are
            delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
            
            # Move delta_W to model's device if different and ensure dtype matches
            if delta_W.device != value.device:
                delta_W = delta_W.to(value.device)
            if delta_W.dtype != value.dtype:
                delta_W = delta_W.to(value.dtype)
            
            # Add delta weights to original weights
            value.data.add_(delta_W)
            
            pbar.set_postfix({"layer": key.split('.')[-2] if '.' in key else key})
    
    return model

def load_and_merge_lora_weight_from_safetensors(
    model: nn.Module,
    lora_weight_path: str,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight"):
    
    # Check if CUDA is available and get first GPU device
    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
        logging.info(f"Loading LoRA weights directly to {target_device}...")
    else:
        target_device = torch.device("cpu")
        logging.info(f"Loading LoRA weights to {target_device}...")
    
    # Use load_file with target device
    from safetensors.torch import load_file
    lora_state_dict = load_file(lora_weight_path, device=str(target_device))
    
    model = load_and_merge_lora_weight(model, lora_state_dict, lora_down_key, lora_up_key)
    logging.info("LoRA weights loaded and merged successfully")
    return model


def save_video(tensor,
               save_file=None,
               fps=30,
               suffix='.mp4',
               nrow=8,
               normalize=True,
               value_range=(-1, 1)):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    try:
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
            torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ],
                             dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(
            cache_file, fps=fps, codec='libx264', quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        logging.info(f'save_video failed, error: {e}')


def save_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1)):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    try:
        tensor = tensor.clamp(min(value_range), max(value_range))
        torchvision.utils.save_image(
            tensor,
            save_file,
            nrow=nrow,
            normalize=normalize,
            value_range=value_range)
        return save_file
    except Exception as e:
        logging.info(f'save_image failed, error: {e}')


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')


def masks_like(tensor, zero=False, generator=None, p=0.2):
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, 0] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, 0]).exp()
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, 0] = torch.zeros_like(u[:, 0])
                v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2

def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    logging.info("Merging video and audio...")
    merge_timer = SimpleTimer("Audio-video merge")
    merge_timer.start()

    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        merge_timer.stop()
        logging.info(f"Audio-video merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        merge_timer.stop()
        logging.error(f"merge_video_audio failed with error: {e}")
        raise