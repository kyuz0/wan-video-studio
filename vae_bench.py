#!/usr/bin/env python3
"""
VAE Performance Benchmarking Script

This script tests VAE encode/decode performance across different frame counts and resolutions
to build a timing prediction model for the Wan video generation pipeline.

Usage:
    python vae_benchmark.py --ckpt_dir ~/Wan2.2-I2V-A14 --device cuda:0
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from pathlib import Path

# Import from the Wan codebase
import sys
sys.path.append('.')
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.vae2_2 import Wan2_2_VAE


def create_test_tensor(frames, height, width, device):
    """Create a test video tensor with specified dimensions."""
    return torch.randn(3, frames, height, width, device=device, dtype=torch.float32)


def benchmark_vae_operation(vae, operation, tensor, warmup_runs=2, test_runs=5):
    """Benchmark a VAE operation (encode or decode) and return average time."""
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            if operation == 'encode':
                _ = vae.encode([tensor])
            else:  # decode
                _ = vae.decode([tensor])
        torch.cuda.synchronize() if tensor.device.type == 'cuda' else None
    
    # Actual timing runs
    times = []
    for _ in range(test_runs):
        start_time = time.time()
        with torch.no_grad():
            if operation == 'encode':
                result = vae.encode([tensor])
            else:  # decode
                result = vae.decode([tensor])
        
        if tensor.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        # Clean up result
        del result
        if tensor.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser(description='Benchmark VAE performance for generate.py usage')
    parser.add_argument('--ckpt_dir', required=True, help='Path to model checkpoints')
    parser.add_argument('--device', default='cuda:0', help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--output', default='vae_benchmark_results.json', help='Output JSON file')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Only i2v-A14B supported sizes (user's actual use case)
    i2v_sizes = [
        (720, 1280, "720x1280"),   
        (1280, 720, "1280x720"),   
        (480, 832, "480x832"),     
        (832, 480, "832x480"),  # User's current config
    ]
    
    # Small frame counts for fast benchmarking (will extrapolate to 81, 121)
    test_frame_counts = [5, 9, 17]  # Quick tests
    target_frame_counts = [81, 121]  # What users actually run
    
    # Generate test configurations
    test_configs = []
    for frames in test_frame_counts:
        for height, width, desc in i2v_sizes:
            test_configs.append((frames, height, width, f"{frames}f_{desc}"))
    
    print(f"Benchmarking VAE 2.1 for i2v-A14B generation")
    print(f"Testing on device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Initialize Wan2.1 VAE (only one used by i2v-A14B)
    vae_path = os.path.join(args.ckpt_dir, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_path):
        print(f"VAE file not found: {vae_path}")
        return
        
    try:
        vae = Wan2_1_VAE(vae_pth=vae_path, device=device)
    except Exception as e:
        print(f"Failed to load VAE: {e}")
        return
    
    results = {
        'device': str(device),
        'task': 'i2v-A14B',
        'vae_version': '2.1', 
        'encode_results': [],
        'decode_results': [],
        'system_info': {
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'gpu_memory_gb': torch.cuda.get_device_properties(device).total_memory / 1e9 if torch.cuda.is_available() else None
        }
    }
    
    print(f"\n=== Benchmarking VAE Encode & Decode Performance ===")
    
    for frames, height, width, desc in test_configs:
        print(f"Testing {desc}: {frames} frames at {height}x{width}")
        
        try:
            # Create test tensor (i2v format: first frame = image, rest = zeros)
            test_tensor = create_test_tensor(frames, height, width, device)
            tensor_size_mb = test_tensor.numel() * test_tensor.element_size() / (1024 * 1024)
            
            print(f"  Tensor size: {tensor_size_mb:.1f} MB")
            
            # Test encoding
            print("  Testing encode...", end=" ", flush=True)
            encode_time, encode_std = benchmark_vae_operation(vae, 'encode', test_tensor)
            print(f"{encode_time:.2f}s ±{encode_std:.2f}s")
            
            # Get encoded tensor for decode test
            with torch.no_grad():
                encoded = vae.encode([test_tensor])[0]
            
            encoded_size_mb = encoded.numel() * encoded.element_size() / (1024 * 1024)
            
            # Test decoding
            print("  Testing decode...", end=" ", flush=True)
            decode_time, decode_std = benchmark_vae_operation(vae, 'decode', encoded)
            print(f"{decode_time:.2f}s ±{decode_std:.2f}s")
            
            # Store results
            results['encode_results'].append({
                'config': desc,
                'frames': frames,
                'height': height, 
                'width': width,
                'tensor_size_mb': tensor_size_mb,
                'time_mean': encode_time,
                'time_std': encode_std,
                'throughput_mb_per_sec': tensor_size_mb / encode_time
            })
            
            results['decode_results'].append({
                'config': desc,
                'frames': frames,
                'height': height,
                'width': width, 
                'encoded_size_mb': encoded_size_mb,
                'time_mean': decode_time,
                'time_std': decode_std
            })
            
            # Cleanup
            del test_tensor, encoded
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  SKIPPED: Out of memory")
                continue
            else:
                raise e
        
        print()
    
    # Create prediction models for both encode and decode
    encode_data = results['encode_results']
    decode_data = results['decode_results']
    
    models = {}
    
    for operation, data in [('encode', encode_data), ('decode', decode_data)]:
        if len(data) >= 6:
            frames_list = [r['frames'] for r in data]
            pixels_list = [r['height'] * r['width'] for r in data]  
            times_list = [r['time_mean'] for r in data]
            
            # Model: time = baseline + frame_factor*frames + pixel_factor*pixels
            X = np.array([[1, f, p] for f, p in zip(frames_list, pixels_list)])
            y = np.array(times_list)
            
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                baseline, frame_factor, pixel_factor = coeffs
                
                # Calculate R-squared
                y_pred = X @ coeffs
                r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                
                models[f'{operation}_model'] = {
                    'baseline': float(baseline),
                    'frame_factor': float(frame_factor), 
                    'pixel_factor': float(pixel_factor),
                    'r_squared': float(r_squared)
                }
                
                print(f"=== {operation.upper()} PREDICTION MODEL ===")
                print(f"Model: time = {baseline:.3f} + {frame_factor:.4f}*frames + {pixel_factor:.8f}*pixels")
                print(f"R-squared: {r_squared:.3f}")
                
            except np.linalg.LinAlgError:
                print(f"Could not create {operation} prediction model")
    
    # Show predictions for actual usage
    if models:
        print(f"\n=== PREDICTIONS FOR ACTUAL USAGE ===")
        for test_frames in target_frame_counts:
            print(f"\n{test_frames} frames:")
            for height, width, desc in i2v_sizes:
                test_pixels = height * width
                
                encode_pred = None
                decode_pred = None
                
                if 'encode_model' in models:
                    m = models['encode_model']
                    encode_pred = m['baseline'] + m['frame_factor'] * test_frames + m['pixel_factor'] * test_pixels
                
                if 'decode_model' in models:
                    m = models['decode_model']
                    decode_pred = m['baseline'] + m['frame_factor'] * test_frames + m['pixel_factor'] * test_pixels
                
                pred_str = f"  {desc}:"
                if encode_pred: pred_str += f" encode {encode_pred:.1f}s"
                if decode_pred: pred_str += f" decode {decode_pred:.1f}s"
                if encode_pred and decode_pred: pred_str += f" total {encode_pred + decode_pred:.1f}s"
                
                print(pred_str)
    
    # Store prediction models
    results.update(models)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("Use this file with generate.py for VAE progress estimation")


if __name__ == "__main__":
    main()