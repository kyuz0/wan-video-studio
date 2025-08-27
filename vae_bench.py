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
    parser = argparse.ArgumentParser(description='Benchmark VAE performance')
    parser.add_argument('--ckpt_dir', required=True, help='Path to model checkpoints')
    parser.add_argument('--device', default='cuda:0', help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--output', default='vae_benchmark_results.json', help='Output JSON file')
    parser.add_argument('--vae_type', choices=['2.1', '2.2', 'both'], default='both', 
                       help='Which VAE to benchmark (default: both)')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Supported sizes from Wan configs
    supported_sizes = [
        (720, 1280, "720x1280"),   # t2v-A14B, i2v-A14B
        (1280, 720, "1280x720"),   # t2v-A14B, i2v-A14B  
        (480, 832, "480x832"),     # t2v-A14B, i2v-A14B
        (832, 480, "832x480"),     # t2v-A14B, i2v-A14B
        (704, 1280, "704x1280"),   # ti2v-5B
        (1280, 704, "1280x704"),   # ti2v-5B
    ]
    
    # Frame counts to test (small counts for fast benchmarking)
    # Note: Frame counts are always 4n+1 in Wan configs
    frame_counts = [5, 9, 17]  # Small counts to extrapolate from
    
    # Generate test configurations
    test_configs = []
    for frames in frame_counts:
        for height, width, desc in supported_sizes:
            test_configs.append((frames, height, width, f"{frames}f_{desc}"))
    
    # VAE configurations to test
    vae_configs = []
    if args.vae_type in ['2.1', 'both']:
        vae_configs.append(('2.1', 'Wan2.1_VAE.pth', Wan2_1_VAE))
    if args.vae_type in ['2.2', 'both']:
        vae_configs.append(('2.2', 'Wan2.2_VAE.pth', Wan2_2_VAE))
    
    all_results = {}
    
    print(f"\nTesting on device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB\n")
    
    for vae_name, vae_file, vae_class in vae_configs:
        print(f"=== Testing VAE {vae_name} ===")
        vae_path = os.path.join(args.ckpt_dir, vae_file)
        
        if not os.path.exists(vae_path):
            print(f"VAE file not found: {vae_path}")
            print("Skipping VAE 2.2 tests\n")
            continue
            
        # Initialize VAE
        try:
            vae = vae_class(vae_pth=vae_path, device=device)
        except Exception as e:
            print(f"Failed to load VAE {vae_name}: {e}")
            continue
        
        results = {
            'vae_version': vae_name,
            'device': str(device),
            'encode_results': [],
            'decode_results': [],
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'gpu_memory_gb': torch.cuda.get_device_properties(device).total_memory / 1e9 if torch.cuda.is_available() else None
            }
        }
        
        for frames, height, width, desc in test_configs:
            print(f"Testing {desc}: {frames} frames at {height}x{width}")
            
            try:
                # Create test tensor
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
                    'encoded_size_mb': encoded.numel() * encoded.element_size() / (1024 * 1024),
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
        
        # Create prediction models for this VAE
        encode_data = results['encode_results']
        if len(encode_data) >= 3:  # Need at least 3 points for good extrapolation
            # Build linear model: time = baseline + (frames * frame_factor) + (pixels * pixel_factor)
            frames_list = [r['frames'] for r in encode_data]
            pixels_list = [r['height'] * r['width'] for r in encode_data]  
            times_list = [r['time_mean'] for r in encode_data]
            
            # Simple multi-variable linear regression
            # We'll use frames as primary predictor since that's what varies most
            import numpy as np
            
            # Model: time = a + b*frames + c*pixels
            X = np.array([[1, f, p] for f, p in zip(frames_list, pixels_list)])
            y = np.array(times_list)
            
            try:
                # Solve for coefficients
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                baseline, frame_factor, pixel_factor = coeffs
                
                # Calculate R-squared
                y_pred = X @ coeffs
                r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                
                print(f"\n=== PREDICTION MODEL for VAE {vae_name} ===")
                print(f"Model: time = {baseline:.3f} + {frame_factor:.4f}*frames + {pixel_factor:.8f}*pixels")
                print(f"R-squared: {r_squared:.3f}")
                
                # Test predictions for common frame counts
                for test_frames in [81, 121]:
                    for height, width, desc in [(832, 480, "832x480"), (720, 1280, "720x1280")]:
                        test_pixels = height * width
                        predicted_time = baseline + frame_factor * test_frames + pixel_factor * test_pixels
                        print(f"  Predicted {test_frames}f {desc}: {predicted_time:.1f}s")
                
                # Store prediction model
                results['prediction_model'] = {
                    'baseline': float(baseline),
                    'frame_factor': float(frame_factor), 
                    'pixel_factor': float(pixel_factor),
                    'r_squared': float(r_squared)
                }
                
            except np.linalg.LinAlgError:
                print(f"Could not create prediction model for VAE {vae_name}")
                
        else:
            print(f"Not enough data points to create prediction model for VAE {vae_name}")
        
        all_results[f'vae_{vae_name}'] = results
        print()
    
    # Save all results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()