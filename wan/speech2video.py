# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from decord import VideoReader
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.s2v.audio_encoder import AudioEncoder
from .modules.s2v.model_s2v import WanModel_S2V, sp_attn_forward_s2v
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import use_cfg, SimpleTimer
from .utils.vae_tiling import tiled_encode, tiled_decode, pixel_to_latent_tiles


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


class WanS2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the speech-to-video generation model components.
        Optimized for Strix Halo unified memory architecture.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        # Force GPU placement for single-GPU usage on Strix Halo
        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False
        elif torch.cuda.is_available():
            self.init_on_cpu = False  # Force GPU for unified memory

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Loading S2V model from {checkpoint_dir}...")
        if not dit_fsdp:
           self.noise_model = WanModel_S2V.from_pretrained(
                checkpoint_dir,
                torch_dtype=self.param_dtype,
                low_cpu_mem_usage=False,   # avoid accelerate’s meta/sharded GPU copies on ROCm
            )
        else:
            self.noise_model = WanModel_S2V.from_pretrained(
                checkpoint_dir,
                torch_dtype=self.param_dtype,
                low_cpu_mem_usage=False,   # avoid accelerate’s meta/sharded GPU copies on ROCm
            )
        logging.info("S2V model loaded successfully")

        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        logging.info("Loading audio encoder...")
        self.audio_encoder = AudioEncoder(
            device=self.device,  # force GPU
            model_id=os.path.join(checkpoint_dir,
                                  "wav2vec2-large-xlsr-53-english"))
        logging.info("Audio encoder loaded successfully")

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.motion_frames = config.transformer.motion_frames
        self.drop_first_motion = config.drop_first_motion
        self.fps = config.sample_fps
        self.audio_sample_m = 0
        self.use_vae_tiling = getattr(config, "use_vae_tiling", False)
        self.vae_tile_px = int(getattr(config, "vae_tile_px", 128))

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """Configure model with Strix Halo optimizations"""
        model.eval().requires_grad_(False)
        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_s2v, block.self_attn)
            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def get_size_less_than_area(self, height, width, target_area=1024 * 704, divisor=64):
        """Calculate optimal size within target area"""
        if height * width <= target_area:
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            max_upper_area = target_area
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area

            min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)
            max_scale = math.sqrt(max_upper_area / (height * width))

        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)

            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            padded_height, padded_width = new_height + pad_height, new_width + pad_width

            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width
        else:
            aspect_ratio = width / height
            target_width = int((target_area * aspect_ratio)**0.5 // divisor * divisor)
            target_height = int((target_area / aspect_ratio)**0.5 // divisor * divisor)

            if target_width >= width or target_height >= height:
                target_width = int(width // divisor * divisor)
                target_height = int(height // divisor * divisor)

            return target_height, target_width

    def encode_audio(self, audio_path, infer_frames):
        """Encode audio with progress tracking"""
        logging.info("Processing audio...")
        audio_timer = SimpleTimer("Audio encoding")
        audio_timer.start()
        
        z = self.audio_encoder.extract_audio_feat(
            audio_path, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device, self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
            
        audio_timer.stop()
        return audio_embed_bucket, num_repeat

    def read_last_n_frames(self, video_path, n_frames, target_fps=16, reverse=False):
        """Read frames from video"""
        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        interval = max(1, round(original_fps / target_fps))
        required_span = (n_frames - 1) * interval
        start_frame = max(0, total_frames - required_span - 1) if not reverse else 0

        sampled_indices = []
        for i in range(n_frames):
            indice = start_frame + i * interval
            if indice >= total_frames:
                break
            else:
                sampled_indices.append(indice)

        return vr.get_batch(sampled_indices).asnumpy()

    def load_pose_cond(self, pose_video, num_repeat, infer_frames, size):
        """Load pose conditioning with memory optimization"""
        HEIGHT, WIDTH = size
        if pose_video is not None:
            logging.info("Processing pose video...")
            pose_seq = self.read_last_n_frames(
                pose_video,
                n_frames=infer_frames * num_repeat,
                target_fps=self.fps,
                reverse=True)

            resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
            crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))

            cond_tensor = torch.from_numpy(pose_seq)
            cond_tensor = cond_tensor.permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
            cond_tensor = crop_opreat(resize_opreat(cond_tensor)).permute(
                1, 0, 2, 3).unsqueeze(0)

            padding_frame_num = num_repeat * infer_frames - cond_tensor.shape[2]
            cond_tensor = torch.cat([
                cond_tensor,
                - torch.ones([1, 3, padding_frame_num, HEIGHT, WIDTH])
            ], dim=2)

            cond_tensors = torch.chunk(cond_tensor, num_repeat, dim=2)
        else:
            cond_tensors = [-torch.ones([1, 3, infer_frames, HEIGHT, WIDTH])]

        COND = []
        vae_timer = SimpleTimer("VAE encoding pose conditions")
        vae_timer.start()
        
        for r in range(len(cond_tensors)):
            cond = cond_tensors[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            _cond = cond.to(dtype=self.param_dtype, device=self.device)
            if getattr(self, "use_vae_tiling", False):
                _lat = tiled_encode(self.vae, _cond[0], tile_px=getattr(self, "vae_tile_px", 128))
                cond_lat = _lat.unsqueeze(0)[:, :, 1:].contiguous().cpu()  # [1, 16, Tl-1, HL, WL]
            else:
                _lat = torch.stack(self.vae.encode(_cond))
                cond_lat = _lat[:, :, 1:].cpu()
            COND.append(cond_lat)
            
        vae_timer.stop()
        return COND

    def get_gen_size(self, size, max_area, ref_image_path, pre_video_path):
        """Calculate generation size"""
        if size is not None:
            HEIGHT, WIDTH = size
        else:
            if pre_video_path:
                ref_image = self.read_last_n_frames(pre_video_path, n_frames=1)[0]
            else:
                ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
            HEIGHT, WIDTH = ref_image.shape[:2]
        HEIGHT, WIDTH = self.get_size_less_than_area(HEIGHT, WIDTH, target_area=max_area)
        return (HEIGHT, WIDTH)

    def generate(
        self,
        input_prompt,
        ref_image_path,
        audio_path,
        num_repeat=1,
        pose_video=None,
        max_area=720 * 1280,
        infer_frames=80,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        init_first_frame=False,
    ):
        r"""
        Generates video from speech with Strix Halo memory optimizations
        """
        # Calculate generation size
        size = self.get_gen_size(
            size=None,
            max_area=max_area,
            ref_image_path=ref_image_path,
            pre_video_path=None)
        HEIGHT, WIDTH = size
        channel = 3

        resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
        crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = transforms.ToTensor()

        # Load reference image
        logging.info(f"Loading reference image: {ref_image_path}")
        ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
        motion_latents = torch.zeros(
            [1, channel, self.motion_frames, HEIGHT, WIDTH],
            dtype=self.param_dtype,
            device=self.device)

        # Process audio
        audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        if num_repeat is None or num_repeat > nr:
            num_repeat = nr
        logging.info(f"Will generate {num_repeat} clips of {infer_frames} frames each")

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))

        # Encode reference image
        logging.info("Encoding reference image...")
        encode_timer = SimpleTimer("Reference image encoding")
        encode_timer.start()
        
        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel_values = ref_pixel_values.to(dtype=self.vae.dtype, device=self.vae.device)
        if getattr(self, "use_vae_tiling", False):
            ref_latents = tiled_encode(self.vae, ref_pixel_values[0], tile_px=getattr(self, "vae_tile_px", 128)).unsqueeze(0)
        else:
            ref_latents = torch.stack(self.vae.encode(ref_pixel_values))
            
        # Encode motion latents
        videos_last_frames = motion_latents.detach()
        drop_first_motion = self.drop_first_motion
        if init_first_frame:
            drop_first_motion = False
            motion_latents[:, :, -6:] = ref_pixel_values

        if getattr(self, "use_vae_tiling", False):
            motion_latents = tiled_encode(self.vae, motion_latents[0], tile_px=getattr(self, "vae_tile_px", 128)).unsqueeze(0)
        else:
            motion_latents = torch.stack(self.vae.encode(motion_latents))
                  
        encode_timer.stop()

        # Load pose conditions
        COND = self.load_pose_cond(
            pose_video=pose_video,
            num_repeat=num_repeat,
            infer_frames=infer_frames,
            size=size)

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Encode text prompts
        logging.info("Encoding text prompts...")
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            # Unload text encoder to save memory
            del self.text_encoder
            torch.cuda.empty_cache()
            gc.collect()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            # Unload text encoder
            del self.text_encoder
            gc.collect()
        logging.info("Text encoding completed")

        out = []
        
        # Generation loop
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
        ):
            for r in range(num_repeat):
                logging.info(f"Generating clip {r + 1}/{num_repeat}")
                clip_timer = SimpleTimer(f"Clip {r + 1}")
                clip_timer.start()
                
                seed_g = torch.Generator(device=self.device)
                seed_g.manual_seed(seed + r)

                lat_target_frames = (infer_frames + 3 + self.motion_frames) // 4 - lat_motion_frames
                target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
                noise = [
                    torch.randn(
                        16,
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        dtype=self.param_dtype,
                        device=self.device,
                        generator=seed_g)
                ]
                max_seq_len = np.prod(target_shape) // 4

                # Setup scheduler
                
                solver = (sample_solver or 'unipc').lower()
                if solver == 'euler':
                    solver = 'unipc'

                if solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False
                    )
                    sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False
                    )
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")


                latents = deepcopy(noise)
                
                # Prepare inputs
                left_idx = r * infer_frames
                right_idx = r * infer_frames + infer_frames
                cond_latents = COND[r] if pose_video else COND[0] * 0
                cond_latents = cond_latents.to(dtype=self.param_dtype, device=self.device)
                audio_input = audio_emb[..., left_idx:right_idx]
                input_motion_latents = motion_latents.clone()

                arg_c = {
                    'context': context[0:1],
                    'seq_len': max_seq_len,
                    'cond_states': cond_latents,
                    "motion_latents": input_motion_latents,
                    'ref_latents': ref_latents,
                    "audio_input": audio_input,
                    "motion_frames": [self.motion_frames, lat_motion_frames],
                    "drop_motion_frames": drop_first_motion and r == 0,
                }
                
                if use_cfg(guide_scale):
                    arg_null = {
                        'context': context_null[0:1],
                        'seq_len': max_seq_len,
                        'cond_states': cond_latents,
                        "motion_latents": input_motion_latents,
                        'ref_latents': ref_latents,
                        "audio_input": 0.0 * audio_input,
                        "motion_frames": [self.motion_frames, lat_motion_frames],
                        "drop_motion_frames": drop_first_motion and r == 0,
                    }

                if offload_model or self.init_on_cpu:
                    self.noise_model.to(self.device)
                    torch.cuda.empty_cache()

                # Diffusion sampling
                for i, t in enumerate(tqdm(timesteps, desc=f"Sampling clip {r + 1}")):
                    latent_model_input = latents[0:1]
                    timestep = torch.stack([t]).to(self.device)

                    noise_pred_cond = self.noise_model(
                        latent_model_input, t=timestep, **arg_c)

                    if use_cfg(guide_scale):
                        noise_pred_uncond = self.noise_model(
                            latent_model_input, t=timestep, **arg_null)
                        noise_pred = [
                            u + guide_scale * (c - u)
                            for c, u in zip(noise_pred_cond, noise_pred_uncond)
                        ]
                    else:
                        noise_pred = noise_pred_cond

                    temp_x0 = sample_scheduler.step(
                        noise_pred[0].unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latents[0] = temp_x0.squeeze(0)

                # Decode latents
                if offload_model:
                    self.noise_model.cpu()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    
                decode_timer = SimpleTimer(f"Decoding clip {r + 1}")
                decode_timer.start()
                
                latents = torch.stack(latents)
                if not (drop_first_motion and r == 0):
                    decode_latents = torch.cat([motion_latents, latents], dim=2)
                else:
                    decode_latents = torch.cat([ref_latents, latents], dim=2)
                    
                if getattr(self, "use_vae_tiling", False):
                    lt = pixel_to_latent_tiles(getattr(self, "vae_tile_px", 128))
                    image = torch.stack([tiled_decode(self.vae, decode_latents[0], latent_tile=lt)])
                else:
                    image = torch.stack(self.vae.decode(decode_latents))
                image = image[:, :, -(infer_frames):]

                if (drop_first_motion and r == 0):
                    image = image[:, :, 3:]

                decode_timer.stop()

                # Update motion frames for next clip
                overlap_frames_num = min(self.motion_frames, image.shape[2])
                videos_last_frames = torch.cat([
                    videos_last_frames[:, :, overlap_frames_num:],
                    image[:, :, -overlap_frames_num:]
                ], dim=2)
                videos_last_frames = videos_last_frames.to(
                    dtype=motion_latents.dtype, device=motion_latents.device)
               if getattr(self, "use_vae_tiling", False):
                    motion_latents = tiled_encode(self.vae, videos_last_frames[0], tile_px=getattr(self, "vae_tile_px", 128)).unsqueeze(0)
                else:
                    motion_latents = torch.stack(self.vae.encode(videos_last_frames))
                
                out.append(image.cpu())
                clip_timer.stop()

        videos = torch.cat(out, dim=2)
        
        # Cleanup
        del noise, latents, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None