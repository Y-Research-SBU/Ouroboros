# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

# @GonzaloMartinGarcia
# The following code is built upon Marigold's run.py, and was adapted to include some new settings
# and normals estimation. All additions made are marked with a # add.

import argparse
import logging
import os
from glob import glob
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline
from load_image import load_exr_image, load_ldr_image

import torchvision

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import random
import sys

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

# Code is from Marigold's util/seed_all.py
def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_paths_from_txt(txt_path):
    """Read paths from a txt file."""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_caption_from_csv(rgb_path, csv_path):
    """Get caption for an RGB path from CSV file."""
    df = pd.read_csv(csv_path)
    match = df[df['file_path'] == rgb_path]
    if len(match) > 0:
        return match.iloc[0]['caption']
    return "an image"  # default caption if not found

def read_line_numbers(file_path):
    """Read line numbers from a file."""
    with open(file_path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image x2rgb estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="GonzaloMG/marigold-e2e-ft-depth", 
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for generation.",
    )
    parser.add_argument(
        "--albedo_path",
        type=str,
        required=True,
        help="Path to the albedo image.",
    )
    parser.add_argument(
        "--normal_path",
        type=str,
        required=True,
        help="Path to the normal image.",
    )
    parser.add_argument(
        "--roughness_path",
        type=str,
        required=True,
        help="Path to the roughness image.",
    )
    parser.add_argument(
        "--metallic_path",
        type=str,
        required=True,
        help="Path to the metallic image.",
    )
    parser.add_argument(
        "--irradiance_path",
        type=str,
        required=True,
        help="Path to the irradiance image.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    parser.add_argument(
        "--modality",
        nargs='+',
        default=['depth'],
        required=True,
    )
    parser.add_argument(
        "--condition",
        nargs='+',
        default=['rgb'],
        help='add conditions to the generation, default is rgb only, to add other condition, use format: --condition rgb cond_1 cond_2 ...'
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    parser.add_argument(
        "--noise",
        type=str,
        default='zeros',
        choices=["gaussian", "pyramid", "zeros"],
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default='trailing',
        choices=["trailing", "leading"],
    ) 

    parser.add_argument(
        "--line_numbers_file",
        type=str,
        help="Path to file containing line numbers to process (optional)"
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to run"
    )

    args = parser.parse_args()

    noise = args.noise
    modality = args.modality
    timestep_spacing = args.timestep_spacing

    checkpoint_path = args.checkpoint
        
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"color_map = {color_map}."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    output_dir_color = os.path.join(output_dir, "colored")
    os.makedirs(output_dir_color, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    unet         = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet_0")   
    vae          = AutoencoderKL.from_pretrained("zheng95z/rgb-to-x", subfolder="vae")  
    text_encoder = CLIPTextModel.from_pretrained("zheng95z/rgb-to-x", subfolder="text_encoder")  
    tokenizer    = CLIPTokenizer.from_pretrained("zheng95z/rgb-to-x", subfolder="tokenizer") 
    scheduler    = DDIMScheduler.from_pretrained("zheng95z/rgb-to-x", timestep_spacing=timestep_spacing, subfolder="scheduler") 
    current_directory = os.path.dirname(os.path.abspath(__file__))
    pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
        checkpoint_path,
        unet=unet, 
        vae=vae, 
        scheduler=scheduler, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        torch_dtype=torch.float16,
        cache_dir=os.path.join(current_directory, "model_cache"),
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  

    pipe = pipe.to(device)
    pipe.unet.eval()

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        prompt = args.prompt
        albedo_path = args.albedo_path
        normal_path = args.normal_path
        roughness_path = args.roughness_path
        metallic_path = args.metallic_path
        irradiance_path = args.irradiance_path
        
        output_dir_color = os.path.join(output_dir, "colored")
        os.makedirs(output_dir_color, exist_ok=True)

        albedo_image = None
        normal_image = None
        roughness_image = None  
        metallic_image = None
        irradiance_image = None

        # Load albedo
        if albedo_path.endswith(".exr"):
            albedo_image = load_exr_image(albedo_path, clamp=True).to("cuda")
        else:
            albedo_image = load_ldr_image(albedo_path, from_srgb=True).to("cuda")

        # Load normal
        if normal_path.endswith(".exr"):
            normal_image = load_exr_image(normal_path, normalize=True).to("cuda")
        else:
            normal_image = load_ldr_image(normal_path, normalize=True).to("cuda")

        # Load roughness
        if roughness_path.endswith(".exr"):
            roughness_image = load_exr_image(roughness_path, clamp=True).to("cuda")
        else:
            roughness_image = load_ldr_image(roughness_path, clamp=True).to("cuda")

        # Load metallic
        if metallic_path.endswith(".exr"):
            metallic_image = load_exr_image(metallic_path, clamp=True).to("cuda")
        else:
            metallic_image = load_ldr_image(metallic_path, clamp=True).to("cuda")

        # Load irradiance
        if irradiance_path.endswith(".exr"):
            irradiance_image = load_exr_image(irradiance_path, tonemaping=True, clamp=True).to("cuda")
        else:
            irradiance_image = load_ldr_image(irradiance_path, from_srgb=True, clamp=True).to("cuda")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        height = 768
        width = 768
        images = [albedo_image, normal_image, roughness_image, metallic_image, irradiance_image]
        for img in images:
            if img is not None:
                height = img.shape[1]
                width = img.shape[2]
                break

        required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
        pipe_outs = []
        generated_image = pipe(
            prompt=prompt,
            albedo=albedo_image,
            normal=normal_image,
            roughness=roughness_image,
            metallic=metallic_image,
            irradiance=irradiance_image,
            num_inference_steps=1,
            height=height,
            width=width,
            generator=generator,
            required_aovs=required_aovs,
            guidance_scale=0,
            image_guidance_scale=0,
            guidance_rescale=0,
        ).images[0]
        pipe_outs.append({"colored": generated_image})

        # Save outputs
        for pipe_out in pipe_outs:
            pred_colored: Image.Image = pipe_out["colored"]
            input_basename = os.path.splitext(os.path.basename(albedo_path))[0]
            colored_save_path = os.path.join(output_dir_color, "color.png")
            pred_colored.save(colored_save_path)

