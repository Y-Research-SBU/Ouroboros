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

import numpy as np
import torch
from PIL import Image

from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from load_image import load_exr_image, load_ldr_image

import torchvision

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import random

EXTENSION_LIST = [".jpg", ".jpeg", ".png", ".exr", ".hdf5"]

# Code is from Marigold's util/seed_all.py
def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="GonzaloMG/marigold-e2e-ft-depth", 
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--input_rgb_path",
        type=str,
        required=True,
        help="Path to the input image.",
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
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
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
        "--num_inference_steps",
        type=int,
        default=1,
        help="Number of inference steps"
    )

    args = parser.parse_args()

    noise = args.noise
    modality = args.modality
    timestep_spacing = args.timestep_spacing

    checkpoint_path = args.checkpoint
    input_rgb_path = args.input_rgb_path
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

    # -------------------- Data --------------------
    # Check if input image exists
    if not os.path.exists(input_rgb_path):
        logging.error(f"Input image not found: '{input_rgb_path}'")
        exit(1)
    
    logging.info(f"Processing image: {input_rgb_path}")

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

    unet         = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")   
    vae          = AutoencoderKL.from_pretrained("zheng95z/rgb-to-x", subfolder="vae")  
    text_encoder = CLIPTextModel.from_pretrained("zheng95z/rgb-to-x", subfolder="text_encoder")  
    tokenizer    = CLIPTokenizer.from_pretrained("zheng95z/rgb-to-x", subfolder="tokenizer") 
    scheduler    = DDIMScheduler.from_pretrained("zheng95z/rgb-to-x", timestep_spacing=timestep_spacing, subfolder="scheduler") 

    current_directory = os.path.dirname(os.path.abspath(__file__))
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
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
        os.makedirs(output_dir, exist_ok=True)

        rgb_path = input_rgb_path

        generator = torch.Generator(device="cuda").manual_seed(seed)

        if rgb_path.endswith(".exr"):
            photo = load_exr_image(rgb_path, tonemaping=True, clamp=True).to("cuda")
        else:
            photo = load_ldr_image(rgb_path, from_srgb=True).to("cuda")

        new_height = photo.shape[1]
        new_width = photo.shape[2]

        if new_width % 8 != 0 or new_height % 8 != 0:
            new_width = new_width // 8 * 8
            new_height = new_height // 8 * 8

        photo = torchvision.transforms.Resize((new_height, new_width))(photo)

        required_aovs = modality
        prompts = {
            "albedo": "Albedo (diffuse basecolor)",
            "normals": "Camera-space Normal",
            "roughness": "Roughness",
            "metallicity": "Metallicness",
            "irradiance": "Irradiance (diffuse lighting)",
        }
        pipe_outs = []
        for m in required_aovs:
            prompt = prompts[m]
            generated_image = pipe(
                prompt=prompt,
                photo=photo,
                num_inference_steps=1,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[m],
            ).images[0][0]
            pipe_outs.append({"modality": m, f"{m}_colored": generated_image})

        for pipe_out in pipe_outs:
            m = pipe_out["modality"]
            pred_colored: Image.Image = pipe_out[f"{m}_colored"]

            modality_output_dir = os.path.join(output_dir, m)
            os.makedirs(modality_output_dir, exist_ok=True)
            
            pred_name = f"{m}.png"
            colored_save_path = os.path.join(modality_output_dir, pred_name)

            pred_colored.save(colored_save_path)

