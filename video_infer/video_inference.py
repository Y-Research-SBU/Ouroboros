import os
import argparse
import random
import numpy as np
import torch
import torchvision
import cv2
from tqdm.auto import tqdm
from diffusers import DDIMScheduler
from PIL import Image

from pipeline_rgb2x_flatten import StableDiffusionAOVMatEstPipeline
from load_image import load_ldr_image
from flatten_models.unet import UNet3DConditionModel


def seed_all(seed: int = 0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_video_from_frames(frame_directory, aov_name):

	video_name = os.path.join(frame_directory, f"{aov_name}_out.mp4")

	# Check if video already exists
	# if so, skip this directory
	if os.path.exists(video_name):
		print(f"Video already exists for {video_name} !")
		return

	images = [img for img in sorted(os.listdir(frame_directory), key=lambda x: (len(x), x))
				if img.endswith(".jpg") or
				img.endswith(".jpeg") or
				img.endswith("png")]

	if len(images) > 0:
		height, width, _ = cv2.imread(os.path.join(frame_directory, images[0])).shape
		
		video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height), True)
		
		print("Generating Video...")
		for image in images:
			curr_frame = cv2.imread(os.path.join(frame_directory, image))
			
			video.write(curr_frame)

		cv2.destroyAllWindows()
		video.release()

		compressed_video_name = os.path.join(frame_directory, f"{aov_name}_out_compressed.mp4")

		print(compressed_video_name)
		# Convert video to higher MPEG-4 compression
		os.system(f'ffmpeg -i "{video_name}" -c:v libx264 -crf 23 -preset medium -y "{compressed_video_name}"')

		print(f"Video created for {frame_directory}")


def load_frames(input_dir: str,
                          num_frames: int,
                          device: torch.device,
                          max_side: int = 1000):

    photos = []
    origin_shape = None

    for idx in range(num_frames):
        img_path = os.path.join(input_dir, f"{idx}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Frame not found: {img_path}")

        photo = load_ldr_image(img_path, from_srgb=True).to(device)
        if origin_shape is None:
            old_height, old_width = photo.shape[1:]
            ratio = old_height / old_width
            if old_height > old_width:
                new_height = max_side
                new_width = int(new_height / ratio)
            else:
                new_width = max_side
                new_height = int(new_width * ratio)
            new_width -= new_width % 8
            new_height -= new_height % 8
            origin_shape = (old_height, old_width)
        photo = torchvision.transforms.Resize((new_height, new_width))(photo)
        photos.append(photo)
    return photos, origin_shape, new_height, new_width


def build_pipeline(checkpoint_path: str,
                   device: torch.device,
                   half_precision: bool,
                   timestep_spacing: str = "trailing"):

    dtype = torch.float16 if half_precision else torch.float32

    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=dtype,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing=timestep_spacing,
    )
    pipe.unet = UNet3DConditionModel.from_pretrained_2d(
        checkpoint_path,
        subfolder="unet_1",
        state_dict=None,
    ).to(dtype=dtype)

    pipe.set_progress_bar_config(disable=True)
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet.eval()
    pipe.to(device)
    return pipe, dtype


def run(args):
    seed_all(args.seed)
    device = torch.device(args.device)

    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normals": "Camera-space Normal",
        "roughness": "Roughness",
        "metallicity": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }
    prompt = prompts[args.required_aovs]
    print(f"[Run] Generating {args.required_aovs} ...")

    pipe, dtype = build_pipeline(
        args.checkpoint_path,
        device=device,
        half_precision=args.half_precision,
        timestep_spacing=args.timestep_spacing,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    photos, origin_shape, new_height, new_width = load_frames(
        args.input_dir,
        args.num_frames,
        device=device,
        max_side=args.max_side,
    )

    window_size, stride = args.window_size, args.stride
    overlap = window_size - stride
    num_iters = (args.num_frames - window_size) // stride + 1
    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        last_latent = None
        latent_common = torch.randn(
            (1, 4, 1, new_height // 8, new_width // 8),
            generator=generator,
            dtype=dtype,
            device=device,
        ).repeat(1, 1, window_size, 1, 1)

        for i in tqdm(range(num_iters), desc="Processing frames"):
            start, end = i * stride, i * stride + window_size
            batch = photos[start:end]
            latents = latent_common.clone()
            if last_latent is not None:
                latents[:, :, :overlap] = (
                    args.scale * latents[:, :, :overlap]
                    + (1 - args.scale) * last_latent[:, :, stride:]
                )
            outputs = pipe(
                prompt=prompt,
                photos=batch,
                num_inference_steps=args.denoise_steps,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[args.required_aovs],
                return_dict=False,
                old_qk=0,
                origin_shape=list(origin_shape),
                latents=latents,
            )

            if last_latent is not None:
                for i in range(overlap, len(outputs[0])):
                    outputs[0][i][0].save(f"{args.save_dir}/frame{start + i}.png")
            else:
                for i in range(0, len(outputs[0])):
                    outputs[0][i][0].save(f"{args.save_dir}/frame{start + i}.png")

        last_latent = outputs[1]

    create_video_from_frames(
        args.save_dir, args.required_aovs
    )


def parse_args():
    parser = argparse.ArgumentParser(description="RGB-to-X Video Generation Script")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=198)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--required_aovs", type=str, default="albedo",
                        choices=["albedo", "normals", "roughness", "metallicity", "irradiance"])
    parser.add_argument("--denoise_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep_spacing", type=str, default="trailing")
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_side", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
