#!/bin/bash

python x2rgb/inference.py \
    --checkpoint="Shanlin/Ouroboros" \
    --modality "rgb" \
    --condition "normals" "albedo" "irradiance" "roughness" "metallicity" \
    --noise "gaussian" \
    --seed 0 \
    --prompt="a bed with a white comforter" \
    --albedo_path="./demo/demo1/Albedo.png" \
    --normal_path="./demo/demo1/Normal.png" \
    --roughness_path="./demo/demo1/Roughness.png" \
    --metallic_path="./demo/demo1/Metallic.png" \
    --irradiance_path="./demo/demo1/Irradiance.png" \
    --output_dir="./output"
