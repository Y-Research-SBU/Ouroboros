#!/bin/bash

python rgb2x/inference.py \
    --checkpoint="Shanlin/Ouroboros" \
    --modality "normals" "albedo" "irradiance" "roughness" "metallicity" \
    --condition "rgb" \
    --noise "gaussian" \
    --seed 0 \
    --input_rgb_path="./demo/demo1/RGB.png" \
    --output_dir="./output" 
