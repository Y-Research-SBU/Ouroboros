#!/bin/bash

python rgb2x/inference.py \
    --checkpoint="" \
    --modality "normals" "albedo" "irradiance" "roughness" "metallicity" \
    --condition "rgb" \
    --noise "gaussian" \
    --seed 0 \
    --input_rgb_path="" \
    --output_dir="" 
