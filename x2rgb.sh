#!/bin/bash

python x2rgb/inference.py \
    --checkpoint="" \
    --modality "rgb" \
    --condition "normals" "albedo" "irradiance" "roughness" "metallicity" \
    --noise "gaussian" \
    --seed 0 \
    --prompt="" \
    --albedo_path="" \
    --normal_path="" \
    --roughness_path="" \
    --metallic_path="" \
    --irradiance_path="" \
    --output_dir="" \