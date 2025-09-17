
# Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://siwensun.github.io/ouroboros-project/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.14461-b31b1b.svg)](https://arxiv.org/abs/2508.14461)

> **Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering**
> 
> [Shanlin Sun](https://siwensun.github.io/), [Yifan Wang](https://yfwang.me/), [Hanwen Zhang](https://github.com/zhw123456789/), [Yifeng Xiong](https://yuukino22.github.io/), [Qin Ren](https://scholar.google.com/citations?user=Tcg-9DcAAAAJ&hl=zh-CN), [Ruogu Fang](https://lab-smile.github.io/), [Xiaohui Xie](https://ics.uci.edu/~xhx/) and [Chenyu You](https://chenyuyou.me/)
> 
> - **Institutions**: University of California, Irvine; Stony Brook University; Huazhong University of Science and Technology; University of Florida 
> - **Contact**: [Shanlin Sun](https://siwensun.github.io/) (shanlins@uci.edu)

## TODO List

- \[x] Release inference codes and checkpoints.
- \[ ] Release training codes.
- \[ ] Release training dataset.

## Installation

### Prerequisites

- Python 3.12
- CUDA-compatible GPU 
- Conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Y-Research-SBU/Ouroboros/tree/main#
cd Ouroboros
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ouroboros
```

## Usage

### RGB to Material Properties (rgb2x)

Estimate material properties from RGB images:

```bash
python rgb2x/inference.py \
    --checkpoint="path/to/checkpoint" \
    --modality "normals" "albedo" "irradiance" "roughness" "metallicity" \
    --condition "rgb" \
    --noise "gaussian" \
    --input_rgb_path="path/to/input.jpg" \
    --output_dir="path/to/output"
```

### Material Properties to RGB (x2rgb)

Generate RGB images from material properties:

```bash
python x2rgb/inference.py \
    --checkpoint="path/to/checkpoint" \
    --modality "rgb" \
    --condition "normals" "albedo" "irradiance" "roughness" "metallicity" \
    --noise "gaussian" \
    --prompt="your text prompt" \
    --albedo_path="path/to/albedo.png" \
    --normal_path="path/to/normal.png" \
    --roughness_path="path/to/roughness.png" \
    --metallic_path="path/to/metallic.png" \
    --irradiance_path="path/to/irradiance.png" \
    --output_dir="path/to/output"
```

## Parameters

### Common Parameters

- `--checkpoint`: Path to model checkpoint or Hugging Face model name
- `--modality`: List of modalities to generate/estimate
- `--condition`: List of conditioning modalities
- `--noise`: Noise type (`gaussian`, `pyramid`, `zeros`)
- `--denoise_steps`: Number of denoising steps

### RGB2X Specific

- `--input_rgb_path`: Path to input RGB image
- `--output_dir`: Output directory for material properties

### X2RGB Specific

- `--prompt`: Text prompt for generation
- `--albedo_path`: Path to albedo image
- `--normal_path`: Path to normal map
- `--roughness_path`: Path to roughness map
- `--metallic_path`: Path to metallic map
- `--irradiance_path`: Path to irradiance map
- `--output_dir`: Output directory for generated RGB

## Citation

If you use this code in your research, please cite with:

```bibtex
@article{sun2025ouroboros,
  title={Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering},
  author={Sun, Shanlin and Wang, Yifan and Zhang, Hanwen and Xiong, Yifeng and Ren, Qin and Fang, Ruogu and Xie, Xiaohui and You, Chenyu},
  journal={arXiv preprint arXiv:2508.14461},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

