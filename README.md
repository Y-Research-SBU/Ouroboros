
<div align="center">

<table>
<tr>
<td><img src="./img/logo.png" height="100" style="object-fit: contain;"></td>
<td>
  <h2>Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering</h2>
</td>
</tr>
</table>


<p>
  <a href="https://siwensun.github.io/">Shanlin Sun</a><sup>1 ★</sup>&nbsp;
  <a href="https://yfwang.me/">Yifan Wang</a><sup>2 ★</sup>&nbsp;
  <a href="https://github.com/zhw123456789/">Hanwen Zhang</a><sup>3 ★</sup>&nbsp;
  <a href="https://yuukino22.github.io/">Yifeng Xiong</a><sup>1</sup>&nbsp;
  <a href="https://scholar.google.com/citations?user=Tcg-9DcAAAAJ&hl=zh-CN">Qin Ren</a><sup>2</sup>&nbsp; <br>
  <a href="https://lab-smile.github.io/">Ruogu Fang</a><sup>4</sup>&nbsp;
  <a href="https://ics.uci.edu/~xhx/">Xiaohui Xie</a><sup>1</sup>&nbsp;
  <a href="https://chenyuyou.me/">Chenyu You</a><sup>2</sup>
</p>

<p>
  <sup>1</sup> University of California, Irvine &nbsp;&nbsp; 
  <sup>2</sup> Stony Brook University &nbsp;&nbsp; 
  <sup>3</sup> Huazhong University of Science and Technology &nbsp;&nbsp; <br>
  <sup>4</sup> University of Florida &nbsp;&nbsp; 
  <sup>★</sup> Equal Contribution <br>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2508.14461">
    <img src="https://img.shields.io/badge/ArXiv-2508.14461-b31b1b?style=flat-square&logo=arxiv&labelColor=000000&logoColor=FFFFFF" alt="Paper">
  </a>
  <a href="https://siwensun.github.io/ouroboros-project/">
    <img src="https://img.shields.io/badge/Project-Website-3B82F6?style=flat-square&logo=googlechrome&logoColor=3B82F6&labelColor=000000" alt="Project Website">
  </a>
  <a href="https://huggingface.co/Y-Research-Group/Ouroboros/tree/main">
    <img src="https://img.shields.io/badge/HuggingFace-Model-FF9A00?style=flat-square&logo=huggingface&labelColor=000000&logoColor=FF9A00" alt="HuggingFace Model">
  </a>
</p>

</div>

# Abstract
While multi-step diffusion models have advanced both forward and inverse rendering, existing approaches often treat these problems independently, leading to cycle inconsistency and slow inference speed. In this work, we present **Ouroboros**, a framework composed of two single-step diffusion models that handle forward and inverse rendering with mutual reinforcement. Our approach extends intrinsic decomposition to both indoor and outdoor scenes and introduces a cycle consistency mechanism that ensures coherence between forward and inverse rendering outputs. Experimental results demonstrate state-of-the-art performance across diverse scenes while achieving substantially faster inference speed compared to other diffusion-based methods. We also demonstrate that Ouroboros can transfer to video decomposition in a training-free manner, reducing temporal inconsistency in video sequences while maintaining high-quality per-frame inverse rendering.
<p align="center">
  <img src="./img/teaser.png" width="100%">
</p>

**Figure:** **Single-step Diffusion Models for Forward and Inverse Rendering in Cycle Consistency**.  
**Left Upper:** Ouroboros decomposes input images into intrinsic maps (albedo, normal, roughness, metallicity, and irradiance). Given these generated intrinsic maps and textual prompts, our neural forward rendering model synthesizes images closely matching the originals.  
**Right Upper:** We extend an end-to-end finetuning technique to diffusion-based neural rendering, outperforming state-of-the-art RGB↔X in both speed and accuracy. The radar plot illustrates numerical comparisons on the InteriorVerse dataset.  
**Bottom:** Our method achieves temporally consistent video inverse rendering without specific finetuning on video data.
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

## Contact

For questions, feedback, or collaboration opportunities, please contact:

**Email**: [shanlins@uci.edu](mailto:shanlins@uci.edu), [chenyu.you@stonybrook.edu](mailto:chenyu.you@stonybrook.edu)

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
