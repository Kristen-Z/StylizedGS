# StylizedGS: Controllable Stylization for 3D Gaussian Splatting

<div align="center">
<a href="https://kristen-z.github.io/stylizedgs/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2404.05220" target="_blank" rel="noopener noreferrer"> <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF"></a>
<a href="https://drive.google.com/file/d/1PuDanf0JnpyCRtiurPIESG80pU_nEBIj/view"> <img src="https://img.shields.io/badge/Demo-blue" alt="Demo"></a>

<p>
    <a href="https://kristen-z.github.io/">Dingxi Zhang</a>   
    ·
    <a href="http://people.geometrylearning.com/yyj/">Yu-Jie Yuan</a>
    · 
    <a herf="https://seancomeon.github.io/">Zhuoxun Chen</a>
    ·
    <a href="https://people.wgtn.ac.nz/fanglue.zhang">Fang-Lue Zhang</a>
    ·
    <a href="https://lynnho.github.io/">Zhenliang He</a>
    ·
    <a href="https://people.ucas.edu.cn/~sgshan">Shiguang Shan</a>
    ·
    <a href="http://www.geometrylearning.com/lin/">Lin Gao<sup>*</sup></a>
</p>

<img src="./assets/teaser.jpg" alt="[Teaser Figure]" style="zoom:80%;" />
</div>

Given a 2D style image, the proposed StylizedGS method can stylize the pre-trained 3D Gaussian Splatting to match the desired style with detailed geometric features and satisfactory visual quality within a few minutes. We also enable users to control several perceptual factors, such as color, the style pattern size (scale), and the stylized regions (spatial), during the stylization to enhance the customization capabilities.

## Setup

### Installation
Clone the repository and install necessary dependencies：

```
conda create -n stylizedgs python==3.10
conda activate stylizedgs
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/fuesd-ssim
pip install -e submodules/simple-knn
```

### Data Preparation
We evaluate the dataset on [LLFF](https://bmild.github.io/llff/), [Tanks and Temples](https://www.tanksandtemples.org/) and [MipNeRF-360](https://jonbarron.info/mipnerf360/) datasets. For convenience, a small subset of preprocessed scene data and reference style images is provided [here](https://drive.google.com/file/d/1U7MTzKAFNY0XbJ4tnr8BwsFHKbedOOyw/view?usp=sharing).

To use custom data, please follow the instructions in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/README.md#processing-your-own-scenes) to process your own scenes.

The `datasets` folder is organized as follows:
```
datasets
|---llff
|   |---flower
|   |---horns
|   |---...
|---tandt
|---mipnerf360
|---styles
|   |---0.jpg
|   |---1.jpg
|   |---...
```

## Quick Start

To run the stylization script on a specific scene, use:
```
bash stylizedGS.sh [DATA_TYPE] [SCENE_NAME] [STYLE_ID]
e.g. bash stylizedGS.sh llff flower 14
```

+ `DATA_TYPE`: dataset name (llff, tandt, or mipnerf360)

+ `SCENE_NAME`: name of the scene folder within the dataset

+ `STYLE_ID`: index of the reference style image in `datasets/styles/`

This command will load the selected scene and style, train the 3DGS representation, perform stylization, and save the rendered results in `output/ckpt_stylegs/[DATA_TYPE]/[SCENE_NAME]_[STYLE_ID]/video`.



## Acknowledgements
Our work is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [ARF](https://github.com/Kai-46/ARF-svox2). We thank the authors for their great work and open-sourcing the code.

## Citation
```
@article{zhang2024stylizedgs,
  title={Stylizedgs: Controllable stylization for 3d gaussian splatting},
  author={Zhang, Dingxi and Yuan, Yu-Jie and Chen, Zhuoxun and Zhang, Fang-Lue and He, Zhenliang and Shan, Shiguang and Gao, Lin},
  journal={arXiv preprint arXiv:2404.05220},
  year={2024}
}
```