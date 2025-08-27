# StylizedGS: Controllable Stylization for 3D Gaussian Splatting

<div align="center">
<a href="https://ivl.cs.brown.edu/research/gigahands.html"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://www.arxiv.org/abs/2412.04244" target="_blank" rel="noopener noreferrer"> <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF"></a>
<p>
    <a href="https://kristen-z.github.io/">Dingxi Zhang</a>   
    ·
    <a href="http://people.geometrylearning.com/yyj/">Yu-Jie Yuan</a>
    · 
    <a>Zhuoxun Chen</a>
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

## Abstract
As XR technology continues to advance rapidly, 3D generation and editing are increasingly crucial. Among these, stylization plays a key role in enhancing the appearance of 3D models. By utilizing stylization, users can achieve consistent artistic effects in 3D editing using a single reference style image, making it a user-friendly editing method. However, recent NeRF-based 3D stylization methods encounter efficiency issues that impact the user experience, and their implicit nature limits their ability to accurately transfer geometric pattern sles. Additionally, the ability for artists to apply flexible control over stylized scenes is considered highly desirable to foster an environment conducive to creative exploration. To address the above issues, we introduce StylizedGS, an efficient 3D neural style transfer framework with adaptable control over perceptual factors based on 3D Gaussian Splatting (3DGS) representation. We propose a filter-based refinement to eliminate floaters that affect the stylization effects in the scene reconstruction process. The nearest neighbor-based style loss is introduced to achieve stylization by fine-tuning the geometry and color parameters of 3DGS, while a depth preservation loss with other regularizations is proposed to prevent the tampering of geometry content. Moreover, facilitated by specially designed losses, StylizedGS enables users to control color, stylized scale, and regions during the stylization to possess customization capabilities. Our method achieves high-quality stylization results characterized by faithful brushstrokes and geometric consistency with flexible controls. Extensive experiments across various scenes and styles demonstrate the effectiveness and efficiency of our method concerning both stylization quality and inference speed.

## Setup

### Installation
Clone the repository and install necessary dependencies：

```
conda create -n stylizedgs python==3.10
conda activate stylizedgs
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### Data Preparation
We evaluate the dataset on [LLFF](https://bmild.github.io/llff/), [Tanks and Temples](https://www.tanksandtemples.org/) and [MipNeRF-360](https://jonbarron.info/mipnerf360/) dataset. Please download the scene data and style imags by running:
```
bash download_data.sh
```
