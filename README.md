##### 🔥 News

- `30/03/2022` :wink: Code of Level-S<sup>2</sup> fM Release

# Level-S<sup>2</sup>fM: Structure from Motion on Neural Level Set of Implicit Surfaces

### [Project Page](https://henry123-boy.github.io/level-s2fm/) | [Paper](https://arxiv.org/pdf/2211.12018.pdf) | [Data](https://henry123-boy.github.io/level-s2fm/)



<img src='https://raw.githubusercontent.com/henry123-boy/henry123-boy.github.io/main/level-s2fm/static/images/teaser1-2.png'/>

> [Level-S<sup>2</sup>fM: Structure from Motion on Neural Level Set of Implicit Surfaces](https://henry123-boy.github.io/level-s2fm/) 
>
>  [Yuxi Xiao](https://henry123-boy.github.io/),  [Nan Xue](https://xuenan.net/),  [Tianfu Wu](https://research.ece.ncsu.edu/ivmcl/), [Gui-Song Xia](http://www.captain-whu.com/xia_En.html)
>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR2023)

## TL;DR: 

Level-S<sup>2</sup> fM presents a neural incremental Structure from Motion(SfM) pipeline based on the neural level sets representation. It leverages the 2D image matches and neural rendering to drive the joint optimization of **Camera Poses**,  **SDF**  and **Radiance Field**.

### Set up a conda environment, and install [CUDA Extension](https://github.com/kwea123/ngp_pl):
```
# create the environment from the yaml
conda env create -f env.yaml
conda activate levels2fm
```

#### Install vren from the [ngp_pl]((https://github.com/kwea123/ngp_pl))

- Clone ngp_pl by 

  ```
  git clone https://github.com/kwea123/ngp_pl
  ```

- get into the directory of ngp_pl

- install the cuda extension

  ```
  pip install models/csrc
  ```

## Prepare Data

In our default setting, Level-S<sup>2</sup> fM depends on the 2D image Matches by SIFT. To leverage existing solutions and avoid redundancy, we directly uses the SIFT matches and pose graph from COLMAP.  We provide our [processed data](https://drive.google.com/file/d/13Ap_UA244OdqPwYlvSpMUcKt3CzrTKfS/view?usp=sharing) in Google Drive. Please download and `unzip` them into the folder `./data` for runing.  


## Reconstruction with Level-S<sup>2</sup>fM 

#### Running Default Version 

In our default version, our Level-S<sup>2</sup> fM uses the **SDF-based Triangulation** and **Neural Bundle Adjustment**, where the SDF plays as a top-down regularization to manage the sparse pointset with feature track and filter the outliers.

```bash
python train.py --group=<group_name_exp> --pipeline=LevelS2fM --yaml=<config file> --name=<exp_name> --data.dataset=<dataset> --data.scene=<scene_name>   --sfm_mode=full --nbv_mode=ours --Ablate_config.dual_field=true                          
```
#### Running with Some Ablations 

Trying our Level-S<sup>2</sup> fM with the traditional triangulation:

```bash
python train.py --group=<group_name_exp> --pipeline=LevelS2fM --yaml=<config file> --name=<exp_name> --data.dataset=<dataset> --data.scene=<scene_name>   --sfm_mode=full --nbv_mode=ours --Ablate_config.dual_field=true --Ablate_config.tri_trad=true
```

Trying our Level-S<sup>2</sup> fM with the traditional Bundle Adjustment:

```bash
python train.py --group=<group_name_exp> --pipeline=LevelS2fM --yaml=<config file> --name=<exp_name> --data.dataset=<dataset> --data.scene=<scene_name>   --sfm_mode=full --nbv_mode=ours --Ablate_config.dual_field=true --Ablate_config.tri_trad=true --Ablate_config.ba_trad=true
```

#### Running with provided Scripts

`cd` into `./scripts`, and run the script file like:

```sh
CUDA_VISIBLE_DEVICES=<GPU> sh train_ETH3D.sh
```

## Creating your own dataset

A complete Instruction is coming soon! 

## Tips

Coming with instrcution.

## Comments

Our Level-S<sup>2</sup> fM provide a new perspective to revisit the traditional sparse reconstruction (SfM) with **Neural Field Representation** and **Neural Rendering**. This work may contribute to let you see the capability of a simple **coordinate MLP** in SfM. However, It's not going to be very mature system yet like  [COLMAP](https://github.com/colmap/colmap), and we are continuing to refine it in the future.

## Acknowledgement

- Thanks to Johannes Schönberger for his excellent work [COLMAP](https://github.com/colmap/colmap).
- Thanks to Thomas Müller for his excellent work [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and  [Instant NGP](https://github.com/NVlabs/instant-ngp) 
- Thanks to Lior Yariv for her excellent work [VolSDF](https://lioryariv.github.io/volsdf/).
- Thanks to AI Aoi for his excellent implementation of [Instant NGP](https://github.com/NVlabs/instant-ngp) by pytorch [ngp_pl](https://github.com/kwea123/ngp_pl).
- Thanks to [Sida Peng](https://pengsida.net/) for his valuable suggestions and discussions for our Level-S<sup>2</sup> fM

## BibTeX

```
@inproceedings{xiao2022level,
  title     = {Level-S\({}^{\mbox{2}}\)fM: Structure from Motion on Neural Level
               Set of Implicit Surfaces},
  author={Yuxi Xiao and Nan Xue and Tianfu Wu and Gui-Song Xia},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
