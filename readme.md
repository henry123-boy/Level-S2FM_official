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

In our default setting, Level-S<sup>2</sup> fM depends on the 2D image Matches by SIFT. To leverage existing solutions and avoid redundancy, we directly uses the SIFT matches and pose graph from COLMAP.  We provide our [processed data](https://henry123-boy.github.io/level-s2fm/) in Google Drive. Please download and `unzip` them into the folder `./data` for runing.  


## Reconstruction with Level-S<sup>2</sup>fM 

#### Running Default Version

```bash
python train.py --group=<group_name_exp> --pipeline=LevelS2fM --yaml=<config file> --name=<exp_name> --data.dataset=<dataset> --data.scene=<scene_name>   --sfm_mode=full --nbv_mode=ours --Ablate_config.dual_field=true                          
```

## Creating your own dataset



## Evaluation

## Tips



## Comments



## BibTeX

```
@article{xiao2022level,
      title={Level-S $\^{} 2$ fM: Structure from Motion on Neural Level Set of Implicit Surfaces},
      author={Xiao, Yuxi and Xue, Nan and Wu, Tianfu and Xia, Gui-Song},
      journal={arXiv preprint arXiv:2211.12018},
      year={2022}
    }
```
