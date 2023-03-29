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
If you'd like to use a different checkpoint, point to it in the config file `configs/train.yaml`, on line 8, after `ckpt_path:`. 

Next, we need to change the config to point to our downloaded (or generated) dataset. If you're using the `clip-filtered-dataset` from above, you can skip this. Otherwise, you may need to edit lines 85 and 94 of the config (`data.params.train.params.path`, `data.params.validation.params.path`). 

Finally, start a training job with the following command:

```
python main.py --name default --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
```


## Creating your own dataset

Our generated dataset of paired images and editing instructions is made in two phases: First, we use GPT-3 to generate text triplets: (a) a caption describing an image, (b) an edit instruction, (c) a caption describing the image after the edit. Then, we turn pairs of captions (before/after the edit) into pairs of images using Stable Diffusion and Prompt-to-Prompt.

## Evaluation

To generate plots like the ones in Figures 8 and 10 in the paper, run the following command:

```
python metrics/compute_metrics.py --ckpt /path/to/your/model.ckpt
```

## Tips

If you're not getting the quality result you want, there may be a few reasons:
1. **Is the image not changing enough?** Your Image CFG weight may be too high. This value dictates how similar the output should be to the input. It's possible your edit requires larger changes from the original image, and your Image CFG weight isn't allowing that. Alternatively, your Text CFG weight may be too low. This value dictates how much to listen to the text instruction. The default Image CFG of 1.5 and Text CFG of 7.5 are a good starting point, but aren't necessarily optimal for each edit. Try:
    * Decreasing the Image CFG weight, or
    * Increasing the Text CFG weight, or
2. Conversely, **is the image changing too much**, such that the details in the original image aren't preserved? Try:
    * Increasing the Image CFG weight, or
    * Decreasing the Text CFG weight
3. Try generating results with different random seeds by setting "Randomize Seed" and running generation multiple times. You can also try setting "Randomize CFG" to sample new Text CFG and Image CFG values each time.
4. Rephrasing the instruction sometimes improves results (e.g., "turn him into a dog" vs. "make him a dog" vs. "as a dog").
5. Increasing the number of steps sometimes improves results.
6. Do faces look weird? The Stable Diffusion autoencoder has a hard time with faces that are small in the image. Try cropping the image so the face takes up a larger portion of the frame.

## Comments

- Our codebase is based on the [Stable Diffusion codebase](https://github.com/CompVis/stable-diffusion).

## BibTeX

```
@article{brooks2022instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  journal={arXiv preprint arXiv:2211.09800},
  year={2022}
}
```
