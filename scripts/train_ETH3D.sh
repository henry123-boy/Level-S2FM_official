#!/bin/bash
__conda_setup="$('/root/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
cd ..
conda activate levels2fm

python train.py --group=ETH3D --pipeline=LevelS2fM --yaml=ETH3D --name=relief --data.dataset=ETH3D --data.scene=relief   --sfm_mode=full --nbv_mode=ours --refine_again=false

python train.py --group=ETH3D --pipeline=LevelS2fM --yaml=ETH3D --name=courtyard --data.dataset=ETH3D --data.scene=courtyard   --sfm_mode=full --nbv_mode=ours --refine_again=false

python train.py --group=ETH3D --pipeline=LevelS2fM --yaml=ETH3D --name=facade --data.dataset=ETH3D --data.scene=facade   --sfm_mode=full --nbv_mode=ours --refine_again=false





python train.py --group=DTU --pipeline=LevelS2fM --yaml=DTU --name=65 --data.dataset=DTU --data.scene=scan65   --sfm_mode=full --nbv_mode=ours --refine_again=false

python train.py --group=DTU --pipeline=LevelS2fM --yaml=DTU --name=24 --data.dataset=DTU --data.scene=scan24   --sfm_mode=full --nbv_mode=ours  --refine_again=false

python train.py --group=DTU --pipeline=LevelS2fM --yaml=DTU --name=37 --data.dataset=DTU --data.scene=scan37   --sfm_mode=full --nbv_mode=ours --refine_again=false



 
