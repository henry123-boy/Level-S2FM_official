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



python train.py --group=BlendedMVS --pipeline=LevelS2fM --yaml=bmvs --name=Fountain --data.dataset=BlendedMVS --data.scene=Fountain   --sfm_mode=full --refine_again=false



 
