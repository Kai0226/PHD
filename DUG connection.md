copy files from local to DUG server:
```
rsync -aP localFile dug:/data/uwa_multimodality/data/
```
HPC FastX Connection Guide for Linux:
https://dugeo.slack.com/docs/T08L8QB7Z/F06AA13J84A

Check space of your home directory:
```
du -chs ~/.[a-zA-Z]*/ ~/*/
```
Since DUG only support 10G space in the home directory, 
```
$ mkdir -p /data/uwa_multimodality/uwa_niuk/env/conda

$ conda create -p /data/uwa_multimodality/uwa_niuk/env/conda/text2video-finetune python=3.10   # specify env install path

$ conda create --name ENV_NAME --clone /data/uwa_multimodality/uwa_niuk/env/conda/text2video-finetune
```

Add the below lines into .bashrc, and source .bashrc
```
## >>> conda initialize >>>
. /data/uwa_multimodality/uwa_niuk/etc/profile.d/conda.sh
## <<< conda initialize <<<
```

change path from your home directory to /data/uwa_multimodality/
```
export CONDA_PKGS_DIRS=/data/uwa_multimodality/uwa_niuk/.conda-pkgs
export PIP_CACHE_DIR=/data/uwa_multimodality/uwa_niuk/.pip-cache

mkdir "$PIP_CACHE_DIR"
mv ~/.cache/pip/* "$PIP_CACHE_DIR/"

mkdir /data/uwa_multimodality/uwa_niuk/tmp
export TMPDIR=/data/uwa_multimodality/uwa_niuk/tmp
```


Set TORCH_HOME to a directory in your /data/ area.
Also do this for environment variables HUGGINGFACE_HUB_CACHE and HF_DATASETS_CACHE
```
mkdir /data/uwa_multimodality/uwa_niuk/hf
mkdir /data/uwa_multimodality/uwa_niuk/torch
mkdir /data/uwa_multimodality/uwa_niuk/hf/datasets
mkdir /data/uwa_multimodality/uwa_niuk/hf/models

export TORCH_HOME=/data/uwa_multimodality/uwa_niuk/torch/
export HF_HOME=/data/uwa_multimodality/uwa_niuk/hf/
export HF_DATASETS_CACHE=/data/uwa_multimodality/uwa_niuk/hf/datasets
export TRANSFORMERS_CACHE=/data/uwa_multimodality/uwa_niuk/hf/models
```

`#module add hpc` will add `#rjs` to your PATH

To initial scripts, 
Add `#conda activate` to the script
```
. /data/uwa_multimodality/uwa_niuk/etc/profile.d/conda.sh 
conda activate text2video-finetune
```

Add `##rj features=v100` or `##rj features=a100` so the job runs on a node which has GPUs
Add `#module add cuda/compat/12.0`, to run cuda 12 workflows by including this in your job scrip.

The compute nodes cannot access internet directly. You'll need to configure the proxy settings in your job script:
```
export http_proxy="http://proxy.per.dug.com:3128"
export https_proxy="http://proxy.per.dug.com:3128"
```

Example of a job script:
```
#!/bin/bash

#rj name=captionvideo queue=uwa_multimodality
#rj features=a100

module add cuda/compat/12.0

export http_proxy="http://proxy.per.dug.com:3128"
export https_proxy="http://proxy.per.dug.com:3128"

. /data/uwa_multimodality/uwa_niuk/etc/profile.d/conda.sh
conda activate text2video-finetune

python /data/uwa_multimodality/uwa_niuk/project/Text-To-Video-Finetuning/Video-BLIP2-Preprocessor/preprocess.py --video_directory /data/uwa_multimodality/uwa_niuk/project/Text-To-Video-Finetuning/data --config_name "secrets-human-blip2" --config_save_name "secrets-human-blip2" --prompt_amount 8

echo "Videos Captioned"

```



```

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/uwahpc/centos8/python/Anaconda3/2024.06/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/uwahpc/centos8/python/Anaconda3/2024.06/etc/profile.d/conda.sh" ]; then
        . "/uwahpc/centos8/python/Anaconda3/2024.06/etc/profile.d/conda.sh"
    else
        export PATH="/uwahpc/centos8/python/Anaconda3/2024.06/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# . /uwahpc/centos8/python/Anaconda3/2024.06/etc/profile.d/conda.sh  # commented out by conda initialize
# conda activate  # commented out by conda initialize

export CONDA_PKGS_DIRS=/group/pmc015/kniu/kai_phd/conda_env/.conda-pkgs
export PIP_CACHE_DIR=/group/pmc015/kniu/kai_phd/conda_env/.pip-cache
export TORCH_HOME=/group/pmc015/kniu/kai_phd/conda_env/torch/
export HF_HOME=/group/pmc015/kniu/kai_phd/conda_env/hf/
export HF_DATASETS_CACHE=/group/pmc015/kniu/kai_phd/conda_env/hf/datasets
export TRANSFORMERS_CACHE=/group/pmc015/kniu/kai_phd/conda_env/hf/models
export CONDA_ENVS_PATH=/group/pmc015/kniu/kai_phd/conda_env
export OUTPUT_PATH=/group/pmc015/kniu/kai_phd/output
export DATA_PATH=/group/pmc015/kniu/kai_phd/data
export MODEL_PATH=/group/pmc015/kniu/kai_phd/models
export EVAL_PATH="/group/pmc015/kniu/kai_phd/eval"
export SAPIENS_ROOT="/group/pmc015/kniu/kai_phd/Video-Generation/third_party/Sapiens/"
export SAPIENS_CHECKPOINT_ROOT="/group/pmc015/kniu/kai_phd/models/Sapiens/sapiens_host"
export SAPIENS_LITE_ROOT="/group/pmc015/kniu/kai_phd/Video-Generation/third_party/Sapiens/lite/"
export SAPIENS_LITE_CHECKPOINT_ROOT="/group/pmc015/kniu/kai_phd/models/Sapiens/sapiens_lite_host"
~                                                                                                      
```

```
salloc -p gpu -n 4 -c 2 --gres=gpu:4
```
