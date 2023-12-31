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
