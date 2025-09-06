# Install conda env
Champ env - 
```
Python 3.10.14
Pytorch 2.2.2+cu121
tensorflow 2.16.1


PyTorch: 2.5.1+cu124
CUDA Toolkit: 12.4 (headers and libs)
GCC >= 9 (you already loaded or installed GCC 12.4.0)
GPU Driver: 550.54.14, which supports CUDA 12.4 on A100
```
```
(conda create -n salix-ai python=3.10)

conda create -p /group/pmc015/kniu/kai_phd/conda_env/champ_text python=3.10

chmod -R 775 /group/pmc015/kniu/kai_phd/conda_env/champ_text/bin/

conda activate ..

salloc -p pophealth --time=5-00:00:00 --mem=40G -n 8 --gres=gpu:a100:1 --ntasks 10

conda search -c conda-forge pytorch
conda search -c conda-forge pytorch=2.3.1 
conda search -c conda-forge tensorflow

conda install "conda-forge/linux-64::pytorch 2.1.2 cuda120_py310h327d3bc_301"
(module load cuda/12.0)
(conda install "conda-forge/linux-64::pytorch 2.4.1 cuda120_py310h5d94b2e_301") - deepseek
(module load cuda/12.4)
(conda install "conda-forge/linux-64::pytorch 2.5.1 cuda126_generic_py310_h478e78a_207") - qwen_rl
(conda install "conda-forge/linux-64::torchvision 0.20.1 cuda126_py310_h47da5a9_4") - finetune_dit
(pip uninstall -y torch torchvision) - qwen_rl
(pip install "torch==2.5.1" "setuptools<71.0.0"  --index-url https://download.pytorch.org/whl/cu124) - qwen_rl
(https://www.philschmid.de/mini-deepseek-r1)

conda install "conda-forge/linux-64::tensorflow 2.16.1 cuda120py310hfaee7bf_0"

pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

pip install numpy==1.23.5

module load gcc/12.4.0
module load cuda/12.4

pip uninstall xformers
git clone https://github.com/facebookresearch/xformers.git
cd xformers
# Optionally set your arch explicitly. For A100: 
export TORCH_CUDA_ARCH_LIST="8.0"

# Make sure submodules are in place
git submodule update --init --recursive

# Optional: set compute capability for an A100
export TORCH_CUDA_ARCH_LIST="8.0"

# Reinstall xFormers using the newer compiler
pip install .


```

# Multiple GPU Usage - UWA KAYA:


-N 2 means apply multiple GPUs from 2 seperate computer nodes, not apply 2 GPUs from multiple computers nodes, that's why it will have the error 
```
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ salloc -p pophealth --mem=80G -N 2 -n 8 --gres=gpu:a100:2
salloc: Job allocation 550543 has been revoked.
salloc: error: Job submit/allocate failed: Requested node configuration is not available

```

hostname - get the hostname of the coputer that host GPUs
```
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ hostname
n006.hpc.uwa.edu.au
```

Use exit to exit the GPU usage session
```
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ exit
srun: error: n006: task 0: Exited with exit code 130
salloc: Relinquishing job allocation 550408
salloc: Job allocation 550408 has been revoked.

```

Check GPUs information. As we can see, the partition 'pophealth' has 2 computer codes that host GPUs, one is '002' which has 2 V100, and one is '006' which has 4 A100.
```
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ sinfo
PARTITION    AVAIL  TIMELIMIT  NODES  STATE NODELIST
work            up 3-00:00:00      2  down* n[023,027]
work            up 3-00:00:00      3  drain n[026,029,032]
work            up 3-00:00:00      5    mix n[010,015-016,024,028]
work            up 3-00:00:00     12   idle n[011-013,017-019,022,025,030-031,033-034]
long            up 7-00:00:00      1    mix n021
long            up 7-00:00:00      1  alloc n020
gpu             up 3-00:00:00     13    mix n[001,003-005,037-044,046]
pophealth       up 15-00:00:0      2   idle n[002,006]
ondemand        up   12:00:00      1  down* n027
ondemand        up   12:00:00      1  drain n026
ondemand        up   12:00:00      2    mix n[024,028]
ondemand        up   12:00:00      1   idle n025
ondemand-gpu    up   12:00:00      8    mix n[036-043]
```

To apply gpu, alwayse use the login nodes, then use 'salloc' command to switch to GPUs computer nodes.
You have to specify the GPUs usage time for A100, the time follows convention '--time=D-HH:MM:SS'
```
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ cd 
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ ls
package.json  package-lock.json
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ pwd
/home/kniu
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ salloc -p pophealth --time=2-00:00:00 --mem=128G -n 8 --gres=gpu:a100:4
salloc: Pending job allocation 550997
salloc: job 550997 queued and waiting for resources
salloc: job 550997 has been allocated resources
salloc: Granted job allocation 550997
(base) bash-4.4$ conda activate champ
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ nvidia-smi
Thu Nov 21 15:12:38 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          Off |   00000000:01:00.0 Off |                    0 |
| N/A   31C    P0             50W /  400W |       0MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          Off |   00000000:41:00.0 Off |                    0 |
| N/A   29C    P0             50W /  400W |       0MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A100-SXM4-40GB          Off |   00000000:81:00.0 Off |                    0 |
| N/A   31C    P0             49W /  400W |       0MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          Off |   00000000:C1:00.0 Off |                    0 |
| N/A   30C    P0             52W /  400W |       0MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```

To get slurm example for running multiple threads:
```
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ module load getexample
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ getexample
kaya
Download an HPC example:
usage:
   getexample <examplename>

Where <examplename> is the name of the example you want to 
download.  This will create a directory named examplename which
you can cd into and hopefully read the README file 
 and the *.slurm file.

For Example:
  getexample helloworld

Possible example names:
/uwahpc/centos8/tools/binary/getexample/kaya_examples
abaqus-explicit           fortranHybrid_gnu  grasp_mpi    hello_mpi_c_intel    kat         molpro  orca       swash
abaqus-exp-usercode       fortranMPI_gnu     gromacs_mpi  hello_mpi_gnu.slurm  lammps_mpi  mrcc    structure  training_hello
fortran_helloworld_intel  gaussian           helloC_gnu   helloOmp_gnuC        lsdyna      nwchem  swan       training_MPI
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ getexample helloC_gnu
kaya
found directory
'/uwahpc/centos8/tools/binary/getexample/kaya_examples/helloC_gnu' -> './helloC_gnu'
'/uwahpc/centos8/tools/binary/getexample/kaya_examples/helloC_gnu/README' -> './helloC_gnu/README'
'/uwahpc/centos8/tools/binary/getexample/kaya_examples/helloC_gnu/helloworld_gnu.slurm' -> './helloC_gnu/helloworld_gnu.slurm'
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ ls
conda  conda_results  helloC_gnu  huggingface  kai_phd  nohup.out  Untitled1.ipynb  Untitled.ipynb
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ cd helloC_gnu/
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ ls
helloworld_gnu.slurm  README
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ less helloworld_gnu.slurm 

```

Use Accelerate for distributed training:

https://huggingface.co/docs/transformers/en/accelerate

https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference

```
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```
Train with a script

If you are running your training from a script, run the following command to create and save a configuration file - choos using 'Multiple GPUs':
```
accelerate config
```
Then launch your training with:
```
accelerate launch train.py
```

```
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ accelerate config
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine                                                                                                                                                                   
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                                                           
multi-GPU                                                                                                                                                                      
How many different machines will you use (use more than 1 for multi-node training)? [1]:                                                                                       
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:                                                 
Do you wish to optimize your script with torch dynamo?[yes/NO]:                                                                                                                
Do you want to use DeepSpeed? [yes/NO]:                                                                                                                                        
Do you want to use FullyShardedDataParallel? [yes/NO]:                                                                                                                         
Do you want to use Megatron-LM ? [yes/NO]:                                                                                                                                     
How many GPU(s) should be used for distributed training? [1]:4                                                                                                                 
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all                                                                           
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
fp16                                                                                                                                                                           
accelerate configuration saved at /group/pmc015/kniu/kai_phd/conda_env/hf/accelerate/default_config.yaml                                                                       
(/group/pmc015/kniu/kai_phd/conda_env/champ) bash-4.4$ accelerate launch Video-Generation/third_party/Champ/train_s1_new.py 
/group/pmc015/kniu/kai_phd/conda_env/champ/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.                                                                                                                     
  torch.utils._pytree._register_pytree_node(
/group/pmc015/kniu/kai_phd/conda_env/champ/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/group/pmc015/kniu/kai_phd/conda_env/champ/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/group/pmc015/kniu/kai_phd/conda_env/champ/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
2024-11-21 20:28:12.652154: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-21 20:28:12.652154: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-21 20:28:12.652155: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-21 20:28:12.657235: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/group/pmc015/kniu/kai_phd/conda_env/champ/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/group/pmc015/kniu/kai_phd/conda_env/champ/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/group/pmc015/kniu/kai_phd/conda_env/champ/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/group/pmc015/kniu/kai_phd/conda_env/champ/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
Number of available GPUs: 4
Device 0: NVIDIA A100-SXM4-40GB
Device 1: NVIDIA A100-SXM4-40GB
Device 2: NVIDIA A100-SXM4-40GB
Device 3: NVIDIA A100-SXM4-40GB

```


```
salloc -p pophealth --time=2-00:00:00 --mem=40G -n 8 --gres=gpu:a100:1 --exclusive
```


# Slurm

Train slurm:
```
#!/bin/bash --login
# SBATCH --job-name=calisthenic_train_50000 # name your job
# SBATCH --output=/group/pmc015/kniu/kai_phd/Video-Generation/slurm/calisthenic_train_50000_output.txt # running logs
# SBATCH --nodes=1 # how many nodes you want ?
# SBATCH --time=4-00:00:00 # how long your task will last
# SBATCH --partition=pophealth # which queue you want to get in
# SBATCH --mem=120G # how many RAM you will need
# SBATCH --gres=gpu:a100:3 # specific filtering
module load cuda/12.4
module load gcc/12.4.0

conda activate champ

# your python script here
python ../third_party/Champ/main.py --run_type train
```
run slurm

```
sbatch inference.slurm
```
check job
```
squeue -u kniu
```
```
 squeue --job 552101
```
cancel job
```
scancel 552099
```

# Copy model output
```
rsync -aP  kniu@kaya.hpc.uwa.edu.au:/group/pmc015/kniu/kai_phd/Video-Generation/output/exp_output_normal_semantic_map/stage1/metrics_combined.png  /media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/output/youtube_new/Champ_new/model_output/exp_output_normal_semantic_map/stage1/metrics_combined.png
```

```
rsync -aP  kniu@kaya.hpc.uwa.edu.au:/group/pmc015/kniu/kai_phd/Video-Generation/output/exp_output_normal_semantic_map/stage1/sanity_check/ /media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/output/youtube_new/Champ_new/model_output/exp_output_normal_semantic_map/stage1/sanity_check/
```

```
rsync -aP  kniu@kaya.hpc.uwa.edu.au:/group/pmc015/kniu/kai_phd/Video-Generation/output/exp_output_normal_semantic_map/stage1/validation/ /media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/output/youtube_new/Champ_new/model_output/exp_output_normal_semantic_map/stage1/validation/
```


```
salloc -p pophealth --time=5-00:00:00 --mem=40G -n 8 --gres=gpu:a100:1 --ntasks 10

```

# Trained Models

```
python main_new_new.py 
```
```
drwxrwxr-x 3 kniu kniu 0 Dec 13 15:04 exp_output_normal                   -- 1 condition - normal - foreground
drwxrwxr-x 4 kniu kniu 0 Dec  2 08:30 exp_output_normal_semantic_map                   -- 2 conditions (normal, semantic maps) - foreground
drwxrwxr-x 4 kniu kniu 0 Nov 28 14:28 exp_output_normal_semantic_map_50000                   -- 2 conditions (normal, semantic maps) - original view
drwxrwxr-x 3 kniu kniu 0 Dec 14 09:05 exp_output_semantic_map                   -- 1 condition - semantic_map - foreground

exp_output_depth                     -- 1 condition - depth - foreground
exp_output_dwpose                   -- 1 condition - keypioints133 - foreground   &                    -- 1 condition - tracking_points - foreground
```

```

exp_output_normal_semantic_map - stage1, stage2
exp_output_normal_semantic_map_50000 - stage1, stage2
exp_output_depth - stage1
exp_output_dwpose  - stage1_keypoints  stage1_tracking_points  stage2_tracking_points  stage2_keypoints
exp_output_normal - stage1, stage2
exp_output_semantic_map - stage1, stage2

generated_video_normal_semantic_map_50000               - 2 condition - original view
generated_video_depth_normal_semantic_map_dwpose_champ               - 4 condition - CHAMP
generated_video_normal_semantic_map               - 2 condition
generated_video_normal               - 1 condition
generated_video_semantic_map               - 1 condition
generated_video_dwpose_keypoints

eval_result_depth_normal_semantic_map_dwpose_champ
eval_result_normal_semantic_map_50000
eval_result_normal_semantic_map               - 2 condition - foreground
eval_result_normal
eval_result_semantic_map

```

# copy the trained model from kaya to local
```
 module load awscli/2.10.3 
aws s3 sync /group/pmc015/kniu/kai_phd/Video-Generation/output  s3://video-generation-calisthenics/train/output/
aws s3 sync /group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ_text s3://video-generation-calisthenics/code/Champ_text

```

```
python main_new_new.py      in 'third_party/Champ' in kaya with env 'champ'  - by setting up 'run_type' for training or inference

rsync -aP  kniu@kaya.hpc.uwa.edu.au:/group/pmc015/kniu/kai_phd/Video-Generation/test/generated_video_normal /media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/output/youtube_new/Champ_new/results/generated_video_normmal

module load ffmpeg/7.0.2    # need to load ffmpeg for evaluation
python evaluate_new_new.py     in 'Calisthenic/evaluate' in Kaya, with env 'champ'

# Evaluate the methods with CHAMP metrics: 'MagicDance/too/metrics/metric_center_new.py'    with magicdance env
```

```
generic-diffusion-feature/feature/example.py   -attention map
''Calisthenic/process/plot_attention.py    - plot attention
```

Have a look gpto output for inference with attention map:

```
Below is an example of how you can modify the code to capture and return the latent attention features from each layer of the diffusion blocks. The main idea is to:

1. Hook into the U-Net(s) or the reference control modules (`ReferenceAttentionControl`) where attention maps are computed.
2. Store the attention maps during the forward pass.
3. After the inference step is complete, retrieve these stored attention maps and return them.

**Key Assumptions**:  
- Your `ReferenceAttentionControl` or U-Net classes provide a way to hook into or retrieve attention maps. If they do not, you will need to add such functionality by modifying their forward methods or by adding forward hooks.
- In the code snippet below, we assume that `ReferenceAttentionControl` has a method `get_attention_maps()` that returns the collected attention maps. If it doesn't exist, you will need to implement a mechanism to store attention maps within `ReferenceAttentionControl` or in the U-Net forward pass itself.
- We also assume that the pipeline call returns only videos by default. We will modify it to return both videos and attention maps.

**Changes made**:
- Add a step in `inference()` to retrieve attention maps from the model's `reference_control_reader`.
- Modify the `inference()` function signature and return value to include attention maps.
- After running the pipeline, collect attention maps and return them.

```python
def inference(
    cfg,
    vae,
    image_enc,
    prompt,
    negative_prompt,
    text_enc,
    tokenizer,
    model,
    scheduler,
    ref_image_pil,
    guidance_pil_group,
    video_length,
    width,
    height,
    device="cuda",
    dtype=torch.float16,
):
    reference_unet = model.reference_unet
    denoising_unet = model.denoising_unet
    guidance_types = cfg.guidance_types
    guidance_types_model = guidance_types
    for i in range(len(guidance_types)):
        if guidance_types[i] == "tracking" or guidance_types[i] == "optical_flow":
            guidance_types_model[i] = "depth"

    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}")
        for g in guidance_types_model
    }

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    pipeline = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        text_encoder=text_enc,
        tokenizer=tokenizer,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        scheduler=scheduler,
        guidance_process_size=cfg.data.get("guidance_process_size", None)
    )
    pipeline = pipeline.to(device, dtype)

    # Run the pipeline as usual
    output = pipeline(
        ref_image_pil,
        prompt,
        negative_prompt,
        guidance_pil_group,
        width,
        height,
        video_length,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
    )
    video = output.videos

    # After pipeline execution, retrieve attention maps
    # Assuming `reference_control_reader` has a method to return attention maps
    # If not, you need to implement a way to store them inside `ReferenceAttentionControl`.
    attention_maps = model.reference_control_reader.get_attention_maps()

    # Cleanup
    del pipeline
    torch.cuda.empty_cache()

    return video, attention_maps
```

**Within `main()`**:  
- Capture the returned attention maps from `inference()`.

```python
def main(cfg, video_name):
    # ... (existing code above remains unchanged)

    result_video_tensor, attention_maps = inference(
        cfg=cfg,
        vae=vae,
        image_enc=image_enc,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        text_enc=text_enc,
        tokenizer=tokenizer,
        model=model,
        scheduler=noise_scheduler,
        ref_image_pil=ref_image_pil,
        guidance_pil_group=guidance_pil_group,
        video_length=video_length,
        width=cfg.width,
        height=cfg.height,
        device="cuda",
        dtype=weight_dtype,
    )

    # Now you have `attention_maps` which you can save or analyze as needed.
    # `attention_maps` might be a dictionary or a list of tensors depending on your `get_attention_maps()` implementation.
    # For example, if you want to save them:
    torch.save(attention_maps, osp.join(save_dir, "attention_maps.pt"))

    # The rest of the code remains the same
    result_video_tensor = resize_tensor_frames(
        result_video_tensor, (ref_image_h, ref_image_w)
    )
    save_videos_grid(result_video_tensor, osp.join(save_dir, "animation.mp4"))

    # ... (rest of your saving code)
```

**Inside `ReferenceAttentionControl` (if needed)**:  
You need to ensure there is a mechanism to store and retrieve attention maps. For example:

```python
class ReferenceAttentionControl:
    def __init__(self, unet, do_classifier_free_guidance, mode, fusion_blocks="full"):
        # ... your existing init code
        self.attention_maps = []

        # You might add forward hooks to attention layers here:
        for block in unet.attention_blocks:
            block.attn.register_forward_hook(self.save_attention_map_hook)

    def save_attention_map_hook(self, module, input, output):
        # Assuming `output` is something like (hidden_states, attention_probs)
        # or that you can access attention weights here.
        # You may need to adjust this code depending on the actual structure of your model.
        self.attention_maps.append(output[1].detach().cpu())

    def get_attention_maps(self):
        return self.attention_maps
```

**Important Notes**:
- The actual implementation of how you retrieve the attention maps depends heavily on how the `UNet` and `ReferenceAttentionControl` classes are structured and how they handle attention. You may need to adapt the above hook logic accordingly.
- If your attention maps are computed directly in the U-Net forward pass, consider placing hooks directly on the attention layers of the U-Net models.
- Ensure that `get_attention_maps()` returns the collected attention maps in the format you need. This could be a list of tensors, a dictionary keyed by layer, etc.

With these changes, after running inference, you will not only have the generated video but also the captured latent attention maps from each layer of the diffusion blocks.


# Calisthenic Video Generation Model
```
/group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ - Orignial CHAMP model modifed for tracking points. The model used in Calisthenic Video Dataset paper
/group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ_text - the architecture modified in JointTextMotionTemporalBasicTransformerBlock class, to accept text embedding (object motion captions) as encoded_hidden_states. The cross attention is calculated between latent_states (motion sequences) and text_embedding. The 1st training stage is fine on 1A100, but the 2rd training stage is OOMM on 4A100s even though using 'gradient_accumulation_steps' 8.
/group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ_text_claude_28jan2025 - The code generated from Claude. It includes an extra attention block to implement cross attention between latent_states and text embedding, which requires a lot more computation. OOMM on 4A100s.
/group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ_text_gpt_25jan2025 - The code generated from GPT. It includes an extra attention block to implement cross attention between latent_states and text embedding, which requires a lot more computation. OOMM on 4A100s.
/home/kai/phd/Video_Generation/git/gym_video_generation/pred/memo/memo - MEMO model. It implement cross attention between video and audio, also replace emb (time embedding) with emotion class. 
```

```
salloc -N 1 --time=01:00:00 --partition=gpu --mem=200G --gres=gpu:v100:2
```

Qwen2_VL_7B - vision vaptioning model training: /home/kai/phd/Video_Generation/git/gym_video_generation/Calisthenics/rl/instructed_fine-tuning/video_captioner/sft_qwen2_VL_7B2.py



# when install zigma
module load cuda/11.8
After 
 conda install "conda-forge/linux-64::pytorch 2.3.1 cuda118_py311h0047a46_300" 
pip uninstall torch
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Finetuning LLM
```

step1_SFT: /home/kai/phd/Video_Generation/git/gym_video_generation/Calisthenics/rl/instructed_fine-tuning/textualized_keypoints_generator/stf_step1.py
step2_SFT: /home/kai/phd/Video_Generation/git/gym_video_generation/Calisthenics/rl/instructed_fine-tuning/textualized_keypoints_generator/stf_step2.py

step1_GRPO: /home/kai/phd/Video_Generation/git/gym_video_generation/Calisthenics/rl/finetune_grpo_step1.py
step1_GRPO_reasoning: /home/kai/phd/Video_Generation/git/gym_video_generation/Calisthenics/rl/finetune_grpo_step1_reasoning.py
step2_GRPP: /home/kai/phd/Video_Generation/git/gym_video_generation/Calisthenics/rl/finetune_grpo_step2.py
reward function: /home/kai/phd/Video_Generation/git/gym_video_generation/Calisthenics/rl/verify_reward_function7.py
*** the weigths for keypoint correctness and keypoint name need to be change, which is to small for step1 (only 1 frame)

SFT LLM models: /media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/models/sft_llm
GRPO LLM Models: /media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/models/grpo_llm
'/group/pmc015/kniu/kai_phd/Video-Generation/Calisthenics/rl/models' in Kaya
```

du -sh */

# How to git add Champ
```
git add Champ
git rm -r --cached Champ_new/mmpose/
git rm -r --cached Champ_new/detectron2/
git rm -r --cached Champ_new/driving_videos/
git rm -r --cached Champ_new/runwayml/
```


```
third_party/Champ   - Calisthenic Video Generation Dataset Paper branch
third_party/Champ_new   - backup
third_party/Champ_keypoints    - experiment replacing motion guiders (imagery skeletons) by json keypoints as the input 
third_party/Champ_text    - exmperiment of adding text into denoising process (similar idea of MEMO - adding motion-aware text attention 
third_party/Champ_text_claude_28jan2025     - the code generated by Claude3.5
third_party/Champ_text_gpt_25jan2025     - the code generated by GPTo1
```

# Notes - 26/05/25
1. multimodal condition attention investigation - 
/home/kai/phd/Video_Generation/git/gym_video_generation/pred/Champ/guidance_attention3.py

2. Temporal guidance attention:
/home/kai/phd/Video_Generation/git/gym_video_generation/pred/Champ/temporal_guidance_attention.py

3. keypoints integrated CHAMP and text integrated CHAMP have been merged in 
https://github.com/Kai0226/Video-Generation/tree/main/third_party

4. It also include processing code of Calisthenic, which including RL (GROP), and multimodal encoders (Autoencoder, VAE and CLIP for text, semantic maps and optical flow)

5. the CHAMP models traind for Calisthenic video dataset have been sync to '/media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/output/youtube_new/Champ_new/output'

6. train, val, and test data in Kaya have been setteled well, 'train' and 'test' have been sycn to s3, but not val.

7. 'results' folder has not been sync to local PC yet.


# Notes - 28/04/25
Attention injection --
1. /home/kai/phd/Video_Generation/git/gym_video_generation/pred/Champ/phd/attention_injection/compute_attention2.py (/home/kai/phd/Video_Generation/git/gym_video_generation/pred/Champ/compute_attention2.py)
2. /home/kai/phd/Video_Generation/git/gym_video_generation/pred/Champ/phd/attention_injection/classify_attention2.py
3. /home/kai/phd/Video_Generation/git/gym_video_generation/pred/Champ/phd/attention_injection/train_attention_correction2.py   --  the model didn't get trained well, overfit too quick
4. /home/kai/phd/Video_Generation/git/gym_video_generation/pred/Champ/phd/attention_injection/inject_attention.py   --  haven't tested this code yet.

5. /home/kai/phd/Video_Generation/git/gym_video_generation/pred/Champ/phd/visualize_attention/guidance_attention7_sapiens3.py  -- CHAMP model attention visualization analysis


# Notes - 05/05/25
TikTok dataset (300 video clips) - attention evaluation
process multimodal conditions: /group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ/process_video_new_run_champ_v2_tiktok.py
run inference: python /group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ/main_new_new_tiktok_dataset.py --run_type inference_champ


# Mount IDRS into local pc
```
smb://store.irds.uwa.edu.au
UNIWA
24188946

sudo mount -t cifs //store.irds.uwa.edu.au/res-pmc-vg202408-p000528 /mnt/irds -o username=24188946,domain=UNIWA

sudo rsync -avh --progress /media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/output/youtube_new/Champ_new/ /mnt/irds/calisthenic/Champ_new/
```
transfering data from aws s3 to IRDS
```
sudo -E aws s3 sync s3://video-generation-calisthenics/train/output/ /mnt/irds/calisthenic/Champ_new/output/
```

If you Input/Output error or Hose is down: 
```
sudo umount /mnt/irds

sudo mount -t cifs //store.irds.uwa.edu.au/res-pmc-vg202408-p000528 /mnt/irds -o username=24188946,domain=UNIWA,iocharset=utf8,rw,nounix,noserverino,vers=3.1.1,hard,intr,actimeo=30

sudo rsync -avh --progress --inplace --partial --append-verify /media/kai/f4b6c365-d543-4751-877a-3b5a123ac025/video_generation/output/youtube_new/Champ_new/output/ /mnt/irds/calisthenic/Champ_new/output/
```
Long timeout
```
sudo rsync -avh --progress --partial --partial-dir=.rsync-partial --timeout=1800 /media/kai/A6AA8433AA840253/data/s3_data/transferred_result_train /mnt/irds/calisthenic/Champ_new/transferred_result_train

```

# Kaya to IRDS
```
cd gio_mount_irds/
dbus-run-session -- bash
./mount_irds.sh

RES-PMC-VG202408-P000528
24188946

source ~/.irds_RES-PMC-VG202408-P000528.conf
echo $MYIRDS

/run/user/11030/gvfs/smb-share:server=store.irds.uwa.edu.au,share=res-pmc-vg202408-p000528

```


# Update 07-May-2025
The current CHAMP dir in Kaya: /group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ  is the code for running inference on TikTok video dataset including 340 vidoe clips. In this codebase, it also includes guidance fusion module (added cross attention to fuse multimodal guidance) in "/group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ/models/champ_model_guidance_fusion.py"

# Sapiens env install

sapiens env:


cuda - 12.1
mmcv - 2.1.0

mmpose - 1.0.0

python - Python 3.10.14
 mediapipe - 0.10.18

  - package mmcv-2.1.0-cuda120py310h7e5e4a0_203 requires pytorch >=2.1.0,<2.2.0a0, but none of the providers can be installed
 
```
conda install "conda-forge/linux-64::pytorch 2.1.0 cuda120_py310h8a81058_302"
conda install "conda-forge/linux-64::mmcv 2.1.0 cuda120py310h7e5e4a0_203"
conda install -c pytorch -c conda-forge torchvision=0.16.0 --no-update-deps
conda install -c pytorch -c conda-forge torchaudio=2.1.0 --no-update-deps
pip install opencv-python tqdm json-tricks
pip install mmpose==1.0.0
pip install mediapipe==0.10.18


(Optional but recommended)
Ensure your PATH and LD_LIBRARY_PATH include CUDA:
   export PATH=/uwahpc/local/cuda-12.0/bin:$PATH
   export LD_LIBRARY_PATH=/uwahpc/local/cuda-12.0/lib64:$LD_LIBRARY_PATH

Install your packages:
   pip install mmcv==2.1.0 mmdet==3.3.0 mmpose==1.0.0
```

# The last upblock layer attention extraction
```
cd /group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ/slurm
sbatch /group/pmc015/kniu/kai_phd/Video-Generation/third_party/Champ/slurm/extract_attention.slurm
 -- which use "inference_extract_attention_tiktok4.py"

python /group/pmc015/kniu/kai_phd/Video-Generation/Calisthenics/attention_investigation/attention_analysis5.py
 -- for analysis
```
# Visualize attention
```
cd /group/pmc015/kniu/kai_phd/Video-Generation/Calisthenics/attention_investigation/attention_visialization/attention_visualizations

# Single video analysis
python champ_attention_visualizer.py --attention_dir /path/to/attention_maps --video_name 00001

# Batch processing multiple videos
python champ_attention_visualizer.py --attention_dir /path/to/attention_maps --batch --video_list 00001 00002 00003

# Limit frames for faster processing
python champ_attention_visualizer.py --attention_dir /path/to/attention_maps --video_name 00001 --max_frames 30


# Quality analysis for single video
python advanced_attention_analysis.py --attention_dir /path/to/attention_maps --video_name 00001 --analysis_type quality

# Hotspot analysis
python advanced_attention_analysis.py --attention_dir /path/to/attention_maps --video_name 00001 --analysis_type hotspots

# Compare multiple videos
python advanced_attention_analysis.py --attention_dir /path/to/attention_maps --video_list 00001 00002 00003 --analysis_type comparison

# Full analysis suite
python advanced_attention_analysis.py --attention_dir /path/to/attention_maps --video_name 00001 --analysis_type all

# Interactive exploration
python interactive_attention_explorer.py --attention_dir /path/to/attention_maps --video_name 00001 --mode explore

# Create summary dashboard
python interactive_attention_explorer.py --attention_dir /path/to/attention_maps --video_list 00001 00002 00003 --mode dashboard
```


```
salloc -N 1 --time=01:00:00 --partition=data-inst --mem=200G --gres=gpu:h100:2
```

```
salloc -p data-inst --gres=gpu:h100:1
```

```
#!/usr/bin/env bash
#SBATCH --job-name=inference_video_clips
#SBATCH --output=/group/pmc015/kniu/video_clips/Wan2.2/slurm/inference_video_clips.txt
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=data-inst
#SBATCH --mem=200G
#SBATCH --gres=gpu:h100:1

# 1) load any system modules you need
module load cuda/12.4
module load gcc/12.4.0

# --- force a safe threading setup for ONNX Runtime ---
export OMP_NUM_THREADS=8        # match the number of CPUs you requested
export MKL_NUM_THREADS=8
export KMP_AFFINITY=none

# 2) source the conda setup script so `conda activate` works
#    adjust this path to match your HPC’s Anaconda installation
eval "$(conda shell.bash hook)"
conda activate wan2gp

cd "/group/pmc015/kniu/video_clips/Wan2.2"

# pick the N-th prompt from a text file with one prompt per line
PROMPT=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" prompts.txt)

python generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --frame_num 97 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --offload_model True \
  --convert_model_dtype \
  --prompt "$PROMPT" \
  --sample_steps 30 \
  --sample_shift 5.0 \
  --sample_guide_scale 5.0 \


```

```
tail -f inference_video_clips.3927_{0..3}.txt

```

```
(/group/pmc015/kniu/kai_phd/conda_env/wan) bash-5.1$ python try_i2v.py /group/pmc015/kniu/video_clips/Wan2GP/manifest.jsonl /group/pmc015/kniu/video_clips/Wan2GP/output/video1
/group/pmc015/kniu/kai_phd/conda_env/wan/lib/python3.10/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 12/12 [01:11<00:00,  6.00s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 12/12 [01:12<00:00,  6.01s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.42it/s]
Loading pipeline components...: 100%|██████████████████████████████████████████████████████| 6/6 [02:28<00:00, 24.78s/it]
100%|████████████████████████████████████████████████████████████████████████████████████| 50/50 [42:39<00:00, 51.20s/it]
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
[1/1] wrote /group/pmc015/kniu/video_clips/Wan2GP/output/video1/a.mp4
(/group/pmc015/kniu/kai_phd/conda_env/wan) bash-5.1$ squeue -u kniu 6718
squeue: error: Unrecognized option: 6718
Usage: squeue [-A account] [--clusters names] [-i seconds] [--job jobid]
              [-n name] [-o format] [--only-job-state] [-p partitions]
              [--qos qos] [--reservation reservation] [--sort fields] [--start]
              [--step step_id] [-t states] [-u user_name] [--usage]
              [-L licenses] [-w nodes] [--federation] [--local] [--sibling]
              [--expand-patterns] [--json=data_parser] [--yaml=data_parser]
              [-ahjlrsv]
(/group/pmc015/kniu/kai_phd/conda_env/wan) bash-5.1$ exit
exit
srun: error: k179: task 0: Exited with exit code 1
salloc: Relinquishing job allocation 6718
salloc: Job allocation 6718 has been revoked.
(/group/pmc015/kniu/kai_phd/conda_env/wan) kniu@kaya01[video_clips]$ salloc -p data-inst --gres=gpu:h100:1
salloc: Granted job allocation 6953
salloc: Nodes k179 are ready for job
(base) bash-5.1$ conda activate wan
(/group/pmc015/kniu/kai_phd/conda_env/wan) bash-5.1$ pwd
/group/pmc015/kniu/video_clips
(/group/pmc015/kniu/kai_phd/conda_env/wan) bash-5.1$ ls
Wan2.2  Wan2GP
(/group/pmc015/kniu/kai_phd/conda_env/wan) bash-5.1$ cd Wan2GP/
(/group/pmc015/kniu/kai_phd/conda_env/wan) bash-5.1$ python infer_t2i2v_wan2gp.py \
  --prompt "neon jungle dreamscape 01 — bioluminescent vines, fractal mushrooms pulsing to 174 BPM, magenta-cyan haze, cinematic drone glide, 16:9 720p" \
  --image_out "/group/pmc015/kniu/video_clips/Wan2GP/test_images" \
  --video_out "/group/pmc015/kniu/video_clips/Wan2GP/test_videos/test_video.mp4" \
  --height 704 --width 1280 \
  --frames 81 \
  --fps 16 \
  --guidance_scale 5.0 \
  --steps 30 \
  --flow_shift 5.0 \
  --teacache 0.25 \
  --seed 123
/group/pmc015/kniu/kai_phd/conda_env/wan/lib/python3.10/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
Loading pipeline components...:   0%|                                                              | 0/7 [00:00<?, ?it/s]You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.20it/s]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 3/3 [00:09<00:00,  3.25s/it]
Loading pipeline components...: 100%|██████████████████████████████████████████████████████| 7/7 [00:12<00:00,  1.84s/it]
[Flux] Generating 1280x704 image...
100%|████████████████████████████████████████████████████████████████████████████████████| 28/28 [00:07<00:00,  3.97it/s]
[INFO] Calling Wan2GP i2v_inference.py:
  /group/pmc015/kniu/kai_phd/conda_env/wan/bin/python /mmfs1/data/group/pmc015/kniu/video_clips/Wan2GP/i2v_inference.py --prompt bioluminescent vines, fractal mushrooms pulsing to 174 BPM, magenta-cyan haze, cinematic drone glide, 16:9 720p --negative-prompt  --input-image /group/pmc015/kniu/video_clips/Wan2GP/test_images/neon_jungle_dreamscape_01.png --output-file /group/pmc015/kniu/video_clips/Wan2GP/test_videos/test_video.mp4 --resolution 1280x704 --frames 81 --steps 30 --guidance-scale 5.0 --flow-shift 5.0 --teacache 0.25 --seed 123
/group/pmc015/kniu/kai_phd/conda_env/wan/lib/python3.10/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.16it/s]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 12/12 [01:12<00:00,  6.02s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 12/12 [01:11<00:00,  5.99s/it]
Loading pipeline components...: 100%|██████████████████████████████████████████████████████| 6/6 [02:29<00:00, 24.87s/it]
Traceback (most recent call last):
  File "/mmfs1/data/group/pmc015/kniu/video_clips/Wan2GP/i2v_inference.py", line 50, in <module>
    main(manifest_path, out_dir)
  File "/mmfs1/data/group/pmc015/kniu/video_clips/Wan2GP/i2v_inference.py", line 19, in main
    with open(manifest_path, "r", encoding="utf-8") as f:
FileNotFoundError: [Errno 2] No such file or directory: '--prompt'
[ERROR] i2v_inference.py failed (exit 1)
(/group/pmc015/kniu/kai_phd/conda_env/wan) bash-5.1$ 

```


```
# paths (adjust if needed)
LORADIR="/group/pmc015/kniu/video_clips/t2i2v_lora/wan2.2"
GGUF_HIGH="$LORADIR/wan22I2VA14BGGUF_a14bHigh.gguf"


python t2i2v.py \
  --prompt "Key — neon jungle dreamscape" \
  --negative_prompt "" \
  --image_out /group/pmc015/kniu/video_clips/t2i2v_lora/seed.png \
  --video_out /group/pmc015/kniu/video_clips/t2i2v_lora/seed.mp4 \
  --t2i-backend flux \
  --flux_model black-forest-labs/FLUX.1-Krea-dev \
  --flux_steps 28 \
  --flux_guidance 3.5 \
  --frames 81 \
  --steps 30 \
  --guidance_scale 5.0 \
  --seed 42 \
  --resolution 1280x704 \
  --transformer_gguf_high /group/pmc015/kniu/video_clips/t2i2v_lora/wan2.2/wan22I2VA14BGGUF_a14bHigh.gguf \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/wan2.2/NSFW-22-H-e8.safetensors@0.8 \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/wan2.2/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise_model.safetensors@1.0

python i2v_inference.py \
  --prompt "neon jungle dreamscape" \
  --input-image /group/pmc015/kniu/video_clips/t2i2v_lora/seed.png \
  --output-file /group/pmc015/kniu/video_clips/t2i2v_lora/seed.mp4 \
  --resolution 1280x704 \
  --frames 81 \
  --steps 30 \
  --guidance-scale 5.0 \
  --seed 42 \
  --transformer_gguf_high /group/pmc015/kniu/video_clips/t2i2v_lora/wan2.2/wan22I2VA14BGGUF_a14bHigh.gguf \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/wan2.2/NSFW-22-H-e8.safetensors@0.8 \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/wan2.2/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise_model.safetensors@1.0

```

```
DEST="/group/pmc015/kniu/video_clips/checkpoints"
URL='https://civitai.com/api/download/models/2091367?type=Model&format=SafeTensor&size=pruned&fp=fp16'
FN='realismIllustriousBy_v50FP16.safetensors'   # choose the final name you want

mkdir -p "$DEST" && cd "$DEST"

# Follow redirects (-L), save as FN (-o), optional retries
curl -L --retry 5 --retry-connrefused -o "$FN" "$URL"


```


```
python t2i2v.py \
  --t2i-backend wan \
  --wan_t2i_model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Jungle — A sexy girl sucks penis" \
  --negative_prompt "low quality, artifacts" \
  --image_out /group/pmc015/kniu/video_clips/t2i2v_lora/seed.png \
  --video_out /group/pmc015/kniu/video_clips/t2i2v_lora/seed.mp4 \
  --width 1280 --height 720 \
  --frames 81 --steps 30 --guidance_scale 5.0 \
  --seed 42 \
  --i2v_model_id Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --transformer_gguf_high /group/pmc015/kniu/video_clips/t2i2v_lora/checkpoints/wan22I2VA14BGGUF_a14bHigh.gguf \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/lora/NSFW-22-H-e8.safetensors@0.8 \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/lora/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise_model.safetensors@1.0 \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/lora/wan2.2-i2v-high-oral-insertion-v1.0.safetensors@1.0


python t2i2v.py \
  --t2i-backend wan \
  --wan_t2i_model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Jungle — embedding:FFF_implied_fingering, no_sex, clothed female, breast press, breasts out, huge erect nipples, clothes lifted, female pubic hair, pain expression, closed eyes, moaning, open mouth, trembling, (motion lines, sound effects), saliva drooling,,,, old, old man, ugly, sanpaku, hairy male, nude male, excited male, faceless, faceless male, (penis out), precum, erection, ((penis to hip)), grabbing another's waist,
,,,,,<lora:Hyji-style--IL:0.7>, mom, long hair, low ponytail, hair over shoulder, black hair, 
, mature female, blush, lipgloss, large breasts, sagging breasts, curvy, thick thighs, , (dutch angle),  
2d, anime coloring, (masterpiece, best quality, amazing quality, very aesthetic), absurdres, ultra-detailed, highly detailed, newest," \
  --negative_prompt "(3d), realistic, (low quality, worst quality, lowres), ((hands, bad anatomy, bad hands)), extra hands, extra fingers, linked arms, conjoined arms, username, sketch, jpeg artifacts, censor, blurry, distorted, signature, watermark, patreon logo, artist name, lipstick, simple background, see-through, shemale, futanari, newhalf, lesbian, homosexual, 2girls, warm, embedding:lazyloli, chibi, femdom, grabbing own breast, (grabbing own ass), gigantic breasts, hair flower, low-tied sidelocks, neck grab, head grab, shoulder grab, extra moles, shiny skin, muscular, muscular male, manly," \
  --image_out /group/pmc015/kniu/video_clips/t2i2v_lora/seed.png \
  --video_out /group/pmc015/kniu/video_clips/t2i2v_lora/seed.mp4 \
  --width 1280 --height 720 \
  --frames 81 --steps 30 --guidance_scale 5.0 \
  --seed 42 \
  --i2v_model_id Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --transformer_gguf_high /group/pmc015/kniu/video_clips/t2i2v_lora/checkpoints/wan22I2VA14BGGUF_a14bHigh.gguf \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/lora/NSFW-22-H-e8.safetensors@0.8 \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/lora/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise_model.safetensors@1.0 \
  --lora /group/pmc015/kniu/video_clips/t2i2v_lora/lora/wan2.2-i2v-high-oral-insertion-v1.0.safetensors@1.0

```
```
CIVITAI_TOKEN='…YOUR_TOKEN…' curl -L --retry 5 --retry-connrefused --fail -C - \
  --output-dir "/group/pmc015/kniu/video_clips/t2i2v_lora/lora" \
  -o "wan2.2_i2v_highnoise_pov_missionary_v1.0.safetensors" \
  -H "Authorization: Bearer $CIVITAI_TOKEN" \
  "https://civitai.com/api/download/models/2098405"

```

```
python t2i2v.py \
  --t2i-backend sdxl \
  --sdxl_model "John6666/realism-illustrious-by-stable-yogi-v45-bf16-sdxl" \
  --i2v_lora "/group/pmc015/kniu/video_clips/t2i2v_lora/lora/NSFW-22-H-e8.safetensors@0.8" \
  --i2v_lora "/group/pmc015/kniu/video_clips/t2i2v_lora/lora/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise_model.safetensors@1.0" \
  --i2v_lora "/group/pmc015/kniu/video_clips/t2i2v_lora/lora/wan2.2-i2v-high-oral-insertion-v1.0.safetensors@1.0" \
  --i2v_lora "/group/pmc015/kniu/video_clips/t2i2v_lora/lora/wan2.2_i2v_highnoise_pov_missionary_v1.0.safetensors@1.0" \
  --i2v_lora "/group/pmc015/kniu/video_clips/t2i2v_lora/lora/WAN-2.2-I2V-POV-Cowgirl-HIGH-v1.0-fixed.safetensors@1.0" \
  --prompt "Jungle — cinematic close-up of dew on leaves, soft dawn light, volumetric rays" \
  --image_out /group/pmc015/kniu/video_clips/t2i2v_lora/seed.png \
  --video_out /group/pmc015/kniu/video_clips/t2i2v_lora/seed.mp4 \
  --frames 81 --steps 30 --guidance_scale 5.0 \
  --i2v_model_id "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
  --transformer_gguf_high "/group/pmc015/kniu/video_clips/t2i2v_lora/checkpoints/wan22I2VA14BGGUF_a14bHigh.gguf"

```
