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
