
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
python ../third_party/Champ/main.py --run_type train  >> output.txt
```

```
sbatch inference.slurm

 squeue -u kniu

 squeue --job 552099

 scancel 552099
```
