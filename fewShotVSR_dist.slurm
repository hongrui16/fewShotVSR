#!/bin/bash

#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq' partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relevant QOS
#SBATCH --job-name=fsv                   # Job name
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/fsv/fewShotVSR/zlog/zslurm/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/fsv/fewShotVSR/zlog/zslurm/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2                # One task per GPU
#SBATCH --gres=gpu:A100.80gb:2             # Request 2 A100 GPUs
#SBATCH --mem=80gb                         # Memory per node
#SBATCH --export=ALL
#SBATCH --time=4-24:00:00                  # Set to 4 days and 24 hours
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task

module load gnu10
module load python

# ---- Sanity check & cleanup ----
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_SIZE
unset PMI_RANK PMI_SIZE PMI_LOCAL_RANK
unset LOCAL_RANK RANK WORLD_SIZE

echo "HOST=$(hostname)"
echo "SLURM_NTASKS=$SLURM_NTASKS  SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
nvidia-smi -L
echo "CVD(raw)=${CUDA_VISIBLE_DEVICES@Q}"

python - <<'PY'
import os, torch
print("CVD:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("device_count:", torch.cuda.device_count())
print("LOCAL_RANK:", repr(os.environ.get("LOCAL_RANK")), 
      "RANK:", repr(os.environ.get("RANK")), 
      "WORLD_SIZE:", repr(os.environ.get("WORLD_SIZE")))
PY



source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/fsv/fewShotVSR

PORT=$(( (RANDOM << 8 | RANDOM) & 0xFFFF ))
echo "Random port: $PORT"


# Launch distributed training with accelerate
accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 --main_process_ip=localhost --main_process_port=$PORT \
    fewShotVSR.py 2> >(grep -v "absl::InitializeLog" | grep -v "gl_context_egl.cc" | grep -v "gl_context.cc" >&2)

# accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 --main_process_ip=localhost --main_process_port=12345 \
#     train_handposeRegressor_dist.py --debug 2> >(grep -v "absl::InitializeLog" | grep -v "gl_context_egl.cc" | grep -v "gl_context.cc" >&2)


# torchrun --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} train_VAE.py --debug 

# torchrun --nnodes=1 --nproc_per_node=1 train_VAE.py --batch_size 10

# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00
# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00