# Explicitly unset ALL relevant environment variables
# Master related
unset MASTER_ADDR
unset MASTER_PORT
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK
unset LOCAL_WORLD_SIZE

# SLURM related (THIS IS CRITICAL)
unset SLURM_JOB_ID
unset SLURM_JOB_NUM_NODES
unset SLURM_JOB_NODELIST
unset SLURM_PROCID
unset SLURM_NTASKS
unset SLURM_LOCALID
# Add any other SLURM_* variables that might be present in your environment
# You can check by running `env | grep SLURM_` if you're unsure what's set.

# NCCL related (for clean runs, only enable if you need specific debugging)
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_SOCKET_IFNAME
unset NCCL_P2P_DISABLE

# Now run your DINOv2 training command
python3 -m torch.distributed.run --nproc_per_node=8 train.py