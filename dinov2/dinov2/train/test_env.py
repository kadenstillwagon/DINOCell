# test_env.py
import os
import torch
import torch.distributed as dist

print(f"PID {os.getpid()}: "
      f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
      f"RANK={os.environ.get('RANK')}, "
      f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
      f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
      f"MASTER_PORT={os.environ.get('MASTER_PORT')}")

# Now try to initialize the process group (optional, just to see if it passes)
try:
    if 'MASTER_ADDR' in os.environ: # Only try to init if env vars are set
        dist.init_process_group(backend="nccl", 
                                rank=int(os.environ['RANK']),
                                world_size=int(os.environ['WORLD_SIZE']),
                                init_method="env://",
                                device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"))
        print(f"PID {os.getpid()}: Successfully initialized process group.")
        dist.destroy_process_group()
except Exception as e:
    print(f"PID {os.getpid()}: Failed to initialize process group: {e}")