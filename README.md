# matmul & attention kernel optimization benchmark
We compare naive CUDA, tiled CUDA, triton, and cuBLAS (NVIDIA) matmul operations as well as naive attention, FlashAttention (Dao, 2023), and PyTorch SDPA kernel implementations on metrics including time (ms), GFLOPs, % peak FLOPs, BW (GB/s), % peak BW
