# matmul & attention kernel optimization benchmark

We compare naive CUDA, tiled CUDA, triton, and cuBLAS (NVIDIA) matmul operations as well as naive attention, FlashAttention (Dao, 2023), and PyTorch SDPA kernel implementations on metrics including time (ms), GFLOPs, % peak FLOPs, BW (GB/s), % peak BW

## Results (T4 GPU)

### Matrix Multiplication (N=1024)

| Implementation | Time (ms) | GFLOPS | % Peak FLOPs |
|----------------|----------:|-------:|-------------:|
| naive CUDA     |       6.5 |    329 |          4.1 |
| tiled CUDA     |       4.5 |    480 |          5.9 |
| Triton         |       1.6 |   1355 |         16.7 |
| cuBLAS         |       0.7 |   3094 |         38.2 |

### Attention (batch=1, heads=8, seq=4096, dim=64)

| Implementation     | Time (ms) | GFLOPS | Notes                                          |
|--------------------|----------:|-------:|------------------------------------------------|
| naive attention    |      20.5 |   1679 | correct baseline                               |
| FlashAttention*    |      25.5 |   1347 | correct (max diff 1e-6)                        |
| PyTorch SDPA       |      11.3 |   3037 | production FA2                                 |

\* Triton implementation — missing vectorized loads + large tiling

Raw results: [`benchmark/t4_results.csv`](benchmark/t4_results.csv)
