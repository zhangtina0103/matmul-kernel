import torch
import triton
import triton.language as tl

// tell triton to compile this python function into GPU code
@triton.jit
def matmul_kernel(
    A, B, C,
    N,
    BLOCK_SIZE: tl.constexpr, // compile time constant
):
    // blockIdx in CUDA -> blocks of threads
    // each program handle entire BLOCK_SIZE^2 tile of output
    row = tl.program_id(0)
    col = tl.program_id(1)

    row_offset = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) // vector, return Triton sensor
    col_offset = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    // initialize matrix of zeros
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, N, BLOCK_SIZE):
        k_offset = k + tl.arange(0, BLOCK_SIZE)
        // mask for bounds
        a_mask = (row_offset[:, None] < N) & (k_offset[None, :] < N)
        a = tl.load(A + row_offset[:, None] * N + k_offset[None, :],
                    mask=a_mask, other=0.0)

        b_mask = (k_offset[:, None] < N) & (col_offset[None, :] < N)
        b = tl.load(B + k_offset[:, None] * N + col_offset[None, :],
                    mask=b_mask, other=0.0)
        // dot product
        acc += tl.dot(a, b)

    c_mask = (row_offset[:, None] < N) & (col_offset[None, :] < N)
    tl.store(C + row_offset[:, None] * N + col_offset[None, :],
             acc, mask=c_mask)


def matmul(A, B):
    N = A.shape[0]
    C = torch.zeros((N, N), device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 32
    grid = (N // BLOCK_SIZE, N // BLOCK_SIZE)
    matmul_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C
