#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32

__global__ void matmulTiled(float* A, float* B, float* C, int N) {
    //declaire shared memory
    // allocate two 32x32 tiles in shared memory -> fast on-chip memory shared by all threatds in block
    // each block get own copy of these tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    // loop over tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // load tileA
        // column index in A for tile
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        // load tileB
        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        // wait for ALL threads finish
        __syncthreads();
        // compute partial sum using tile
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        // wait before loading next tile
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float* d_A; float* d_B; float* d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    matmulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[1023][1023] = %f\n", h_C[1023 * N + 1023]);

    free(h_A); free(h_B); free(h_C);
}
