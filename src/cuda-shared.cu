/*
 * Based on materials from:
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 * https://enccs.github.io/OpenACC-CUDA-beginners/2.02_cuda-heat-equation/
 * changed 23 nov 2022 - vad@fct.unl.pt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef PNG
#include "pngwriter.h"
#endif

/* Convert 2D index layout to unrolled 1D layout
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 * \returns An index in the unrolled 1D array.
 */
__device__ int getIndex(const int i, const int j, const int width) {
    return i * width + j;
}

void initTemp(float *T, int h, int w) {
    // Initializing the data with heat from top side
    // all other points at zero
    for (int i = 0; i < w; i++) {
        T[i] = 100.0;
    }
}

/* write_pgm - write a PGM image ascii file
 */
void write_pgm(FILE *f, float *img, int width, int height, int maxcolors) {
    // header
    fprintf(f, "P2\n%d %d %d\n", width, height, maxcolors);
    // data
    for (int l = 0; l < height; l++) {
        for (int c = 0; c < width; c++) {
            int p = (l * width + c);
            fprintf(f, "%d ", (int) (img[p]));
        }
        putc('\n', f);
    }
}

/* write heat map image
*/
void writeTemp(float *T, int h, int w, int n) {
    char filename[64];
#ifdef PNG
    sprintf(filename, "heat_%06d.png", n);
    save_png(T, h, w, filename, 'c');
#else
    sprintf(filename, "cuda_heat_%06d.pgm", n);
    FILE *f = fopen(filename, "w");
    write_pgm(f, T, w, h, 100);
    fclose(f);
#endif
}

__global__ void compute(const float *Tn, float *Tnp1, int nx, int ny, float aXdt, float h2) {
    extern __shared__ float s_Tn[];
    int BLOCK_SIZE_X = blockDim.x;
    int BLOCK_SIZE_Y = blockDim.y;

    // Global indices
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // Shared memory indices
    int s_i = threadIdx.x + 1;
    int s_j = threadIdx.y + 1;
    int s_ny = BLOCK_SIZE_Y + 2;

    // Load data into shared memory
    s_Tn[getIndex(s_i, s_j, s_ny)] = Tn[getIndex(i, j, ny)];

    // Top border
    if (s_i == 1 && i!=0) {
        s_Tn[getIndex(s_i - 1, s_j, s_ny)] = Tn[getIndex(blockIdx.x * blockDim.x - 1, j, ny)];
    }
    // Bottom border
    if (s_i == BLOCK_SIZE_X && i != nx - 1 ) {
        s_Tn[getIndex(BLOCK_SIZE_X+1 , s_j, s_ny)] = Tn[getIndex((blockIdx.x + 1) * blockDim.x, j, ny)];
    }
    // Left border
    if (s_j == 1 && j != 0) {
        s_Tn[getIndex(s_i, s_j - 1, s_ny)] = Tn[getIndex(i, blockIdx.y * blockDim.y - 1, ny)];
    }
    // Right border
    if (s_j == BLOCK_SIZE_Y && j != ny - 1) {
        s_Tn[getIndex(s_i, BLOCK_SIZE_Y+1 , s_ny)] = Tn[getIndex(i, (blockIdx.y + 1) * blockDim.y, ny)];
    }

    __syncthreads();

    if (i > 0 && i < nx - 1) {
        if (j > 0 && j < ny - 1) {
            float tij = s_Tn[getIndex(s_i, s_j, s_ny)];
            float tim1j = s_Tn[getIndex(s_i - 1, s_j, s_ny)];
            float tijm1 = s_Tn[getIndex(s_i, s_j - 1, s_ny)];
            float tip1j = s_Tn[getIndex(s_i + 1, s_j, s_ny)];
            float tijp1 = s_Tn[getIndex(s_i, s_j + 1, s_ny)];

            Tnp1[getIndex(i, j, ny)] = tij + aXdt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
        }
    }
}

double timedif(struct timespec *t, struct timespec *t0) {
    return (t->tv_sec-t0->tv_sec)+1.0e-9*(double)(t->tv_nsec-t0->tv_nsec);
}

int main(int argc, char* argv[]) {
    int threads = atoi(argv[1]);
    int BLOCK_SIZE_X = threads;
    int BLOCK_SIZE_Y = threads;

    const int nx = 200;   // Width of the area
    const int ny = 200;   // Height of the area

    const float a = 0.5;     // Diffusion constant

    const float h = 0.005;   // h=dx=dy  grid spacing

    const float h2 = h * h;

    const float dt = h2 / (4.0 * a); // Largest stable time step
    const int numSteps = 100000;      // Number of time steps to simulate (time=numSteps*dt)
    const int outputEvery = 100000;   // How frequently to write output image

    int numElements = nx * ny;

    // Allocate two sets of data for current and next timesteps
    float *Tn = (float *) calloc(numElements, sizeof(float));

    // Initializing the data for T0
    initTemp(Tn, nx, ny);

    float *d_Tn;
    float *d_Tnp1;

    cudaMalloc(&d_Tn, numElements * sizeof(float));
    cudaMalloc(&d_Tnp1, numElements * sizeof(float));

    dim3 numBlocks(nx / BLOCK_SIZE_X + 1, ny / BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);


    printf("Simulated time: %g (%d steps of %g)\n", numSteps * dt, numSteps, dt);
    printf("Simulated surface: %gx%g (in %dx%g divisions)\n", nx * h, ny * h, nx, h);
    writeTemp(Tn, nx, ny, 0);

    // Timing
    struct timespec t0, t;
    /*start*/
    clock_gettime(CLOCK_MONOTONIC, &t0);

    cudaMemcpy(d_Tn, Tn, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Tnp1, Tn, numElements * sizeof(float), cudaMemcpyHostToDevice);

    int smem = (BLOCK_SIZE_X + 2) * (BLOCK_SIZE_Y + 2) * sizeof(float);
    // Main loop
    for (int n = 0; n <= numSteps; n++) {
        compute<<<numBlocks, threadsPerBlock, smem>>>(d_Tn, d_Tnp1, nx, ny, a * dt, h2);

        // Write the output if needed
        if ((n + 1) % outputEvery == 0) {
            cudaMemcpy(Tn, d_Tn, numElements * sizeof(float), cudaMemcpyDeviceToHost);
            cudaError_t errorCode = cudaGetLastError();

            if (errorCode != cudaSuccess) {
                printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
                exit(1);
            }
            writeTemp(Tn, nx, ny, n + 1);
        }

        // Swapping the pointers for the next timestep
        float *t = d_Tn;
        d_Tn = d_Tnp1;
        d_Tnp1 = t;
    }

    // Timing
    /*end*/
    clock_gettime(CLOCK_MONOTONIC, &t);
    printf("It took %f seconds\n", timedif(&t, &t0) );

    // Release the memory
    free(Tn);

    cudaFree(d_Tn);
    cudaFree(d_Tnp1);

    return 0;
}
