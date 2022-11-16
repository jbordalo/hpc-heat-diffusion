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

#define BLOCK_SIZE 256

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

    //Global index
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const int indexMod = index % ny;

    //SM index
    int s = threadIdx.x + ny;

    //load data into SM
    s_Tn[s] = Tn[index];
    //TOP
    if (s < 2 * ny)
        s_Tn[s - ny] = Tn[index - ny];
    //BOTTOM
    if (s > BLOCK_SIZE && s < BLOCK_SIZE + ny )
        s_Tn[s + ny] = Tn[index + ny];
    //LEFT
    if (s == ny)
        s_Tn[s - 1] = Tn[index - 1];
    //RIGHT
    if (s == BLOCK_SIZE+ny-1)
        s_Tn[s + 1] = Tn[index + 1];

    __syncthreads();

    if (index > nx && indexMod < ny-1 && indexMod > 0 && index < nx*ny-ny) {
            float tij = s_Tn[s];
            float tim1j = s_Tn[s - ny];
            float tijm1 = s_Tn[s - 1];
            float tip1j = s_Tn[s + ny];
            float tijp1 = s_Tn[s + 1];

            Tnp1[index] = tij + aXdt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
    }
}

int main() {
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

    int nb = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("aSimulated time: %g (%d steps of %g)\n", numSteps * dt, numSteps, dt);
    printf("Simulated surface: %gx%g (in %dx%g divisions)\n", nx * h, ny * h, nx, h);
    writeTemp(Tn, nx, ny, 0);

    // Timing
    clock_t start = clock();

    cudaMemcpy(d_Tn, Tn, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Tnp1, Tn, numElements * sizeof(float), cudaMemcpyHostToDevice);

    int smem = (BLOCK_SIZE+2*ny)*sizeof(int);
    // Main loop
    for (int n = 0; n <= numSteps; n++) {
        compute<<<nb, BLOCK_SIZE, smem>>>(d_Tn, d_Tnp1, nx, ny, a * dt, h2);

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
    clock_t finish = clock();
    printf("It took %f seconds\n", (double) (finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(Tn);

    cudaFree(d_Tn);
    cudaFree(d_Tnp1);

    return 0;
}
