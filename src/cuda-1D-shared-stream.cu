#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef PNG
#include "pngwriter.h"
#endif

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

__global__ void compute(const float *Tn, float *Tnp1, int nx, int ny, float aXdt, float h2, int offset) {
    extern __shared__ float s_Tn[];
    int BLOCK_SIZE = blockDim.x;

    // Global index
    const int index = threadIdx.x + blockIdx.x * blockDim.x + offset;
    const int indexMod = index % ny;

    // Shared memory index
    int s = threadIdx.x + ny;

    // load data into SM
    s_Tn[s] = Tn[index];
    // Top
    if (s < 2 * ny)
        s_Tn[s - ny] = Tn[index - ny];
    // Bottom
    if (s > BLOCK_SIZE && s < BLOCK_SIZE + ny)
        s_Tn[s + ny] = Tn[index + ny];
    // Left
    if (s == ny)
        s_Tn[s - 1] = Tn[index - 1];
    // Right
    if (s == BLOCK_SIZE + ny - 1)
        s_Tn[s + 1] = Tn[index + 1];

    __syncthreads();

    if (index > nx && indexMod < ny - 1 && indexMod > 0 && index < nx * ny - ny) {
        float tij = s_Tn[s];
        float tim1j = s_Tn[s - ny];
        float tijm1 = s_Tn[s - 1];
        float tip1j = s_Tn[s + ny];
        float tijp1 = s_Tn[s + 1];

        Tnp1[index] = tij + aXdt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
    }
}

double timedif(struct timespec *t, struct timespec *t0) {
    return (t->tv_sec-t0->tv_sec)+1.0e-9*(double)(t->tv_nsec-t0->tv_nsec);
}

int main(int argc, char* argv[]) {
    int BLOCK_SIZE = atoi(argv[1]);
    int STREAM_SIZE = atoi(argv[2]);

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

    printf("Simulated time: %g (%d steps of %g)\n", numSteps * dt, numSteps, dt);
    printf("Simulated surface: %gx%g (in %dx%g divisions)\n", nx * h, ny * h, nx, h);
    writeTemp(Tn, nx, ny, 0);

    // Timing
    struct timespec t0, t;
    /*start*/
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int nStreams = numElements / STREAM_SIZE;
    int smem = (BLOCK_SIZE + 2 * ny) * sizeof(float);

    nb = (nb + nStreams - 1) / nStreams;

    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

    int offset;

    for (int i = 0; i < nStreams; ++i) {
        offset = i * STREAM_SIZE;
        if (offset - ny >= 0) offset -= ny;
        int size = STREAM_SIZE;
        if (offset + STREAM_SIZE + ny < numElements) size += ny;
        cudaMemcpyAsync(&d_Tn[offset], &Tn[offset],
                        sizeof(float) * size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_Tnp1[offset],  &Tn[offset],
                        sizeof(float) * size, cudaMemcpyHostToDevice, stream[i]);

    }

    for (int n = 0; n <= numSteps; n++) {
        for (int i = 0; i < nStreams; ++i) {
            offset = i * STREAM_SIZE;
            compute<<<nb, BLOCK_SIZE, smem, stream[i]>>>(d_Tn, d_Tnp1, nx, ny, a * dt, h2, offset);
        }

        if ((n + 1) % outputEvery == 0) {
            for (int i = 0; i < nStreams; ++i) {
                offset = i * STREAM_SIZE;
                cudaMemcpyAsync(&Tn[offset], &d_Tn[offset],
                                sizeof(float) * STREAM_SIZE, cudaMemcpyDeviceToHost, stream[i]);
                cudaError_t errorCode = cudaGetLastError();
                if (errorCode != cudaSuccess) {
                    printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
                    exit(1);
                }
            }
            cudaDeviceSynchronize();
            writeTemp(Tn, nx, ny, n + 1);
        }

        // Swapping the pointers for the next timestep
        float *temp = d_Tn;
        d_Tn = d_Tnp1;
        d_Tnp1 = temp;
    }

    // Timing
    /*end*/
    clock_gettime(CLOCK_MONOTONIC, &t);
    printf("It took %f seconds\n", timedif(&t, &t0) );
    // Release the memory
    free(Tn);

    for (int i = 0; i < nStreams; ++i)
        cudaStreamDestroy(stream[i]);

    cudaFree(d_Tn);
    cudaFree(d_Tnp1);

    return 0;
}
