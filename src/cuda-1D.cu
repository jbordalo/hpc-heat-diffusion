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

__global__ void compute(const float *Tn, float *Tnp1, int nx, int ny, float aXdt, float h2) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const int indexMod = index % ny;
    if (index > nx && indexMod < ny-1 && indexMod > 0 && index < nx*ny-ny) {
            float tij = Tn[index];
            float tim1j = Tn[index - ny];
            float tijm1 = Tn[index - 1];
            float tip1j = Tn[index + ny];
            float tijp1 = Tn[index + 1];

            Tnp1[index] = tij + aXdt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
    }
}

double timedif(struct timespec *t, struct timespec *t0) {
    return (t->tv_sec-t0->tv_sec)+1.0e-9*(double)(t->tv_nsec-t0->tv_nsec);
}

int main(int argc, char* argv[]) {
    int BLOCK_SIZE = atoi(argv[1]);

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

    cudaMemcpy(d_Tn, Tn, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Tnp1, Tn, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Main loop
    for (int n = 0; n <= numSteps; n++) {
        compute<<<nb, BLOCK_SIZE>>>(d_Tn, d_Tnp1, nx, ny, a * dt, h2);

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
