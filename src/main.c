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
int getIndex(const int i, const int j, const int width) {
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
    sprintf(filename, "heat_%06d.pgm", n);
    FILE *f = fopen(filename, "w");
    write_pgm(f, T, w, h, 100);
    fclose(f);
#endif
}

int main() {
    const int nx = 200;      // Width of the area
    const int ny = 200;      // Height of the area

    const float a = 0.5;     // Diffusion constant

    const float h = 0.005;   // h=dx=dy  grid spacing

    const float h2 = h * h;

    const float dt = h2 / (4.0 * a);  // Largest stable time step
    const int numSteps = 100000;      // Number of time steps to simulate (time=numSteps*dt)
    const int outputEvery = 100000;   // How frequently to write output image

    int numElements = nx * ny;

    // Allocate two sets of data for current and next timesteps
    float *Tn = (float *) calloc(numElements, sizeof(float));
    float *Tnp1 = (float *) calloc(numElements, sizeof(float));

    // Initializing the data for T0
    initTemp(Tn, nx, ny);

    // Fill in the data on the next step to ensure that the boundaries are identical.
    memcpy(Tnp1, Tn, numElements * sizeof(float));

    printf("Simulated time: %g (%d steps of %g)\n", numSteps * dt, numSteps, dt);
    printf("Simulated surface: %gx%g (in %dx%g divisions)\n", nx * h, ny * h, nx, h);
    writeTemp(Tn, nx, ny, 0);

    // Timing
    clock_t start = clock();

    // Main loop
    for (int n = 0; n <= numSteps; n++) {
        // Going through the entire area for one step
        // (borders stay at the same fixed temperatures)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                const int index = getIndex(i, j, ny);
                float tij = Tn[index];
                float tim1j = Tn[getIndex(i - 1, j, ny)];
                float tijm1 = Tn[getIndex(i, j - 1, ny)];
                float tip1j = Tn[getIndex(i + 1, j, ny)];
                float tijp1 = Tn[getIndex(i, j + 1, ny)];

                Tnp1[index] = tij + a * dt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
            }
        }
        // Write the output if needed
        if ((n + 1) % outputEvery == 0)
            writeTemp(Tnp1, nx, ny, n + 1);

        // Swapping the pointers for the next timestep
        float *t = Tn;
        Tn = Tnp1;
        Tnp1 = t;
    }

    // Timing
    clock_t finish = clock();
    printf("It took %f seconds\n", (double) (finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(Tn);
    free(Tnp1);

    return 0;
}
