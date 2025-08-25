/* =================================================================
mandelbrot-serial.c

Written by Frederick Fung for NCI OpenMP Workshop March 2022, updated
by Frederick Fung 2025

This program computes Mandelbrot set within a pre-defined region. 
For plotting, the output is filled as the iteration goes.

Output: an 2D array. Each element corresponds to a pixel to be drawn.  


Compile: gcc -g -Wall -O3 -o mandelbrot-serial mandelbrot-serial.c 

Usage: ./mandelbrot-serial

.....................................................................

Produced for NCI Training. 

Frederick Fung 2022, 2025
====================================================================*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


enum { MAXITER = 100 };
static const double XMIN = -2.0, XMAX = 0.47;
static const double YMIN = -1.12, YMAX = 1.12;


typedef struct { double re, im; } Complex;

/* Return the first iteration that escapes; if never escapes, return MAXITER */
static int mandelbrot_escape(Complex c) {
    Complex z = {0.0, 0.0};
    for (int i = 0; i < MAXITER; ++i) {
        /* z = z^2 + c */
        double zr2 = z.re * z.re;
        double zi2 = z.im * z.im;
        double two_zr_zi = 2.0 * z.re * z.im;

        z.re = zr2 - zi2 + c.re;
        z.im = two_zr_zi + c.im;

        /* escape test: |z|^2 > 4,  diverges */
        if (z.re * z.re + z.im * z.im > 4.0)
            return i + 1; 
    }
    return MAXITER;
}

static void gen_mandelbrot(int points) {
    if (points <= 0) {
        fprintf(stderr, "points must be positive\n");
        return;
    }

    char path[128];

    snprintf(path, sizeof(path), "mandelbrot_set_%d.csv", points);
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("fopen");
        return;
    }

    /* Include both endpoints by using N samples over a closed interval */
    const double stepx = (points > 1) ? (XMAX - XMIN) / (points - 1) : 0.0;
    const double stepy = (points > 1) ? (YMAX - YMIN) / (points - 1) : 0.0;

    for (int i = 0; i < points; ++i) {
        double x = XMIN + i * stepx;
        for (int j = 0; j < points; ++j) {
            double y = YMIN + j * stepy;
            Complex c = { x, y };
            int iters = mandelbrot_escape(c);
            fprintf(fp, (j + 1 < points) ? "%d," : "%d\n", iters);
        }
    }

    if (fclose(fp) != 0) perror("fclose");
}

int main(void) {
    int npoints_array[] = {100, 1000, 10000};
    int ncases = (int)(sizeof(npoints_array) / sizeof(npoints_array[0]));

    for (int k = 0; k < ncases; ++k) {
        int pts = npoints_array[k];
        printf("Resolution #%d: %d x %d\n", k, pts, pts);
        gen_mandelbrot(pts);
    }
    return 0;
}
