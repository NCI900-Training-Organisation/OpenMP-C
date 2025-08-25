/* =================================================================
fd_laplace-serial.c

Solve a model 2D Poisson equaton with Dirichlet boundary condition.

-Delta u = 2pi^2 * sin(pi x)sin(pi y) in [0,1]^2
       u = sin(pi x) sin(y) on boundary

The problem is discretised over a uniform mesh by finite difference 
method and the resulting linear system is solved by choices of Jacobi
or Gauss-Seidel.


Compile:  gcc -fopenmp -g -Wall -O3 -lm -o fd_laplace fd_laplace.

Usage:  ./fdd_laplace-omp size tolerance method

Produced for NCI Training. 

Frederick Fung 2022, 2025
====================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double bnd_fc(int i, int j, double h) {
    const double x = i * h, y = j * h;
    return sin(M_PI * x) * sin(M_PI * y);
}
static inline double rhs_fc(int i, int j, double h) {
    const double x = i * h, y = j * h;
    return (2.0 * M_PI * M_PI) * sin(M_PI * x) * sin(M_PI * y);
}

/* L2 norm of residual r = A u - f on interior  */
static double l2_residual(int n, double h,
                          double (*restrict u)[n],
                          const double (*restrict f)[n],
                          int normalize)
{
    double sum = 0.0;
    const double invh2 = 1.0 / (h * h);
    for (int i = 1; i < n - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            const double Au =
                (4.0 * u[i][j] - u[i-1][j] - u[i+1][j] - u[i][j-1] - u[i][j+1]) * invh2;
            const double r = Au - f[i][j];
            sum += r * r;
        }
    }
    double res = sqrt(sum);
    if (normalize) res /= (n - 2) * (n - 2);
    return res;
}

/* Jacobi: updates grid in-place  */
static int Jacobi(double tol, int max_iter, int n,
                  double (*restrict grid)[n],
                  const double (*restrict rhs)[n],
                  double h)
{
    double (*next)[n] = malloc((size_t)n * (size_t)n * sizeof **next);
    if (!next) { perror("malloc"); return -1; }

    memcpy(next, grid, (size_t)n * (size_t)n * sizeof **grid);

    const double h2_4 = 0.25 * h * h;
    int iter = 0;
    double res = l2_residual(n, h, grid, rhs, 0);

    while (res > tol && iter < max_iter) {
        ++iter;
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                next[i][j] = h2_4 * rhs[i][j]
                           + 0.25 * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]);
            }
        }
        double (*tmp)[n] = grid; grid = next; next = tmp;

        if (iter % 1000 == 0){
            res = l2_residual(n, h, grid, rhs, 0);
            printf("Residual after %d iteration: %.10f\n", iter, res);
        }

    }
    memcpy(next, grid, (size_t)n * (size_t)n * sizeof **grid);
    free(next);
    return iter;
}

/* Gauss–Seidel (in-place) */
static int GaussSeidel(double tol, int max_iter, int n,
                       double (*restrict grid)[n],
                       const double (*restrict rhs)[n],
                       double h)
{
    const double h2_4 = 0.25 * h * h;
    int iter = 0;
    double res = l2_residual(n, h, grid, rhs, 0);

    while (res > tol && iter < max_iter) {
        ++iter;
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                grid[i][j] = h2_4 * rhs[i][j]
                           + 0.25 * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]);
            }
        }
        if (iter % 1000 == 0){
            res = l2_residual(n, h, grid, rhs, 0);
            printf("Residual after %d iteration: %.10f\n", iter, res);
        }
    }
    return iter;
}

int main(int argc, char *argv[]){
    int size;
    double space;
    double tolerance;
    const char *method;

    if (argc == 4){
        size = atoi(argv[1]);
        tolerance = atof(argv[2]);
        method = argv[3];

        if (size < 3) {  /* prevent division by zero and degenerate grid */
            fprintf(stderr, "size must be >= 3\n");
            return 1;
        }

        if (strcmp(method, "Gauss-Seidel") == 0){
            printf("%s METHOD IS IN USE\n", method);
        } else if (strcmp(method, "Jacobi") == 0){
            printf("%s METHOD IS IN USE\n", method);
        } else {
            printf("Not a supported method\n");
            return 1;
        }

        FILE *fp = fopen("laplace-soln.dat","w");
        if (!fp) { perror("fopen"); return 1; }

        space = 1.0 / (size - 1);

        double (*grid)[size] = malloc(sizeof *grid * size);
        double (*rhs )[size] = malloc(sizeof *rhs  * size);
        if (!grid || !rhs) { perror("malloc"); free(grid); free(rhs); fclose(fp); return 1; }

        /* set up the boundary condition and rhs*/
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                if (i == 0 || j == 0 || i == size-1 || j == size-1){
                    grid[i][j] = bnd_fc(i, j, space);
                    rhs[i][j]  = 0.0;
                } else {
                    grid[i][j] = 0.0;
                    rhs[i][j]  = rhs_fc(i, j, space);
                }
            }
        }

        if (strcmp(method, "Gauss-Seidel") == 0)
            GaussSeidel(tolerance, /*max_iter=*/1000000, size, grid, rhs, space);
        else
            Jacobi     (tolerance, /*max_iter=*/1000000, size, grid, rhs, space);

        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++)
                fprintf(fp, "%.5f\t", grid[i][j]);
            fprintf(fp, "\n");
        }

        fclose(fp);
        free(grid);
        free(rhs);
        return 0;
    } else {
        printf("Usage: %s [size] [tolerance] [method]\n", argv[0]);
        return 1;
    }
}
