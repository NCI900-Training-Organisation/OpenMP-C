/* =================================================================
monte-carlo-pi-openmp.c

Written by Frederick Fung for NCI OpenMP Workshop March 2022

This program approximates the pi value by Monte-Carlo method. 

The code is accelerated by openmp multi-threading. 

Compile: gcc -fopenmp -g -Wall -O3 -lm -o monte-carlo-pi-openmp monte-carlo-pi-openmp.c 

Usage: ./monte-carlo-pi-openmp

.....................................................................

Produced for NCI Training. 

Frederick Fung 2022
4527FD1D
====================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#define MATH_PI acos(-1.0)

uint64_t calc_pi(uint64_t samples) {
    uint64_t count = 0;

    #pragma omp parallel
    {
        unsigned int seed =(unsigned)time(NULL)^ (2654435761u * (unsigned)(omp_get_thread_num() + 1)) ^ (unsigned)samples;

        #pragma omp for reduction(+:count) schedule(static)
        for (long long i = 0; i < (long long)samples; ++i) {
            double x = rand_r(&seed) / (double)RAND_MAX;
            double y = rand_r(&seed) / (double)RAND_MAX;
            if (x * x + y * y <= 1.0) {
                ++count;
            }
        }
    } // implicit barrier

    return count;
}

int main(void) {
    int trials[] = {10, 100, 1000, 10000, 100000, 1000000,
                    10000000, 100000000, 1000000000};

    printf("MATH Pi %.15f\n", MATH_PI);
    printf("/////////////////////////////////////////////////////\n");

    size_t ntrials = sizeof(trials) / sizeof(trials[0]);
    for (size_t i = 0; i < ntrials; ++i) {
        uint64_t samples = (uint64_t)trials[i];

        double t0 = omp_get_wtime();
        uint64_t hits = calc_pi(samples);
        double t1 = omp_get_wtime();

        double pi_est = 4.0 * (double)hits / (double)samples;
        double err = fabs(pi_est - MATH_PI);

        printf("Sampling points %d; Hit numbers %d; Approx Pi %f\n", samples, hit, pi_est);  

    }
    return 0;
}