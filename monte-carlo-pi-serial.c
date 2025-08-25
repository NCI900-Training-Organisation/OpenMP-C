/* =================================================================
monte-carlo-pi-serial.c

Written by Frederick Fung for NCI OpenMP Workshop March 2022, updated
by Frederick Fung in 2025

This program approximates the pi value by Monte-Carlo method. 

Compile: gcc -g -Wall -O3 -lm -o monte-carlo-pi-serial monte-carlo-pi-serial.c 

Usage: ./monte-carlo-pi-serial

.....................................................................
Produced for NCI Training. 

Frederick Fung 2022, 2025
====================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define MATH_PI acos(-1.0)

uint64_t calc_pi(uint64_t samples) {
    double x, y;
    unsigned int seed = (unsigned)time(NULL) ^ (unsigned)(samples * 2654435761u);
    uint64_t count = 0;

    for (uint64_t i = 0; i < samples; ++i) {
        x = rand_r(&seed) / (double)RAND_MAX;
        y = rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0) ++count;
    }
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
        uint64_t hits = calc_pi(samples);
        double pi_est = 4.0 * (double)hits / (double)samples;

        printf("Sampling points %ld; Hit numbers %ld; Approx Pi %f\n", samples, hits, pi_est);  

    }
    return 0;
}