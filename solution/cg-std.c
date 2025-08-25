/* =================================================================
cg-std.c

Written by Frederick Fung for NCI OpenMP Workshop 2022

This program implements the standard conjugate gradient method, that 
is viewed as the derivation from Lanczos algorithm and a variant of 
Direct Incomplete Orthogonalisation Method. For details, see Iterative
Methods for Linear Systems by Yousef Saad and Matrix Computation by
Gene H. Goloub, Charles F. Van Loan.

The accepted linear system format for input: MatrixMarket.

The code is accelerated by openmp multi-threading. 

Compile: gcc -fopenmp -g -Wall -O3 -lm -o cg-std cg-std.c 

Example usage: ./cg-std 1e-5 < msc04515.dat 

.....................................................................

Produced for NCI Training. 

Frederick Fung 2022
4527FD1D
====================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include<time.h>
//#include<omp.h>

//#define DEBUG

double dotProduct( double *a,  double *b, int dim, int length){
  	double sum = 0.0;
	int i;
	for (i = 0; i < dim; i++) {
		sum += (a[i] * b[i]);		
	}
    

	return sum;
	

}

double *scalar(double *dest, double *v, double s, int dim){
  	int i;
	for (i = 0; i < dim; i++) {
		dest[i] = s * v[i];
	}
	return dest;

}

 double *vectoradd( double *dest, double *a,  double *b, int dim){
	int i;
	for (i = 0; i < dim; i++) {
		dest[i] = a[i] + b[i];
	}
	return dest;
}

void assignVector(double *a,  double *b, int dim){
  int i;
	for (i = 0; i < dim; i++) {
		a[i] = b[i];
	}

}

double *matrixVector(double *dest,  double *matrix,   double *v, int dim){
int i, j;
	for (i = 0; i < dim; i++) {
		dest[i] = 0.0f;
		for (j = 0; j < dim; j++) {
			dest[i] += matrix[i*dim+j] * v[j];
		}
	}
	return dest;
}

 double *vectorsubstract( double *dest,   double *a,  double *b, int dim){
	int i;
	for (i = 0; i < dim; i++) {
		dest[i] = a[i] - b[i];
	}
	return dest;
}

void CG(double *x, 
        double *r, 
		double *p,
		double *Ap,
		double *tmp_vec, 
		double *matrix,
		double *rhs,
		const int dim, 
		const int length,
		double res_prev,
		double res_new,
		double alpha,
		double *tolerance)
{


int i;
for (i=0; i< dim; i++){
  x[i] = 0.0L;
  r[i] = rhs[i];
}

r = vectorsubstract(r, rhs, matrixVector(tmp_vec, matrix, x, dim), dim);

assignVector(p, r, dim);

/* res_prev = r_i^T r_i */
res_prev = dotProduct(r, r, dim, length); 


for (i=0; i<dim; i++){

     /* Ap_i */
    Ap = matrixVector(Ap, matrix, p, dim);

    /* alpha = r_i^Tr_i / p_i^TAp_i */
    alpha = res_prev / dotProduct(p, Ap, dim, length);
	
	#ifdef DEBUG
    printf("Alpha %f rsold %f dot %f\n", alpha, res_prev, dotProduct(p, Ap, dim, length));
	#endif
    
	/* x_i+1 = x_i + alpha * p */
    x = vectoradd(x, x, scalar(tmp_vec, p, alpha, dim), dim);

    /* r_i+1 = r_i - alpha * Ap */
    r = vectorsubstract(r, r, scalar(tmp_vec, Ap, alpha, dim), dim);
    
	/* stopping criterion */
    if (res_prev< *tolerance) break;
    
	/* res_new = r_i+1^T r_i+1 */
    res_new = dotProduct(r, r, dim, length);
    
	/* p_j+1 = r_j+1 + res_new/ res_prev p_j */
    p = vectoradd(p, r, scalar(tmp_vec, p, res_new / res_prev, dim), dim); 

    res_prev =res_new;

    printf("The norm of the residual calculated directly from the definition of residual: \n");

    printf("iter %d, Res %lf\n", i, res_new);

    
    } /* end of iteration */
  
	  
} /* end of CG */


int main (int argc, char *argv[])
{
  int i, dim; 

  double tolerance;

  /* parse arguments */
  if (argc >1)
{
	tolerance = atof(argv[1]);

}
else {
        tolerance =1e-5;
		printf("==== No tolerance specified. Set default value to be 1e-5 \n");

	  }


int length; /* length of data, #non-zeros */
int M; /* number of rows */
int N; /* number of columns */
scanf("%d %d %d", &M, &N, &length);

dim  = M;

/* Read the matrix problem size */
printf("sizes %d %d %d %d \n", M, N, length, dim);

/* read the matrix market file */
int rows, cols;
 double *matrix = (double*) malloc(M * N * sizeof( double));
 double *rhs = (double*) malloc(M * sizeof( double));


/* Read the matrix problem */
for (i = 0; i< length; i++){
     double entry;	
   int index ;
  if (scanf("%d %d %lf", &rows, &cols, &entry)== 3) {
      index = (rows-1)*N +cols-1;
      matrix[index] = entry;
  }

  else 
  printf("Error rows %d, cols %d\n", rows, cols);

}

/* Initialise RHS vector */
for (i = 0; i< M; i++){
	rhs[i] = 1.0L;
}

#ifdef DEBUG
printf("=== PRINTING MATRIX === \n");
	  for (i = 0; i < 10; i++) {
	  	  printf("i entry %d %.20lf\n", i, matrix[i]);
	  }
#endif
	  


double *x = (double*) malloc(dim * sizeof(double));

double *Ap = (double*) malloc(dim * sizeof(double));

double *p = (double*) malloc(dim * sizeof(double));

double *r = (double*) malloc(dim * sizeof(double));


double *tmp_vec = (double*) malloc(dim * sizeof(double));

double res_prev = 0.0f;
double res_new = 0.0f;
double alpha = 0.0f;

struct timespec start, finish;
double elapsed;

clock_gettime(CLOCK_MONOTONIC, &start);

CG(x, r, p, Ap, tmp_vec, matrix, rhs, dim, length, res_prev, res_new, alpha, &tolerance);

clock_gettime(CLOCK_MONOTONIC, &finish);

elapsed = (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
printf("Time spent in CG %f seconds \n", elapsed);


free(matrix);
free(rhs);
free(x);
free(Ap);
free(p);
free(r);
free(tmp_vec);


return 0;
}