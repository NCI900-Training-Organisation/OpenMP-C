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

Frederick Fung 2022
4527FD1D
====================================================================*/


#include<stdio.h>
#include <stdlib.h>
#include<string.h>
#include<math.h>

float bnd_fc(int x, int y, double space){
    /* Boundary function */

    double value = sin(M_PI * x * space) * sin(M_PI *y *space);
    //printf ("value %f\n", value);
    return value;
}

float rhs_fc(int x, int y, double space){
    /* rhs sin(pi*x)sin(pi*y) */
    double value =(2.0* M_PI*M_PI) * sin( M_PI * x * space) * sin ( M_PI * y * space); 
    return value;
}


float l2_residual(int size, double space, double grid[size][size], double rhs[size][size]){
    
    /* l2 norm of residual by a given approximation */
    double residual = 0.0f;

    for (int i = 1; i < size -1 ; i++ ){
        for (int j = 1; j < size - 1; j++){
            
            /* residual = Ax^k -b */
            double diff = (4*grid[i][j]-grid[i-1][j]-grid[i+1][j]-grid[i][j+1]-grid[i][j-1])/ (space *space) -rhs[i][j];

            residual += pow(diff, 2);


        }
      }
    
    
    return sqrt(residual);

    
}


void Jacobi(double tolerance, int size, double grid[size][size], double rhs[size][size], double space){
    
    double (*grid_tmp)[size] = malloc(sizeof *rhs *size);

    double residual = 1.0f;
    double *p_grid, *p_grid_tmp, *p_rhs, *swap;
    int iter = 0;
    int i, j;


    memcpy(&grid_tmp[0][0], &grid[0][0], size *size *sizeof(grid[0][0]));
    p_grid = &grid[0][0];
    p_grid_tmp = &grid_tmp[0][0];
    p_rhs = &rhs[0][0];
    
    /* iterates until tolerance is reached */
    do {
   
  
    iter +=1;
    
    for (i = 1; i< size -1; i++){    
        for ( j = 1; j< size -1; j++){
        
        /* Update new approximation */
        *(p_grid_tmp + i *size +j) =  space * space * (*(p_rhs +i * size +j)) *0.25 + ( *(p_grid + ((i-1)* size) +j) + *(p_grid + ((i+1)*size) +j)
                                                                              +  *(p_grid + (i *size) +j -1) + *(p_grid + (i*size) +j+1) ) * 0.25;
         }     
    }
         swap = p_grid_tmp;
         p_grid_tmp = p_grid;
         p_grid = swap;

         /* Print out residuals after every 100 iterations */
         if (iter % 1000 == 0){
         residual = l2_residual(size, space, grid, rhs);
        
         printf("Residual after %d iteratio: %.10f\n", iter, residual);
        }


    }
   
   while (residual > tolerance);
}

void GS(double tolerance, int size, double grid[size][size], double rhs[size][size], double space){


    double residual = 1.0f;
    double *p_grid,  *p_rhs;
    int iter = 0;
    int i, j;


    
    p_grid = &grid[0][0];
    p_rhs = &rhs[0][0];
        
    /* iterates until tolerance is reached */
    do {
   
    iter +=1;
    
    for (i = 1; i< size -1; i++){
       for ( j = 1; j< size -1; j++){
           
        /* Update new approximation */
        *(p_grid + i *size +j) =  space * space * (*(p_rhs +i * size +j)) *0.25 + ( *(p_grid + ((i-1)* size) +j) + *(p_grid + ((i+1)*size) +j)
                                                                              +  *(p_grid + (i *size) +j -1) + *(p_grid + (i*size) +j+1) ) * 0.25;
      }     
    }
     

        /* Print out residuals after every 100 iterations */
        if (iter % 1000 == 0){
        residual = l2_residual(size, space, grid, rhs);
        
        printf("Residual after %d iteratio: %.10f\n", iter, residual);
        }


    }
   
   while (residual > tolerance);
}

int main(int argc, char *argv[]){


int size;

double space;

double tolerance;

char *method;


/* parse arguments */
if (argc == 4){
size = atof(argv[1]);

tolerance = atof(argv[2]);

method = argv[3];

if ((strcmp(method, "Gauss-Seidel") == 0)){
   printf("%s METHOD IS IN USE   \n ", method);}
else if ((strcmp(method, "Jacobi") == 0)){
    printf("%s METHOD IS IN USE \n", method);
}
else {
    
    printf( "Not a valid method\n");
    exit(1);
}


FILE *fp;
//clrscr();
fp=fopen("laplace-soln.dat","w"); //output will be stored in this file

space = (double) 1 / (size-1);

double (*grid)[size] = malloc(sizeof *grid * size);

double (*rhs)[size] = malloc(sizeof *rhs * size);




/* Initial mesh */


for (int i = 0; i< size; i++){
    for (int j = 0 ; j< size; j++){
        
        if (i == 0 || j == 0 || i == size-1 || j == size - 1 ){
            
            grid[i][j] = bnd_fc(i, j,  space);
            rhs[i][j] = 0.0;
         
        }
        else
       {
        grid[i][j] = 0.0;
        rhs[i][j] = rhs_fc(i,j, space);
       }
    }
    

      
    }


/* Smoothers */
if (strcmp(method, "Gauss-Seidel")==0)
GS(tolerance, size, grid, rhs, space);
else{
Jacobi(tolerance, size, grid, rhs, space);}

/* Output data  */
for (int i = 0; i< size; i++){
  for (int j = 0; j<size; j++)
               fprintf(fp,"%.5f\t",grid[i][j]);
            fprintf(fp,"\n");
}

fclose(fp);
exit(0);
}
else {
    printf("Usage: %s [size] [tolerance] [method] \n", argv[0]);
    exit(1);
}

}