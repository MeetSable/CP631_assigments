#include "mod_matvec.h"
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
* 0 is failure
* 1 is success
*/ 
#define MAX_LINE_LENGTH 50

int Read_input(char* input_path, int* mg, int* ng, float** local_a, float** local_x)
{
	FILE *fptr = NULL;
	fptr = fopen(input_path, "r");
	if(fptr == NULL)
	{
		printf("Failed to read file\n");
		return 0;
	}

	char line[MAX_LINE_LENGTH];
	char *token, *rest;
	fgets(line, MAX_LINE_LENGTH, fptr);
	int m, n;
	token = strtok_r(line, " ", &rest);
	m = atoi(token);
	token = strtok_r(line, " ", &rest);
	n = atoi(token);
	*mg = m, *ng = n;
	*local_a = (float*)malloc(m*n*sizeof(float));
	*local_x = (float*)malloc(n*sizeof(float));
	memset(*local_a, 0, m*n*sizeof(float));
	memset(*local_x, 0, n*sizeof(float));
	int i, j;
	for(i = 0 ; i < m ; i++)
	{
		fgets(line, MAX_LINE_LENGTH, fptr);
		token = strtok_r(line, " ", &rest);
		for(j = 0 ; j < n ; j++)
		{
			(*local_a)[i*n + j] = atof(token);
			token = strtok_r(NULL, " ", &rest);
		}
	}
	fgets(line, MAX_LINE_LENGTH, fptr);
	token = strtok_r(line, " ", &rest);
	for(i = 0 ; i < n ; i++)
	{
		(*local_x)[i] = atof(token);
		token = strtok_r(NULL, " ", &rest);
	}
	fclose(fptr);
	return 1;
}

void DistributeData(float** local_a, float** local_x, int m, int n, int local_m, int local_n, int my_rank, int p)
{
	if(my_rank != 0)
	{
		*local_a = (float*)malloc(local_m*n*sizeof(float));
		*local_x = (float*)malloc(n*sizeof(float));
		memset(*local_a, 0, local_m*n*sizeof(float));
		memset(*local_x, 0, n*sizeof(float));
	}
	MPI_Scatter(
		*local_a,
		local_m*n,
		MPI_FLOAT,
		*local_a,
		local_m*n,
		MPI_FLOAT,
		0,
		MPI_COMM_WORLD
	);
	MPI_Scatter(
		*local_x,
		local_n,
		MPI_FLOAT,
		*local_x,
		local_n,
		MPI_FLOAT,
		0,
		MPI_COMM_WORLD
	);
}

void Parallel_matrix_vector_prod(float* local_a, float* local_x, float* global_x, float* local_y, int m, int n, int local_m, int local_n)
{
	int i, j;
	MPI_Allgather(local_x, local_n, MPI_FLOAT, global_x, local_n, MPI_FLOAT, MPI_COMM_WORLD);
	for(i = 0 ; i < local_m ; i++)
	{
		local_y[i] = 0.0;
		for(j = 0; j < n; j++)
			local_y[i] = local_y[i] + local_a[i*m + j]*global_x[j];
	}
	// gather solution
	MPI_Gather(local_y, local_m, MPI_FLOAT, local_y, local_m, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void Print_matrix(char* title, float* mat, int m, int n)
{
	int i,j;
	printf("%s\n", title);
	for(i = 0 ; i < m ; i++)
	{
		for(j = 0 ; j < n ; j++)
		{
			printf("%4.1f ", mat[i*n + j]);
		}
		printf("\n");
	}
}
void Print_vector(char* title, float* vec, int n)
{
	int i;
	printf("%s\n", title);
	for(i = 0 ; i < n ; i++)
		printf("%4.1f ", vec[i]);
	printf("\n");
}
