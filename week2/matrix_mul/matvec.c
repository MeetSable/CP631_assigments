#include <mpi.h>
#include <stdio.h>
#include "matvec.h"

void Read_matrix(char* prompt, LOCAL_MATRIX_T local_A, int local_m, int n, int my_rank, int p)
{
	int i, j;
	LOCAL_MATRIX_T temp;
	for(i = 0 ; i < p*local_m ; i++)
		for(j = n; j < MAX_ORDER; j++)
			temp[i][j] = 0.0;
	if (my_rank == 0)
	{
		printf("%s\n", prompt);
		for(i = 0; i < p*local_m; i++)
			for(j = 0; j < n; j++)
				scanf("%f", &temp[i][j]);
	}
	MPI_Scatter(temp, local_m*MAX_ORDER, MPI_FLOAT, local_A, local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void Read_vector(char* prompt, float local_x[], int local_n, int my_rank, int p)
{
	int i;
	float temp[MAX_ORDER];
	if (my_rank == 0)
	{
		printf("%s\n", prompt);
		for(i = 0 ; i < p*local_n ; i++)
			scanf("%f", &temp[i]);
	}
	MPI_Scatter(temp, local_n, MPI_FLOAT, local_x, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void Parallel_matrix_vector_prod(LOCAL_MATRIX_T local_A, int m, int n, float local_x[], float global_x[], float local_y[], int local_m, int local_n)
{
	int i, j;
	MPI_Allgather(local_x, local_n, MPI_FLOAT, global_x, local_n, MPI_FLOAT, MPI_COMM_WORLD);
	for(i = 0 ; i < local_m ; i++)
	{
		local_y[i] = 0.0;
		for(j = 0 ; j < n ; j++)
			local_y[i] = local_y[i] + local_A[i][j]*global_x[j];
	}
}

void Print_vector(char* title, float local_y[], int local_m, int my_rank, int p)
{
	int i;
	float temp[MAX_ORDER];
	MPI_Gather(local_y, local_m, MPI_FLOAT, temp, local_m, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if(my_rank == 0)
	{
		printf("%s\n", title);
		for(i = 0 ; i < p*local_m; i++)
			printf("%4.1f ", temp[i]);
		printf("\n");
	}
}

void Print_matrix(char* title, LOCAL_MATRIX_T local_A, int local_m, int n, int my_rank, int p)
{
	int i, j;
	float temp[MAX_ORDER][MAX_ORDER];
	MPI_Gather(local_A, local_m*MAX_ORDER, MPI_FLOAT, temp, local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if(my_rank == 0)
	{
		printf("%s\n", title);
		for(i = 0 ; i < p*local_m; i++)
		{
			for(j = 0; j< n; j++)
				printf("%4.1f ", temp[i][j]);
			printf("\n");
		}
	}
}
