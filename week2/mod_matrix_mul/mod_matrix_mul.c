#include <mpi.h>
#include "mod_matvec.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv)
{
	int my_rank, p;
	float* local_a;
	float* global_x;
	float* local_x;
	float* local_y;
	int m , n, local_m, local_n;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int stat = 1;
	if(my_rank == 0)
	{
		char *input_path = (char*)"input";
		stat = Read_input(input_path, &m, &n, &local_a, &local_x);
		Print_matrix((char*)"Read matrix:", local_a, m, n);
		Print_vector((char*)"Read vector:", local_x, n);
	}
	MPI_Bcast(&stat, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(stat == 0)
	{
		free(local_a);
		free(local_x);
		free(local_y);
		free(global_x);
		MPI_Finalize();
		return -1;
	}
	//
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	local_m = m/p;
	local_n = n/p;
	DistributeData(&local_a, &local_x, m, n, local_m, local_n, my_rank, p);
	global_x = (float*)malloc(n*sizeof(float));
	local_y = (float*)malloc(n*sizeof(float));
	memset(global_x, 0, n*sizeof(float));
	memset(local_y, 0, n*sizeof(float));
	Parallel_matrix_vector_prod(local_a, local_x, global_x, local_y, m, n, local_m, local_n);
	if(my_rank == 0)
		Print_vector((char*)"Calculated vector:", local_y, n);
	// 

	free(local_a);
	free(local_x);
	free(local_y);
	free(global_x);
	MPI_Finalize();
	return 0;
}
