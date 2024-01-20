#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
	int my_rank;
	int p;
	MPI_Status status;
	// variables for algorithm
	int broadcast_integer = -1;
	int spacing;
	int stage;
	int source;
	int dest;
	int tag = 0;
	// variables for algorithm
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	// broadcast algorithm starts
	if(my_rank == 0) broadcast_integer = 100;
	spacing = p; stage = 0;
	while(spacing > 1)
	{
		if(my_rank % spacing == 0)
		{
			dest = my_rank + spacing/2;
			printf("%d sends to %d, %d \n", my_rank, dest, stage);
			MPI_Send(&broadcast_integer, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
		}
		else if(my_rank % (spacing/2) == 0)
		{
			source = my_rank - spacing/2;
			printf("%d receives from %d, %d \n", my_rank, source, stage);
			MPI_Recv(&broadcast_integer, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
		}
		spacing = spacing/2;
		stage = stage + 1;
	}
	MPI_Finalize();
	// broadcast algorithm ends
	return 0;
}
