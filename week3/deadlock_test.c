#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
	int my_rank;
	int p;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if(p != 2)
	{
		// Can only run on 2 process
		printf("Run on 2 process!!!");
		MPI_Finalize();
		return 0;
	}

	if(argc < 1)
	{
		// provide 
		printf("Provide max size kb (will be multiplied with 1024)!!");
		MPI_Finalize();
		return 0;
	}

	int max_size = atoi(argv[1]) * 1024; 
	int *send_msg, *recv_msg;
	// after each iteration message size is doubled
	// code stops printing msg size after reaching deadlock
	// on my system it stopped on 1024 msg size
	for(int i = 1 ; i < max_size ; i*=2)
	{
		long msg_size = (long)i;
		if(my_rank == 0)
		{
			printf("Sending msg of size %ld\n", msg_size);
			send_msg = (int *)malloc(msg_size * sizeof(int));
			recv_msg = (int *)malloc(msg_size * sizeof(int));
			MPI_Send(send_msg, msg_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
			MPI_Recv(recv_msg, msg_size, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else
		{
			// printf("Sending msg of size %ld\n", msg_size);
			send_msg = (int *)malloc(msg_size * sizeof(int));
			recv_msg = (int *)malloc(msg_size * sizeof(int));
			MPI_Send(send_msg, msg_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
			MPI_Recv(recv_msg, msg_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
	free(send_msg);
	free(recv_msg);

	MPI_Finalize();

	return 0;
}
