#include <stdio.h>
#include <mpi.h>

#define MSG_LENGTH 8192 

int main(int argc, char** argv)
{
	int my_rank;
	int p;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if(p < 3)
	{
		printf("There should be at least 3 processes!!!");
		MPI_Finalize();
		return 0;
	}
	
	int msg[MSG_LENGTH] = {111 * (my_rank + 1)};
	int r_msg_1[MSG_LENGTH], r_msg_2[MSG_LENGTH];

	int source = my_rank;
	int dest_1 = (my_rank + 1) % p; // to keep it withing 0 to p-1
	int dest_2 = (my_rank + p - 1) % p; // incrementing with p then -1 to keep it above zero
	
	MPI_Request req[4];
	MPI_Status stat[4];
	
	// it's working for small msg lengths
	MPI_Isend(msg, MSG_LENGTH, MPI_INT, dest_1, my_rank, MPI_COMM_WORLD, &req[0]);
	MPI_Isend(msg, MSG_LENGTH, MPI_INT, dest_2, my_rank, MPI_COMM_WORLD, &req[1]);
	MPI_Irecv(r_msg_1, MSG_LENGTH, MPI_INT, dest_1, dest_1, MPI_COMM_WORLD, &req[2]);
	MPI_Irecv(r_msg_2, MSG_LENGTH, MPI_INT, dest_2, dest_2, MPI_COMM_WORLD, &req[3]);

	MPI_Waitall(4, req, stat);

	printf("p%d communicated with %d & %d\n", my_rank, dest_1, dest_2);

	MPI_Finalize();
	return 0;
}
