#pragma once

int Read_input(char* input_path, int* m, int* n, float** local_a, float** local_x);
void DistributeData(float** local_a, float** local_x, int m, int n, int local_m, int local_n, int my_rank, int p);
void Parallel_matrix_vector_prod(float* local_a, float* local_x, float* global_x, float* local_y, int m, int n, int local_m, int local_n);
void Print_matrix(char* title, float* mat, int m, int n);
void Print_vector(char* title, float* vec, int n);
