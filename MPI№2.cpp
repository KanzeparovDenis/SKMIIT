#include "mpi.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath> 
using namespace std;


#define EXACT 0.002747252747


double f(double x, double y, double z) 
	return x * y * y * z * z * z;
}


int main(int argc, char *argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	double eps = atof(argv[1]); 
	int n = 2;
	int stop = 0;
	if (rank == 0)
	{
		double start = MPI_Wtime(); 
		double ctime = 0;
		double res;
		while (stop != 1) 
		{
			
			int seed = 2; // фиксация зерна генератора
			//int seed = 20;
			//int seed = 200;
			//int seed = 2000;
			srand(seed);
			n *= 2; 
			int s = int(n / (size - 1)); 
			int d = n - s * (size - 1);
			MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); 
			MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
			double total = 0; 
			double tmtotal = 0; 
			double* Points = new double[n * 3]; 
			double startcr = MPI_Wtime(); 
			for (int i = 0; i < 3 * n; i += 3)
			{
				double x = (double) rand() / RAND_MAX;
				double y = (double) rand() / RAND_MAX;
				double z = (double) rand() / RAND_MAX;
				Points[i] = x;
				Points[i + 1] = y;
				Points[i + 2] = z;
			}
			double stopcr = MPI_Wtime();
			for (int i = 1; i < size; ++i) 
			{
				int left = 3 * s * (i - 1);
				if (i == (size - 1))
				{
					MPI_Send(&Points[left], 3 * (s + d), MPI_DOUBLE, i, 20, MPI_COMM_WORLD); 
				}
				else
				{
					MPI_Send(&Points[left], 3 * s, MPI_DOUBLE, i, 10, MPI_COMM_WORLD);
				}
			}
			for (int i = 1; i < size; total += tmtotal, ++i)
			{
				MPI_Recv(&tmtotal, 1, MPI_DOUBLE, i, 30, MPI_COMM_WORLD, &status); 
			}
			total = total / n; 
			stop = fabs(total - EXACT) < eps;
			MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
			res = total; 
			ctime += stopcr - startcr; 
			delete[] Points;
		}
		double dif = fabs(res - EXACT);
		double end = MPI_Wtime();
		printf("Difference = %3.10f, Result = %3.10f, Number of Points = %d, Time = %3.10f, Creation Time = %3.10f \n", dif, res, n, end - start, ctime);
	}
	else
	{

		while (stop != 1)
		{
			int s;
			int d;
			MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); 
			MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (rank == (size - 1))
			{
				double* po = new double[(s + d) * 3]; 
				MPI_Recv(po, 3 * (s + d), MPI_DOUBLE, 0, 20, MPI_COMM_WORLD, &status); 
				double tempresult = 0;
				for (int i = 0; i < 3 * (s + d); i += 3)
				{
					if ((po[i] > po[i + 1]) && (po[i + 2] < po[i + 1] * po[i]) && (1 > po[i])) 
					{
						tempresult += f(po[i], po[i + 1], po[i + 2]); 
					}
				}
				MPI_Send(&tempresult, 1, MPI_DOUBLE, 0, 30, MPI_COMM_WORLD); 
				delete[] po;
			}
			else 
			{
				double* po = new double[s * 3];
				MPI_Recv(po, 3 * s, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &status);
				double tempresult = 0;
				for (int i = 0; i < 3 * s; i += 3)
				{
					if ((po[i] > po[i + 1]) && (po[i + 2] < po[i + 1] * po[i]) && (1 > po[i]))
					{
						tempresult += f(po[i], po[i + 1], po[i + 2]);
					}
				}
				MPI_Send(&tempresult, 1, MPI_DOUBLE, 0, 30, MPI_COMM_WORLD);
				delete[] po;
			}
			MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}
	MPI_Finalize();
}