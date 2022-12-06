#include <iostream>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib> 
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <omp.h>

int M, N;
int n_procs, rank, n_min, n_max, m_min, m_max, len_x_cell, len_y_cell, num_x_cell, num_y_cell, i_cell, j_cell; 

double uf(int i, int j, double h1, double h2) {
	double x = i * h1;
	double y = j * h2;
	double u = sqrt(4. + x * y);
	return u;
}

double Fun(int i, int j, double h1, double h2) {
	double x = i * h1;
	double y = j * h2;
	double u = sqrt(4. + x * y);
	double result = (0.25 * (x * x + y * y) / (u * u * u)) + (u * (x + y)); 
	return result;
}

double psi_rl(int i, int j, double h1, double h2) {
	double x = i * h1;
	double y = j * h2;
	double u = uf(i, j, h1, h2);
	double t = y / (2. * u);
	if(i == 0) {
		return u - t;
	} else if(i == M) {
		return u + t;
	}
	return u + t;
}

double psi_tb(int i, int j, double h1, double h2) {
	double x = i * h1;
	double y = j * h2;
	double u = uf(i, j, h1, h2);
	double t = x / (2. * u);
	if(j == 0) {
		return u - t;
	} else if(j == N) {
		return u + t;
	}
	return u + t;
}

double qf(int i, int j, double h1, double h2) {
	double x = i * h1;
	double y = j * h2;
	return x + y;
}

double wx(double ** w, int i, int j, double h1) {
	return (1. / (h1)) * ((w[i + 1][j] - w[i][j]) / h1 - ((w[i][j] - w[i-1][j]) / h1));
}

double wy(double ** w, int i, int j, double h2) {
	return (1. / (h2)) * ((w[i][j + 1] - w[i][j]) / h2 - ((w[i][j] - w[i][j - 1]) / h2));
}

double **init_matrix(int w, int h) {
    double **result = new double *[w + 1];
    for (int i = 0; i <= w; i++) {
        result[i] = new double[h + 1];

        for(int j = 0; j <= h; ++j) {
        	result[i][j] = 0;
        }
        // memset(result[i], 0, h * sizeof(double));
    }
    return result;
}

void free_matrix(double **matrix, int M, int N) {
	for (int i = 0; i <= M; ++i)
	    delete [] matrix[i];
	delete [] matrix;
}

double** Am(double ** w, double **ar, int M, int N, double h1, double h2) {
	for(int i = 1; i <= M - 1; ++i) {
		double x = i * h1;
		for(int j = 1; j <= N - 1; ++j) {
			double y = j * h2;
			ar[i][j] =  - wx(w, i, j, h1) - wy(w, i, j, h2) + qf(i, j, h1, h2) * w[i][j];
		}
	} // internal points


	for(int j = 1; j <= N-1; ++j) {
		ar[M][j] = (2. / h1 / h1) * (w[M][j] - w[M-1][j]) + 
		(qf(M, j, h1, h2) + (2. / h1)) * w[M][j] - 
		wy(w, M, j, h2);

		ar[0][j] = (2. / h1 / h1) * (w[0][j] - w[1][j]) + 
		(qf(0, j, h1, h2) + (2. / h1)) * w[0][j] - 
		wy(w, 0, j, h2);
	}
	for(int i = 1; i <= M-1; ++i) {
		ar[i][N] = (2. / h2 / h2) * (w[i][N] - w[i][N-1]) + 
		(qf(i, N, h1, h2) + (2. / h2)) * w[i][N] - 
		wx(w, i, N, h1);

		ar[i][0] = (2. / h2 / h2) * (w[i][0] - w[i][1]) + 
		(qf(i, 0, h1, h2) + (2. / h2)) * w[i][0] - wx(w, i, 0, h1);
	} // border points

	ar[0][0] = -(2. / h1 / h1) * (w[1][0] - w[0][0]) -
	(2. / h2 / h2) * (w[0][1] - w[0][0]) + 
	(qf(0, 0, h1, h2) + (2./ h1) + (2./h2)) * w[0][0];

	ar[M][0] = (2. / h1 / h1) * (w[M][0] - w[M-1][0]) -
	(2. / h2 / h2) * (w[M][1] - w[M][0]) + 
	(qf(M, 0, h1, h2) + (2./ h1) + (2./h2)) * w[M][0];

	ar[M][N] = (2. / h1 / h1) * (w[M][N] - w[M-1][N]) +
	(2. / h2 / h2) * (w[M][N] - w[M][N-1]) + 
	(qf(M, N, h1, h2) + (2./ h1) + (2./h2)) * w[M][N];

	ar[0][N] = -(2. / h1 / h1) * (w[1][N] - w[0][N]) +
	(2. / h2 / h2) * (w[0][N] - w[0][N-1]) + 
	(qf(0, N, h1, h2) + (2./ h1) + (2./h2)) * w[0][N]; //corner points


	return ar;
}

void Am_mpi(double ** w, double **ar, 
	int M, int N, double h1, double h2,
	int m_min, int m_max, int n_min, int n_max) {

	//center
	#pragma omp parallel for
	for(int i = std::max(1, m_min); i <= std::min(m_max, M - 1); ++i) {
		double x = i * h1;
		#pragma omp parallel for
		for(int j = std::max(1, n_min); j <= std::min(n_max, N - 1); ++j) {
			double y = j * h2;
			ar[i][j] =  - wx(w, i, j, h1) - wy(w, i, j, h2) + qf(i, j, h1, h2) * w[i][j];
		}
	} // internal points
	//left border
	if(m_min == 0) {
		#pragma omp parallel for
		for(int j = std::max(1, n_min); j <= std::min(n_max, N - 1); ++j) {
			ar[0][j] = (2. / h1 / h1) * (w[0][j] - w[1][j]) + 
			(qf(0, j, h1, h2) + (2. / h1)) * w[0][j] - 
			wy(w, 0, j, h2);
		}
	}

	//right border
	if(m_max == M) {
		#pragma omp parallel for
		for(int j = std::max(1, n_min); j <= std::min(n_max, N - 1); ++j) {
			ar[M][j] = (2. / h1 / h1) * (w[M][j] - w[M-1][j]) + 
			(qf(M, j, h1, h2) + (2. / h1)) * w[M][j] - 
			wy(w, M, j, h2);
		}
	}

	//bottom border
	if(n_min == 0) {
		#pragma omp parallel for
		for(int i = std::max(1, m_min); i <= std::min(m_max, M - 1); ++i) {
			ar[i][0] = (2. / h2 / h2) * (w[i][0] - w[i][1]) + 
			(qf(i, 0, h1, h2) + (2. / h2)) * w[i][0] - wx(w, i, 0, h1);
		} // border points
	}

	//bottom border
	if(n_max == N) {
		#pragma omp parallel for
		for(int i = std::max(1, m_min); i <= std::min(m_max, M - 1); ++i) {
			ar[i][N] = (2. / h2 / h2) * (w[i][N] - w[i][N-1]) + 
			(qf(i, N, h1, h2) + (2. / h2)) * w[i][N] - wx(w, i, N, h1);
		} // border points
	}

	//corners
	if(m_min == 0 && n_min == 0) {
		ar[0][0] = -(2. / h1 / h1) * (w[1][0] - w[0][0]) -
		(2. / h2 / h2) * (w[0][1] - w[0][0]) + 
		(qf(0, 0, h1, h2) + (2./ h1) + (2./h2)) * w[0][0];
	}

	if(m_max == M && n_min == 0) {
		ar[M][0] = (2. / h1 / h1) * (w[M][0] - w[M-1][0]) -
		(2. / h2 / h2) * (w[M][1] - w[M][0]) + 
		(qf(M, 0, h1, h2) + (2./ h1) + (2./h2)) * w[M][0];
	}

	if(m_max == M and n_max == N) {
		ar[M][N] = (2. / h1 / h1) * (w[M][N] - w[M-1][N]) +
		(2. / h2 / h2) * (w[M][N] - w[M][N-1]) + 
		(qf(M, N, h1, h2) + (2./ h1) + (2./h2)) * w[M][N];	
	}

	if(m_min == 0 and n_max == N) {
		ar[0][N] = -(2. / h1 / h1) * (w[1][N] - w[0][N]) +
		(2. / h2 / h2) * (w[0][N] - w[0][N-1]) + 
		(qf(0, N, h1, h2) + (2./ h1) + (2./h2)) * w[0][N];
	}

}

void init_B_matrix(double **B, int M, int N, double h1, double h2) {

	for(int i = 1; i <= M-1; ++i) {
		for(int j = 1; j <= N-1; ++j) {
			B[i][j] = Fun(i, j, h1, h2);
		}
	}
	for(int j = 1; j <= N-1; ++j) {
		B[M][j] = Fun(M, j, h1, h2) + 2. * psi_rl(M, j, h1, h2) / h1;
		B[0][j] = Fun(0, j, h1, h2) + 2. * psi_rl(0, j, h1, h2) / h1;
	} // top bottom
	for(int i = 1; i <= M-1; ++i) {
		B[i][N] = Fun(i, N, h1, h2) + 2. * psi_tb(i, N, h1, h2) / h2;
		B[i][0] = Fun(i, 0, h1, h2) + 2. * psi_tb(i, 0, h1, h2) / h2;
	}// right left
	B[0][0] = Fun(0, 0, h1, h2) + ((1. / h1) + (1. / h2)) * (psi_rl(0, 1, h1, h2) + psi_tb(1, 0, h1, h2));
	B[M][0] = Fun(M, 0, h1, h2) + ((1. / h1) + (1. / h2)) * (psi_rl(M, 1, h1, h2) + psi_tb(M-1, 0, h1, h2));
	B[M][N] = Fun(M, N, h1, h2) + ((1. / h1) + (1. / h2)) * (psi_rl(M, N-1, h1, h2) + psi_tb(M-1, N, h1, h2));
	B[0][N] = Fun(0, N, h1, h2) + ((1. / h1) + (1. / h2)) * (psi_rl(0, N-1, h1, h2) + psi_tb(1, N, h1, h2));
}

void init_mpi_cells(int n_procs, int rank, int M, int N, 
		int &n_min, int &n_max, int &m_min, int &m_max,
		int &len_x_cell, int &len_y_cell,
		int &num_x_cell, int &num_y_cell,
		int &i_cell, int &j_cell) {

	for (num_x_cell = sqrt(n_procs); n_procs % num_x_cell != 0; --num_x_cell) {} // init num_x_cells
	num_y_cell = n_procs / num_x_cell;


	i_cell = rank / num_y_cell; // 0..num_x_cell - 1
	j_cell =  rank % num_y_cell; // 0..num_y_cell - 1

	if(i_cell < num_x_cell - 1) {
		len_x_cell = (M + 1) / num_x_cell;
	} else {
		len_x_cell = (M + 1) / num_x_cell + (M + 1) % num_x_cell;
	}	

	if(j_cell < num_y_cell - 1) {
		len_y_cell = (N + 1) / num_y_cell;
	} else {
		len_y_cell = (N + 1) / num_y_cell + (N + 1) % num_y_cell;
	}

	m_min = i_cell * (M + 1) / num_x_cell;
	n_min = j_cell * (N + 1) / num_y_cell;

	m_max = m_min + len_x_cell - 1;
	n_max = n_min + len_y_cell - 1;
}

double scalar_mult(double ** u, double ** v,
 int M, int N, double h1, double h2) {
	double result = 0.;
	double sum = 0.;
	for(int i = 0; i <= M; ++i) {
		sum = 0.;
		for(int j = 0; j <= N; ++j) {
			double rho = 1.;
			if(i == 0 || i == M) {
				rho *= 0.5;
			}
			if(j == 0 || j == N) {
				rho *= 0.5;
			}
			sum += rho * u[i][j] * v[i][j];
		}
		result += sum;
	}
	return result * h1 * h2;
}

double scalar_mult_mpi(double ** u, double ** v,
 int M, int N, double h1, double h2,
 int m_min, int m_max, int n_min, int n_max) {
	double result = 0.;
	double sum = 0.;
	#pragma omp parallel for reduction (+:result)
	for(int i = m_min; i <= m_max; ++i) {
		sum = 0.;

		for(int j = n_min; j <= n_max; ++j) {
			double rho = 1.;
			if(i == 0 || i == M) {
				rho *= 0.5;
			}
			if(j == 0 || j == N) {
				rho *= 0.5;
			}
			sum += rho * h1 * h2 * u[i][j] * v[i][j];
		}
		result += sum;
	}
	return result;
}

void send_recv_borders(double **w,
	double *bottom_buf, double *top_buf, double *left_buf, double *right_buf,
	double *bottom_buf_r, double *top_buf_r, double *left_buf_r, double *right_buf_r,
	int m_min, int m_max, int n_min, int n_max,
	MPI_Request *send_status,
	MPI_Status *recv_status) {
 
	// sending bottom border
    if (m_max != M) {
        for (int i = 0; i < len_y_cell; ++i) {
            bottom_buf[i] = w[m_max][n_min + i];
        }
        // std::cout << "bottom rank: " << rank << " " << (i_cell + 1) * num_y_cell + j_cell << " " << m_max << " " << M << std::endl;
        MPI_Isend(bottom_buf, len_y_cell, MPI_DOUBLE, (i_cell + 1) * num_y_cell + j_cell , 123, MPI_COMM_WORLD, &send_status[0]);
    }

    // sending top border
    if (m_min != 0) {
        for (int i = 0; i < len_y_cell; ++i) {
            top_buf[i] = w[m_min][n_min + i];
        }
        // std::cout << "top rank: " << rank << " " << (i_cell - 1) * num_y_cell + j_cell << std::endl;
        MPI_Isend(top_buf, len_y_cell, MPI_DOUBLE, (i_cell - 1) * num_y_cell + j_cell, 123, MPI_COMM_WORLD, &send_status[1]);
    }

    // sending left border
    if (n_min != 0) {
        for (int i = 0; i < len_x_cell; ++i) {
            left_buf[i] = w[m_min + i][n_min];
        }
        // std::cout << "left rank: " << rank << " " << (i_cell) * num_y_cell + j_cell - 1 << std::endl;
        MPI_Isend(left_buf, len_x_cell, MPI_DOUBLE, i_cell * num_y_cell + j_cell - 1, 123, MPI_COMM_WORLD, &send_status[2]);
    }

    // sending right border
    if (n_max != N) {
        for (int i = 0; i < len_x_cell; ++i) {
            right_buf[i] = w[m_min + i][n_max];
        }
        // std::cout << "right rank: " << rank << " " << (i_cell) * num_y_cell + j_cell + 1 << std::endl;
        MPI_Isend(right_buf, len_x_cell, MPI_DOUBLE, i_cell * num_y_cell + j_cell + 1, 123, MPI_COMM_WORLD, &send_status[3]);
    }

    //receive bottom border
    if (m_max != M) {
    	MPI_Recv(bottom_buf_r, len_y_cell, MPI_DOUBLE, (i_cell + 1) * num_y_cell + j_cell , 123, MPI_COMM_WORLD, &recv_status[0]);
        for (int i = 0; i < len_y_cell; ++i) {
            w[m_max + 1][n_min + i] = bottom_buf_r[i];
        }
    }

    //receive top border
    if (m_min != 0) {
    	MPI_Recv(top_buf_r, len_y_cell, MPI_DOUBLE, (i_cell - 1) * num_y_cell + j_cell , 123, MPI_COMM_WORLD, &recv_status[1]);
        for (int i = 0; i < len_y_cell; ++i) {
            w[m_min - 1][n_min + i] = top_buf_r[i];
        }
    }

    //receive left border
    if (n_min != 0) {
    	MPI_Recv(left_buf_r, len_x_cell, MPI_DOUBLE, i_cell * num_y_cell + j_cell - 1 , 123, MPI_COMM_WORLD, &recv_status[2]);
        for (int i = 0; i < len_x_cell; ++i) {
            w[m_min + i][n_min - 1] = left_buf_r[i];
        }
    }


    //receive right border
    if (n_max != N) {
    	MPI_Recv(right_buf_r, len_x_cell, MPI_DOUBLE, i_cell * num_y_cell + j_cell + 1 , 123, MPI_COMM_WORLD, &recv_status[3]);
        for (int i = 0; i < len_x_cell; ++i) {
            w[m_min + i][n_max + 1] = right_buf_r[i];
        }
    }

} 

int main(int argc, char **argv) {
	double eps = 1e-6;
	M = 500;
	N = 500;
	M = atoi(argv[1]);
	N = atoi(argv[2]);
	// int M(200), N(200);
	double h1 = 4. / M, h2 = 3. / N;

	double **r = init_matrix(M, N);
	double **Am_r = init_matrix(M, N);
	double **B = init_matrix(M, N);
	double **w_k = init_matrix(M, N);
	double **w_diff = init_matrix(M, N);

	init_B_matrix(B, M, N, h1, h2);

	MPI_Init(&argc, &argv);
	double start_time = MPI_Wtime();
	MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	init_mpi_cells(n_procs, rank, M, N, 
		n_min, n_max, m_min, m_max, 
		len_x_cell, len_y_cell,
		num_x_cell, num_y_cell,
		i_cell, j_cell);

	double *bottom_buf = new double[len_y_cell];
	double *right_buf = new double[len_x_cell]; 
	double *top_buf = new double[len_y_cell];
	double *left_buf = new double[len_x_cell];  

	double *bottom_buf_r = new double[len_y_cell];
	double *right_buf_r = new double[len_x_cell]; 
	double *top_buf_r = new double[len_y_cell];
	double *left_buf_r = new double[len_x_cell];   

	MPI_Request send_status[4];
    MPI_Status recv_status[4];   


	double tau = 0;
	double Am_r_r_scalar_cell = 0;
	double Am_r_r = 0;
	double Am_r_Am_r_scalar_cell = 0;
	double Am_r_Am_r = 0;
	double norm = 1;
	int n_iter = 0;
	do {

		Am_mpi(w_k, r, M, N, h1, h2,
			m_min, m_max, n_min, n_max);

		for(int i = m_min; i <= m_max; ++i) {
			for(int j = n_min; j <= n_max; ++j) {
				r[i][j] -= B[i][j];
			}	
		} // r = Aw(k) - B
		Am_mpi(r, Am_r, M, N, h1, h2,
			m_min, m_max, n_min, n_max);
		Am_r_r_scalar_cell = scalar_mult_mpi(Am_r, r, M, N, h1, h2,
			m_min, m_max, n_min, n_max);

		MPI_Allreduce(&Am_r_r_scalar_cell, &Am_r_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		Am_r_Am_r_scalar_cell = scalar_mult_mpi(Am_r, Am_r, M, N, h1, h2,
			m_min, m_max, n_min, n_max);

		MPI_Allreduce(&Am_r_Am_r_scalar_cell, &Am_r_Am_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		tau = Am_r_r / Am_r_Am_r;

		for(int i = m_min; i <= m_max; ++i) {
			for(int j = n_min; j <= n_max; ++j) {
				w_diff[i][j] = w_k[i][j] - (w_k[i][j] - tau * r[i][j]);
			} 
		} // w_diff

		for(int i = m_min; i <= m_max; ++i) {
			for(int j = n_min; j <= n_max; ++j) {
				w_k[i][j] -=  tau * r[i][j];
			} 
		} // w_{k+1} = ...

		send_recv_borders(w_k, 
			bottom_buf, top_buf, left_buf, right_buf,
			bottom_buf_r, top_buf_r, left_buf_r, right_buf_r,
			m_min, m_max, n_min, n_max,
			send_status, recv_status);

		double scalar_w_diff = 0;

		scalar_w_diff = scalar_mult_mpi(w_diff, w_diff, M, N, h1, h2,
			m_min, m_max, n_min, n_max);

		MPI_Allreduce(&scalar_w_diff, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		++n_iter;
		norm = sqrt(norm);

	} while(norm >= eps);

	double max_error = 0.;
	for(int i = m_min; i <= m_max; ++i) {
		for(int j = n_min; j <= n_max; ++j) {
			double t = w_k[i][j] - uf(i, j, h1, h2);
			if(t < 0) {
				t = -t;
			}
			if(t > max_error) {
				max_error = t;
			}
		}
	}

	double end_time = MPI_Wtime();
	double time = end_time - start_time;
	double max_time;
	MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	double max_max_error;
	MPI_Allreduce(&max_error, &max_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if(rank == 0) {
		// std::cout << "mpi + openmp version" << std::endl;
		std::cout << "M: " << M << " N: " << N << std::endl;
		std::cout << "n_procs: " << n_procs << std::endl;
		std::cout << "Time: " << max_time << std::endl;
		std::cout << "Max error: " << max_error << std::endl;
	}

	MPI_Finalize();


	return 0;
}