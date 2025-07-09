#include "function.h"

void mul(const double* A, const double* B, double* C, size_t sz) {
    memset(C, 0, sz * sz * sizeof(double));
    for (size_t i = 0; i < sz; ++i) {
        for (size_t k = 0; k < sz; ++k) {
            for (size_t j = 0; j < sz; ++j) {
                C[i * sz + j] += A[i * sz + k] * B[k * sz + j];
            }
        }
    }
}

void blockcachemul(const double* A, const double* B, double* C, size_t sz) {
    const size_t BLOCK_SIZE = 64; 
    memset(C, 0, sz * sz * sizeof(double));
    for (size_t i1 = 0; i1 < sz; i1 += BLOCK_SIZE) {
        for (size_t k1 = 0; k1 < sz; k1 += BLOCK_SIZE) {
            for (size_t j1 = 0; j1 < sz; j1 += BLOCK_SIZE) {
                size_t i_end = std::min(i1 + BLOCK_SIZE, sz);
                size_t k_end = std::min(k1 + BLOCK_SIZE, sz);
                size_t j_end = std::min(j1 + BLOCK_SIZE, sz);
                for (size_t i2 = i1; i2 < i_end; ++i2) {
                    const double* A_row = &A[i2 * sz];
                    double* C_row = &C[i2 * sz];
                    for (size_t k2 = k1; k2 < k_end; ++k2) {
                        double A_val = A_row[k2];
                        const double* B_row = &B[k2 * sz];
                        for (size_t j2 = j1; j2 < j_end; ++j2) {
                            C_row[j2] += A_val * B_row[j2];
                        }
                    }
                }
            }
        }
    }
}

void matrixchainmul(const double* A, const double* B, double* C, size_t sz) {
    int ProcNum, ProcRank;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (sz % ProcNum != 0) {
        if (ProcRank == 0) {
            std::cerr << "Matrix size must be divisible by the number of processes" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    const int dim = sz;
    const int ProcPartsize = dim / ProcNum;
    const int ProcPartElem = ProcPartsize * dim;
    double* bufA = new double[ProcPartElem];
    double* bufB = new double[ProcPartElem];
    double* bufC = new double[ProcPartElem]();  
    double* B_transposed = nullptr; // временная транспонированная матрциа
    if (ProcRank == 0) {
        B_transposed = new double[dim * dim];
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                B_transposed[j * dim + i] = B[i * dim + j];
            }
        }
    }
    MPI_Scatter(A, ProcPartElem, MPI_DOUBLE, bufA, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B_transposed, ProcPartElem, MPI_DOUBLE, bufB, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (ProcRank == 0) {
        delete[] B_transposed;
    }
    for (int i = 0; i < ProcPartsize; ++i) {
        for (int j = 0; j < ProcPartsize; ++j) {
            double temp = 0.0;
            for (int k = 0; k < dim; ++k) {
                temp += bufA[i * dim + k] * bufB[j * dim + k];
            }
            bufC[i * dim + (ProcRank * ProcPartsize + j)] += temp;
        }
    }
    for (int step = 1; step < ProcNum; ++step) {
        int next_rank = (ProcRank + 1) % ProcNum;
        int prev_rank = (ProcRank - 1 + ProcNum) % ProcNum;
        MPI_Sendrecv_replace(
            bufB, ProcPartElem, MPI_DOUBLE,
            next_rank, 0,
            prev_rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        int owner_rank = (ProcRank - step + ProcNum) % ProcNum;
        for (int i = 0; i < ProcPartsize; ++i) {
            for (int j = 0; j < ProcPartsize; ++j) {
                double temp = 0.0;
                for (int k = 0; k < dim; ++k) {
                    temp += bufA[i * dim + k] * bufB[j * dim + k];
                }
                bufC[i * dim + (owner_rank * ProcPartsize + j)] += temp;
            }
        }
    }
    MPI_Gather(bufC, ProcPartElem, MPI_DOUBLE, C, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete[] bufA;
    delete[] bufB;
    delete[] bufC;
}

void generatematrix(double* matrix, size_t sz) {
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<double> coef_gen(-524288, 524288);
    for (size_t i = 0; i < sz * sz; ++i) {
        matrix[i] = double(coef_gen(e));
    }
}

double error(const double* A, const double* B, size_t sz) {
    double er = -1;
    for (size_t i = 0; i < sz * sz; ++i)
        if (abs(A[i] - B[i]) > er)
            er = abs(A[i] - B[i]);
    return er;
}