#include "function.h"
#include <gtest.h>

const size_t MSIZE = 1024;

class MatrixTest :public::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        A = new double[MSIZE * MSIZE];
        B = new double[MSIZE * MSIZE];
        C1 = new double[MSIZE * MSIZE];
        C2 = new double[MSIZE * MSIZE];
        if (myid == 0) {
            generatematrix(A, MSIZE);
            generatematrix(B, MSIZE);
            mul(A, B, C1, MSIZE);
        }
        MPI_Bcast(A, MSIZE * MSIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B, MSIZE * MSIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(C1, MSIZE * MSIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    void TearDown() override
    {
        delete[] A;
        delete[] B;
        delete[] C1;
        delete[] C2;
    }
    double* A = nullptr;
    double* B = nullptr;
    double* C1 = nullptr;
    double* C2 = nullptr;
    size_t sz = MSIZE;
    int myid;
    int numprocs;
};

TEST_F(MatrixTest, CacheFriendlyMultiplication)
{
    if (myid == 0) {
        mul(A, B, C2, sz);
        double err = error(C1, C2, sz);
        EXPECT_LT(err, 1.e-6) << "Cache friendly multiplication error: " << error;
    }
}

TEST_F(MatrixTest, MatrixChainMultiplication) {
    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    matrixchainmul(A, B, C2, sz);
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0) {
        double err = error(C1, C2, sz);
        EXPECT_LT(err, 1.e-6) << "Chain MPI multiplication error: " << error;
    }
}