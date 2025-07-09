#include "function.h"
#include <chrono>

int main(int *argc, char** argv) {
    MPI_Init(argc, &argv);
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    const size_t sz = 1024;     
    const size_t trials = 1;    
    double total_time = 0.0;
    double* A = nullptr;
    double* B = nullptr;
    double* C_base = nullptr;
    double* C_cache = nullptr;
    double* C_chain = nullptr;
    double* C_cannon = nullptr;
    if (ProcRank == 0) {
        std::cout << "Matrix size: " << sz << "x" << sz << std::endl;
        std::cout << "MPI processes: " << ProcNum << std::endl;
        std::cout << "Trials per method: " << trials << std::endl;
        A = new double[sz * sz];
        B = new double[sz * sz];
        C_base = new double[sz * sz]();
        C_cache = new double[sz * sz]();
        C_chain = new double[sz * sz]();
        C_cannon = new double[sz * sz]();
        generatematrix(A, sz);
        generatematrix(B, sz);
    }
    if (ProcRank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t t = 0; t < trials; ++t) {
            mul(A, B, C_base, sz);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "\nBase sequential: " << diff.count() / trials << " sec" << std::endl;
    }
    if (ProcRank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t t = 0; t < trials; ++t) {
            blockcachemul(A, B, C_cache, sz);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Cache-friendly: " << diff.count() / trials << " sec" << std::endl;
        std::cout << "Cache vs Base error: ";
        std::cout << error(C_base, C_cache, sz) << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_chain = std::chrono::high_resolution_clock::now();
    for (size_t t = 0; t < trials; ++t) {
        matrixchainmul(A, B, C_chain, sz);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_chain = std::chrono::high_resolution_clock::now();
    if (ProcRank == 0) {
        std::chrono::duration<double> diff_chain = end_chain - start_chain;
        std::cout << "\nMatrix Chain (MPI): " << diff_chain.count() / trials << " sec" << std::endl;
        std::cout << "Chain vs Base error: ";
        std::cout << error(C_base, C_chain, sz) << std::endl;
    }
    if (ProcRank == 0) {
        delete[] A;
        delete[] B;
        delete[] C_base;
        delete[] C_cache;
        delete[] C_chain;
        delete[] C_cannon;
    }
    MPI_Finalize();
    return 0;
}
