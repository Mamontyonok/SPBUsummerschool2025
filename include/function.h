#pragma once
#include <iostream>
#include <random>
#include <cstring>
#include <vector>
#include "mpi.h"

void mul(const double* A, const double* B, double* C, size_t sz);
void blockcachemul(const double* A, const double* B, double* C, size_t sz);
double error(const double* A, const double* B, size_t sz);
void matrixchainmul(const double* A, const double* B, double* C, size_t sz);
void generatematrix(double* matrix, size_t sz);
