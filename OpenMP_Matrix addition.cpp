/*
*	Mohamad Chamanmotlagh
*	06/06/2021
*/
#define N 100000000

#include <iostream>
#include <omp.h>
#include <cmath>

using namespace std;


void fillArray(int *A);

void add(int *A, int *B, int *res);

void printArray(int *A);

int main(int argc, char *argv[]) {
    int *A = (int *) malloc(N * sizeof(int));
    int *B = (int *) malloc(N * sizeof(int));
    int *C = (int *) malloc(N * sizeof(int));

    srand(0);
    fillArray(A);
    fillArray(B);

    //printArray(A);
    //printArray(B);


    volatile double start_time = omp_get_wtime();
    add(A, B, C);
    //printArray(C);
    volatile double end_time = omp_get_wtime();
    cout << "\nElapsed time: " << ceil((end_time - start_time) * 1000) << " ms\n";

    free(A);
    free(B);
    free(C);
	return EXIT_SUCCESS;
}

void fillArray(int *A) {
    for (int i = 0; i < N; ++i) {
            A[i] = rand() % 5;
    }
}

void add(int *A, int *B, int *res) {
#pragma omp parallel for
    for (int i = 0; i < N; i++)
            res[i] = A[i] + B[i];
}

void printArray(int *A) {
    cout << "[";
    for (int i = 0; i < N; ++i) {
        if (i != 0)
            cout << " ";
            cout << A[i];
                cout << " ";
    }
    cout << "]\n";
}