//
// Created by mohamad on ۲۰۲۱/۵/۲۷.
//
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace std;

#define threads_count 5

void allocate(int **);

void freeArr(int **);

void fillArray(int **);

void printArray(int **arr);

int determinantOfMatrix(int **mat);

int N = 128;
volatile int threads = 1;

int main() {
    int serialTimeSums = 0;
    int parallelTimeSums = 0;
    for (int i = 0; i < 4; ++i) {
        switch (i) {
            case 0:
                N = 128;
                break;
            case 1:
                N = 256;
                break;
            case 2:
                N = 512;
                break;
            case 3:
                N = 1024;
                break;
        }

        int **mat = (int **) malloc(N * sizeof(int *));
        allocate(mat);
        fillArray(mat);

        threads = 1;
        volatile double start_time = omp_get_wtime();
        determinantOfMatrix(mat);
        volatile double end_time = omp_get_wtime();
        cout << "Elapsed time using " << threads << " thread on " << N << " :" << ceil((end_time - start_time) * 1000)
             << " ms\n";
        serialTimeSums += ceil((end_time - start_time) * 1000);

        threads = threads_count;
        start_time = omp_get_wtime();
        determinantOfMatrix(mat);
        end_time = omp_get_wtime();
        cout << "Elapsed time using " << threads << " threads on " << N << " :" << ceil((end_time - start_time) * 1000)
             << " ms\n";
        parallelTimeSums += ceil((end_time - start_time) * 1000);
        freeArr(mat);
    }
    cout << "\033[1;32m" << "\nAverage speedup using " << threads_count << " threads: "
         << ((double) serialTimeSums / (double) parallelTimeSums);
    return 0;
}

void allocate(int **arr) {
    if (arr == NULL) {
        fprintf(stderr, "Out of memory");
        exit(0);
    }
    for (int i = 0; i < N; i++) {
        arr[i] = (int *) malloc(N * sizeof(int));
        if (arr[i] == NULL) {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }
}
void freeArr(int **arr) {
    if (arr == NULL)
        return;
    for (int i = 0; i < N; i++) {
        if (arr[i] != NULL) {
            free(arr[i]);
        }
    }
}

void fillArray(int **arr) {
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            arr[i][j] = rand() % 2 + 1;
        }
    }
}

void printArray(int **arr) {
    srand(time(NULL));
    cout << "[";
    for (int i = 0; i < N; ++i) {
        if (i != 0)
            cout << " ";
        for (int j = 0; j < N; ++j) {
            cout << arr[i][j];
            if (j != N - 1)
                cout << " ";
        }
        if (i != N - 1)
            cout << "\n";
    }
    cout << "]\n";
}

void luDecomposition(int **a, int **l, int **u) {
#pragma omp parallel num_threads(threads) if(threads > 1)
    {
#pragma omp master
        {
            for (int i = 0; i < N; i++) {
#pragma omp task
                for (int j = 0; j < N; j++) {
                    if (j < i)
                        l[j][i] = 0;
                    else {
                        l[j][i] = a[j][i];
                        for (int k = 0; k < i; k++) {
                            l[j][i] = l[j][i] - l[j][k] * u[k][i];
                        }
                    }
                }
#pragma omp task
                for (int j = 0; j < N; j++) {
                    if (j < i)
                        u[i][j] = 0;
                    else if (j == i)
                        u[i][j] = 1;
                    else {
                        u[i][j] = a[i][j] / (l[i][i] == 0 ? 1 : l[i][i]);
                        for (int k = 0; k < i; k++) {
                            u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / (l[i][i] == 0 ? 1 : l[i][i]));
                        }
                    }
                }
            }
        }
    }
}

int determinantOfMatrix(int **mat) {
    int **L = (int **) malloc(N * sizeof(int *));
    allocate(L);
    int **U = (int **) malloc(N * sizeof(int *));
    allocate(U);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            L[j][j] = 0;
            U[j][j] = 0;
        }
    }
    luDecomposition(mat, L, U);
    int det = 1;
    for (int i = 0; i < N; ++i) {
        det *= L[i][i] * U[i][i];
    }
    freeArr(L);
    freeArr(U);

    return det;
}