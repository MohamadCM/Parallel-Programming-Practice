//
// Created by mohamad on ۲۰۲۱/۵/۶.
//
#define dimension 128
#define n_threads 1

#include <iostream>
#include <ctime>
#include <omp.h>

using namespace std;

void init_fill_array(int ***array, bool fill);


void print3D(int ***);


int main() {
    int size;
    int ***A = (int ***) malloc(dimension * sizeof(int **));
    int ***B = (int ***) malloc(dimension * sizeof(int **));
    init_fill_array(A, true);
    init_fill_array(B, true);

    size = (dimension * dimension * dimension * 4) /
           1000000; // Size of the 3D array is the product of size of each array
    printf("Processing %d MB for 3 matrices", size);

    int ***C = (int ***) malloc(dimension * sizeof(int ***));
    init_fill_array(C, false);

    volatile double start_time = omp_get_wtime();
	
    for (int l = 0; l < dimension; ++l) {
#pragma omp parallel for num_threads(n_threads)
    for (int m = 0; m < dimension; ++m) {

        for (int i = 0; i < dimension; i++) { // 2D multiply
            for (int j = 0; j < dimension; j++) {
                for (int k = 0; k < dimension; k++) {
                    C[m][i][j] += ((A[l][m][k]) * (B[k][i][j]));
                }
            }
        }

        }
    }
    volatile double end_time = omp_get_wtime();


    //cout << "A: ";
    //print3D(A);
    //cout<<"B: ";
    //print3D(B);
    //cout<<"C: ";
    //print3D(C);

    cout << "\nElapsed time: " << (end_time - start_time) << "\n";
    return 0;
}

void init_fill_array(int ***array, bool fill) {
    if (array == NULL) {
        fprintf(stderr, "Out of memory");
        exit(0);
    }

    for (int i = 0; i < dimension; i++) {
        array[i] = (int **) malloc(dimension * sizeof(int *));

        if (array[i] == NULL) {
            fprintf(stderr, "Out of memory");
            exit(0);
        }

        for (int j = 0; j < dimension; j++) {
            array[i][j] = (int *) malloc(dimension * sizeof(int));
            if (array[i][j] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }
    }
    /* Filing array with random values in range of 0 - 9 */
    srand(time(NULL));
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            for (int k = 0; k < dimension; ++k) {
                if (fill)
                    array[i][j][k] = rand() % 10;
                else
                    array[i][j][k] = 0;
            }
        }
    }
}

void init_array(int ****array) {
    if (array == NULL) {
        fprintf(stderr, "Out of memory");
        exit(0);
    }

    for (int i = 0; i < dimension; i++) {
        array[i] = (int ***) malloc(dimension * sizeof(int **));

        if (array[i] == NULL) {
            fprintf(stderr, "Out of memory");
            exit(0);
        }

        for (int j = 0; j < dimension; j++) {
            array[i][j] = (int **) malloc(dimension * sizeof(int *));
            if (array[i][j] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
            for (int k = 0; k < dimension; ++k) {
                array[i][j][k] = (int *) malloc(dimension * sizeof(int));
                if (array[i][j] == NULL) {
                    fprintf(stderr, "Out of memory");
                    exit(0);
                }
            }
        }
    }
}

void print3D(int ***arr) {
    int i, j, k;
    cout << "\n";
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            for (k = 0; k < dimension; k++) {
                cout << "[" << i << "][" << j << "][" << k << "] = " << arr[i][j][k] << endl;
            }
        }
    }
    cout << "\n";
}
