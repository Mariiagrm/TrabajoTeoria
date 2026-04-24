#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include "matrix_io.h"

//  Strassen helpers 

int verificar_resultados(Matrix A, Matrix B) {
    if (A.size != B.size) {
        printf("Error: dimensiones incompatibles (%dx%d vs %dx%d)\n",
               A.size, A.size, B.size, B.size);
        return -1;
    }
    double epsilon = 1e-6;
    for (int i = 0; i < A.size; i++)
        for (int j = 0; j < A.size; j++)
            if (fabs(A.data[i][j] - B.data[i][j]) > epsilon) {
                printf("Error en [%d][%d]: A=%f B=%f\n", i, j,
                       A.data[i][j], B.data[i][j]);
                return 0;
            }
    return 1;
}

int proxima_potencia_2(int n) {
    int p = 1;
    if (n && !(n & (n - 1))) return n;
    while (p < n) p <<= 1;
    return p;
}

Matrix crear_matriz(int size) {
    Matrix m;
    m.size = size;
    m.data = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
        m.data[i] = (double *)calloc(size, sizeof(double));
    return m;
}

Matrix crear_aleatoria(int size) {
    Matrix m = crear_matriz(size);
    srand((unsigned)time(NULL));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            m.data[i][j] = (double)rand() / RAND_MAX;
    return m;
}

void liberar_matriz(Matrix m) {
    for (int i = 0; i < m.size; i++) free(m.data[i]);
    free(m.data);
}

Matrix sumar(Matrix A, Matrix B) {
    Matrix C = crear_matriz(A.size);
    for (int i = 0; i < A.size; i++)
        for (int j = 0; j < A.size; j++)
            C.data[i][j] = A.data[i][j] + B.data[i][j];
    return C;
}

Matrix restar(Matrix A, Matrix B) {
    Matrix C = crear_matriz(A.size);
    for (int i = 0; i < A.size; i++)
        for (int j = 0; j < A.size; j++)
            C.data[i][j] = A.data[i][j] - B.data[i][j];
    return C;
}

Matrix multiplicacion_clasica(Matrix A, Matrix B) {
    Matrix C = crear_matriz(A.size);
    for (int i = 0; i < A.size; i++)
        for (int j = 0; j < A.size; j++)
            for (int k = 0; k < A.size; k++)
                C.data[i][j] += A.data[i][k] * B.data[k][j];
    return C;
}

