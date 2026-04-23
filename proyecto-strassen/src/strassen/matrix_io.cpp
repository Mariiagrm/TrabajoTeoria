#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include "matrix_io.h"

// ── Strassen helpers ──────────────────────────────────────────────────────────

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

// ── Nx3 point-cloud I/O ───────────────────────────────────────────────────────

double **cargar_puntos_sin_padding(const char *ruta_bin, int *out_rows, int *out_cols) {
    *out_rows = 0;
    *out_cols = 0;

    FILE *f = fopen(ruta_bin, "rb");
    if (!f) { perror("Error al abrir binario"); return NULL; }

    int dims[2];
    if (fread(dims, sizeof(int), 2, f) != 2) {
        printf("Error: cabecera inválida en %s\n", ruta_bin);
        fclose(f);
        return NULL;
    }

    int rows = dims[0];
    int cols = dims[1];

    double **data = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        data[i] = (double *)malloc(cols * sizeof(double));
        if (fread(data[i], sizeof(double), cols, f) != (size_t)cols) {
            printf("Error leyendo fila %d\n", i);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    *out_rows = rows;
    *out_cols = cols;
    printf("Puntos cargados: %d filas x %d columnas\n", rows, cols);
    return data;
}

void exportar_puntos_csv(const char *ruta_csv, double **data, int rows, int cols) {
    FILE *f = fopen(ruta_csv, "w");
    if (!f) { perror("Error al crear CSV"); return; }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j > 0) fprintf(f, ",");
            fprintf(f, "%.10f", data[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("CSV exportado (%dx%d): %s\n", rows, cols, ruta_csv);
}
