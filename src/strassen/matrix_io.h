
#ifndef MATRIX_IO_H
#define MATRIX_IO_H

// Square matrix used by Strassen (size is always a power of 2)
typedef struct {
    int size;
    double **data;
} Matrix;

// --- Strassen helpers ---
int verificar_resultados(Matrix A, Matrix B);
int proxima_potencia_2(int n);
Matrix crear_matriz(int size);
Matrix crear_aleatoria(int size);
void liberar_matriz(Matrix m);
Matrix sumar(Matrix A, Matrix B);
Matrix restar(Matrix A, Matrix B);
Matrix multiplicacion_clasica(Matrix A, Matrix B);


#endif // MATRIX_IO_H