
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_io.h"  // Para cargar matrices con padding
#include "strassen_3d.h" // Implementación de Strassen (secuencial y paralela)
#include "omp.h"
#include <string> // Cambiado de "String.h" a <string>
#include <stdio.h>



    // EXPORTAR RESULTADO PARA PYTHON
    




// 3. EL ALGORITMO DE STRASSEN (Secuencial)
Matrix strassen_secuencial(Matrix A, Matrix B) {
    int n = A.size;

    if (n <= 64) {
        return multiplicacion_clasica(A, B);
    }

    int mitad = n / 2;
    Matrix A11 = crear_matriz(mitad), A12 = crear_matriz(mitad), A21 = crear_matriz(mitad), A22 = crear_matriz(mitad);
    Matrix B11 = crear_matriz(mitad), B12 = crear_matriz(mitad), B21 = crear_matriz(mitad), B22 = crear_matriz(mitad);

    // Dividir las matrices en 4 cuadrantes
    for (int i = 0; i < mitad; i++) {
        for (int j = 0; j < mitad; j++) {
            A11.data[i][j] = A.data[i][j];
            A12.data[i][j] = A.data[i][j + mitad];
            A21.data[i][j] = A.data[i + mitad][j];
            A22.data[i][j] = A.data[i + mitad][j + mitad];

            B11.data[i][j] = B.data[i][j];
            B12.data[i][j] = B.data[i][j + mitad];
            B21.data[i][j] = B.data[i + mitad][j];
            B22.data[i][j] = B.data[i + mitad][j + mitad];
        }
    }

    // Calcular las 7 multiplicaciones de Strassen (P1 a P7)
    Matrix S1 = restar(B12, B22); Matrix P1 = strassen_secuencial(A11, S1);
    Matrix S2 = sumar(A11, A12);  Matrix P2 = strassen_secuencial(S2, B22);
    Matrix S3 = sumar(A21, A22);  Matrix P3 = strassen_secuencial(S3, B11);
    Matrix S4 = restar(B21, B11); Matrix P4 = strassen_secuencial(A22, S4);
    Matrix S5 = sumar(A11, A22);  
    Matrix S6 = sumar(B11, B22); Matrix P5 = strassen_secuencial(S5, S6);
    Matrix S7 = restar(A12, A22); 
    Matrix S8 = sumar(B21, B22); Matrix P6 = strassen_secuencial(S7, S8);
    Matrix S9 = restar(A11, A21); 
    Matrix S10 = sumar(B11, B12); Matrix P7 = strassen_secuencial(S9, S10);

    // Combinar resultados en C11, C12, C21, C22
    Matrix C11 = sumar(restar(sumar(P5, P4), P2), P6);
    Matrix C12 = sumar(P1, P2);
    Matrix C21 = sumar(P3, P4);
    Matrix C22 = restar(restar(sumar(P5, P1), P3), P7);

    // Unir en la matriz final C
    Matrix C = crear_matriz(n);
    for (int i = 0; i < mitad; i++) {
        for (int j = 0; j < mitad; j++) {
            C.data[i][j] = C11.data[i][j];
            C.data[i][j + mitad] = C12.data[i][j];
            C.data[i + mitad][j] = C21.data[i][j];
            C.data[i + mitad][j + mitad] = C22.data[i][j];
        }
    }

    // Liberar TODA la memoria temporal 
    liberar_matriz(A11); liberar_matriz(A12); liberar_matriz(A21); liberar_matriz(A22);
    liberar_matriz(B11); liberar_matriz(B12); liberar_matriz(B21); liberar_matriz(B22);
    liberar_matriz(P1); liberar_matriz(P2); liberar_matriz(P3); liberar_matriz(P4);
    liberar_matriz(P5); liberar_matriz(P6); liberar_matriz(P7);
    liberar_matriz(S1); liberar_matriz(S2); liberar_matriz(S3); liberar_matriz(S4);
    liberar_matriz(S5); liberar_matriz(S6); liberar_matriz(S7); liberar_matriz(S8);
    liberar_matriz(S9); liberar_matriz(S10);
    liberar_matriz(C11); liberar_matriz(C12); liberar_matriz(C21); liberar_matriz(C22);

    return C;
}



// 3. EL ALGORITMO DE STRASSEN (Paralelo con OpenMP y Logs a Fichero)
// IMPORTANTE: Recuerda actualizar la firma en tu "strassen_3d.h"
Matrix strassen_paralelo(Matrix A, Matrix B, FILE *log_file) {
    int n = A.size;

    if (n <= 64) {
        #pragma omp critical
        {
            fprintf(log_file, "[Base] Alcanzado caso base n=%d en hilo %d\n", n, omp_get_thread_num());
            fflush(log_file);
        }
        return multiplicacion_clasica(A, B);
    }

    // Proteger la escritura en el archivo para evitar condiciones de carrera entre hilos
    #pragma omp critical
    {
        fprintf(log_file, "[División] Dividiendo matrices de tamaño %d en hilo %d\n", n, omp_get_thread_num());
        fflush(log_file); // Forzar la escritura inmediata
    }

    int mitad = n / 2;
    Matrix A11 = crear_matriz(mitad), A12 = crear_matriz(mitad), A21 = crear_matriz(mitad), A22 = crear_matriz(mitad);
    Matrix B11 = crear_matriz(mitad), B12 = crear_matriz(mitad), B21 = crear_matriz(mitad), B22 = crear_matriz(mitad);

    // Dividir las matrices en 4 cuadrantes
    for (int i = 0; i < mitad; i++) {
        for (int j = 0; j < mitad; j++) {
            A11.data[i][j] = A.data[i][j];
            A12.data[i][j] = A.data[i][j + mitad];
            A21.data[i][j] = A.data[i + mitad][j];
            A22.data[i][j] = A.data[i + mitad][j + mitad];

            B11.data[i][j] = B.data[i][j];
            B12.data[i][j] = B.data[i][j + mitad];
            B21.data[i][j] = B.data[i + mitad][j];
            B22.data[i][j] = B.data[i + mitad][j + mitad];
        }
    }

    // Calcular las matrices intermedias S1..S10
    Matrix S1 = restar(B12, B22); 
    Matrix S2 = sumar(A11, A12);  
    Matrix S3 = sumar(A21, A22);  
    Matrix S4 = restar(B21, B11); 
    Matrix S5 = sumar(A11, A22);  Matrix S6 = sumar(B11, B22); 
    Matrix S7 = restar(A12, A22); Matrix S8 = sumar(B21, B22); 
    Matrix S9 = restar(A11, A21); Matrix S10 = sumar(B11, B12); 

    Matrix P1, P2, P3, P4, P5, P6, P7;

    #pragma omp critical
    {
        fprintf(log_file, "[Tareas] Lanzando 7 tareas paralelas para n=%d\n", n);
        fflush(log_file);
    }

    // 7 tareas paralelas (Pasamos log_file a las llamadas recursivas)
    #pragma omp task shared(P1)
    { P1 = strassen_paralelo(A11, S1, log_file); }

    #pragma omp task shared(P2)
    { P2 = strassen_paralelo(S2, B22, log_file); }

    #pragma omp task shared(P3)
    { P3 = strassen_paralelo(S3, B11, log_file); }

    #pragma omp task shared(P4)
    { P4 = strassen_paralelo(A22, S4, log_file); }

    #pragma omp task shared(P5)
    { P5 = strassen_paralelo(S5, S6, log_file); }

    #pragma omp task shared(P6)
    { P6 = strassen_paralelo(S7, S8, log_file); }

    #pragma omp task shared(P7)
    { P7 = strassen_paralelo(S9, S10, log_file); }

    // Esperar a que las 7 tareas terminen
    #pragma omp taskwait
    
    #pragma omp critical
    {
        fprintf(log_file, "[Combinar] Combinando submatrices P para n=%d\n", n);
        fflush(log_file);
    }

    // Combinar resultados en C11, C12, C21, C22
    Matrix C11 = sumar(restar(sumar(P5, P4), P2), P6);
    Matrix C12 = sumar(P1, P2);
    Matrix C21 = sumar(P3, P4);
    Matrix C22 = restar(restar(sumar(P5, P1), P3), P7);

    // Unir en la matriz final C
    Matrix C = crear_matriz(n);
    for (int i = 0; i < mitad; i++) {
        for (int j = 0; j < mitad; j++) {
            C.data[i][j] = C11.data[i][j];
            C.data[i][j + mitad] = C12.data[i][j];
            C.data[i + mitad][j] = C21.data[i][j];
            C.data[i + mitad][j + mitad] = C22.data[i][j];
        }
    }


    // Liberar memoria temporal
    liberar_matriz(A11); liberar_matriz(A12); liberar_matriz(A21); liberar_matriz(A22);
    liberar_matriz(B11); liberar_matriz(B12); liberar_matriz(B21); liberar_matriz(B22);
    liberar_matriz(P1); liberar_matriz(P2); liberar_matriz(P3); liberar_matriz(P4);
    liberar_matriz(P5); liberar_matriz(P6); liberar_matriz(P7);
    liberar_matriz(S1); liberar_matriz(S2); liberar_matriz(S3); liberar_matriz(S4);
    liberar_matriz(S5); liberar_matriz(S6); liberar_matriz(S7); liberar_matriz(S8);
    liberar_matriz(S9); liberar_matriz(S10);
    liberar_matriz(C11); liberar_matriz(C12); liberar_matriz(C21); liberar_matriz(C22);

    return C;
}