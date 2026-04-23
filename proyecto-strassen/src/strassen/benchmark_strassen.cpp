
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "matrix_io.h"
#include "strassen_3d.h"

#define STRASSEN_BENCHMARK_SIZE 

static void benchmark_strassen(double *t_sec_out, double *t_par_out, int n) {
    printf("\n===== BENCHMARK STRASSEN %dx%d =====\n", n, n);

    Matrix A = crear_aleatoria(n);
    Matrix B = crear_aleatoria(n);

    printf("  Strassen secuencial...\n");
    double t0 = omp_get_wtime();
    Matrix C_seq = strassen_secuencial(A, B);
    *t_sec_out = omp_get_wtime() - t0;
    printf("  Tiempo secuencial : %.4f s\n", *t_sec_out);

    printf("  Strassen paralelo...\n");
    FILE *log = fopen("../../results/strassen_paralelo.log", "w");
    Matrix C_par;
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        { C_par = strassen_paralelo(A, B, log); }
    }
    *t_par_out = omp_get_wtime() - t0;
    if (log) fclose(log);
    printf("  Tiempo paralelo   : %.4f s\n", *t_par_out);
    printf("  Speedup           : %.2fx\n", *t_sec_out / *t_par_out);

    if (verificar_resultados(C_seq, C_par))
        printf("  [ OK ] Resultados coinciden.\n");
    else
        printf("  [ X ] Error matematico en Strassen.\n");

    liberar_matriz(A); liberar_matriz(B);
    liberar_matriz(C_seq); liberar_matriz(C_par);
}

int main(int argc, char *argv[]) {
    int n = (argc >= 2) ? atoi(argv[1]) : STRASSEN_BENCHMARK_SIZE;

    double t_sec = 0.0, t_par = 0.0;
    benchmark_strassen(&t_sec, &t_par, n);

    FILE *csv = fopen("../../results/tiempos.csv", "a");
    if (csv) {
        fprintf(csv, "strassen_secuencial,%.6f,%d\n", t_sec, n);
        fprintf(csv, "strassen_paralelo,%.6f,%d\n",   t_par, n);
        fclose(csv);
        printf("\nTiempos de Strassen exportados a ../../results/tiempos.csv\n");
    }

    return 0;
}
