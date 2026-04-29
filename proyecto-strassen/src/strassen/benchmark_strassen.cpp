
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <omp.h>
#include "matrix_io.h"
#include "strassen_3d.h"

#define CSV_PATH "../../results/tiempos.csv"
#define LOG_PATH "../../results/strassen_paralelo.log"

static const int DEFAULT_SIZES[] = {128, 256, 512, 1024, 2048, 4096, 8192};
static const int DEFAULT_NUM_SIZES = sizeof(DEFAULT_SIZES) / sizeof(DEFAULT_SIZES[0]);

static const int DEFAULT_THREADS[] = {1, 2, 4, 8, 16};
static const int DEFAULT_NUM_THREADS = sizeof(DEFAULT_THREADS) / sizeof(DEFAULT_THREADS[0]);

static int csv_is_empty(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 1;
    return st.st_size == 0;
}

static int csv_has_expected_header(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[256];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);
    return strstr(buf, "tiempo_secuencial_s") != NULL &&
           strstr(buf, "tiempo_paralelo_s")   != NULL &&
           strstr(buf, "speedup")             != NULL &&
           strstr(buf, "hilos")               != NULL;
}

static void benchmark_strassen(double *t_sec_out, double *t_sec_omp_out, double *t_par_out, int n, int num_hilos) {
    printf("\n............... BENCHMARK STRASSEN %dx%d | %d hilos ...............\n",
           n, n, num_hilos);

    Matrix A = crear_aleatoria(n);
    Matrix B = crear_aleatoria(n);

    printf("  Strassen secuencial...\n");
    double t0 = omp_get_wtime();
    Matrix C_seq = strassen_secuencial(A, B);
    *t_sec_out = omp_get_wtime() - t0;
    printf("  Tiempo secuencial : %.4f s\n", *t_sec_out);

    printf("  Strassen paralelo (%d hilos)...\n", num_hilos);
    FILE *log = fopen(LOG_PATH, "w");
    Matrix C_par;
    omp_set_num_threads(num_hilos);
    t0 = omp_get_wtime();
    #pragma omp parallel num_threads(num_hilos)
    {
        #pragma omp single
        { C_par = strassen_paralelo(A, B, log); }
    }
    *t_par_out = omp_get_wtime() - t0;

    printf("Strassen secuencial con OpenMP (paralelismo solo en el primer nivel)...\n");
    t0 = omp_get_wtime();
    Matrix C_seq_omp = strassen_secuencial_openMP(A, B, log);
    *t_sec_omp_out = omp_get_wtime() - t0;
    if (log) fclose(log);
    printf("  Tiempo secuencial OpenMP : %.4f s\n", *t_sec_omp_out);
    printf("  Tiempo paralelo   : %.4f s\n", *t_par_out);
    printf("  Speedup  (Secuencial vs Paralelo)         : %.2fx\n", *t_sec_out / *t_par_out);
    printf("  Speedup  (Paralelo vs Secuencial OpenMP)         : %.2fx\n", *t_par_out / *t_sec_omp_out  );
    printf("  Speedup  ( Secuencial vs Secuencial OpenMP)         : %.2fx\n", *t_sec_out / *t_sec_omp_out);

    if (verificar_resultados(C_seq, C_par) and verificar_resultados(C_seq, C_seq_omp))
        printf("  [ OK ] Resultados coinciden.\n");
    else
        printf("  [ X ] Error matematico en Strassen.\n");

    liberar_matriz(A); liberar_matriz(B);
    liberar_matriz(C_seq); liberar_matriz(C_par); liberar_matriz(C_seq_omp);
}

static void parse_int_list(const char *s, int *out, int max_n, int *n_out) {
    int n = 0;
    const char *p = s;
    while (*p && n < max_n) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0) out[n++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    *n_out = n;
}

int main(int argc, char *argv[]) {
    const int *sizes = DEFAULT_SIZES;
    int num_sizes = DEFAULT_NUM_SIZES;

    const int *threads = DEFAULT_THREADS;
    int num_threads = DEFAULT_NUM_THREADS;

    int custom_sizes[16];
    int custom_threads[16];
    int num_custom_s = 0, num_custom_t = 0;

    // Uso:
    //   ./benchmark_strassen                    -> tamaños y hilos por defecto
    //   ./benchmark_strassen --sizes 512,1024 --threads 1,2,4,8
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            parse_int_list(argv[++i], custom_sizes, 16, &num_custom_s);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            parse_int_list(argv[++i], custom_threads, 16, &num_custom_t);
        } else {
            int v = atoi(argv[i]);
            if (v > 0 && num_custom_s < 16) custom_sizes[num_custom_s++] = v;
        }
    }
    if (num_custom_s > 0) { sizes = custom_sizes; num_sizes = num_custom_s; }
    if (num_custom_t > 0) { threads = custom_threads; num_threads = num_custom_t; }

    int need_header = csv_is_empty(CSV_PATH) || !csv_has_expected_header(CSV_PATH);
    FILE *csv = fopen(CSV_PATH, "a");
    if (!csv) {
        fprintf(stderr, "No se pudo abrir %s\n", CSV_PATH);
        return 1;
    }
    if (need_header) {
        fprintf(csv, "n,hilos,tiempo_secuencial_s,tiempo_paralelo_s, tiempo_secuencial_openmp_s,speedup (Secuencial vs Paralelo),speedup (Paralelo vs Secuencial OpenMP),speedup (Secuencial vs Secuencial OpenMP)\n");
    }

    printf("Barrido: %d tamanos x %d configuraciones de hilos. Max hilos disponibles: %d\n",
           num_sizes, num_threads, omp_get_max_threads());

    for (int s = 0; s < num_sizes; ++s) {
        int n = sizes[s];
        for (int h = 0; h < num_threads; ++h) {
            int p = threads[h];
            double t_sec = 0.0, t_par = 0.0, t_sec_omp = 0.0;
            benchmark_strassen(&t_sec, &t_sec_omp, &t_par, n, p);

            double speedup = (t_par > 0.0) ? (t_sec / t_par) : 0.0;
            fprintf(csv, "%d,%d,%.6f,%.6f,%.4f,%.2f,%.2f,%.2f\n",
                    n, p, t_sec, t_par, t_sec_omp, speedup,  t_par/t_sec_omp, t_sec/t_sec_omp);
            fflush(csv);
        }
    }

    fclose(csv);
    printf("\nTiempos exportados a %s\n", CSV_PATH);
    return 0;
}
