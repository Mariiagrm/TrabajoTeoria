
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

//  PARTE 1: Rotación 3D con multiplicación estándar paralela (Nx3 × 3x3) 

static void rotar_puntos(double **puntos, double **rotados, int N,
                         double R[3][3]) {
    #pragma omp parallel for num_threads(8) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
            double acc = 0.0;
            for (int k = 0; k < 3; k++)
                acc += puntos[i][k] * R[k][j];
            rotados[i][j] = acc;
        }
    }
}

//  Nx3 point-cloud I/O 

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

//  main 

int main(void) {

    //  1. Cargar nube de puntos Nx3 (sin padding) 
    printf("--- PASO 2: Reconstruccion de imagen 3D mediante multiplicacion de matrices ---");

    printf("Cargando nube de puntos 3D desde ../results/cloud_points/puntos_3d.bin...\n");
    int N = 0, cols = 0;
    double **puntos = cargar_puntos_sin_padding("../results/cloud_points/puntos_3d.bin",
                                                &N, &cols);
    if (!puntos || cols != 3) {
        printf("Error: se esperan puntos Nx3 (obtenido Nx%d).\n", cols);
        return 1;
    }
    printf("Nube cargada: %d puntos 3D.\n", N);

    //  2. Exportar puntos originales 
    printf("Exportando puntos originales a CSV...\n");
    exportar_puntos_csv("../results/cloud_points/puntos_originales.csv", puntos, N, 3);
    printf("Puntos originales guardados en ../results/cloud_points/puntos_originales.csv\n");

    //  3. Rotación 45° alrededor del eje Z (Nx3 × 3x3, paralela) 
    double angulo = 45.0 * M_PI / 180.0;
    double R[3][3] = {
        { cos(angulo), -sin(angulo), 0.0 },
        { sin(angulo),  cos(angulo), 0.0 },
        { 0.0,          0.0,         1.0 }
    };

    double **rotados = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
        rotados[i] = (double *)malloc(3 * sizeof(double));

    printf("\nRotando %d puntos con 8 hilos (Nx3 x 3x3)...\n", N);
    double t_rot_ini = omp_get_wtime();
    rotar_puntos(puntos, rotados, N, R);
    double t_rot = omp_get_wtime() - t_rot_ini;
    printf("Rotacion completada en %.6f s\n", t_rot);

    exportar_puntos_csv("../results/cloud_points/puntos_rotados.csv", rotados, N, 3);


    //  Liberar memoria 
    for (int i = 0; i < N; i++) { free(puntos[i]); free(rotados[i]); }
    free(puntos);
    free(rotados);

    return 0;
}
