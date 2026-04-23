
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "constants.h"

// ── PARTE 1: Rotación 3D con multiplicación estándar paralela (Nx3 × 3x3) ────

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

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {

    // ── 1. Cargar nube de puntos Nx3 (sin padding) ──────────────────────────
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

    // ── 2. Exportar puntos originales ────────────────────────────────────────
    printf("Exportando puntos originales a CSV...\n");
    exportar_puntos_csv("../results/cloud_points/puntos_originales.csv", puntos, N, 3);
    printf("Puntos originales guardados en ../results/cloud_points/puntos_originales.csv\n");

    // ── 3. Rotación 45° alrededor del eje Z (Nx3 × 3x3, paralela) ───────────
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

    // ── 4. Exportar tiempo de rotación a CSV (resetea el fichero) ────────────
    FILE *csv = fopen("../results/tiempos.csv", "w");
    if (csv) {
        fprintf(csv, "algoritmo,tiempo_s,n\n");
        fprintf(csv, "rotacion_paralela_nx3,%.6f,%d\n", t_rot, N);
        fclose(csv);
        printf("\nTiempo de rotacion exportado a ../results/tiempos.csv\n");
    }

    // ── Liberar memoria ───────────────────────────────────────────────────────
    for (int i = 0; i < N; i++) { free(puntos[i]); free(rotados[i]); }
    free(puntos);
    free(rotados);

    return 0;
}
