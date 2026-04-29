# Benchmark: Algoritmo de Strassen con OpenMP

## Implementaciones comparadas

| Implementación | Función | Paralelismo |
|---|---|---|
| **Secuencial** | `strassen_secuencial(A, B)` | Ninguno |
| **Paralelo Tasks** | `strassen_paralelo(A, B, log)` | `omp task` recursivo en todos los niveles |
| **Sections (1er nivel)** | `strassen_secuencial_openMP(A, B, log)` | `omp parallel sections` solo en el primer nivel; recursión secuencial |

La diferencia clave entre las dos versiones paralelas: **Tasks** crea nuevas tareas en cada nivel recursivo (puede generar miles de tareas), mientras que **Sections** paraleliza únicamente las 7 multiplicaciones del nivel raíz y desciende de forma secuencial, evitando la sobresubscripción de hilos.

---

## Configuración del experimento

- **Tamaños de matriz:** 128 × 128, 256 × 256, 512 × 512, 1024 × 1024, 2048 × 2048
- **Hilos:** 1, 2, 4, 8, 16
- **Caso base:** multiplicación clásica para n ≤ 64
- **Métrica:** tiempo de pared (`omp_get_wtime`)

---

## Resultados por tamaño de matriz

### n = 128

| Hilos | Secuencial (s) | Paralelo Tasks (s) | Sections 1er nivel (s) | Speedup Sec→Tasks | Speedup Sec→Sections |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.0055 | 0.0068 | 0.0054 | 0.80x | 1.02x |
| 2  | 0.0044 | 0.0046 | 0.0032 | 0.94x | 1.37x |
| 4  | 0.0052 | 0.0042 | 0.0022 | 1.24x | **2.34x** |
| 8  | 0.0040 | 0.0069 | 0.0049 | 0.59x | 0.83x |
| 16 | 0.0052 | 0.0082 | 0.0026 | 0.63x | 2.02x |

> Para matrices pequeñas el overhead de sincronización domina. Tasks es incluso más lento que secuencial con 8–16 hilos.

---

### n = 256

| Hilos | Secuencial (s) | Paralelo Tasks (s) | Sections 1er nivel (s) | Speedup Sec→Tasks | Speedup Sec→Sections |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.0303 | 0.0232 | 0.0251 | 1.30x | 1.21x |
| 2  | 0.0275 | 0.0181 | 0.0140 | 1.52x | 1.97x |
| 4  | 0.0210 | 0.0146 | 0.0116 | 1.45x | 1.82x |
| 8  | 0.0260 | 0.0099 | 0.0119 | 2.62x | 2.18x |
| 16 | 0.0277 | 0.0100 | 0.0086 | **2.78x** | **3.21x** |

> A partir de n=256 el paralelismo empieza a compensar el overhead.

---

### n = 512

| Hilos | Secuencial (s) | Paralelo Tasks (s) | Sections 1er nivel (s) | Speedup Sec→Tasks | Speedup Sec→Sections |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.1342 | 0.1333 | 0.1166 | 1.01x | 1.15x |
| 2  | 0.1283 | 0.0985 | 0.0819 | 1.30x | 1.57x |
| 4  | 0.1228 | 0.0618 | 0.0597 | 1.99x | 2.06x |
| 8  | 0.1269 | 0.0550 | 0.0494 | 2.31x | 2.57x |
| 16 | 0.1267 | 0.0480 | 0.0441 | **2.64x** | **2.87x** |

---

### n = 1024

| Hilos | Secuencial (s) | Paralelo Tasks (s) | Sections 1er nivel (s) | Speedup Sec→Tasks | Speedup Sec→Sections |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.9196 | 1.0167 | 0.8717 | 0.90x | 1.06x |
| 2  | 0.8903 | 0.6177 | 0.5649 | 1.44x | 1.58x |
| 4  | 0.8999 | 0.4174 | 0.3569 | 2.16x | 2.52x |
| 8  | 0.9235 | 0.3130 | 0.2943 | 2.95x | 3.14x |
| 16 | 1.0106 | 0.3700 | 0.2888 | 2.73x | **3.50x** |

> Con 16 hilos, Tasks empeora respecto a 8 hilos (sobresubscripción en recursión profunda). Sections mantiene la mejora.

---

### n = 2048

| Hilos | Secuencial (s) | Paralelo Tasks (s) | Sections 1er nivel (s) | Speedup Sec→Tasks | Speedup Sec→Sections |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 7.522 | 9.005 | 6.828 | 0.84x | 1.10x |
| 2  | 7.853 | 5.754 | 7.232 | 1.36x | 1.09x |
| 4  | 7.194 | 2.916 | 2.824 | 2.47x | 2.55x |
| 8  | 7.323 | 2.782 | 3.237 | **2.63x** | 2.26x |
| 16 | 7.105 | 2.999 | 2.832 | 2.37x | 2.51x |

> A n=2048 con 2 hilos, Sections es casi secuencial (solo 7 secciones disponibles pero el trabajo por sección es enorme y desbalanceado).

---

## Mejor speedup conseguido por tamaño

| n | Mejor tiempo (s) | Implementación | Hilos | Speedup vs Secuencial |
|:---:|:---:|:---:|:---:|:---:|
| 128  | 0.0022 | Sections | 4  | 2.34x |
| 256  | 0.0086 | Sections | 16 | 3.21x |
| 512  | 0.0441 | Sections | 16 | 2.87x |
| 1024 | 0.2888 | Sections | 16 | 3.50x |
| 2048 | 2.782  | Tasks    | 8  | 2.63x |

---

## Análisis y conclusiones

**Sections gana en la mayoría de tamaños** porque evita el overhead de creación de tareas OMP en cada nivel recursivo. Con `omp task` recursivo, para n=1024 se generan potencialmente cientos de tareas anidadas; con Sections solo se crean 7 hilos una vez.

**Tasks puntúa mejor a n=2048 con 8 hilos** porque a ese tamaño la recursión tiene más niveles con trabajo real, y Tasks puede aprovechar mejor el paralelismo en los niveles intermedios que Sections ignora.

**El paralelismo apenas ayuda con n=128**: el tiempo de divide, sincronización y unión de resultados supera al cómputo útil. El umbral de rentabilidad está alrededor de n=256.

**Saturación a 16 hilos**: la mejora de 8→16 hilos es pequeña o nula porque el algoritmo solo expone 7 tareas independientes en el primer nivel. El límite teórico de Sections es speedup=7 (Ley de Amdahl con fracción paralela = 7/7).

---

## Gráficas generadas

| Gráfica | Descripción |
|---|---|
| [speedup_vs_hilos.png](results/plots/speedup_vs_hilos.png) | Speedup de cada implementación frente al número de hilos |
| [treeStrassen.png](results/plots/treeStrassen.png) | Árbol de recursión de Strassen |
| [resultado_3d.png](results/plots/resultado_3d.png) | Resultado de la reconstrucción 3D |

---

*Datos completos en [results/tiempos.csv](results/tiempos.csv)*
