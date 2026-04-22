#ifndef CONSTANTS_H
#define CONSTANTS_H

// Size of the square matrices used in the Strassen benchmark.
// Memory per matrix = STRASSEN_BENCHMARK_SIZE² × 8 bytes.
// 2048 → 32 MB/matrix, safe for sequential + parallel (8 threads).
#define STRASSEN_BENCHMARK_SIZE 2048

#endif // CONSTANTS_H
