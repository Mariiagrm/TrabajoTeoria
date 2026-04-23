#ifndef STRASSEN_3D_H
#define STRASSEN_3D_H

Matrix strassen_secuencial(Matrix A, Matrix B);
Matrix strassen_paralelo(Matrix A, Matrix B, FILE *log_file);   

#endif // STRASSEN_3D_H
