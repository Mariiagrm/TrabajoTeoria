# Configuración del compilador
CXX  = g++

# BANDERAS CLAVE:
# -O3: Optimización máxima
# -fopenmp: Activa el soporte para procesamiento multinúcleo
# -MMD -MP: Genera archivos .d con las dependencias exactas de cada .cpp automáticamente
CXXFLAGS  = -Wall -Wextra -std=c++17 -O3 -fopenmp -MMD -MP $(shell pkg-config --cflags opencv4)

# Arquitecturas CUDA: Turing(75), Ampere(80/86), Ada(89), Hopper(90) + PTX para forward-compat.
# Se puede sobreescribir con: make CUDA_GENCODE="-arch=sm_89"

# Banderas del enlazador (Linker)
LDFLAGS      = -fopenmp $(shell pkg-config --libs opencv4)
NVCC_LDFLAGS = $(shell pkg-config --libs opencv4) -lstdc++fs


TARGET1    = benchmark_strassen

# Lista de archivos fuente
SRCS1 = benchmark_strassen.cpp matrix_io.cpp strassen_3d.cpp

# Generar lista de objetos (.o) y de dependencias (.d) a partir de los fuentes
OBJS1 = $(SRCS1:.cpp=.o)
DEPS1 = $(OBJS1:.o=.d)

# Regla principal (Enlazado)
$(TARGET1): $(OBJS1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(OBJS1) $(LDFLAGS)



# Regla para compilar cada archivo .cpp a .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Magia negra del Makefile: Incluye los archivos .d generados por GCC
-include $(DEPS1)


# Limpiar archivos temporales (.o, .d) y ejecutable
.PHONY: clean run

clean:
	rm -f $(OBJS1) $(DEPS1) $(TARGET1)


# Ejecutar el programa
run-$(TARGET1): $(TARGET1)
	./$(TARGET1)

