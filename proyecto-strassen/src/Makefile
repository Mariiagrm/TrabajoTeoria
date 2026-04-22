# Configuración del compilador
CXX  = g++
NVCC = nvcc

# BANDERAS CLAVE:
# -O3: Optimización máxima
# -fopenmp: Activa el soporte para procesamiento multinúcleo
# -MMD -MP: Genera archivos .d con las dependencias exactas de cada .cpp automáticamente
CXXFLAGS  = -Wall -Wextra -std=c++17 -O3 -fopenmp -MMD -MP $(shell pkg-config --cflags opencv4)
NVCCFLAGS = -std=c++17 -O3 -arch=native --expt-relaxed-constexpr \
            $(shell pkg-config --cflags opencv4 | sed 's/-I/-I/g')

# Banderas del enlazador (Linker)
LDFLAGS      = -fopenmp $(shell pkg-config --libs opencv4)
NVCC_LDFLAGS = $(shell pkg-config --libs opencv4) -lstdc++fs

TARGET1    = strassen_app
TARGET2    = utils
TARGET3    = cuda_sift

# Lista de archivos fuente
SRCS1 = main.cpp matrix_io.cpp strassen_3d.cpp
SRCS2 = utils.cpp
SRCS3 = cuda_sift.cu

# Generar lista de objetos (.o) y de dependencias (.d) a partir de los fuentes
OBJS1 = $(SRCS1:.cpp=.o)
DEPS1 = $(OBJS1:.o=.d)
OBJS2 = $(SRCS2:.cpp=.o)
DEPS2 = $(OBJS2:.o=.d)

# Regla principal (Enlazado)
$(TARGET1): $(OBJS1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(OBJS1) $(LDFLAGS)

$(TARGET2): $(OBJS2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(OBJS2) $(LDFLAGS)

$(TARGET3): $(SRCS3)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET3) $(SRCS3) $(NVCC_LDFLAGS)

# Regla para compilar cada archivo .cpp a .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Magia negra del Makefile: Incluye los archivos .d generados por GCC
-include $(DEPS1)
-include $(DEPS2)

# Limpiar archivos temporales (.o, .d) y ejecutable
.PHONY: clean run

NVCC_AVAILABLE := $(shell command -v nvcc 2>/dev/null)

all: $(TARGET1) $(TARGET2)
ifdef NVCC_AVAILABLE
all: $(TARGET3)
endif

clean:
	rm -f $(OBJS1) $(DEPS1) $(TARGET1)
	rm -f $(OBJS2) $(DEPS2) $(TARGET2)
	rm -f $(TARGET3)

# Ejecutar el programa
run-$(TARGET1): $(TARGET1)
	./$(TARGET1)

run-$(TARGET2): $(TARGET2)
	./$(TARGET2)

run-$(TARGET3): $(TARGET3)
	./$(TARGET3)