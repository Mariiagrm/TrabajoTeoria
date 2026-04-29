# Configuración del compilador
CXX  = g++
NVCC = nvcc

# BANDERAS CLAVE:
# -O3: Optimización máxima
# -fopenmp: Activa el soporte para procesamiento multinúcleo
# -MMD -MP: Genera archivos .d con las dependencias exactas de cada .cpp automáticamente
CXXFLAGS  = -Wall -Wextra -std=c++17 -O3 -fopenmp -MMD -MP $(shell pkg-config --cflags opencv4)

# Arquitecturas CUDA: Turing(75), Ampere(80/86), Ada(89), Hopper(90) + PTX para forward-compat.
# Se puede sobreescribir con: make CUDA_GENCODE="-arch=sm_89"
CUDA_GENCODE ?= -gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_86,code=sm_86 \
                -gencode arch=compute_89,code=sm_89 \
                -gencode arch=compute_90,code=sm_90 \
                -gencode arch=compute_90,code=compute_90
NVCCFLAGS = -std=c++17 -O3 $(CUDA_GENCODE) --expt-relaxed-constexpr \
            $(shell pkg-config --cflags opencv4 | sed 's/-I/-I/g')

# Banderas del enlazador (Linker)
LDFLAGS      = -fopenmp $(shell pkg-config --libs opencv4)
NVCC_LDFLAGS = $(shell pkg-config --libs opencv4) -lstdc++fs

TARGET1    = program # Nombre del ejecutable principal
TARGET2    = utils
TARGET3    = cuda_sift

# Lista de archivos fuente
SRCS1 = main.cpp 
SRCS2 = utils.cpp
SRCS3 = cuda_sift.cu

# Generar lista de objetos (.o) y de dependencias (.d) a partir de los fuentes
OBJS1 = $(SRCS1:.cpp=.o)
DEPS1 = $(OBJS1:.o=.d)
OBJS2 = $(SRCS2:.cpp=.o)
DEPS2 = $(OBJS2:.o=.d)
OBJS4 = $(SRCS4:.cpp=.o)
DEPS4 = $(OBJS4:.o=.d)

# Regla principal (Enlazado)
$(TARGET1): $(OBJS1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(OBJS1) $(LDFLAGS)

$(TARGET2): $(OBJS2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(OBJS2) $(LDFLAGS)

$(TARGET3): $(SRCS3)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET3) $(SRCS3) $(NVCC_LDFLAGS)

$(TARGET4): $(OBJS4)
	$(CXX) $(CXXFLAGS) -o $(TARGET4) $(OBJS4) $(LDFLAGS)

# Regla para compilar cada archivo .cpp a .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Magia negra del Makefile: Incluye los archivos .d generados por GCC
-include $(DEPS1)
-include $(DEPS2)
-include $(DEPS4)

# Limpiar archivos temporales (.o, .d) y ejecutable
.PHONY: clean run

NVCC_AVAILABLE := $(shell nvcc -V 2>/dev/null)

all: $(TARGET1) $(TARGET2) $(TARGET4)
ifdef NVCC_AVAILABLE
all: $(TARGET3)
endif

clean:
	rm -f $(OBJS1) $(DEPS1) $(TARGET1)
	rm -f $(OBJS2) $(DEPS2) $(TARGET2)
	rm -f $(TARGET3)
	rm -f $(OBJS4) $(DEPS4) $(TARGET4)

# Ejecutar el programa
run-$(TARGET1): $(TARGET1)
	./$(TARGET1)

run-$(TARGET2): $(TARGET2)
	./$(TARGET2)

run-$(TARGET3): $(TARGET3)
	./$(TARGET3)

run-$(TARGET4): $(TARGET4)
	./$(TARGET4)