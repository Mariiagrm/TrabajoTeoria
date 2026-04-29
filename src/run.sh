#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [ -d env ]; then
    PYTHON="env/bin/python3"
fi

#  1. Compilar todo 
echo "=============================="
echo " PASO 0: make all"
echo "=============================="
make all

#  2. Preprocesar imágenes y reconstrucción 3D (genera results/cloud_points/puntos_3d.bin) 
echo ""
echo "=============================="
echo " PASO 1: reconstrucción 3D"
echo "=============================="
python3 utils.py 1

#  3. Rotación con multiplicacion de matrices (lee puntos_3d.bin, escribe puntos_rotados.csv) 
echo ""
echo "=============================="
echo " PASO 2: Transformación de ángulos con multiplicación de matrices (C++)"
echo "=============================="
./program

#  4. Visualización 
echo ""
echo "=============================="
echo " PASO 3: visualización"
echo "=============================="
python3 utils.py 2

#Abrir en navegador (si no hay entorno gráfico, sólo informa)
xdg-open ../results/plots/resultado_3d.html 2>/dev/null \
    || echo "[info] No hay navegador disponible. Abre: results/plots/resultado_3d.html desde el host."

echo ""
echo "Completado."