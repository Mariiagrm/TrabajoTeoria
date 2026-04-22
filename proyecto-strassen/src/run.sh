#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [ -d env ]; then
    PYTHON="env/bin/python3"
fi

# ── 1. Compilar todo ────────────────────────────────────────────────────────
echo "=============================="
echo " PASO 0: make all"
echo "=============================="
make all

# ── 2. Preprocesar imágenes y reconstrucción 3D (genera data_binaria/puntos_3d.bin) ──
echo ""
echo "=============================="
echo " PASO 1: reconstrucción 3D"
echo "=============================="
python3 utils.py 1

# ── 3. Rotación con Strassen (lee puntos_3d.bin, escribe puntos_rotados.csv) ──
echo ""
echo "=============================="
echo " PASO 2: Strassen app (C++)"
echo "=============================="
./strassen_app

# ── 4. Visualización ────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo " PASO 3: visualización"
echo "=============================="
python3 utils.py 2

#Abrir en navegador
xdg-open ../data_binaria/resultado_3d.html

echo ""
echo "Completado."