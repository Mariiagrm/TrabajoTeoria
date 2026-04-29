"""
Genera graficas a partir de results/tiempos.csv producido por benchmark_strassen.
  - speedup_vs_hilos.png   : speedup = t_sec / t_par, con linea ideal y=p
"""

import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent.parent / "results" / "tiempos.csv"
OUT_DIR  = HERE.parent.parent / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED = {"n", "hilos", "tiempo_secuencial_s", "tiempo_paralelo_s", "speedup"}


def cargar_filas(path: Path):
    """
    El CSV puede tener cabeceras antiguas mezcladas. Buscamos la cabecera nueva
    y parseamos solo las filas que van a partir de ella.
    """
    filas = []
    header = None
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if EXPECTED.issubset(set(row)):
                header = row
                continue
            if header is None:
                continue
            if len(row) != len(header):
                continue
            try:
                fila = dict(zip(header, row))
                filas.append({
                    "n":       int(fila["n"]),
                    "hilos":   int(fila["hilos"]),
                    "t_sec":   float(fila["tiempo_secuencial_s"]),
                    "t_par":   float(fila["tiempo_paralelo_s"]),
                    "speedup": float(fila["speedup"]),
                })
            except (ValueError, KeyError):
                continue
    if not filas:
        raise RuntimeError(f"No se encontraron filas validas en {path}")
    return filas


def agrupar(filas, clave):
    grupos = {}
    for f in filas:
        grupos.setdefault(f[clave], []).append(f)
    for k in grupos:
        grupos[k].sort(key=lambda x: x["hilos" if clave == "n" else "n"])
    return dict(sorted(grupos.items()))



def plot_speedup(por_n):
    plt.figure(figsize=(9, 6))
    max_p = 1
    for n, filas in por_n.items():
        xs = [f["hilos"] for f in filas]
        ys = [f["speedup"] for f in filas]
        max_p = max(max_p, max(xs))
        plt.plot(xs, ys, marker="o", label=f"n={n}")
    ideal_x = [1, max_p]
    plt.plot(ideal_x, ideal_x, "k--", alpha=0.5, label="Ideal (y=p)")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("Numero de hilos (p)")
    plt.ylabel("Speedup = t_sec / t_par")
    plt.title("Strassen paralelo: speedup vs hilos")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(title="Tamano")
    out = OUT_DIR / "speedup_vs_hilos.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"-> {out}")




def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"No existe {CSV_PATH}. Ejecuta antes ./benchmark_strassen")
    filas = cargar_filas(CSV_PATH)
    print(f"Cargadas {len(filas)} filas desde {CSV_PATH}")

    por_n     = agrupar(filas, "n")
    por_hilos = agrupar(filas, "hilos")

    plot_speedup(por_n)

    print(f"\nGraficas guardadas en {OUT_DIR}")


if __name__ == "__main__":
    main()
