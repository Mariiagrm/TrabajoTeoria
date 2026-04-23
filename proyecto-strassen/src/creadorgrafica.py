import matplotlib.pyplot as plt
import networkx as nx

log_data = """[División] Dividiendo matrices de tamaño 8 en hilo 1
[Tareas] Lanzando 7 tareas paralelas para n=8
[División] Dividiendo matrices de tamaño 4 en hilo 5
[División] Dividiendo matrices de tamaño 4 en hilo 3
[División] Dividiendo matrices de tamaño 4 en hilo 7
[División] Dividiendo matrices de tamaño 4 en hilo 1
[División] Dividiendo matrices de tamaño 4 en hilo 2
[Tareas] Lanzando 7 tareas paralelas para n=4
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 1
[Combinar] Combinando submatrices P para n=4
[División] Dividiendo matrices de tamaño 4 en hilo 0
[División] Dividiendo matrices de tamaño 4 en hilo 1
[Tareas] Lanzando 7 tareas paralelas para n=4
[Tareas] Lanzando 7 tareas paralelas para n=4
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 1
[Base] Alcanzado caso base n=2 en hilo 0
[Combinar] Combinando submatrices P para n=4
[Combinar] Combinando submatrices P para n=4
[Tareas] Lanzando 7 tareas paralelas para n=4
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 0
[Combinar] Combinando submatrices P para n=4
[Tareas] Lanzando 7 tareas paralelas para n=4
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 5
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 5
[Base] Alcanzado caso base n=2 en hilo 3
[Tareas] Lanzando 7 tareas paralelas para n=4
[Base] Alcanzado caso base n=2 en hilo 5
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 4
[Combinar] Combinando submatrices P para n=4
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 7
[Base] Alcanzado caso base n=2 en hilo 5
[Base] Alcanzado caso base n=2 en hilo 7
[Base] Alcanzado caso base n=2 en hilo 3
[Tareas] Lanzando 7 tareas paralelas para n=4
[Base] Alcanzado caso base n=2 en hilo 3
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 6
[Base] Alcanzado caso base n=2 en hilo 5
[Base] Alcanzado caso base n=2 en hilo 2
[Base] Alcanzado caso base n=2 en hilo 0
[Base] Alcanzado caso base n=2 en hilo 3
[Combinar] Combinando submatrices P para n=4
[Combinar] Combinando submatrices P para n=4
[Combinar] Combinando submatrices P para n=8"""

import re

# Parse logs
root_thread = None
n4_threads = []
n2_threads = []

for line in log_data.split('\n'):
    if "n=8 en hilo" in line or "tamaño 8 en hilo" in line:
        m = re.search(r"hilo (\d+)", line)
        if m: root_thread = m.group(1)
    elif "tamaño 4 en hilo" in line:
        m = re.search(r"hilo (\d+)", line)
        if m: n4_threads.append(m.group(1))
    elif "n=2 en hilo" in line:
        m = re.search(r"hilo (\d+)", line)
        if m: n2_threads.append(m.group(1))

# Build Tree Coordinates manually for 7-ary tree
# 1 root
# 7 children
# 49 grandchildren

G = nx.DiGraph()
pos = {}
labels = {}
colors = []

# Root
G.add_node("root")
pos["root"] = (0, 2)
labels["root"] = f"N=8\n(Hilo {root_thread})"
colors.append("lightblue")

# N=4 level
for i in range(7):
    node_id = f"n4_{i}"
    G.add_node(node_id)
    G.add_edge("root", node_id)
    x = -3 + i * 1.0
    pos[node_id] = (x, 1)
    t = n4_threads[i] if i < len(n4_threads) else "?"
    labels[node_id] = f"N=4\n(Hilo {t})"
    colors.append("lightgreen")
    
    # N=2 level
    for j in range(7):
        idx = i * 7 + j
        child_id = f"n2_{idx}"
        G.add_node(child_id)
        G.add_edge(node_id, child_id)
        
        # Position 49 nodes evenly
        x_c = -3.43 + idx * 0.143
        pos[child_id] = (x_c, 0)
        t2 = n2_threads[idx] if idx < len(n2_threads) else "?"
        labels[child_id] = f"{t2}" # Only thread to fit
        colors.append("salmon")

plt.figure(figsize=(16, 8))
nx.draw(G, pos, with_labels=False, node_size=1500, node_color=colors, edge_color="gray", arrows=True)

# Draw custom labels
for node, (x, y) in pos.items():
    if "n2_" in node:
        plt.text(x, y, labels[node], fontsize=8, ha='center', va='center')
    else:
        plt.text(x, y, labels[node], fontsize=10, ha='center', va='center', fontweight='bold')

plt.title("Árbol de Recursión de Strassen (N=8) - Asignación de Hilos", fontsize=16)
plt.figtext(0.5, 0.05, "Naranja: Casos Base (N=2, mostrando el Hilo), Verde: Tareas N=4, Azul: Raíz N=8", ha="center", fontsize=12)
import os
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "plots", "arbol_strassen.png")
os.makedirs(os.path.dirname(_out), exist_ok=True)
plt.savefig(_out, bbox_inches="tight")
plt.close()