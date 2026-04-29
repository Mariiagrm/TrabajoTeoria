import os
import time
import numpy as np
from pathlib import Path
from PIL import Image # Usaremos Pillow para JPG/PNG
import cv2 # OpenCV para procesamiento de imágenes y reconstrucción 3D
import matplotlib.pyplot as plt
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor


def ruta(subruta):
    # 1. Navegación de rutas dinámica (de /app/src a /app/data/test)
    archivo_actual = Path(__file__).resolve()
    directorio_app = archivo_actual.parent.parent # Sube de src/ a app/
    
    return directorio_app / subruta
  

def reconstruccion_directorio(ruta_directorio, K):
    """
    Reconstrucción 3D secuencial para todas las imágenes de un directorio.
    Procesa pares consecutivos (0-1, 1-2, ...) acumulando la pose global
    para que todos los puntos queden en el mismo sistema de referencia.

    Parámetros:
    - ruta_directorio: Ruta al directorio con las imágenes.
    - K: Matriz intrínseca de la cámara (3x3 numpy array).

    Retorna:
    - puntos_3d_global: Nube de puntos acumulada (Nx3 numpy array).
    """
    

    print(f"--- Iniciando Reconstrucción Secuencial con imágenes ---")

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    # Precomputar descriptores para todas las imágenes

    
    print("Procedemos a limpiar las imágenes con la gaussiana y el umbral de otsu...")
    resultado = subprocess.run(
        ["./utils", str(ruta_directorio)],
        
    )
    if resultado.returncode != 0:
        print(f"Error al ejecutar utils (código {resultado.returncode}), intentando compilar...")
        subprocess.run("make utils", shell=True)
        subprocess.run(["./utils", str(ruta_directorio)])

    print("Preprocesando imágenes y calculando descriptores SIFT... (esto puede tardar un poco)")
    print("Alternativa más rapida con kernel de CUDA, si tienes GPU compatible [S/N]: \n ", end="")
    
    keypoints_list, descriptores_list = [], []

    proj_dir = Path(__file__).resolve().parent.parent  # project root
    ruta_procesadas = proj_dir / "data" / "clean_images"
    print(f"Buscando imágenes preprocesadas en: {ruta_procesadas}")

    # Leer imágenes procesadas (pobladas por cuda_sift o por la ruta CPU)
    archivos = sorted([f for f in os.listdir(ruta_procesadas)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    
    respuesta = input().strip().lower()

    if len(archivos) < 2:
        raise ValueError("Se necesitan al menos 2 imágenes en el directorio.")

    if respuesta == 's':
        print("Ejecutando preprocesamiento optimizado con GPU (CUDA)...")
        print(f"Encontradas {len(archivos)} imágenes. Calculando descriptores SIFT con CUDA...")

        time_start = time.time()
        resultado = subprocess.run(
            ["./cuda_sift", str(ruta_procesadas)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time_end = time.time()
        if resultado.returncode != 0:
            print(f"Error al ejecutar cuda_sift (código {resultado.returncode}):\n{resultado.stderr}")
            sys.exit(1)
        print(resultado.stdout)
        print(f"Tiempo total de cálculo de descriptores SIFT: {time_end - time_start:.2f} segundos")

      
        # Leer los .sift binarios escritos por cuda_sift y reconstruir
        # (keypoints, descriptors) 
        kp_dtype = np.dtype([
            ("x",        np.float32),
            ("y",        np.float32),
            ("sigma",    np.float32),
            ("angle",    np.float32),
            ("response", np.float32),
            ("octave",   np.int32),
        ])
        resultados = []
        for archivo in archivos:
            
            cuda_trans = proj_dir / "data" / "procesadas_filtros_cuda"

            sift_path = cuda_trans / (Path(archivo).stem + ".sift")
            with open(sift_path, "rb") as f:
                n = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                if n == 0:
                    resultados.append(([], None))
                    continue
                kp_raw = np.fromfile(f, dtype=kp_dtype, count=n)
                desc = np.fromfile(f, dtype=np.float32, count=n * 128).reshape(n, 128)
            keypoints = [
                cv2.KeyPoint(
                    x=float(k["x"]),
                    y=float(k["y"]),
                    size=float(k["sigma"]) * 2.0,
                    angle=float(np.degrees(k["angle"])),
                    response=float(k["response"]),
                    octave=int(k["octave"]),
                )
                for k in kp_raw
            ]
            resultados.append((keypoints, desc))
        print("Descriptores SIFT calculados (CUDA).")


    else:
        print("Ejecutando preprocesamiento secuencial en CPU...")
        print(f"Encontradas {len(archivos)} imágenes. Calculando descriptores SIFT con 2 hilos...")

        def _procesar(archivo):
            img = cv2.imread(str(ruta_procesadas / archivo), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"No se pudo cargar: {archivo}")
            sift_local = cv2.SIFT_create()
            return sift_local.detectAndCompute(img, None)

        time_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            resultados = list(executor.map(_procesar, archivos))
        time_end = time.time()
        print(f"Tiempo total de cálculo de descriptores SIFT: {time_end - time_start:.2f} segundos")

 
    keypoints_list   = [r[0] for r in resultados]
    descriptores_list = [r[1] for r in resultados]

    #Cámara 0 como origen del sistema de referencia global
    R_global = np.eye(3)
    t_global = np.zeros((3, 1))
    todos_puntos_3d = []

    for i in range(len(archivos) - 1):
        print(f"\nPar {i+1}/{len(archivos)-1}: {archivos[i]} <-> {archivos[i+1]}")

        kp1, des1 = keypoints_list[i], descriptores_list[i]
        kp2, des2 = keypoints_list[i + 1], descriptores_list[i + 1]

        matches = flann.knnMatch(des1, des2, k=2)
        buenos_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

        if len(buenos_matches) < 8:
            print(f"   -> Solo {len(buenos_matches)} matches, saltando par.")
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in buenos_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in buenos_matches])
        print(f"   -> Puntos comunes: {len(pts1)}")

        E, mascara = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None or mascara is None:
            print("   -> Matriz esencial no calculable, saltando par.")
            continue

        pts1_inliers = pts1[mascara.ravel() == 1]
        pts2_inliers = pts2[mascara.ravel() == 1]

        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

        # Proyección en coordenadas globales
        P1 = K @ np.hstack((R_global, t_global))
        R_global_new = R_rel @ R_global
        t_global_new = R_rel @ t_global + t_rel
        P2 = K @ np.hstack((R_global_new, t_global_new))

        puntos_4d = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
        puntos_3d = (puntos_4d[:3] / puntos_4d[3]).T
        todos_puntos_3d.append(puntos_3d)

        R_global = R_global_new
        t_global = t_global_new

    if not todos_puntos_3d:
        raise RuntimeError("No se generaron puntos 3D.")

    puntos_3d_final = np.vstack(todos_puntos_3d)
    print(f"\n--- Reconstrucción finalizada: {len(puntos_3d_final)} puntos 3D en total ---")
    return puntos_3d_final

#Para evitar errores de OOM 
#               total       usado       libre  compartido   búf/caché  disponible
#Mem:            15Gi       3,9Gi        10Gi       526Mi       1,4Gi        11Gi
#Inter:         4,0Gi       2,1Gi       1,9Gi
# Como tengo (dejando margen para el SO) 7 * 10^9 GB = 28/3 * N^2 * 8 bytes => N = sqrt((7 * 10^9 * 3) / (28 * 8)) ≈ 16384
def exportar_puntos_a_binario(puntos_3d, ruta_salida, max_puntos=16384):
    """
    Guarda la nube de puntos 3D (Nx3) en un archivo binario para C.
    Limita a max_puntos para evitar matrices NxN gigantes en Strassen.
    """
    print(f"Exportando {len(puntos_3d)} puntos 3D a binario...")
    print("Eliminar outliers: quedarse con puntos dentro de 3 desviaciones estándar en cada eje")
    antes = len(puntos_3d)
    for eje in range(3):
        media = np.median(puntos_3d[:, eje])
        mad   = np.median(np.abs(puntos_3d[:, eje] - media))
        limite = 6 * mad if mad > 0 else 3 * np.std(puntos_3d[:, eje])
        mascara = np.abs(puntos_3d[:, eje] - media) < limite
        puntos_3d = puntos_3d[mascara]
    print(f"Filtrado de outliers: {antes} → {len(puntos_3d)} puntos conservados.")

    if len(puntos_3d) > max_puntos:
        indices = np.random.choice(len(puntos_3d), max_puntos, replace=False)
        puntos_3d = puntos_3d[indices]
        print(f"Submuestreados {max_puntos} puntos de la nube original.")

    # 1. Crear directorios primero
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    with open(ruta_salida, 'wb') as f:
        # 2. Guardar la forma (filas, columnas) como enteros de 32 bits
        np.array(puntos_3d.shape, dtype=np.int32).tofile(f)
        
        # 3. Guardar los datos de la matriz como double (64 bits)
        puntos_3d.astype(np.float64).tofile(f)
        
    print(f"Nube de puntos guardada para C en: {ruta_salida} | Dimensiones: {puntos_3d.shape}")

def visualizar_resultado():
    print("--- Iniciando visualización de resultados ---")
    import plotly.graph_objects as go

    # 1. Rutas de los archivos
    ruta_csv = ruta("results/cloud_points/puntos_rotados.csv")
    ruta_bin = ruta("results/cloud_points/puntos_originales.csv")

    if not ruta_csv.exists():
        print(f"Error: No se encuentra el archivo {ruta_csv}")
        print("Asegúrate de ejecutar primero el programa en C++.")
        return

    # 2. Cargar CSVs (recortados a Nx3 por el código C++)
    print("Cargando nube de puntos rotada...")
    puntos_rotados = np.genfromtxt(ruta_csv, delimiter=",", invalid_raise=False)
    puntos_rotados = puntos_rotados[:, :3]

    print("Cargando nube de puntos original...")
    puntos_originales = np.genfromtxt(ruta_bin, delimiter=",", invalid_raise=False)
    puntos_originales = puntos_originales[:, :3]

    # 3. HTML interactivo con plotly (funciona sin display, abre en cualquier navegador)
    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter3d(
        x=puntos_originales[:, 0],
        y=puntos_originales[:, 2],
        z=puntos_originales[:, 1],
        mode='markers',
        marker=dict(size=2, color='gray', opacity=0.5),
        name='Originales'
    ))
    fig_html.add_trace(go.Mesh3d(
        x=puntos_rotados[:, 0],
        y=puntos_rotados[:, 2],
        z=puntos_rotados[:, 1],
        alphahull=5,
        color='red',
        opacity=0.5,
        flatshading=True,
        name='Rotados NX3 x 3X3'
    ))
    fig_html.update_layout(
        title='Reconstrucción 3D: Original vs Rotación multiplicacion de matrices',
        scene=dict(
            xaxis_title='Eje X',
            yaxis_title='Eje Z (Profundidad)',
            zaxis_title='Eje Y (Altura)'
        )
    )
    ruta_html = ruta("results/plots/resultado_3d.html")
    ruta_html.parent.mkdir(parents=True, exist_ok=True)
    fig_html.write_html(str(ruta_html))
    print(f"Gráfico interactivo guardado en: {ruta_html}  (abrir en navegador)")

    # 4. PNG estático como respaldo
    fig_png = plt.figure(figsize=(12, 8))
    ax = fig_png.add_subplot(111, projection='3d')
    ax.scatter(puntos_originales[:, 0], puntos_originales[:, 2], puntos_originales[:, 1],
               c='gray', marker='.', s=20, alpha=0.5, label='Originales')
    ax.plot_trisurf(puntos_rotados[:, 0], puntos_rotados[:, 2], puntos_rotados[:, 1],
                    color='red', alpha=0.5, linewidth=0.2, edgecolor='darkred',
                    label='Rotados NX3 x 3X3')
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Z (Profundidad)')
    ax.set_zlabel('Eje Y (Altura)')
    ax.set_title('Reconstrucción 3D: Original vs Rotación multiplicacion de matrices')
    ax.set_box_aspect([1, 1, 1])
    plt.legend()
    plt.tight_layout()
    ruta_guardado = ruta("results/plots/resultado_3d.png")
    plt.savefig(ruta_guardado)
    print(f"Gráfico estático guardado en:     {ruta_guardado}")
    if os.environ.get("DISPLAY"):
        plt.show()


# Ejecución
if __name__ == "__main__":

# Verificar que se haya pasado un parámetro
    if len(sys.argv) < 2:
        print("Uso: python3 script.py [parametro]")
        print("   1: Filtrado, reconstrucción 3D y creación de binario")
        print("   2: Visualización de resultados (Original vs Rotado)")
        sys.exit(1)

    parametro = sys.argv[1]
 


    if parametro == "1":
        print("--- PASO 1: Procesamiento de Imágenes y Generación de Binario ---")

        # exportar_imagenes_a_binario('data/train', 'data_binaria/training')
        # Matriz intrínseca estimada para imágenes de 5712x4284 px.
        # fx=fy ~ image_width es una aproximación razonable sin calibración.
        # cx, cy = mitad exacta de la imagen.
        IMG_W, IMG_H = 5712, 4284
        f = float(IMG_W)         
        cx = IMG_W / 2.0           # 2856.0
        cy = IMG_H / 2.0           # 2142.0
        K_ejemplo = np.array([[f,   0.0, cx],
                              [0.0,   f, cy],
                              [0.0, 0.0, 1.0]])
                            

        # 2. Reconstrucción 3D desde todas las fotos del directorio
        ruta_fotos = ruta("data")
        puntos_3d_resultado = reconstruccion_directorio(ruta_fotos, K_ejemplo)

        # 3. Mandar el resultado 3D a C
        exportar_puntos_a_binario(puntos_3d_resultado, ruta("results/cloud_points/puntos_3d.bin"))
    elif parametro =="2":
        print("--- PASO 3: Visualización de Resultados tras multiplicacion de matrices ---")

        visualizar_resultado()

    else:
        print(f"Parámetro '{parametro}' no reconocido. Use 1 o 2.")



    