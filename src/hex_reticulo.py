
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import os

# ==============================
# Utilidades básicas
# ==============================

def punto_en_poligono(punto, poligono):
    return poligono.contains(Point(punto))

# ==============================
# Generación de retícula hexagonal con validaciones
# ==============================

def generar_reticulo_hexagonal(vertices, distancia_exacta, punto_inicial=None):
    """
    Genera puntos en retícula hexagonal dentro de un polígono.

    - vertices: lista de tuplas (x, y) en orden perimetral
    - distancia_exacta: distancia entre centros de hexágonos adyacentes
    - punto_inicial: tupla (x, y) opcional; si está dentro, se usa como centro del retículo

    Retorna: lista de tuplas (x, y) de puntos dentro del polígono
    """
    poligono = Polygon(vertices)

    # Métricas del polígono
    bounds = poligono.bounds
    ancho_poligono = bounds[2] - bounds[0]
    alto_poligono = bounds[3] - bounds[1]
    area_poligono = poligono.area

    print("=== ANÁLISIS DEL POLÍGONO ===")
    print("Ancho del polígono: " + str(round(ancho_poligono, 2)) + " unidades")
    print("Alto del polígono: " + str(round(alto_poligono, 2)) + " unidades")
    print("Área del polígono: " + str(round(area_poligono, 2)) + " unidades²")
    print("Distancia solicitada: " + str(distancia_exacta) + " unidades")

    # Advertencia si la distancia es mayor que la dimensión mínima
    dimension_minima = min(ancho_poligono, alto_poligono)
    if distancia_exacta > dimension_minima:
        print("")
        print("⚠️  ADVERTENCIA: La distancia solicitada (" + str(distancia_exacta) + ") es mayor que la dimensión mínima del polígono (" + str(round(dimension_minima, 2)) + ")")
        print("Es muy probable que no se generen puntos o se generen muy pocos.")
        print("Recomendación: Usar una distancia menor a " + str(round(dimension_minima * 0.8, 2)) + " unidades")

    # Estimación de conteo (área de celda hexagonal)
    area_por_punto = distancia_exacta ** 2 * np.sqrt(3) / 2
    puntos_estimados = int(area_poligono / area_por_punto)
    print("Puntos estimados (aproximado): " + str(puntos_estimados))

    if puntos_estimados < 1:
        print("")
        print("❌ ERROR: La distancia es demasiado grande para este polígono.")
        print("No se generarán puntos dentro del polígono.")
        print("Distancia máxima recomendada: " + str(round(np.sqrt(area_poligono * 2 / np.sqrt(3)), 2)) + " unidades")
        return []

    print("")
    print("=== GENERANDO RETÍCULO ===")

    puntos = []

    # Centro: punto inicial válido o centroide
    if punto_inicial is not None and punto_en_poligono(punto_inicial, poligono):
        centro = np.array(punto_inicial)
        print("Usando punto inicial como centro: (" + str(round(centro[0], 2)) + ", " + str(round(centro[1], 2)) + ")")
    else:
        centro = np.array(poligono.centroid.coords[0])
        print("Usando centroide como centro: (" + str(round(centro[0], 2)) + ", " + str(round(centro[1], 2)) + ")")

    # Geometría del retículo hexagonal
    paso_vertical = distancia_exacta * np.sqrt(3) / 2.0

    # Estimar radio máximo desde el centro a los vértices para acotar el grid
    radio_max = 0.0
    for v in vertices:
        d = np.linalg.norm(np.array(v) - centro)
        if d > radio_max:
            radio_max = d

    num_filas = int(np.ceil(radio_max / paso_vertical)) + 2

    for i in range(-num_filas, num_filas + 1):
        for j in range(-num_filas, num_filas + 1):
            x = centro[0] + distancia_exacta * (j + (i % 2) * 0.5)
            y = centro[1] + paso_vertical * i
            candidato = (x, y)
            if punto_en_poligono(candidato, poligono):
                puntos.append(candidato)

    if len(puntos) == 0:
        print("")
        print("❌ RESULTADO: No se generaron puntos dentro del polígono.")
        print("La distancia especificada (" + str(distancia_exacta) + ") es demasiado grande.")
        print("Intente con una distancia menor.")
    elif len(puntos) < 3:
        print("")
        print("⚠️  ADVERTENCIA: Se generaron muy pocos puntos (" + str(len(puntos)) + "). Considere reducir la distancia.")
    else:
        print("")
        print("✅ ÉXITO: Se generaron " + str(len(puntos)) + " puntos correctamente.")

    return puntos

# ==============================
# Visualización con conteo de puntos
# ==============================

def plot_poligono_y_puntos(vertices, puntos, punto_inicial=None, distancia_exacta=None):
    poligono = Polygon(vertices)

    x, y = poligono.exterior.xy
    plt.fill(x, y, alpha=0.5, fc='lightblue', ec='darkblue', label='Polígono')

    num_puntos_generados = len(puntos)
    num_punto_inicial = 1 if (punto_inicial is not None and punto_en_poligono(punto_inicial, poligono)) else 0
    num_total_puntos = num_puntos_generados + num_punto_inicial

    if num_puntos_generados > 0:
        puntos_array = np.array(puntos)
        plt.scatter(puntos_array[:, 0], puntos_array[:, 1], c='green', label='Puntos generados (' + str(num_puntos_generados) + ')', s=50)

    if num_punto_inicial > 0:
        plt.scatter(punto_inicial[0], punto_inicial[1], c='red', label='Punto inicial (1)', zorder=5, s=100)

    titulo = "Retículo hexagonal - Total: " + str(num_total_puntos) + " puntos"
    if distancia_exacta is not None:
        titulo = titulo + " (d=" + str(distancia_exacta) + ")"
    plt.title(titulo)

    plt.text(0.02, 0.98, "Puntos dentro del polígono: " + str(num_total_puntos),
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if num_punto_inicial > 0:
        plt.text(0.02, 0.92, "Punto inicial: 1",
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.text(0.02, 0.86, "Puntos generados: " + str(num_puntos_generados),
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

# ==============================
# Utilidad: generar polígono estrella (pruebas)
# ==============================

def generar_estrella(centro, radio_exterior, radio_interior, num_puntas=5):
    vertices = []
    angulo_paso = 2 * np.pi / (num_puntas * 2)
    for i in range(num_puntas * 2):
        angulo = i * angulo_paso - np.pi / 2.0
        radio = radio_exterior if i % 2 == 0 else radio_interior
        x = centro[0] + radio * np.cos(angulo)
        y = centro[1] + radio * np.sin(angulo)
        vertices.append((x, y))
    return vertices

# ==============================
# Carga de Excel y utilidades de I/O
# ==============================

def cargar_vertices_desde_excel(ruta_excel):
    """
    Carga un Excel y detecta automáticamente dos columnas numéricas para X e Y.
    Retorna lista de vértices (x, y) en el orden en que aparecen, nombres de columnas y el dataframe.
    """
    df = pd.read_excel(ruta_excel)

    # Detectar columnas numéricas
    cols_num = [c for c in df.columns if df[c].dtype.kind in ['i', 'u', 'f']]
    if len(cols_num) < 2:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        cols_num = [c for c in df.columns if df[c].dtype.kind in ['i', 'u', 'f']]

    if len(cols_num) < 2:
        raise ValueError('No se encontraron al menos dos columnas numéricas en el archivo de Excel.')

    col_x = cols_num[0]
    col_y = cols_num[1]

    xs = df[col_x].dropna().values
    ys = df[col_y].dropna().values

    if len(xs) != len(ys):
        n = min(len(xs), len(ys))
        xs = xs[:n]
        ys = ys[:n]

    vertices = list(zip(xs, ys))
    return vertices, (col_x, col_y), df


def exportar_puntos_csv(puntos, ruta_csv):
    df = pd.DataFrame(puntos, columns=['x', 'y'])
    df.to_csv(ruta_csv, index=False)
    print('Exportado CSV en ' + ruta_csv)

# ==============================
# Ejecución principal (usa poligono.xlsx por defecto)
# ==============================

def main():
    ruta_excel = 'poligono.xlsx'
    if not os.path.exists(ruta_excel):
        print('No se encontró el archivo ' + ruta_excel + '. Colócalo en el mismo directorio y vuelve a ejecutar.')
        return

    print('Cargando vértices desde ' + ruta_excel)
    vertices, (col_x, col_y), df = cargar_vertices_desde_excel(ruta_excel)
    print('Primeras filas del Excel:')
    print(df.head())
    print('Columnas detectadas para coordenadas: ' + str(col_x) + ' y ' + str(col_y))

    distancia = 45.0
    punto_inicial = None  # Cambia a una tupla (x, y) si deseas forzar un punto inicial

    puntos = generar_reticulo_hexagonal(vertices, distancia, punto_inicial=punto_inicial)

    # Visualizar
    plot_poligono_y_puntos(vertices, puntos, punto_inicial=punto_inicial, distancia_exacta=distancia)

    # Exportar
    salida_csv = 'puntos_generados.csv'
    exportar_puntos_csv(puntos, salida_csv)

if __name__ == '__main__':
    main()
