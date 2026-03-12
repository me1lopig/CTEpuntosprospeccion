import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import math

def generar_distribucion_puntos(kmz_filename, distancia_metros):
    # --- 1. Leer Polígono del KMZ ---
    with zipfile.ZipFile(kmz_filename, 'r') as kmz:
        kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
        with kmz.open(kml_filename, 'r') as kml_file:
            root = ET.parse(kml_file).getroot()

    # Extraer coordenadas
    ns = {'kml': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    polygon = root.find('.//kml:Polygon', ns) if ns else root.find('.//Polygon')
    coords_text = (polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns) if ns 
                   else polygon.find('.//outerBoundaryIs/LinearRing/coordinates')).text.strip()
    
    # Lista de coordenadas (Lon, Lat)
    coords_geo = np.array([(float(p.split(',')[0]), float(p.split(',')[1])) 
                           for p in coords_text.split()])

    # --- 2. Proyección a Metros (Plana Local) ---
    # Usamos una aproximación local para no depender de librerías GIS pesadas
    lat_media = np.mean(coords_geo[:, 1])
    lat_rad = np.radians(lat_media)
    # Factor de conversión (aprox. para latitudes medias)
    m_por_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
    m_por_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    
    min_lon, min_lat = np.min(coords_geo, axis=0)
    
    # Convertir polígono a metros (X, Y)
    poly_x = (coords_geo[:, 0] - min_lon) * m_por_deg_lon
    poly_y = (coords_geo[:, 1] - min_lat) * m_por_deg_lat
    poly_m = np.column_stack((poly_x, poly_y))

    # --- 3. Generar Malla Hexagonal (Tresbolillo) ---
    min_x, min_y = np.min(poly_m, axis=0)
    max_x, max_y = np.max(poly_m, axis=0)
    
    dy = distancia_metros * math.sin(math.pi/3) # Altura triángulo equilátero
    dx = distancia_metros
    
    puntos_m = []
    filas = int((max_y - min_y) / dy) + 2
    cols = int((max_x - min_x) / dx) + 2
    
    for fila in range(filas):
        y = min_y + fila * dy
        offset_x = (distancia_metros / 2) if fila % 2 == 1 else 0
        for col in range(cols):
            x = min_x + col * dx + offset_x
            puntos_m.append([x, y])
            
    puntos_m = np.array(puntos_m)
    
    # --- 4. Filtrar Puntos (Dentro del Polígono) ---
    ruta_poligono = Path(poly_m)
    mascara = ruta_poligono.contains_points(puntos_m)
    puntos_finales_m = puntos_m[mascara]

    # Convertir puntos finales de vuelta a Lat/Lon para exportar o mostrar
    puntos_finales_lon = (puntos_finales_m[:, 0] / m_por_deg_lon) + min_lon
    puntos_finales_lat = (puntos_finales_m[:, 1] / m_por_deg_lat) + min_lat

    # --- 5. Graficar ---
    plt.figure(figsize=(10, 8))
    plt.plot(coords_geo[:, 0], coords_geo[:, 1], 'k-', label='Límite Polígono')
    plt.fill(coords_geo[:, 0], coords_geo[:, 1], alpha=0.1, color='green')
    plt.scatter(puntos_finales_lon, puntos_finales_lat, c='red', s=15, label='Puntos')
    plt.title(f"Distribución a {distancia_metros}m (Total: {len(puntos_finales_m)} puntos)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    filename_img = f"distribucion_{distancia_metros}m.png"
    plt.savefig(filename_img)
    print(f"Imagen guardada como: {filename_img}")
    return len(puntos_finales_m)

# --- EJECUTAR ---
archivo = 'Poligono de prueba.kmz'
DISTANCIA = 30  # <--- CAMBIA ESTE VALOR (en metros)
total = generar_distribucion_puntos(archivo, DISTANCIA)