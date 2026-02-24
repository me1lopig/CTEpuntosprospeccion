import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import math
import utm
import pandas as pd
import io

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Generador de Malla Agrícola/Topográfica", layout="wide")
st.title("📍 Generador de Puntos sobre Polígono (KMZ)")
st.markdown("Sube tu archivo KMZ, define la distancia entre puntos y descarga las coordenadas en formato UTM.")

def procesar_kmz(uploaded_file, distancia_metros):
    # --- 1. Leer Polígono del KMZ (desde memoria) ---
    with zipfile.ZipFile(uploaded_file, 'r') as kmz:
        kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
        with kmz.open(kml_filename, 'r') as kml_file:
            root = ET.parse(kml_file).getroot()

    ns = {'kml': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    polygon = root.find('.//kml:Polygon', ns) if ns else root.find('.//Polygon')
    coords_text = (polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns) if ns 
                   else polygon.find('.//outerBoundaryIs/LinearRing/coordinates')).text.strip()
    
    coords_geo = np.array([(float(p.split(',')[0]), float(p.split(',')[1])) for p in coords_text.split()])

    # --- 2. Proyección Local a Metros ---
    lat_media = np.mean(coords_geo[:, 1])
    lat_rad = np.radians(lat_media)
    m_por_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
    m_por_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    
    min_lon, min_lat = np.min(coords_geo, axis=0)
    poly_x = (coords_geo[:, 0] - min_lon) * m_por_deg_lon
    poly_y = (coords_geo[:, 1] - min_lat) * m_por_deg_lat
    poly_m = np.column_stack((poly_x, poly_y))

    # --- 3. Generar Malla Hexagonal ---
    min_x, min_y = np.min(poly_m, axis=0)
    max_x, max_y = np.max(poly_m, axis=0)
    
    dy = distancia_metros * math.sin(math.pi/3)
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
    
    # --- 4. Filtrar Puntos ---
    ruta_poligono = Path(poly_m)
    mascara = ruta_poligono.contains_points(puntos_m)
    puntos_finales_m = puntos_m[mascara]

    puntos_finales_lon = (puntos_finales_m[:, 0] / m_por_deg_lon) + min_lon
    puntos_finales_lat = (puntos_finales_m[:, 1] / m_por_deg_lat) + min_lat

    # --- 5. Exportación a UTM y DataFrame ---
    datos_exportar = []
    for lon, lat in zip(puntos_finales_lon, puntos_finales_lat):
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        datos_exportar.append({
            'ID_Punto': len(datos_exportar) + 1,
            'Latitud': lat,
            'Longitud': lon,
            'UTM_X_Este': round(easting, 2),
            'UTM_Y_Norte': round(northing, 2),
            'Huso_UTM': f"{zone_number}{zone_letter}"
        })
    df = pd.DataFrame(datos_exportar)

    # --- 6. Crear Gráfico ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(coords_geo[:, 0], coords_geo[:, 1], 'k-', label='Límite Polígono')
    ax.fill(coords_geo[:, 0], coords_geo[:, 1], alpha=0.1, color='green')
    ax.scatter(puntos_finales_lon, puntos_finales_lat, c='red', s=15, label='Puntos')
    ax.set_title(f"Distribución a {distancia_metros}m (Total: {len(puntos_finales_m)} puntos)")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.axis('equal')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    return df, fig

# --- INTERFAZ DE USUARIO (UI) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Configuración")
    archivo_kmz = st.file_uploader("Sube tu archivo .kmz", type=['kmz'])
    distancia = st.number_input("Distancia entre puntos (metros):", min_value=1.0, value=30.0, step=1.0)
    
with col2:
    st.header("2. Resultados")
    if archivo_kmz is not None:
        with st.spinner('Calculando puntos y coordenadas UTM...'):
            df_resultados, figura = procesar_kmz(archivo_kmz, distancia)
            
            # Mostrar métricas
            st.success(f"¡Cálculo completado! Se han generado **{len(df_resultados)}** puntos dentro del polígono.")
            
            # Mostrar gráfico
            st.pyplot(figura)
            
            # Botón para descargar CSV
            csv = df_resultados.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
            st.download_button(
                label="📥 Descargar Coordenadas (CSV / Excel)",
                data=csv,
                file_name=f'coordenadas_{int(distancia)}m.csv',
                mime='text/csv',
            )
            
            # Vista previa de datos
            st.write("Vista previa de los datos:")
            st.dataframe(df_resultados.head())
    else:
        st.info("👈 Por favor, sube un archivo KMZ en el panel de la izquierda para comenzar.")