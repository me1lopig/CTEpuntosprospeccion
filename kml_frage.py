import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import math
import pandas as pd
import sys

# Attempt to import pyproj for accurate UTM conversion
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Attempt to import ezdxf for DXF creation
try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False

print(f"Libraries available: pyproj={HAS_PYPROJ}, ezdxf={HAS_EZDXF}")

def parse_kmz(filename):
    """Parses KMZ to get polygon coordinates (assuming single polygon for simplicity)."""
    with zipfile.ZipFile(filename, 'r') as kmz:
        kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
        with kmz.open(kml_filename, 'r') as kml_file:
            tree = ET.parse(kml_file)
            root = tree.getroot()

    ns = {'kml': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    prefix = 'kml:' if ns else ''

    polygons = []
    for pm in root.findall(f'.//{prefix}Placemark', ns):
        name = pm.find(f'{prefix}name', ns).text if pm.find(f'{prefix}name', ns) is not None else "Unnamed"
        poly = pm.find(f'.//{prefix}Polygon', ns)
        if poly is not None:
            coord_tag = poly.find(f'.//{prefix}outerBoundaryIs/{prefix}LinearRing/{prefix}coordinates', ns)
            if coord_tag is not None and coord_tag.text:
                coords = []
                for p in coord_tag.text.strip().split():
                    parts = p.split(',')
                    coords.append((float(parts[0]), float(parts[1]))) # Lon, Lat
                polygons.append({'name': name, 'coords': np.array(coords)})
    return polygons

def latlon_to_utm_manual(lat, lon):
    """
    Manual conversion from Lat/Lon (WGS84) to UTM.
    Useful if pyproj is not available.
    """
    a = 6378137
    f = 1/298.257223563
    k0 = 0.9996
    
    phi = math.radians(lat)
    lam = math.radians(lon)
    
    zone_number = int((lon + 180) / 6) + 1
    lam0 = math.radians(((zone_number - 1) * 6 - 180) + 3)
    
    e = math.sqrt(2*f - f*f)
    e2 = e*e
    e4 = e2*e2
    e6 = e4*e2
    
    nu = a / math.sqrt(1 - e2 * math.sin(phi)**2)
    
    A = (lam - lam0) * math.cos(phi)
    A2 = A*A
    A3 = A2*A
    A4 = A3*A
    A5 = A4*A
    A6 = A5*A
    
    T = math.tan(phi)**2
    C = e2 / (1 - e2) * math.cos(phi)**2
    
    M = a * ((1 - e2/4 - 3*e4/64 - 5*e6/256) * phi - 
             (3*e2/8 + 3*e4/32 + 45*e6/1024) * math.sin(2*phi) + 
             (15*e4/256 + 45*e6/1024) * math.sin(4*phi) - 
             (35*e6/3072) * math.sin(6*phi))
    
    easting = 500000 + k0 * nu * (A + (1-T+C)*A3/6 + (5-18*T+T*T+72*C-58*e2)*A5/120)
    northing = k0 * (M + nu * math.tan(phi) * (A2/2 + (5-T+9*C+4*C*C)*A4/24 + (61-58*T+T*T+600*C-330*e2)*A6/720))
    
    if lat < 0:
        northing += 10000000
        
    return easting, northing, zone_number

def get_utm_coords(lat, lon):
    if HAS_PYPROJ:
        # Calculate zone
        zone_number = int((lon + 180) / 6) + 1
        south = lat < 0
        proj_str = f"+proj=utm +zone={zone_number} +ellps=WGS84 {'+south' if south else ''}"
        proj = pyproj.Proj(proj_str)
        easting, northing = proj(lon, lat)
        return easting, northing, zone_number
    else:
        return latlon_to_utm_manual(lat, lon)

def generate_simple_dxf(filename, polygon_coords, points, points_ids):
    """
    Generates a simple DXF file (R12 format) manually without ezdxf.
    """
    with open(filename, 'w') as f:
        # Header
        f.write("0\nSECTION\n2\nHEADER\n0\nENDSEC\n")
        
        # Tables (Layers)
        f.write("0\nSECTION\n2\nTABLES\n")
        f.write("0\nTABLE\n2\nLAYER\n")
        
        # Layer: contorno (Color 1 = Red)
        f.write("0\nLAYER\n2\ncontorno\n70\n0\n62\n1\n6\nCONTINUOUS\n0\n")
        # Layer: puntos_de_prospeccion (Color 3 = Green)
        f.write("0\nLAYER\n2\npuntos_de_prospeccion\n70\n0\n62\n3\n6\nCONTINUOUS\n0\n")
        # Layer: etiquetas (Color 7 = White)
        f.write("0\nLAYER\n2\netiquetas\n70\n0\n62\n7\n6\nCONTINUOUS\n0\n")
        
        f.write("0\nENDTAB\n0\nENDSEC\n")
        
        # Entities
        f.write("0\nSECTION\n2\nENTITIES\n")
        
        # 1. Polygon (LWPOLYLINE)
        f.write("0\nPOLYLINE\n8\ncontorno\n66\n1\n") # 66=1 means vertex follow
        for lon, lat in polygon_coords:
            # Convert to UTM for DXF? Usually CAD is in meters (UTM) or local.
            # User asked for UTM in CSV. CAD is usually projected.
            # We will use UTM coordinates for the CAD drawing so it's measurable.
            x, y, _ = get_utm_coords(lat, lon)
            f.write(f"0\nVERTEX\n8\ncontorno\n10\n{x}\n20\n{y}\n30\n0.0\n")
        f.write("0\nSEQEND\n")
        
        # 2. Points
        for idx, (lat, lon) in zip(points_ids, points):
            x, y, _ = get_utm_coords(lat, lon)
            
            # Point entity
            f.write(f"0\nPOINT\n8\npuntos_de_prospeccion\n10\n{x}\n20\n{y}\n30\n0.0\n")
            
            # Text entity (Number)
            f.write(f"0\nTEXT\n8\npuntos_de_prospeccion\n")
            f.write(f"10\n{x + 2}\n20\n{y + 2}\n30\n0.0\n") # Offset text slightly
            f.write(f"40\n5.0\n") # Text height
            f.write(f"1\n{idx}\n") # Text content
            
        f.write("0\nENDSEC\n0\nEOF\n")

def generate_hex_grid(polygon_coords_latlon, distance_meters):
    # Convert to meters (Local projection for grid generation)
    ref_lat = np.mean(polygon_coords_latlon[:, 1])
    lat_rad = np.radians(ref_lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    
    min_lon = np.min(polygon_coords_latlon[:, 0])
    min_lat = np.min(polygon_coords_latlon[:, 1])
    
    poly_x = (polygon_coords_latlon[:, 0] - min_lon) * m_per_deg_lon
    poly_y = (polygon_coords_latlon[:, 1] - min_lat) * m_per_deg_lat
    poly_m = np.column_stack((poly_x, poly_y))
    
    # Generate Grid
    min_x, min_y = np.min(poly_m, axis=0)
    max_x, max_y = np.max(poly_m, axis=0)
    
    dy = distance_meters * math.sin(math.pi/3)
    dx = distance_meters
    
    xs, ys = [], []
    rows = int((max_y - min_y) / dy) + 2
    cols = int((max_x - min_x) / dx) + 2
    
    for row in range(rows):
        y = min_y + row * dy
        x_offset = (distance_meters / 2) if row % 2 == 1 else 0
        for col in range(cols):
            x = min_x + col * dx + x_offset
            xs.append(x)
            ys.append(y)
            
    points_m = np.column_stack((xs, ys))
    
    # Filter
    path = Path(poly_m)
    mask = path.contains_points(points_m, radius=0.1)
    points_m = points_m[mask]
    
    # Convert back to LatLon
    if len(points_m) > 0:
        points_lon = (points_m[:, 0] / m_per_deg_lon) + min_lon
        points_lat = (points_m[:, 1] / m_per_deg_lat) + min_lat
        return np.column_stack((points_lat, points_lon))
    else:
        return np.array([])

# --- Main Processing ---
filename = 'Poligono de prueba.kmz'
DISTANCE = 50 # Defaulting to 50 as per discussion, or keep context.

polygons = parse_kmz(filename)
# Assuming 1 polygon for simplicity or process all
all_data = []
all_points_latlon = []
all_points_ids = []

# Prepare Plot
plt.figure(figsize=(10, 10))

global_id = 1

if polygons:
    poly = polygons[0] # Process the first/main one
    coords = poly['coords']
    
    # Plot Polygon
    plt.plot(coords[:,0], coords[:,1], 'k-', linewidth=2)
    plt.fill(coords[:,0], coords[:,1], alpha=0.1, color='gray')
    
    # Generate Points
    points_latlon = generate_hex_grid(coords, DISTANCE)
    
    for pt in points_latlon:
        lat, lon = pt
        
        # Calculate UTM
        utm_e, utm_n, zone = get_utm_coords(lat, lon)
        
        all_data.append({
            'ID': global_id,
            'Latitud': lat,
            'Longitud': lon,
            'UTM_Este': utm_e,
            'UTM_Norte': utm_n,
            'Zona': zone
        })
        
        # Plot Point
        plt.scatter(lon, lat, c='red', s=10)
        plt.annotate(str(global_id), (lon, lat), xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        all_points_latlon.append((lat, lon))
        all_points_ids.append(global_id)
        
        global_id += 1

    # Save outputs
    # 1. Image
    plt.title(f"Distribución de Puntos (d={DISTANCE}m)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('mapa_numerado.png')
    
    # 2. CSV
    df = pd.DataFrame(all_data)
    df.to_csv('puntos_prospeccion.csv', index=False)
    
    # 3. DXF
    # Note: DXF uses UTM coordinates for metric scale
    generate_simple_dxf('geometria_y_puntos.dxf', coords, all_points_latlon, all_points_ids)

    print("Files created: mapa_numerado.png, puntos_prospeccion.csv, geometria_y_puntos.dxf")
    print(df.head())
else:
    print("No polygon found")