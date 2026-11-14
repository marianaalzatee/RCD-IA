# ============================================================
# RCD IA - APLICACION STREAMLIT
# Integra las etapas dinamicas del notebook RCD_IA.ipynb
# ============================================================

import json
import math
import re
import time
import unicodedata
from pathlib import Path
from html import escape

import pandas as pd
import requests
import streamlit as st
import folium
import streamlit.components.v1 as components

from actualizacion_datos import (
    actualizar_generadores, actualizar_generadores_desde_archivo,
    actualizar_gestores_desde_archivo, registrar_eventos,
)

from modelo_ia import (
    entrenar_dbscan, comparar_radios, analizar_cercania_gestores,
    generar_hallazgos_territoriales,
)


# ------------------------------------------------------------
# 1. CONFIGURACION Y RUTAS
# ------------------------------------------------------------

st.set_page_config(
    page_title="RCD IA",
    page_icon="♻️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
RESULTADOS_DIR = BASE_DIR / "resultados"

ARCHIVO_GENERADORES = DATOS_DIR / "generador_rcd.geojson"
ARCHIVO_GESTORES = DATOS_DIR / "gestor_rcd.geojson"
# Busca el CSV primero en datos/ y, si no está allí, en la carpeta principal.
RUTAS_RECOMENDACIONES = [
    DATOS_DIR / "recomendaciones_RCD_IA.csv",
    BASE_DIR / "recomendaciones_RCD_IA.csv",
]
ARCHIVO_RECOMENDACIONES = next(
    (ruta for ruta in RUTAS_RECOMENDACIONES if ruta.exists()),
    RUTAS_RECOMENDACIONES[0]
)

RESULTADOS_DIR.mkdir(exist_ok=True)


def normalizar_id(valor):
    """Compara IDs numéricos ignorando ceros iniciales sin alterar su visualización."""
    if pd.isna(valor):
        return ""
    texto = str(valor).strip()
    if texto.endswith(".0"):
        texto = texto[:-2]
    if texto.isdigit():
        return texto.lstrip("0") or "0"
    return texto


# ------------------------------------------------------------
# 2. CARGA DE GEOJSON
# ------------------------------------------------------------

def leer_geojson(ruta, encoding):
    """Lee GeoJSON con tolerancia a las codificaciones de las fuentes originales y nuevas."""
    try:
        with open(ruta, "r", encoding=encoding) as archivo:
            return json.load(archivo)
    except UnicodeDecodeError:
        codificacion_alterna = "utf-8-sig" if encoding.lower() in ("latin1", "iso-8859-1") else "latin1"
        with open(ruta, "r", encoding=codificacion_alterna) as archivo:
            return json.load(archivo)


def geojson_a_dataframe(datos):
    registros = []

    for feature in datos.get("features", []):
        propiedades = feature.get("properties") or {}
        geometria = feature.get("geometry") or {}
        coordenadas = geometria.get("coordinates")

        registro = propiedades.copy()

        if geometria.get("type") == "Point" and coordenadas:
            registro["longitud"] = float(coordenadas[0])
            registro["latitud"] = float(coordenadas[1])
        else:
            registro["longitud"] = None
            registro["latitud"] = None

        registros.append(registro)

    return pd.DataFrame(registros)


@st.cache_data
def cargar_bases():
    datos_generadores = leer_geojson(ARCHIVO_GENERADORES, "latin1")
    datos_gestores = leer_geojson(ARCHIVO_GESTORES, "utf-8")

    generadores = geojson_a_dataframe(datos_generadores)
    gestores = geojson_a_dataframe(datos_gestores)

    for tabla in (generadores, gestores):
        tabla["longitud"] = pd.to_numeric(tabla["longitud"], errors="coerce")
        tabla["latitud"] = pd.to_numeric(tabla["latitud"], errors="coerce")
        tabla["coordenadas_validas"] = (
            tabla["longitud"].between(-75, -73)
            & tabla["latitud"].between(4, 5)
        )

    generadores = generadores[generadores["coordenadas_validas"]].copy()
    gestores = gestores[gestores["coordenadas_validas"]].copy()

    # La etapa 9 del notebook utiliza estas columnas para los generadores.
    generadores["longitud_geo"] = generadores["longitud"]
    generadores["latitud_geo"] = generadores["latitud"]

    # Conservar el ID original para mostrarlo y crear una clave de comparación
    # que ignora ceros iniciales (por ejemplo, 000541 y 541).
    if "idproy" in generadores.columns:
        generadores["idproy"] = generadores["idproy"].astype(str).str.strip()
        generadores["_idproy_match"] = generadores["idproy"].map(normalizar_id)
    if "id_sitio_d" in gestores.columns:
        gestores["id_sitio_d"] = gestores["id_sitio_d"].map(normalizar_id)

    return generadores.reset_index(drop=True), gestores.reset_index(drop=True)


@st.cache_data
def cargar_recomendaciones():
    if not ARCHIVO_RECOMENDACIONES.exists():
        raise FileNotFoundError(
            "No se encontró recomendaciones_RCD_IA.csv. "
            "Guárdalo en la carpeta datos o junto al archivo app.py."
        )

    tabla = pd.read_csv(ARCHIVO_RECOMENDACIONES, encoding="utf-8-sig")
    columnas_requeridas = [
        "idproy", "Nombre_gestor", "Codigo_residuo", "Tipo_residuo",
        "ID_gestor", "orden_recomendacion", "distancia_carretera_km",
        "tiempo_ruta_minutos", "puntaje_equilibrado"
    ]
    faltantes = [columna for columna in columnas_requeridas if columna not in tabla.columns]
    if faltantes:
        raise ValueError(
            "El CSV de recomendaciones no contiene estas columnas: "
            + ", ".join(faltantes)
        )

    tabla["idproy"] = tabla["idproy"].astype(str).str.strip()
    tabla["_idproy_match"] = tabla["idproy"].map(normalizar_id)
    tabla["ID_gestor"] = tabla["ID_gestor"].map(normalizar_id)
    tabla["Codigo_residuo"] = tabla["Codigo_residuo"].astype(str).str.strip().str.upper()
    for columna in [
        "orden_recomendacion", "distancia_carretera_km",
        "tiempo_ruta_minutos", "puntaje_equilibrado"
    ]:
        tabla[columna] = pd.to_numeric(tabla[columna], errors="coerce")

    return tabla.dropna(
        subset=["orden_recomendacion", "distancia_carretera_km",
                "tiempo_ruta_minutos", "puntaje_equilibrado"]
    ).copy()


# ------------------------------------------------------------
# 3. MATRIZ DE COMPATIBILIDAD (REGLAS DEL NOTEBOOK, ETAPA 3)
# ------------------------------------------------------------

MATERIALES = {
    "R01": {
        "nombre": "Concreto y residuos cementicios",
        "palabras": ["concreto", "concretos", "cemento", "cementos",
                     "mezcla de cemento", "mezclas de cemento",
                     "residuos cementicios"]
    },
    "R02": {
        "nombre": "Ladrillos, bloques y cerámicos",
        "palabras": ["ladrillo", "ladrillos", "bloque", "bloques",
                     "mamposteria", "ceramica", "ceramicas"]
    },
    "R03": {
        "nombre": "Material de excavación y tierras",
        "palabras": ["excavacion", "tierras", "material de excavacion",
                     "rocas de excavacion"]
    },
    "R04": {
        "nombre": "Arenas, gravas y materiales pétreos",
        "palabras": ["arena", "arenas", "grava", "gravas", "gravilla",
                     "gravillas", "roca", "rocas", "petreo", "petreos"]
    },
    "R05": {"nombre": "Madera", "palabras": ["madera"]},
    "R06": {
        "nombre": "Metales",
        "palabras": ["metales", "metal", "chatarra", "acero", "hierro",
                     "cobre", "aluminio", "zinc", "estano"]
    },
    "R07": {"nombre": "Plástico y PVC", "palabras": ["plastico", "pvc"]},
    "R08": {"nombre": "Papel y cartón", "palabras": ["papel", "carton"]},
    "R09": {"nombre": "Drywall", "palabras": ["drywall"]},
    "R10": {
        "nombre": "Icopor",
        "palabras": ["icopor", "poliestireno expandido"]
    },
    "R11": {"nombre": "Vidrio", "palabras": ["vidrio", "vidrios"]},
    "R12": {
        "nombre": "Mezclas asfálticas",
        "palabras": ["mezcla asfaltica", "mezclas asfalticas", "asfalto"]
    }
}

PALABRAS_MEZCLADOS_CONFIRMADOS = [
    "escombros mezclados", "residuos mezclados",
    "rcd mezclados", "rcd mixtos"
]
PALABRAS_GENERALES_RCD = [
    "aprovechamiento y tratamiento de rcd",
    "tratamiento de rcd",
    "gestion integral de rcd"
]


def normalizar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    return re.sub(r"\s+", " ", texto).strip()


def construir_matriz_compatibilidad(gestores):
    matriz = []

    for _, gestor in gestores.iterrows():
        descripcion_original = gestor.get("mat_a_disp", "")
        descripcion = normalizar_texto(descripcion_original)

        for codigo, material in MATERIALES.items():
            coincidencias = []

            for palabra in material["palabras"]:
                palabra_norm = normalizar_texto(palabra)
                patron = r"\b" + re.escape(palabra_norm) + r"\b"
                if re.search(patron, descripcion):
                    coincidencias.append(palabra)

            nivel_evidencia = "Sin evidencia"
            if coincidencias:
                estado = "Compatible reportado"
                evidencia = ", ".join(sorted(set(coincidencias)))
                # Para arenas/gravas/petreos distinguimos una mencion
                # especifica del material de una referencia amplia a RCD petreos.
                if codigo == "R04":
                    terminos_especificos = [
                        "arena", "arenas", "grava", "gravas",
                        "gravilla", "gravillas", "roca", "rocas"
                    ]
                    menciones_especificas = [
                        termino for termino in terminos_especificos
                        if re.search(r"\b" + re.escape(termino) + r"\b", descripcion)
                    ]
                    nivel_evidencia = (
                        "Específica" if menciones_especificas else "General"
                    )
                else:
                    nivel_evidencia = "Específica"
            elif any(p in descripcion for p in PALABRAS_GENERALES_RCD):
                estado = "Pendiente de verificacion"
                evidencia = (
                    "Descripcion general de tratamiento o "
                    "aprovechamiento de RCD"
                )
            else:
                estado = "No identificado como compatible"
                evidencia = "No se encontro evidencia textual"

            matriz.append({
                "ID_gestor": gestor.get("id_sitio_d"),
                "Nombre_gestor": gestor.get("nomb_disp_", "Sin nombre"),
                "Categoria": gestor.get("categoria_", "Sin categoria"),
                "Modalidad": "Separado",
                "Codigo_residuo": codigo,
                "Tipo_residuo": material["nombre"],
                "Estado_compatibilidad": estado,
                "Nivel_evidencia": nivel_evidencia,
                "Evidencia": evidencia,
                "Descripcion_original": descripcion_original,
                "Aprobacion_reportada": gestor.get("aprobacion")
            })

        coincidencia_mezcla = [
            p for p in PALABRAS_MEZCLADOS_CONFIRMADOS if p in descripcion
        ]
        coincidencia_general = [
            p for p in PALABRAS_GENERALES_RCD if p in descripcion
        ]

        if coincidencia_mezcla:
            estado_mix = "Compatible reportado"
            evidencia_mix = ", ".join(coincidencia_mezcla)
        elif coincidencia_general:
            estado_mix = "Pendiente de verificacion"
            evidencia_mix = (
                "Mencion general de RCD; no confirma recepcion "
                "de residuos mezclados"
            )
        elif "escombros" in descripcion:
            estado_mix = "Pendiente de verificacion"
            evidencia_mix = (
                "Menciona escombros, pero no especifica si recibe "
                "mezclas de residuos"
            )
        else:
            estado_mix = "No identificado como compatible"
            evidencia_mix = (
                "No se encontro evidencia de recepcion de residuos mezclados"
            )

        matriz.append({
            "ID_gestor": gestor.get("id_sitio_d"),
            "Nombre_gestor": gestor.get("nomb_disp_", "Sin nombre"),
            "Categoria": gestor.get("categoria_", "Sin categoria"),
            "Modalidad": "Mezclado",
            "Codigo_residuo": "RCD_MIX",
            "Tipo_residuo": "RCD mezclados",
            "Estado_compatibilidad": estado_mix,
            "Nivel_evidencia": "No clasificada",
            "Evidencia": evidencia_mix,
            "Descripcion_original": descripcion_original,
            "Aprobacion_reportada": gestor.get("aprobacion")
        })

    return pd.DataFrame(matriz)


# ------------------------------------------------------------
# 4. DISTANCIAS Y FILTRO DINAMICO POR EVIDENCIA (ETAPA 9.6.4)
# ------------------------------------------------------------

def calcular_distancia(lat1, lon1, lat2, lon2):
    radio_tierra = 6371.0
    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return radio_tierra * 2 * math.asin(math.sqrt(a))


def calcular_gestores_cercanos(generador, gestores):
    filas = []

    for _, gestor in gestores.iterrows():
        if pd.isna(gestor["longitud"]) or pd.isna(gestor["latitud"]):
            continue

        distancia = calcular_distancia(
            float(generador["latitud_geo"]),
            float(generador["longitud_geo"]),
            float(gestor["latitud"]),
            float(gestor["longitud"])
        )

        filas.append({
            "id_sitio_d": gestor["id_sitio_d"],
            "nomb_disp_": gestor["nomb_disp_"],
            "categoria_": gestor.get("categoria_", "Sin categoria"),
            "longitud": gestor["longitud"],
            "latitud": gestor["latitud"],
            "distancia_linea_recta_km": distancia
        })

    return pd.DataFrame(filas).sort_values(
        "distancia_linea_recta_km"
    ).reset_index(drop=True)


def filtrar_gestores_por_material(residuos, modalidad, matriz, gestores_cercanos):
    """Evalua cada residuo por separado para permitir gestores distintos."""
    modalidad_matriz = "Mezclado" if "mezcl" in str(modalidad).lower() else "Separado"
    resultados = {}

    for residuo in residuos:
        codigo = str(residuo["Codigo_residuo"]).strip()
        filas = matriz[
            (matriz["Modalidad"] == modalidad_matriz)
            & (matriz["Codigo_residuo"].astype(str) == codigo)
        ].copy()

        if filas.empty:
            resultados[codigo] = (pd.DataFrame(), pd.DataFrame())
            continue

        # Una fila por gestor y material. La matriz se construye con una
        # sola fila por combinacion gestor-material, pero drop_duplicates
        # protege frente a duplicados de origen.
        filas = filas.drop_duplicates(subset=["ID_gestor", "Codigo_residuo"])
        compatibles_ids = filas.loc[
            filas["Estado_compatibilidad"] == "Compatible reportado", "ID_gestor"
        ].dropna().unique()
        pendientes_ids = filas.loc[
            filas["Estado_compatibilidad"] == "Pendiente de verificacion", "ID_gestor"
        ].dropna().unique()

        compatibles = gestores_cercanos[
            gestores_cercanos["id_sitio_d"].isin(compatibles_ids)
        ].copy()
        pendientes = gestores_cercanos[
            gestores_cercanos["id_sitio_d"].isin(pendientes_ids)
        ].copy()

        # Incorporar la evidencia textual para que sea visible en la interfaz.
        columnas_evidencia = ["ID_gestor", "Estado_compatibilidad", "Nivel_evidencia", "Evidencia", "Descripcion_original"]
        compatibles = compatibles.merge(
            filas[columnas_evidencia], left_on="id_sitio_d", right_on="ID_gestor", how="left"
        )
        pendientes = pendientes.merge(
            filas[columnas_evidencia], left_on="id_sitio_d", right_on="ID_gestor", how="left"
        )

        if "Nivel_evidencia" in compatibles.columns:
            compatibles["Prioridad_evidencia"] = compatibles["Nivel_evidencia"].map(
                {"Específica": 0, "General": 1}
            ).fillna(2)
            compatibles = compatibles.sort_values(
                ["Prioridad_evidencia", "distancia_linea_recta_km"]
            ).reset_index(drop=True)
        else:
            compatibles = compatibles.sort_values("distancia_linea_recta_km").reset_index(drop=True)
        pendientes = pendientes.sort_values("distancia_linea_recta_km").reset_index(drop=True)
        if not compatibles.empty:
            compatibles.insert(0, "Orden_recomendacion", range(1, len(compatibles) + 1))

        resultados[codigo] = (compatibles, pendientes)

    return resultados


# ------------------------------------------------------------
# 5. RUTAS OSRM (ETAPA 9.7)
# ------------------------------------------------------------

def calcular_rutas_generador_dinamico(generador, recomendaciones):
    resultados = []

    for _, gestor in recomendaciones.iterrows():
        lon_origen = float(generador["longitud_geo"])
        lat_origen = float(generador["latitud_geo"])
        lon_destino = float(gestor["longitud"])
        lat_destino = float(gestor["latitud"])

        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{lon_origen},{lat_origen};{lon_destino},{lat_destino}"
            "?overview=false"
        )

        try:
            respuesta = requests.get(url, timeout=30)
            respuesta.raise_for_status()
            datos = respuesta.json()

            if datos.get("code") == "Ok" and datos.get("routes"):
                ruta = datos["routes"][0]
                resultados.append({
                    "Nombre_gestor": gestor["nomb_disp_"],
                    "ID_gestor": gestor["id_sitio_d"],
                    "Categoria": gestor["categoria_"],
                    "Nivel_evidencia": gestor.get("Nivel_evidencia", "No clasificada"),
                    "Distancia_linea_recta_km": gestor["distancia_linea_recta_km"],
                    "Distancia_carretera_km": ruta["distance"] / 1000,
                    "Tiempo_estimado_min": ruta["duration"] / 60
                })
        except requests.RequestException:
            continue

        time.sleep(0.2)

    rutas = pd.DataFrame(resultados)

    if not rutas.empty:
        rutas = rutas.sort_values(
            "Distancia_carretera_km"
        ).reset_index(drop=True)
        rutas.insert(0, "Orden_recomendacion", range(1, len(rutas) + 1))

    return rutas


# ------------------------------------------------------------
# 6. EMISIONES (FORMULA DE LA ETAPA 6 DEL NOTEBOOK)
# ------------------------------------------------------------

def calcular_emisiones(rutas, residuos, capacidad, consumo, combustible):
    cantidad_total = sum(float(r["Cantidad_toneladas"]) for r in residuos)

    factores = {
        "Diésel B2": 10.149,
        "Gasolina": 8.8085
    }
    litros_por_galon = 3.78541
    factor_kg_litro = factores[combustible] / litros_por_galon
    numero_viajes = math.ceil(cantidad_total / capacidad)

    resultados = rutas.copy()
    resultados["Distancia_total_km"] = (
        resultados["Distancia_carretera_km"] * 2 * numero_viajes
    )
    resultados["Combustible_litros"] = (
        resultados["Distancia_total_km"] * consumo / 100
    )
    resultados["Emisiones_CO2_kg"] = (
        resultados["Combustible_litros"] * factor_kg_litro
    )
    resultados["Emisiones_CO2_ton"] = (
        resultados["Emisiones_CO2_kg"] / 1000
    )

    if "Orden_recomendacion" in resultados.columns:
        # Conserva el orden equilibrado validado en el notebook (50 % distancia, 50 % tiempo).
        resultados = resultados.sort_values(
            "Orden_recomendacion"
        ).reset_index(drop=True)
    elif "Nivel_evidencia" in resultados.columns:
        resultados["Prioridad_evidencia"] = resultados["Nivel_evidencia"].map(
            {"Específica": 0, "General": 1}
        ).fillna(2)
        resultados = resultados.sort_values(
            ["Prioridad_evidencia", "Distancia_carretera_km"]
        ).reset_index(drop=True)
    else:
        resultados = resultados.sort_values(
            "Distancia_carretera_km"
        ).reset_index(drop=True)
    resultados["Orden_emisiones"] = range(1, len(resultados) + 1)

    resultados.to_csv(
        RESULTADOS_DIR / "emisiones_por_gestor.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return resultados, numero_viajes, factor_kg_litro


# ------------------------------------------------------------
# 7. MAPA DINAMICO (ESTRUCTURA DE LA ETAPA 9.10)
# ------------------------------------------------------------

def crear_mapa(generador, rutas, emisiones, residuos):
    mapa_datos = rutas.merge(
        emisiones[["ID_gestor", "Emisiones_CO2_kg"]],
        on="ID_gestor",
        how="left"
    )

    mapa_datos = mapa_datos.merge(
        gestores[
            ["id_sitio_d", "nomb_disp_", "longitud", "latitud", "categoria_"]
        ],
        left_on="ID_gestor",
        right_on="id_sitio_d",
        how="left"
    )

    mapa_datos = mapa_datos.dropna(
        subset=["longitud", "latitud"]
    ).copy()
    if "Orden_recomendacion" in mapa_datos.columns:
        mapa_datos = mapa_datos.sort_values(
            "Orden_recomendacion"
        ).reset_index(drop=True)
    elif "Nivel_evidencia" in mapa_datos.columns:
        mapa_datos["Prioridad_evidencia"] = mapa_datos["Nivel_evidencia"].map(
            {"Específica": 0, "General": 1}
        ).fillna(2)
        mapa_datos = mapa_datos.sort_values(
            ["Prioridad_evidencia", "Distancia_carretera_km"]
        ).reset_index(drop=True)
    else:
        mapa_datos = mapa_datos.sort_values(
            "Distancia_carretera_km"
        ).reset_index(drop=True)

    if mapa_datos.empty:
        return None, 0, 0

    lat0 = float(generador["latitud_geo"])
    lon0 = float(generador["longitud_geo"])

    mapa = folium.Map(location=[lat0, lon0], zoom_start=11, tiles=None)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, HERE, Garmin, OpenStreetMap contributors",
        name="Esri World Street Map"
    ).add_to(mapa)

    folium.Marker(
        [lat0, lon0],
        tooltip=f"Generador: {generador['nombproy']}",
        popup=f"Generador: {escape(str(generador['nombproy']))}",
        icon=folium.Icon(color="blue", icon="home", prefix="fa")
    ).add_to(mapa)

    rutas_dibujadas = 0
    errores = 0
    puntos = [[lat0, lon0]]

    for posicion, gestor in mapa_datos.iterrows():
        lon = float(gestor["longitud"])
        lat = float(gestor["latitud"])
        puntos.append([lat, lon])

        recomendado = posicion == 0
        color = "green" if recomendado else "orange"

        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{lon0},{lat0};{lon},{lat}"
            "?overview=full&geometries=geojson"
        )

        try:
            respuesta = requests.get(url, timeout=30)
            datos = respuesta.json()

            if datos.get("code") == "Ok" and datos.get("routes"):
                coordenadas = datos["routes"][0]["geometry"]["coordinates"]
                puntos_ruta = [[latitud, longitud] for longitud, latitud in coordenadas]

                folium.PolyLine(
                    puntos_ruta,
                    color=color,
                    weight=5 if recomendado else 3,
                    opacity=0.85,
                    tooltip=str(gestor["Nombre_gestor"])
                ).add_to(mapa)
                rutas_dibujadas += 1
            else:
                errores += 1
        except requests.RequestException:
            errores += 1

        nombre = escape(str(gestor["Nombre_gestor"]))
        popup = (
            f"<b>{'GESTOR RECOMENDADO' if recomendado else 'GESTOR ALTERNATIVO'}</b><br>"
            f"<b>{nombre}</b><br>"
            f"Distancia: {gestor['Distancia_carretera_km']:.2f} km<br>"
            f"Tiempo: {gestor['Tiempo_estimado_min']:.2f} min<br>"
            f"Emisiones: {gestor['Emisiones_CO2_kg']:.2f} kg CO2"
        )

        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup, max_width=300),
            tooltip=nombre,
            icon=folium.Icon(color=color, icon="industry", prefix="fa")
        ).add_to(mapa)

    mapa.fit_bounds(puntos, padding=(40, 40))
    folium.LayerControl().add_to(mapa)

    # Leyenda fija para que el usuario identifique los colores del mapa.
    leyenda = """
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                background: white; padding: 12px 15px; border: 1px solid #999;
                border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.25);
                font-size: 13px; line-height: 1.7;">
      <b>Leyenda del mapa</b><br>
      <span style="color:#2878c8; font-size:17px;">●</span>
      Generador de RCD<br>
      <span style="color:#2e9d50; font-size:17px;">●</span>
      Gestor recomendado (mejor puntaje equilibrado de distancia y tiempo)<br>
      <span style="color:#f39c35; font-size:17px;">●</span>
      Gestores alternativos
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(leyenda))

    mapa.save(RESULTADOS_DIR / "mapa_dinamico_rcdia.html")
    return mapa, rutas_dibujadas, errores


# ------------------------------------------------------------
# 8. INTERFAZ
# ------------------------------------------------------------

st.markdown(
    "<h1>♻️ RCD IA</h1>",
    unsafe_allow_html=True
)
st.title("Sistema inteligente de apoyo a la gestión de residuos de construcción y demolición")
st.write(
    "RCD IA permite identificar gestores potencialmente compatibles, "
    "analizar rutas de transporte, estimar emisiones de CO₂ y generar "
    "recomendaciones para la gestión de residuos de construcción y demolición."
)

try:
    generadores, gestores = cargar_bases()
    recomendaciones_notebook = cargar_recomendaciones()
    matriz_compatibilidad = construir_matriz_compatibilidad(gestores)
    st.success(
        f"Bases cargadas: {len(generadores)} generadores, "
        f"{len(gestores)} gestores y {len(recomendaciones_notebook):,} recomendaciones validadas."
    )
except Exception as error:
    st.error("No fue posible cargar las bases de datos.")
    st.exception(error)
    st.stop()


# ------------------------------------------------------------
# 8.0 ACTUALIZACIÓN DE INFORMACIÓN
# ------------------------------------------------------------

with st.expander("🔄 Actualización de información", expanded=False):
    st.write(
        "Consulta las fuentes disponibles e incorpora nuevas versiones de las bases. "
        "Los archivos existentes solo se reemplazan después de validar que la descarga "
        "contiene registros geográficos utilizables."
    )
    st.caption(
        "Generadores: consulta automática del servicio geográfico de la SDA. "
        "Gestores: carga controlada de un GeoJSON oficial, porque el conjunto público "
        "disponible puede no reflejar el directorio mensual más reciente."
    )
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("Actualizar generadores desde SDA", key="actualizar_generadores_sda"):
            try:
                evento = actualizar_generadores(ARCHIVO_GENERADORES)
                registrar_eventos([evento], RESULTADOS_DIR / "historial_actualizaciones.json")
                st.success(f"Generadores actualizados automáticamente: {evento['registros']:,} registros.")
                st.cache_data.clear()
                st.rerun()
            except Exception as error:
                st.warning(
                    "No fue posible actualizar desde el servicio de la SDA. "
                    "La última base válida se conserva. Puedes incorporar una versión "
                    "oficial mediante la carga manual que aparece abajo. "
                    f"Detalle técnico: {error}"
                )

        st.markdown("**Alternativa: cargar generadores manualmente**")
        archivo_generadores_nuevo = st.file_uploader(
            "Selecciona el GeoJSON actualizado de generadores",
            type=["geojson", "json"],
            key="archivo_generadores_actualizacion",
            help=(
                "Carga un GeoJSON FeatureCollection con geometrías Point y coordenadas "
                "geográficas (longitud/latitud) de Bogotá. La base actual no se reemplaza "
                "hasta que el archivo supere la validación."
            ),
        )
        if st.button(
            "Validar e incorporar generadores",
            key="actualizar_generadores_archivo",
            disabled=archivo_generadores_nuevo is None,
        ):
            try:
                evento = actualizar_generadores_desde_archivo(
                    archivo_generadores_nuevo.getvalue(), ARCHIVO_GENERADORES
                )
                registrar_eventos([evento], RESULTADOS_DIR / "historial_actualizaciones.json")
                st.success(
                    f"Generadores incorporados desde archivo: {evento['registros']:,} "
                    "registros geográficos válidos."
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as error:
                st.error(
                    "El archivo no se incorporó. La última base válida se conserva. "
                    f"Revisa el GeoJSON. Detalle: {error}"
                )
    with col_act2:
        archivo_gestores_nuevo = st.file_uploader(
            "Cargar GeoJSON oficial de gestores", type=["geojson", "json"],
            key="archivo_gestores_actualizacion"
        )
        if st.button("Validar e incorporar gestores", key="actualizar_gestores_archivo", disabled=archivo_gestores_nuevo is None):
            try:
                evento = actualizar_gestores_desde_archivo(archivo_gestores_nuevo.getvalue(), ARCHIVO_GESTORES)
                registrar_eventos([evento], RESULTADOS_DIR / "historial_actualizaciones.json")
                st.success(f"Gestores actualizados: {evento['registros']:,} registros.")
                st.cache_data.clear()
                st.rerun()
            except Exception as error:
                st.error(f"No se actualizó la base de gestores. Se conserva el archivo anterior. Detalle: {error}")
    historial = RESULTADOS_DIR / "historial_actualizaciones.json"
    if historial.exists():
        try:
            historial_df = pd.DataFrame(json.loads(historial.read_text(encoding="utf-8")))
            st.markdown("**Historial de actualizaciones realizadas**")
            st.dataframe(historial_df, use_container_width=True, hide_index=True)
        except Exception:
            st.caption("El historial existe, pero no pudo leerse.")
    else:
        st.info("Todavía no hay actualizaciones registradas desde la aplicación.")
    st.warning(
        "La descarga automática depende de la disponibilidad y del certificado SSL del "
        "servicio de la SDA. Si falla, puedes cargar manualmente un GeoJSON oficial de "
        "generadores. La carga manual y la de gestores validan las coordenadas antes de "
        "reemplazar las bases vigentes; los PDF no se convierten automáticamente."
    )


# ------------------------------------------------------------
# 8.1 INTELIGENCIA TERRITORIAL (DBSCAN)
# ------------------------------------------------------------

with st.expander("🧠 Inteligencia territorial: análisis para la planificación de la Secretaría", expanded=False):
    st.write(
        "Este módulo utiliza **DBSCAN (aprendizaje automático no supervisado)** para detectar "
        "concentraciones espaciales de generadores de RCD. Después compara cada concentración "
        "con la ubicación de los gestores registrados para apoyar la identificación de zonas "
        "que podrían requerir revisión o seguimiento por parte de la Secretaría Distrital de Ambiente."
    )
    st.info(
        "Alcance: se analizan registros y distancias geográficas en línea recta. Los datos no "
        "incluyen toneladas generadas, capacidad disponible de los gestores ni demanda atendida; "
        "por tanto, una distancia alta no demuestra una brecha de servicio."
    )
    colp1, colp2 = st.columns(2)
    with colp1:
        radio_dbscan = st.selectbox("Radio de vecindad DBSCAN", [250, 500, 750, 1000], index=0, key="radio_dbscan_app", format_func=lambda x: f"{x} m")
    with colp2:
        min_pts_dbscan = st.selectbox("Mínimo de generadores por concentración", [3, 4, 5, 6, 8, 10], index=2, key="min_pts_dbscan_app")
    if st.button("Ejecutar análisis territorial", key="ejecutar_dbscan_app"):
        try:
            with st.spinner("Detectando concentraciones y comparando su cercanía a gestores..."):
                generadores_limpios_app = generadores.copy()
                if "nombproy" in generadores_limpios_app.columns and "idproy" in generadores_limpios_app.columns:
                    generadores_limpios_app["_longitud_nombre"] = generadores_limpios_app["nombproy"].fillna("").astype(str).str.strip().str.len()
                    generadores_limpios_app = generadores_limpios_app.sort_values("_longitud_nombre", ascending=False).drop_duplicates(subset="idproy", keep="first").drop(columns="_longitud_nombre")
                clasificados_app, resumen_app, metricas_app = entrenar_dbscan(generadores_limpios_app, eps_metros=int(radio_dbscan), min_samples=int(min_pts_dbscan))
                resumen_territorial = analizar_cercania_gestores(resumen_app, gestores)
                comparacion_app = comparar_radios(generadores_limpios_app, min_samples=int(min_pts_dbscan))
            st.session_state["dbscan_clasificados"] = clasificados_app
            st.session_state["dbscan_resumen"] = resumen_territorial
            st.session_state["dbscan_metricas"] = metricas_app
            st.session_state["dbscan_comparacion"] = comparacion_app
            st.session_state["dbscan_radio"] = int(radio_dbscan)
            st.success("Análisis territorial ejecutado correctamente.")
        except Exception as error:
            st.error(f"No fue posible ejecutar el análisis territorial: {error}")

    if "dbscan_metricas" in st.session_state:
        metricas_app = st.session_state["dbscan_metricas"]
        resumen_terr = st.session_state["dbscan_resumen"]
        a, b, c, d = st.columns(4)
        a.metric("Generadores analizados", f"{metricas_app['generadores_analizados']:,}")
        b.metric("Concentraciones", f"{metricas_app['clusters']:,}")
        c.metric("Generadores agrupados", f"{metricas_app['agrupados']:,}")
        d.metric("Registros aislados", f"{metricas_app['ruido']:,}")
        st.caption(f"Agrupados: {metricas_app['porcentaje_agrupado']:.2f}% | Ruido: {metricas_app['porcentaje_ruido']:.2f}% | Radio: {metricas_app['radio_metros']} m | Mínimo: {metricas_app['min_samples']} puntos")
        st.markdown("**Hallazgos automáticos del análisis**")
        for hallazgo in generar_hallazgos_territoriales(resumen_terr, metricas_app):
            st.write("• " + hallazgo)
        st.markdown("**Concentraciones priorizadas para revisión territorial**")
        columnas_vista = [c for c in ["cluster_dbscan", "numero_generadores", "prioridad_analisis", "gestor_mas_cercano", "distancia_gestor_mas_cercano_m", "gestores_a_1000m", "gestores_a_2000m", "gestores_a_5000m"] if c in resumen_terr.columns]
        st.dataframe(resumen_terr[columnas_vista].head(20).rename(columns={"cluster_dbscan":"Clúster", "numero_generadores":"Generadores", "prioridad_analisis":"Prioridad de análisis", "gestor_mas_cercano":"Gestor más cercano", "distancia_gestor_mas_cercano_m":"Distancia (m)", "gestores_a_1000m":"Gestores <1 km", "gestores_a_2000m":"Gestores <2 km", "gestores_a_5000m":"Gestores <5 km"}), use_container_width=True, hide_index=True)
        st.caption("La prioridad es una regla exploratoria: combina tamaño del clúster y distancia al gestor más cercano, comparados con las medianas del conjunto. No es una predicción de déficit.")
        st.markdown("**Mapa de concentraciones y gestores registrados**")

        # Streamlit vuelve a ejecutar todo el archivo cuando el usuario
        # interactúa con otros controles. Recuperamos el resultado de DBSCAN
        # desde session_state para que siga disponible después del clic.
        clasificados_app = st.session_state["dbscan_clasificados"]

        centro = [
            float(clasificados_app["latitud"].mean()),
            float(clasificados_app["longitud"].mean())
        ]

        mapa_territorial = folium.Map(
            location=centro,
            zoom_start=11,
            tiles=None
        )

        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Esri, HERE, Garmin, OpenStreetMap contributors",
            name="Mapa base (Esri)",
            overlay=False,
            control=True
        ).add_to(mapa_territorial)

        # Colores asociados a la prioridad exploratoria de cada clúster.
        colores = {
            "Alta": "#e62920",
            "Media": "#f5a623",
            "Baja": "#1745e8"
        }

        # Capa de clústeres: el tamaño del círculo representa el número
        # de generadores y el color representa la prioridad de análisis.
        grupo_clusters = folium.FeatureGroup(
            name="Concentraciones de generadores",
            show=True
        )

        for _, fila in resumen_terr.iterrows():
            tam = int(fila["numero_generadores"])
            prioridad = str(fila.get("prioridad_analisis", "Baja"))
            color = colores.get(prioridad, colores["Baja"])

            distancia = fila.get(
                "distancia_gestor_mas_cercano_m",
                float("nan")
            )
            distancia_texto = (
                f"{float(distancia) / 1000:.2f} km"
                if pd.notna(distancia)
                else "No disponible"
            )
            nombre_gestor = escape(
                str(fila.get("gestor_mas_cercano", "No identificado"))
            )

            popup = f"""
            <div style="font-family:Arial; width:230px;">
                <b>Clúster {int(fila['cluster_dbscan'])}</b><br>
                Generadores registrados: <b>{tam}</b><br>
                Prioridad de análisis: <b>{escape(prioridad)}</b><br>
                Gestor más cercano: {nombre_gestor}<br>
                Distancia en línea recta: {distancia_texto}
            </div>
            """

            folium.CircleMarker(
                location=[
                    float(fila["latitud_centro"]),
                    float(fila["longitud_centro"])
                ],
                radius=max(5, min(18, 4 + tam ** 0.5)),
                color=color,
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.55,
                popup=folium.Popup(popup, max_width=300),
                tooltip=f"Clúster {int(fila['cluster_dbscan'])}: {tam} generadores"
            ).add_to(grupo_clusters)

        grupo_clusters.add_to(mapa_territorial)

        # Capa de gestores registrados.
        grupo_gestores = folium.FeatureGroup(
            name="Gestores registrados",
            show=True
        )

        for _, gestor in gestores.iterrows():
            nombre = next(
                (
                    str(gestor[c])
                    for c in (
                        "nombre_sitio",
                        "nom_establecimiento",
                        "nombre",
                        "razon_social",
                        "Nombre_gestor",
                        "nombgestor"
                    )
                    if c in gestores.columns and pd.notna(gestor[c])
                ),
                "Gestor registrado"
            )

            folium.CircleMarker(
                location=[
                    float(gestor["latitud"]),
                    float(gestor["longitud"])
                ],
                radius=5,
                color="#218c27",
                weight=2,
                fill=True,
                fill_color="#218c27",
                fill_opacity=0.9,
                tooltip=escape(nombre)
            ).add_to(grupo_gestores)

        grupo_gestores.add_to(mapa_territorial)
        folium.LayerControl(position="topright", collapsed=True).add_to(
            mapa_territorial
        )

        # Leyenda fija dentro del mapa.
        leyenda_html = """
        <div style="
            position: fixed;
            bottom: 28px;
            right: 22px;
            z-index: 9999;
            width: 255px;
            max-width: calc(100% - 55px);
            box-sizing: border-box;
            background: rgba(255,255,255,0.97);
            padding: 14px 16px;
            border: 2px solid #b8b8b8;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.18);
            font-family: Arial, sans-serif;
            color: #222;
            font-size: 13px;
            line-height: 1.35;
        ">
            <div style="font-size:18px;font-weight:bold;margin-bottom:12px;">
                Leyenda
            </div>

            <div style="font-weight:bold;margin-bottom:8px;">
                Concentraciones de generadores<br>
                (Clústeres DBSCAN)
            </div>

            <div style="display:flex;align-items:center;margin:9px 0;">
                <span style="display:inline-block;width:17px;height:17px;
                    border-radius:50%;background:#e62920;border:2px solid #c91e17;
                    margin-right:12px;flex-shrink:0;"></span>
                <span><b>Prioridad alta</b><br>Mayor prioridad de revisión territorial</span>
            </div>

            <div style="display:flex;align-items:center;margin:9px 0;">
                <span style="display:inline-block;width:17px;height:17px;
                    border-radius:50%;background:#f5a623;border:2px solid #d98b0c;
                    margin-right:12px;flex-shrink:0;"></span>
                <span><b>Prioridad media</b><br>Prioridad intermedia de revisión</span>
            </div>

            <div style="display:flex;align-items:center;margin:9px 0;">
                <span style="display:inline-block;width:17px;height:17px;
                    border-radius:50%;background:#1745e8;border:2px solid #1035b9;
                    margin-right:12px;flex-shrink:0;"></span>
                <span><b>Prioridad baja</b><br>Menor prioridad relativa de revisión</span>
            </div>

            <hr style="border:0;border-top:1px solid #ddd;margin:12px 0;">

            <div style="font-weight:bold;margin-bottom:8px;">Gestores de RCD</div>
            <div style="display:flex;align-items:center;margin:8px 0;">
                <span style="display:inline-block;width:12px;height:12px;
                    border-radius:50%;background:#218c27;border:2px solid #176b1d;
                    margin-right:15px;margin-left:3px;flex-shrink:0;"></span>
                Gestor registrado
            </div>

            <hr style="border:0;border-top:1px solid #ddd;margin:12px 0;">

            <div style="font-size:11px;color:#555;">
                <b>Tamaño de los círculos:</b> proporcional al número de
                generadores identificados en cada clúster.
            </div>
            <div style="font-size:10px;color:#777;margin-top:7px;">
                La prioridad es exploratoria; no demuestra una brecha de
                servicio ni falta de capacidad de los gestores.
            </div>
        </div>
        """

        mapa_territorial.get_root().html.add_child(
            folium.Element(leyenda_html)
        )

        components.html(
            mapa_territorial._repr_html_(),
            height=650,
            scrolling=False
        )
        st.markdown("**Comparación de parámetros DBSCAN**")
        st.dataframe(st.session_state["dbscan_comparacion"], use_container_width=True, hide_index=True)
        if st.button("Guardar resultados territoriales", key="guardar_dbscan_app"):
            RESULTADOS_DIR.mkdir(exist_ok=True)
            st.session_state["dbscan_clasificados"].to_csv(RESULTADOS_DIR / "generadores_clasificados.csv", index=False, encoding="utf-8-sig")
            resumen_terr.to_csv(RESULTADOS_DIR / "caracterizacion_y_priorizacion_clusters.csv", index=False, encoding="utf-8-sig")
            st.session_state["dbscan_comparacion"].to_csv(RESULTADOS_DIR / "comparacion_parametros_dbscan.csv", index=False, encoding="utf-8-sig")
            st.success(f"Resultados guardados en: {RESULTADOS_DIR}")


st.divider()
st.header("1. Selección del generador")

# El usuario selecciona directamente un registro del catálogo.
# La opción "Otro" habilita el ingreso de una dirección no registrada.
generadores_catalogo = generadores.reset_index(drop=True).copy()
opciones_generador = ["Selecciona un generador..."] + [
    f"{str(f.get('nombproy', 'Sin nombre'))} | {str(f.get('dirproy', 'Sin dirección'))} | ID: {str(f.get('idproy', ''))} | Registro {i+1}"
    for i, (_, f) in enumerate(generadores_catalogo.iterrows())
] + ["Otro (ingresar dirección manualmente)"]

seleccion_generador = st.selectbox(
    "Selecciona un generador del catálogo",
    opciones_generador,
    index=0,
    key="selector_catalogo_generadores"
)

generador_seleccionado = None

if seleccion_generador == "Selecciona un generador...":
    st.caption("Elige un generador registrado o selecciona «Otro» si no aparece en el catálogo.")

elif seleccion_generador == "Otro (ingresar dirección manualmente)":
    st.info(
        "Si el proyecto no aparece en la base de datos, ingresa su dirección. "
        "La aplicación buscará sus coordenadas y utilizará ese punto como origen."
    )
    direccion_manual = st.text_input(
        "Dirección del proyecto o lugar de generación",
        placeholder="Ej. Calle 100 # 15-20, Bogotá, Colombia",
        key="direccion_generador_manual"
    )
    nombre_manual = st.text_input(
        "Nombre o identificador del proyecto (opcional)",
        placeholder="Ej. Obra edificio residencial",
        key="nombre_generador_manual"
    )

    if st.button("Ubicar dirección", key="geocodificar_direccion"):
        if not direccion_manual.strip():
            st.warning("Escribe una dirección antes de intentar ubicarla.")
        else:
            consulta = direccion_manual.strip()
            if "bogota" not in normalizar_texto(consulta) and "colombia" not in normalizar_texto(consulta):
                consulta += ", Bogotá, Colombia"

            try:
                respuesta_geo = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": consulta, "format": "jsonv2", "limit": 1, "countrycodes": "co"},
                    headers={"User-Agent": "RCD-IA-Universidad-Project/1.0"},
                    timeout=20
                )
                respuesta_geo.raise_for_status()
                resultados_geo = respuesta_geo.json()

                if resultados_geo:
                    resultado_geo = resultados_geo[0]
                    st.session_state["generador_manual_geocodificado"] = {
                        "nombproy": nombre_manual.strip() or "Generador ingresado manualmente",
                        "idproy": "MANUAL",
                        "dirproy": direccion_manual.strip(),
                        "longitud_geo": float(resultado_geo["lon"]),
                        "latitud_geo": float(resultado_geo["lat"]),
                        "longitud": float(resultado_geo["lon"]),
                        "latitud": float(resultado_geo["lat"]),
                        "coordenadas_validas": True,
                        "origen_registro": "Dirección ingresada manualmente"
                    }
                    st.session_state["direccion_manual_resultado"] = resultado_geo.get("display_name", consulta)
                else:
                    st.session_state.pop("generador_manual_geocodificado", None)
                    st.warning(
                        "No se pudo ubicar esa dirección. Intenta agregar barrio, localidad "
                        "o ciudad, y verifica la escritura."
                    )
            except requests.RequestException as error:
                st.error(f"No fue posible consultar el servicio de geocodificación: {error}")

    if "generador_manual_geocodificado" in st.session_state:
        generador_seleccionado = st.session_state["generador_manual_geocodificado"]
        st.success("Dirección localizada. Se utilizarán estas coordenadas para calcular distancias y rutas.")
        st.caption(f"Ubicación encontrada: {st.session_state.get('direccion_manual_resultado', '')}")
        st.write(
            f"**Generador:** {generador_seleccionado['nombproy']}  \
"
            f"**Dirección ingresada:** {generador_seleccionado['dirproy']}  \
"
            f"**Coordenadas:** {generador_seleccionado['latitud_geo']:.6f}, "
            f"{generador_seleccionado['longitud_geo']:.6f}"
        )

else:
    indice = opciones_generador.index(seleccion_generador) - 1
    generador_seleccionado = generadores_catalogo.iloc[indice].to_dict()
    # Si se cambia de un generador manual a uno del catálogo, limpiar la ubicación anterior.
    st.session_state.pop("generador_manual_geocodificado", None)
    st.session_state.pop("direccion_manual_resultado", None)
    st.success("Generador del catálogo seleccionado correctamente.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nombre", str(generador_seleccionado.get("nombproy", "Sin nombre")))
    c2.metric("ID del proyecto", str(generador_seleccionado.get("idproy", "Sin ID")))
    c3.markdown(f"**Dirección**\n\n{generador_seleccionado.get('dirproy', 'Sin dirección')}")

st.divider()
st.header("2. Caracterización de los residuos")

modalidad = st.radio(
    "¿Cómo se encuentran los residuos?",
    ["Separados", "Mezclados"],
    horizontal=True
)

residuos = []

if modalidad == "Separados":
    codigos = st.multiselect(
        "Selecciona los materiales",
        list(MATERIALES.keys()),
        format_func=lambda codigo: f"{codigo} - {MATERIALES[codigo]['nombre']}"
    )

    for codigo in codigos:
        cantidad = st.number_input(
            f"Cantidad de {MATERIALES[codigo]['nombre']} (toneladas)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            key=f"cantidad_{codigo}"
        )
        if cantidad > 0:
            residuos.append({
                "Codigo_residuo": codigo,
                "Tipo_residuo": MATERIALES[codigo]["nombre"],
                "Cantidad_toneladas": float(cantidad)
            })
else:
    cantidad = st.number_input(
        "Cantidad total de residuos mezclados (toneladas)",
        min_value=0.0,
        value=0.0,
        step=0.5
    )
    descripcion = st.text_area(
        "Describe la composición conocida de la mezcla",
        placeholder="Ej. Mezcla de concreto, ladrillos y material de excavación."
    )
    if cantidad > 0:
        residuos.append({
            "Codigo_residuo": "RCD_MIX",
            "Tipo_residuo": "RCD mezclados",
            "Cantidad_toneladas": float(cantidad),
            "Descripcion": descripcion
        })

if residuos:
    st.info(
        f"Cantidad total registrada: "
        f"{sum(r['Cantidad_toneladas'] for r in residuos):.2f} toneladas."
    )
    st.dataframe(pd.DataFrame(residuos), use_container_width=True)

st.divider()
st.header("3. Información del transporte")

col1, col2, col3 = st.columns(3)
with col1:
    combustible = st.selectbox("Combustible", ["Diésel B2", "Gasolina"])

with col2:
    capacidad = st.number_input(
        "Capacidad del vehículo (toneladas)",
        min_value=0.0,
        value=10.0,
        step=0.5
    )

with col3:
    consumo = st.number_input(
        "Consumo (L/100 km)",
        min_value=0.0,
        value=25.0,
        step=0.5
    )


st.divider()
st.header("4. Generar recomendación")

if st.button("Analizar gestión de RCD", type="primary", use_container_width=True):

    if generador_seleccionado is None:
        st.warning("Primero debes buscar y seleccionar un generador.")
    elif not residuos:
        st.warning("Selecciona al menos un material e ingresa una cantidad mayor que cero.")
    elif modalidad == "Mezclados" and not residuos[0].get("Descripcion", "").strip():
        st.warning("Describe la composición conocida de los residuos mezclados.")
    else:
        with st.spinner("Evaluando compatibilidad por material..."):
            cercanos = calcular_gestores_cercanos(generador_seleccionado, gestores)
            evaluacion_por_material = filtrar_gestores_por_material(
                residuos, modalidad, matriz_compatibilidad, cercanos
            )

        st.subheader("Resultados de compatibilidad por material")
        total_compatibles = sum(len(par[0]) for par in evaluacion_por_material.values())
        total_pendientes = sum(len(par[1]) for par in evaluacion_por_material.values())
        c1, c2 = st.columns(2)
        c1.metric("Compatibilidades descriptivas reportadas", total_compatibles)
        c2.metric("Pendientes de verificación", total_pendientes)
        st.caption(
            "IMPORTANTE: la compatibilidad y la recomendación son resultados distintos. "
            "La compatibilidad identifica gestores cuya descripción reportada menciona "
            "el material (o requiere verificación); es un filtro descriptivo y no determina "
            "el orden final. La recomendación selecciona y ordena hasta tres gestores "
            "usando la tabla validada del notebook, con un puntaje equilibrado de distancia "
            "por carretera y tiempo estimado (50 % y 50 %). Un gestor puede ser compatible "
            "sin quedar entre las tres alternativas recomendadas."
        )
        st.caption(
            "Un mismo gestor puede aparecer para varios materiales. La aplicación evalúa "
            "cada material y su transporte por separado; no construye rutas entre centros "
            "de aprovechamiento ni consolida toda la carga en un único viaje."
        )

        resultados_emisiones = []
        mapas_por_material = []

        for residuo in residuos:
            codigo = str(residuo["Codigo_residuo"])
            nombre_material = str(residuo["Tipo_residuo"])
            cantidad_material = float(residuo["Cantidad_toneladas"])
            compatibles, pendientes = evaluacion_por_material.get(
                codigo, (pd.DataFrame(), pd.DataFrame())
            )

            st.markdown(f"---")
            st.subheader(f"{nombre_material} — {cantidad_material:.2f} toneladas")
            especificos = (
                int((compatibles["Nivel_evidencia"] == "Específica").sum())
                if "Nivel_evidencia" in compatibles.columns else len(compatibles)
            )
            generales = (
                int((compatibles["Nivel_evidencia"] == "General").sum())
                if "Nivel_evidencia" in compatibles.columns else 0
            )
            st.markdown("**Compatibilidad descriptiva**")
            c_esp, c_gen, c_pend = st.columns(3)
            c_esp.metric("Específica reportada", especificos)
            c_gen.metric("General (confirmar)", generales)
            c_pend.metric("Pendiente de verificación", len(pendientes))
            st.caption(
                "Estos conteos describen coincidencias encontradas en la información de los gestores. "
                "No equivalen al número de alternativas recomendadas ni al orden de preferencia."
            )

            if not pendientes.empty:
                with st.expander(f"Pendientes de verificación — {nombre_material}"):
                    columnas_p = ["nomb_disp_", "categoria_", "distancia_linea_recta_km", "Evidencia"]
                    st.dataframe(
                        pendientes[[c for c in columnas_p if c in pendientes.columns]],
                        use_container_width=True, hide_index=True
                    )

            if compatibles.empty:
                st.info(
                    "La matriz descriptiva no encontró coincidencias textuales para este material. "
                    "La recomendación se consultará de todas formas en la tabla validada del notebook."
                )
            else:
                with st.expander(f"Evidencia de compatibilidad descriptiva — {nombre_material}"):
                    columnas_e = ["nomb_disp_", "categoria_", "Nivel_evidencia", "Evidencia", "Descripcion_original"]
                    st.dataframe(
                        compatibles[[c for c in columnas_e if c in compatibles.columns]],
                        use_container_width=True, hide_index=True
                    )

            # Para generadores del catálogo, las alternativas y su orden vienen
            # directamente de la tabla validada en el notebook. No se recalculan
            # distancias ni tiempos por OSRM en cada consulta.
            if str(generador_seleccionado.get("idproy", "")) != "MANUAL":
                id_generador = normalizar_id(
                    generador_seleccionado.get("idproy", "")
                )
                rutas_material = recomendaciones_notebook[
                    (recomendaciones_notebook["_idproy_match"] == id_generador)
                    & (recomendaciones_notebook["Codigo_residuo"] == codigo.upper())
                ].copy()

                if not rutas_material.empty:
                    rutas_material = rutas_material.merge(
                        gestores[["id_sitio_d", "categoria_"]],
                        left_on="ID_gestor", right_on="id_sitio_d", how="left"
                    )
                    rutas_material = rutas_material.rename(columns={
                        "categoria_": "Categoria",
                        "distancia_carretera_km": "Distancia_carretera_km",
                        "tiempo_ruta_minutos": "Tiempo_estimado_min",
                        "orden_recomendacion": "Orden_recomendacion",
                        "puntaje_equilibrado": "Puntaje_equilibrado"
                    })
                    rutas_material["Nivel_evidencia"] = "Compatibilidad reportada"
                    rutas_material["Distancia_linea_recta_km"] = float("nan")
                    rutas_material = rutas_material.dropna(
                        subset=["Distancia_carretera_km", "Tiempo_estimado_min"]
                    ).sort_values("Orden_recomendacion").reset_index(drop=True)
            else:
                # Para una dirección manual no existe una fila en la tabla del notebook;
                # se conserva el cálculo dinámico como alternativa exploratoria.
                with st.spinner(f"Calculando rutas para {nombre_material}..."):
                    rutas_material = calcular_rutas_generador_dinamico(
                        generador_seleccionado, compatibles
                    )

            if rutas_material.empty:
                if codigo == "RCD_MIX":
                    st.warning(
                        "La tabla validada del notebook no contiene recomendaciones confirmadas "
                        "para residuos mezclados. Los gestores pendientes se muestran solo para "
                        "verificación; no se presentan como opciones compatibles."
                    )
                else:
                    st.warning(
                        f"No hay recomendaciones validadas para {nombre_material} "
                        "en este generador, o no fue posible asociar sus coordenadas."
                    )
                continue

            # Las emisiones se calculan con la cantidad de este material,
            # no con la suma de todos los materiales del proyecto.
            emisiones_material, viajes_material, factor = calcular_emisiones(
                rutas_material, [residuo], capacidad, consumo, combustible
            )
            # Estas columnas pueden venir ya incluidas en la tabla de recomendaciones.
            # Asignarlas evita el ValueError de insertar una columna duplicada.
            emisiones_material["Codigo_residuo"] = codigo
            emisiones_material["Material"] = nombre_material
            emisiones_material["Cantidad_toneladas_material"] = cantidad_material
            resultados_emisiones.append(emisiones_material)

            mejor = emisiones_material.iloc[0]
            st.markdown("### ⭐ Recomendación principal")
            nivel_mejor = str(mejor.get("Nivel_evidencia", "No clasificada"))
            if nivel_mejor == "Específica":
                criterio_evidencia = (
                    "su descripción menciona específicamente el material solicitado"
                )
            elif nivel_mejor == "General":
                criterio_evidencia = (
                    "solo se encontró una referencia general al material o a su categoría; "
                    "se requiere confirmar la recepción antes de contratar"
                )
            else:
                criterio_evidencia = "su evidencia requiere revisión adicional"
            st.success(
                f"Para **{nombre_material}**, la alternativa recomendada es "
                f"**{mejor['Nombre_gestor']}**. La selección corresponde al orden "
                "equilibrado calculado en el notebook, que combina distancia por carretera "
                "y tiempo estimado de viaje. Las emisiones se presentan como indicador adicional."
            )
            st.caption(
                "Las alternativas y su orden provienen de recomendaciones_RCD_IA.csv. "
                "El puntaje equilibrado combina distancia y tiempo (50 % y 50 %). "
                "Las emisiones se calculan en esta aplicación con los parámetros de transporte "
                "ingresados. Esto no sustituye la verificación de autorización, capacidad "
                "ni condiciones reales de recepción del gestor."
            )
            st.markdown("**Gestor recomendado**")
            st.markdown(
                f"<div style='padding:14px 16px; border:1px solid #d8e6dd; "
                f"border-left:5px solid #2e9d50; border-radius:8px; "
                f"font-size:20px; font-weight:600; overflow-wrap:anywhere;'>"
                f"{escape(str(mejor['Nombre_gestor']))}</div>",
                unsafe_allow_html=True
            )
            b, c, d = st.columns(3)
            b.metric("Distancia por trayecto", f"{mejor['Distancia_carretera_km']:.2f} km")
            c.metric("Tiempo por trayecto", f"{mejor['Tiempo_estimado_min']:.1f} min")
            d.metric("Emisiones estimadas", f"{mejor['Emisiones_CO2_kg']:.2f} kg CO₂")
            st.caption(
                f"Viajes estimados para este material: {viajes_material}. "
                f"Factor aplicado: {factor:.4f} kg CO₂/L."
            )

            columnas = [
                "Orden_emisiones", "Nombre_gestor", "Categoria", "Nivel_evidencia",
                "Distancia_carretera_km", "Tiempo_estimado_min", "Emisiones_CO2_kg"
            ]
            if "Puntaje_equilibrado" in emisiones_material.columns:
                columnas.insert(4, "Puntaje_equilibrado")
            st.dataframe(
                emisiones_material[[c for c in columnas if c in emisiones_material.columns]].rename(columns={
                    "Orden_emisiones": "Orden", "Nombre_gestor": "Gestor",
                    "Categoria": "Categoría", "Distancia_carretera_km": "Distancia (km)",
                    "Tiempo_estimado_min": "Tiempo (min)", "Emisiones_CO2_kg": "Emisiones (kg CO₂)",
                    "Nivel_evidencia": "Compatibilidad", "Puntaje_equilibrado": "Puntaje equilibrado"
                }),
                use_container_width=True, hide_index=True
            )

            # Guardar resultados independientes por material.
            emisiones_material.to_csv(
                RESULTADOS_DIR / f"emisiones_{codigo}.csv", index=False, encoding="utf-8-sig"
            )
            rutas_material.to_csv(
                RESULTADOS_DIR / f"rutas_{codigo}.csv", index=False, encoding="utf-8-sig"
            )

            st.markdown(f"**Mapa de rutas — {nombre_material}**")
            with st.spinner("Construyendo mapa interactivo..."):
                mapa, dibujadas, errores = crear_mapa(
                    generador_seleccionado, rutas_material, emisiones_material, [residuo]
                )
            if mapa is not None:
                components.html(mapa.get_root().render(), height=650, scrolling=True)
                st.caption(
                    f"Gestores representados: {len(emisiones_material)} | "
                    f"Rutas dibujadas: {dibujadas} | Errores: {errores}"
                )

        if resultados_emisiones:
            consolidado = pd.concat(resultados_emisiones, ignore_index=True)
            consolidado.to_csv(
                RESULTADOS_DIR / "emisiones_consolidadas_por_material.csv",
                index=False, encoding="utf-8-sig"
            )
            st.divider()
            st.subheader("Resumen general de la gestión")
            resumen = consolidado.sort_values(
                ["Codigo_residuo", "Orden_emisiones"]
            ).groupby("Codigo_residuo", as_index=False).first()[
                ["Material", "Cantidad_toneladas_material", "Nombre_gestor",
                 "Distancia_carretera_km", "Tiempo_estimado_min", "Emisiones_CO2_kg"]
            ]
            st.dataframe(
                resumen.rename(columns={
                    "Material": "Material", "Cantidad_toneladas_material": "Cantidad (t)",
                    "Nombre_gestor": "Gestor recomendado", "Distancia_carretera_km": "Distancia (km)",
                    "Tiempo_estimado_min": "Tiempo (min)", "Emisiones_CO2_kg": "Emisiones (kg CO₂)"
                }), use_container_width=True, hide_index=True
            )
            st.success(
                "Análisis terminado. La recomendación se presenta por material; "
                "por tanto, madera y arena pueden tener gestores diferentes."
            )
        else:
            st.warning(
                "No fue posible generar recomendaciones con evidencia para ninguno de los materiales. "
                "Revisa la descripción reportada de los gestores o verifica opciones por otra fuente."
            )

        st.info(
            "Las recomendaciones para generadores del catálogo provienen de la tabla validada "
            "en el notebook y priorizan un puntaje equilibrado de distancia y tiempo. Los tiempos "
            "son estimaciones de ruta y no incorporan tráfico en tiempo real. Las emisiones se "
            "estiman con el combustible, capacidad y consumo ingresados, para viajes de ida y vuelta. "
            "La tabla actual no contiene recomendaciones confirmadas para residuos mezclados. "
            "Siempre deben verificarse autorizaciones vigentes, capacidad real, condiciones de "
            "recepción y restricciones de acceso para el vehículo."
        )

st.divider()
with st.expander("Estado de las bases de datos"):
    st.write(f"Generadores cargados: {len(generadores)}")
    st.write(f"Gestores cargados: {len(gestores)}")
    st.write(f"Registros de la matriz de compatibilidad: {len(matriz_compatibilidad)}")

st.caption("RCD IA | Herramienta de apoyo para la gestión de residuos de construcción y demolición.")