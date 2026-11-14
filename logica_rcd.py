"""Funciones comunes de limpieza y validación para RCD IA."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def normalizar_id(valor) -> str:
    """Normaliza identificadores para cruces sin ceros iniciales."""
    if pd.isna(valor):
        return ""
    texto = str(valor).strip()
    if texto.endswith(".0"):
        texto = texto[:-2]
    if texto.isdigit():
        return texto.lstrip("0") or "0"
    return texto


def leer_geojson_dataframe(ruta: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Convierte un GeoJSON de puntos a DataFrame, conservando propiedades."""
    with open(ruta, "r", encoding=encoding) as archivo:
        datos = json.load(archivo)
    filas = []
    for feature in datos.get("features", []):
        fila = (feature.get("properties") or {}).copy()
        geometria = feature.get("geometry") or {}
        coordenadas = geometria.get("coordinates")
        if geometria.get("type") == "Point" and coordenadas and len(coordenadas) >= 2:
            fila["longitud"] = coordenadas[0]
            fila["latitud"] = coordenadas[1]
        else:
            fila["longitud"] = None
            fila["latitud"] = None
        filas.append(fila)
    return pd.DataFrame(filas)


def limpiar_generadores(generadores: pd.DataFrame) -> pd.DataFrame:
    """Replica la depuración del notebook: conserva el nombre más completo por idproy."""
    df = generadores.copy()
    if "idproy" not in df.columns:
        raise ValueError("La base de generadores no contiene la columna idproy.")
    if "nombproy" not in df.columns:
        df["nombproy"] = ""
    df["_longitud_nombre"] = df["nombproy"].fillna("").astype(str).str.strip().str.len()
    df = df.sort_values("_longitud_nombre", ascending=False)
    df = df.drop_duplicates(subset="idproy", keep="first").drop(columns="_longitud_nombre")
    for col in ["longitud", "latitud"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def validar_recomendaciones(tabla: pd.DataFrame) -> None:
    requeridas = {
        "idproy", "Nombre_gestor", "Codigo_residuo", "Tipo_residuo", "ID_gestor",
        "orden_recomendacion", "distancia_carretera_km", "tiempo_ruta_minutos",
        "puntaje_equilibrado"
    }
    faltantes = sorted(requeridas - set(tabla.columns))
    if faltantes:
        raise ValueError("Faltan columnas en recomendaciones_RCD_IA.csv: " + ", ".join(faltantes))
