"""Actualización segura de las bases geográficas de RCD IA.

Generadores: consulta el servicio REST oficial de la SDA.
Gestores: permite incorporar un GeoJSON oficial descargado/validado por el usuario.
Nunca reemplaza la última base válida si la descarga o validación falla.
"""
from __future__ import annotations
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
import requests
import certifi
import ssl
from requests.adapters import HTTPAdapter

try:
    import truststore
except ImportError:  # permite seguir usando certifi si no se instaló el extra
    truststore = None

# La SDA publica el mismo servicio con dos variantes de host/ruta.
# Se prueban en orden, sin desactivar nunca la validación SSL.
URLS_GENERADORES = (
    "https://www.secretariadeambiente.gov.co/arcgis/rest/services/Calidad_Suelo/generadorrcd/MapServer/0/query",
    "https://secretariadeambiente.gov.co/arcgis/rest/services/Calidad_suelo/generadorrcd/MapServer/0/query",
)
URL_GENERADORES = URLS_GENERADORES[0]  # compatibilidad


def _validar_geojson(datos: Any, tipo: str) -> tuple[bool, str, int]:
    if not isinstance(datos, dict) or datos.get("type") != "FeatureCollection":
        return False, "El archivo no es un GeoJSON FeatureCollection válido.", 0
    features = datos.get("features")
    if not isinstance(features, list) or len(features) == 0:
        return False, "La fuente no contiene registros.", 0
    validos = 0
    for f in features:
        if not isinstance(f, dict):
            continue
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates")
        if geom.get("type") != "Point" or not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            continue
        if -75 <= lon <= -73 and 4 <= lat <= 5:
            validos += 1
    minimo = 10 if tipo == "generadores" else 1
    if validos < minimo:
        return False, f"Solo se encontraron {validos} puntos dentro del área esperada de Bogotá; se conserva la base anterior.", validos
    return True, f"GeoJSON válido: {validos} puntos con coordenadas dentro del área esperada.", validos


def _guardar_atomico(ruta: Path, contenido: bytes) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fd, temporal = tempfile.mkstemp(prefix=ruta.stem + "_", suffix=".tmp", dir=ruta.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contenido)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporal, ruta)
    finally:
        if os.path.exists(temporal):
            os.remove(temporal)


def _crear_sesion_segura() -> requests.Session:
    """Crea una sesión HTTPS usando también el almacén de certificados del sistema.

    En Windows esto permite confiar en certificados raíz instalados por el sistema
    (por ejemplo, los de un proxy institucional o antivirus), sin desactivar SSL.
    Si truststore no está instalado, requests usa su almacén habitual (certifi).
    """
    ca_bundle = os.environ.get("RCDIA_CA_BUNDLE", "").strip()
    if ca_bundle:
        ruta_ca = Path(ca_bundle).expanduser()
        if not ruta_ca.is_file():
            raise FileNotFoundError(
                "La ruta configurada en RCDIA_CA_BUNDLE no existe. "
                "Verifica el archivo de certificado CA en formato PEM."
            )

    session = requests.Session()
    session.headers.update({"User-Agent": "RCD-IA/1.0 (actualizacion de datos geograficos)"})

    if truststore is not None and not ca_bundle:
        class AdaptadorAlmacenSistema(HTTPAdapter):
            def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
                pool_kwargs["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                return super().init_poolmanager(connections, maxsize, block, **pool_kwargs)

        session.mount("https://", AdaptadorAlmacenSistema())

    return session


def descargar_generadores(timeout: int = 90) -> tuple[dict, str]:
    """Prueba las dos direcciones oficiales de SDA con SSL siempre activado."""
    ca_bundle = os.environ.get("RCDIA_CA_BUNDLE", "").strip()
    if ca_bundle and not Path(ca_bundle).expanduser().is_file():
        raise FileNotFoundError(
            "La ruta configurada en RCDIA_CA_BUNDLE no existe. "
            "Verifica el archivo de certificado CA en formato PEM."
        )
    verificacion_ssl = str(Path(ca_bundle).expanduser()) if ca_bundle else True
    errores = []
    page_size = 1500

    for endpoint in URLS_GENERADORES:
        session = _crear_sesion_segura()
        offset, features = 0, []
        try:
            while True:
                params = {
                    "where": "1=1", "outFields": "*", "returnGeometry": "true",
                    "outSR": "4326", "f": "geojson", "resultOffset": offset,
                    "resultRecordCount": page_size, "orderByFields": "OBJECTID"
                }
                response = session.get(
                    endpoint, params=params, timeout=(15, timeout), verify=verificacion_ssl
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("error"):
                    raise RuntimeError(payload["error"].get("message", "Error del servicio ArcGIS"))
                page = payload.get("features", [])
                if not isinstance(page, list):
                    raise RuntimeError("La respuesta del servicio no contiene una lista de registros.")
                features.extend(page)
                if len(page) < page_size:
                    break
                offset += len(page)
                if offset > 100000:
                    raise RuntimeError("La consulta superó el límite de seguridad de registros.")
            if not features:
                raise RuntimeError("El servicio respondió, pero no devolvió registros.")
            return {"type": "FeatureCollection", "features": features}, endpoint
        except requests.exceptions.SSLError as exc:
            errores.append(f"{endpoint}: error SSL ({exc})")
        except (requests.exceptions.RequestException, ValueError, RuntimeError) as exc:
            errores.append(f"{endpoint}: {exc}")
        finally:
            session.close()

    detalle = "\n".join(errores)
    if all("ssl" in e.lower() or "certificate" in e.lower() for e in errores):
        raise requests.exceptions.SSLError(
            "No fue posible validar el certificado SSL en ninguna de las direcciones oficiales. "
            "Se mantuvo activa la verificación. Detalle: " + detalle
        )
    raise RuntimeError("No fue posible descargar los generadores desde la SDA. " + detalle)

def actualizar_generadores(ruta_destino: str | Path) -> dict:
    ruta = Path(ruta_destino)
    datos, url = descargar_generadores()
    ok, mensaje, n = _validar_geojson(datos, "generadores")
    if not ok:
        raise ValueError(mensaje)
    contenido = json.dumps(datos, ensure_ascii=False).encode("utf-8")
    _guardar_atomico(ruta, contenido)
    return {"base": "generadores", "estado": "actualizada", "registros": n,
            "fecha_consulta": datetime.now().astimezone().isoformat(timespec="seconds"),
            "fuente": url, "mensaje": mensaje}


def _decodificar_geojson(contenido: bytes) -> dict:
    """Lee GeoJSON con codificaciones comunes de exportaciones GIS en Windows.

    No altera la validación JSON ni geográfica. Una vez leído, el archivo se
    normaliza a UTF-8 al guardarlo en disco.
    """
    if not isinstance(contenido, (bytes, bytearray)):
        raise ValueError("El archivo cargado no contiene bytes legibles.")

    errores = []
    # UTF-8 y UTF-16 cubren exportaciones estándar; cp1252/latin-1 son
    # frecuentes en archivos GIS antiguos con tildes o caracteres especiales.
    for codificacion in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            texto = bytes(contenido).decode(codificacion)
            datos = json.loads(texto)
            if not isinstance(datos, dict):
                raise ValueError("La raíz del JSON no es un objeto.")
            return datos
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errores.append(f"{codificacion}: {exc}")

    detalle = " | ".join(errores[-2:])
    raise ValueError(
        "No se pudo interpretar el archivo como JSON. Verifica que sea un GeoJSON "
        "válido y no un archivo comprimido o una página web descargada. " + detalle
    )


def actualizar_generadores_desde_archivo(archivo, ruta_destino: str | Path) -> dict:
    """Incorpora manualmente un GeoJSON de generadores validado por el usuario.

    Acepta UTF-8, UTF-16, Windows-1252 y Latin-1. La base vigente solo se
    reemplaza después de validar el contenido y las coordenadas.
    """
    contenido = archivo if isinstance(archivo, (bytes, bytearray)) else archivo.read()
    datos = _decodificar_geojson(contenido)

    ok, mensaje, n = _validar_geojson(datos, "generadores")
    if not ok:
        raise ValueError(mensaje)

    # Se conserva la estructura completa de propiedades y geometrías del GeoJSON.
    contenido_normalizado = json.dumps(datos, ensure_ascii=False).encode("utf-8")
    _guardar_atomico(Path(ruta_destino), contenido_normalizado)
    return {
        "base": "generadores",
        "estado": "actualizada_manual",
        "registros": n,
        "fecha_consulta": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fuente": "Archivo GeoJSON de generadores cargado manualmente",
        "mensaje": mensaje,
    }


def actualizar_gestores_desde_archivo(archivo, ruta_destino: str | Path) -> dict:
    """Incorpora GeoJSON oficial de gestores con detección de codificación."""
    contenido = archivo if isinstance(archivo, (bytes, bytearray)) else archivo.read()
    datos = _decodificar_geojson(contenido)
    ok, mensaje, n = _validar_geojson(datos, "gestores")
    if not ok:
        raise ValueError(mensaje)
    _guardar_atomico(Path(ruta_destino), json.dumps(datos, ensure_ascii=False).encode("utf-8"))
    return {"base": "gestores", "estado": "actualizada", "registros": n,
            "fecha_consulta": datetime.now().astimezone().isoformat(timespec="seconds"),
            "fuente": "Archivo GeoJSON oficial cargado por el usuario", "mensaje": mensaje}


def registrar_eventos(eventos: list[dict], ruta_historial: str | Path) -> None:
    ruta = Path(ruta_historial)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    anterior = []
    if ruta.exists():
        try:
            anterior = json.loads(ruta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            anterior = []
    anterior.extend(eventos)
    ruta.write_text(json.dumps(anterior, ensure_ascii=False, indent=2), encoding="utf-8")
