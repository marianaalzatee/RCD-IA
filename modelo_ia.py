"""Modelo de aprendizaje no supervisado DBSCAN para inteligencia territorial RCD."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

RADIO_TIERRA_M = 6_371_000
RADIO_SELECCIONADO_M = 250
MIN_SAMPLES = 5


def entrenar_dbscan(generadores: pd.DataFrame, eps_metros: int = RADIO_SELECCIONADO_M,
                    min_samples: int = MIN_SAMPLES):
    """Entrena DBSCAN Haversine siguiendo el radio final de 250 m del notebook.

    Se conserva el filtro espacial del notebook (Bogotá: lon -75 a -73, lat 4 a 5).
    Devuelve (registros_etiquetados, resumen_clusters, métricas).
    """
    df = generadores.copy()
    for col in ("longitud", "latitud"):
        if col not in df.columns:
            raise ValueError(f"La base no contiene la coordenada {col}.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[
        df["longitud"].between(-75, -73) & df["latitud"].between(4, 5)
    ].copy().reset_index(drop=True)
    if len(df) < min_samples:
        raise ValueError(f"Solo hay {len(df)} registros con coordenadas válidas; se requieren al menos {min_samples}.")
    coordenadas_rad = np.radians(df[["latitud", "longitud"]].to_numpy())
    modelo = DBSCAN(eps=eps_metros / RADIO_TIERRA_M, min_samples=min_samples, metric="haversine")
    df["cluster_dbscan"] = modelo.fit_predict(coordenadas_rad)
    df["es_punto_nucleo"] = False
    df.loc[modelo.core_sample_indices_, "es_punto_nucleo"] = True
    agrupados = df[df["cluster_dbscan"] != -1]
    resumen = (agrupados.groupby("cluster_dbscan")
        .agg(numero_generadores=("cluster_dbscan", "size"),
             latitud_centro=("latitud", "mean"), longitud_centro=("longitud", "mean"),
             latitud_min=("latitud", "min"), latitud_max=("latitud", "max"),
             longitud_min=("longitud", "min"), longitud_max=("longitud", "max"))
        .reset_index().sort_values("numero_generadores", ascending=False).reset_index(drop=True))
    resumen["porcentaje_total"] = resumen["numero_generadores"] / len(df) * 100
    diametros = []
    for cluster_id, grupo in agrupados.groupby("cluster_dbscan"):
        coords = np.radians(grupo[["latitud", "longitud"]].to_numpy())
        diametro = float(haversine_distances(coords).max() * RADIO_TIERRA_M)
        diametros.append((cluster_id, diametro))
    if diametros:
        resumen["diametro_maximo_m"] = resumen["cluster_dbscan"].map(dict(diametros))
    else:
        resumen["diametro_maximo_m"] = pd.Series(dtype=float)
    ruido = int((df["cluster_dbscan"] == -1).sum())
    n_clusters = int(df.loc[df["cluster_dbscan"] != -1, "cluster_dbscan"].nunique())
    metricas = {
        "radio_metros": int(eps_metros), "min_samples": int(min_samples),
        "generadores_analizados": int(len(df)), "clusters": n_clusters,
        "ruido": ruido, "agrupados": int(len(df) - ruido),
        "porcentaje_agrupado": float((len(df) - ruido) / len(df) * 100),
        "porcentaje_ruido": float(ruido / len(df) * 100)
    }
    return df, resumen, metricas


def comparar_radios(generadores: pd.DataFrame, radios=(250, 500, 750, 1000), min_samples=MIN_SAMPLES):
    """Compara radios como en la etapa 10.2.6 del notebook."""
    resultados = []
    for radio in radios:
        _, _, m = entrenar_dbscan(generadores, int(radio), min_samples)
        resultados.append(m)
    return pd.DataFrame(resultados)


def analizar_cercania_gestores(resumen_clusters: pd.DataFrame, gestores: pd.DataFrame,
                               radios_alerta=(1000, 2000, 5000)) -> pd.DataFrame:
    """Calcula distancia en línea recta desde cada centroide a gestores registrados.

    La cercanía es un indicador exploratorio: no mide capacidad, tráfico ni cobertura real.
    """
    if resumen_clusters.empty:
        return resumen_clusters.copy()
    g = gestores.copy()
    for col in ("longitud", "latitud"):
        if col not in g.columns:
            raise ValueError(f"La base de gestores no contiene la coordenada {col}.")
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=["longitud", "latitud"]).copy()
    if g.empty:
        raise ValueError("No hay gestores con coordenadas válidas para comparar.")
    nombre_col = next((c for c in ("nombre_sitio", "nom_establecimiento", "nombre", "razon_social", "Nombre_gestor", "nombgestor", "nomb_disp_") if c in g.columns), None)
    id_col = next((c for c in ("id_sitio_d", "ID_gestor", "id_gestor", "id") if c in g.columns), None)
    nombres = g[nombre_col].fillna("Gestor sin nombre").astype(str) if nombre_col else pd.Series(["Gestor registrado"]*len(g), index=g.index)
    ids = g[id_col].fillna("").astype(str) if id_col else pd.Series([""]*len(g), index=g.index)
    coords_g = np.radians(g[["latitud", "longitud"]].to_numpy())
    salida = resumen_clusters.copy()
    distancias_min, nombres_min, ids_min = [], [], []
    conteos = {int(r): [] for r in radios_alerta}
    for _, cluster in salida.iterrows():
        punto = np.radians([[float(cluster["latitud_centro"]), float(cluster["longitud_centro"]) ]])
        d = haversine_distances(punto, coords_g)[0] * RADIO_TIERRA_M
        idx = int(np.argmin(d))
        distancias_min.append(float(d[idx])); nombres_min.append(nombres.iloc[idx]); ids_min.append(ids.iloc[idx])
        for radio in radios_alerta:
            conteos[int(radio)].append(int((d <= int(radio)).sum()))
    salida["gestor_mas_cercano"] = nombres_min
    salida["id_gestor_mas_cercano"] = ids_min
    salida["distancia_gestor_mas_cercano_m"] = distancias_min
    for radio, valores in conteos.items():
        salida[f"gestores_a_{radio}m"] = valores
    # Regla transparente de priorización exploratoria: tamaño y distancia por encima
    # de la mediana de los clústeres. No representa una brecha de servicio comprobada.
    med_tam = salida["numero_generadores"].median()
    med_dist = salida["distancia_gestor_mas_cercano_m"].median()
    salida["tamano_superior_mediana"] = salida["numero_generadores"] >= med_tam
    salida["distancia_superior_mediana"] = salida["distancia_gestor_mas_cercano_m"] >= med_dist
    condiciones = salida["tamano_superior_mediana"].astype(int) + salida["distancia_superior_mediana"].astype(int)
    salida["prioridad_analisis"] = condiciones.map({2: "Alta", 1: "Media", 0: "Baja"})
    return salida


def generar_hallazgos_territoriales(resumen: pd.DataFrame, metricas: dict) -> list[str]:
    """Produce hallazgos descriptivos y trazables; no genera afirmaciones causales."""
    if resumen.empty:
        return ["No se identificaron clústeres con los parámetros seleccionados."]
    mayor = resumen.iloc[0]
    alta = int((resumen.get("prioridad_analisis", pd.Series(dtype=str)) == "Alta").sum())
    lejos = resumen["distancia_gestor_mas_cercano_m"].idxmax() if "distancia_gestor_mas_cercano_m" in resumen else None
    hallazgos = [
        f"Se identificaron {metricas['clusters']} clústeres entre {metricas['generadores_analizados']:,} registros analizados; {metricas['porcentaje_agrupado']:.1f}% quedó agrupado y {metricas['porcentaje_ruido']:.1f}% fue clasificado como ruido.",
        f"El clúster de mayor tamaño reúne {int(mayor['numero_generadores'])} generadores (clúster {int(mayor['cluster_dbscan'])}).",
    ]
    if "prioridad_analisis" in resumen.columns:
        hallazgos.append(f"{alta} clúster(es) quedaron en prioridad alta de análisis según la regla exploratoria de tamaño y distancia al gestor más cercano.")
    if lejos is not None:
        fila = resumen.loc[lejos]
        hallazgos.append(f"La mayor distancia en línea recta a un gestor registrado es {fila['distancia_gestor_mas_cercano_m']/1000:.2f} km (clúster {int(fila['cluster_dbscan'])}).")
    hallazgos.append("Estos resultados orientan dónde profundizar el diagnóstico. No permiten concluir falta de capacidad, toneladas disponibles ni ausencia de cobertura: esas variables no están en las bases analizadas.")
    return hallazgos
