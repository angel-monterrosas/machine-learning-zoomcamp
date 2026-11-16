"""Predictor de inferencia: carga bundle y predice con inputs simples."""
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
BUNDLE_PATH = MODEL_DIR / "bundle.pkl"

# Constantes de validación
MIN_ANIO, MAX_ANIO = 2015, 2030
MIN_MES, MAX_MES = 1, 12

# Versión del formato del bundle esperada.
# Debe coincidir con BUNDLE_FORMAT_VERSION en src/train.py
BUNDLE_FORMAT_EXPECTED = 2

TipoFlujo = Literal["Exportaciones", "Importaciones"]
TipoBien = Literal["Consumo", "Intermedio", "Capital"]


class Predictor:
    """Clase para cargar el modelo en producción y realizar inferencias."""

    def __init__(self, bundle_path: Path = BUNDLE_PATH):
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto del modelo en {bundle_path}. "
                "Ejecute 'python -m src.train' para generarlo."
            )

        logger.info(f"Cargando artefactos del modelo desde: {bundle_path}")
        bundle = joblib.load(bundle_path)

        # Validar que el bundle tenga el formato correcto (con nombres de catálogos)
        fmt = bundle.get("format_version", 1)
        if fmt < BUNDLE_FORMAT_EXPECTED:
            raise RuntimeError(
                f"El bundle en {bundle_path} tiene formato v{fmt}, "
                f"pero se espera v{BUNDLE_FORMAT_EXPECTED}. "
                "Reentrena el modelo ejecutando 'uv run python -m src.train'."
            )

        self.pipeline = bundle["pipeline"]
        self.pais_lookup: dict = bundle["pais_lookup"]
        self.winner_name: str = bundle["winner_name"]
        self.version: str = bundle["version"]
        self.metrics_val: dict = bundle.get("metrics_val", {})

        logger.info(f"Modelo '{self.winner_name}' v{self.version} cargado correctamente.")

    def predict_one(
        self,
        pais_o_d: str,
        tipo: TipoFlujo,
        tipo_bien: TipoBien,
        anio: int,
        mes: int,
    ) -> dict:
        """
        Genera una predicción para un flujo comercial específico.

        Args:
            pais_o_d: Código ISO-3 del país.
            tipo: Flujo comercial (Exportaciones o Importaciones).
            tipo_bien: Categoría del bien (Consumo, Intermedio, Capital).
            anio: Año de la predicción.
            mes: Mes de la predicción (1-12).

        Returns:
            Diccionario con el valor predicho y metadatos enriquecidos.

        Raises:
            ValueError: Si los parámetros de entrada están fuera de rango o son inválidos.
        """
        pais_o_d = pais_o_d.upper()

        # Validaciones
        if pais_o_d not in self.pais_lookup:
            raise ValueError(
                f"País no válido: '{pais_o_d}'. "
                f"Consulte el catálogo de los {len(self.pais_lookup)} países disponibles."
            )
        if not (MIN_MES <= mes <= MAX_MES):
            raise ValueError(f"El campo 'mes' debe estar entre {MIN_MES} y {MAX_MES}.")
        if not (MIN_ANIO <= anio <= MAX_ANIO):
            raise ValueError(f"El campo 'anio' debe estar entre {MIN_ANIO} y {MAX_ANIO}.")

        info = self.pais_lookup[pais_o_d]

        # Construcción del vector de características (solo lo que el modelo necesita)
        row = {
            "PAIS_O_D": pais_o_d,
            "CLAVE_CONTINENTE": info.get("clave_continente"),  # Puede ser None -> pandas NaN
            "CLAVE_REGION": info.get("clave_region"),          # Puede ser None -> pandas NaN
            "TIPO": tipo,
            "TIPO_BIEN": tipo_bien,
            "ANIO": anio,
            "MES": mes,
            "MES_SIN": np.sin(2 * np.pi * mes / 12),
            "MES_COS": np.cos(2 * np.pi * mes / 12),
            "es_periodo_ajustado": int(2023 <= anio <= 2025),
        }

        X = pd.DataFrame([row])
        y_log = self.pipeline.predict(X)[0]
        y = float(np.expm1(y_log))

        # Advertencia para periodos con menor certidumbre
        warning = None
        if anio >= 2026:
            warning = (
                "Predicción para 2026 o posterior: los datos de este periodo tienen estatus "
                "'Cifras Revisadas' y el modelo presenta menor precisión en este rango "
                "(R2 test aprox. 0.76). Se recomienda interpretar el resultado con cautela."
            )

        # Respuesta enriquecida (incluye nombres legibles de catálogos)
        return {
            "predicted_val_usd_miles": round(y, 2),
            "pais_o_d": pais_o_d,
            "clave_continente": info.get("clave_continente"),
            "clave_region": info.get("clave_region"),
            "nombre": info.get("nombre", pais_o_d),
            "nombre_continente": info.get("nombre_continente"),
            "nombre_region": info.get("nombre_region"),
            "tipo": tipo,
            "tipo_bien": tipo_bien,
            "anio": anio,
            "mes": mes,
            "model": self.winner_name,
            "version": self.version,
            "warning": warning,
        }

    def list_paises(self) -> list[dict]:
        """
        Retorna el catálogo de países disponibles ordenado alfabéticamente.
        Usa un contrato estricto para evitar errores de validación en Pydantic.
        """
        return sorted(
            [
                {
                    "pais_o_d": k,
                    "clave_continente": v.get("clave_continente"),
                    "clave_region": v.get("clave_region"),
                    "nombre": v.get("nombre", k),
                    "nombre_continente": v.get("nombre_continente"),
                    "nombre_region": v.get("nombre_region"),
                }
                for k, v in self.pais_lookup.items()
            ],
            key=lambda x: x["pais_o_d"],
        )

    def metadata(self) -> dict:
        """Retorna los metadatos y métricas de validación del modelo en producción."""
        return {
            "model": self.winner_name,
            "version": self.version,
            "n_paises": len(self.pais_lookup),
            "metrics_val": self.metrics_val,
        }


# Singleton para la API
_predictor = None


def get_predictor() -> Predictor:
    """Retorna la instancia única del predictor (Patrón Singleton)."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor