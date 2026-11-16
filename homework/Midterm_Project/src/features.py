"""Feature engineering para el modelo BCMM."""
import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["TIPO", "TIPO_BIEN", "CLAVE_CONTINENTE", "CLAVE_REGION"]
NUMERIC_COLS = ["ANIO", "MES", "MES_SIN", "MES_COS", "es_periodo_ajustado"]
TARGET = "VAL_USD"


def add_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    """Codificación cíclica del mes (seno/coseno)."""
    df = df.copy()
    df["MES_SIN"] = np.sin(2 * np.pi * df["MES"] / 12)
    df["MES_COS"] = np.cos(2 * np.pi * df["MES"] / 12)
    return df


def temporal_split(df: pd.DataFrame):
    """
    Split temporal:
      - train: 2015-2023
      - val:   2024
      - test:  2025-2026
    """
    train = df[df["ANIO"] <= 2023].copy()
    val   = df[df["ANIO"] == 2024].copy()
    test  = df[df["ANIO"] >= 2025].copy()
    return train, val, test


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformaciones de features (sin codificar aún)."""
    df = add_cyclic_month(df)
    return df