"""Carga y limpieza de la Balanza Comercial de Mercancías de México (BCMM)."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Códigos que no representan países reales (agregados continentales/regionales o no identificados)
AGGREGATES = ["AME", "ASI", "EUR", "ZZZ", "EMU", "YUG", "RKE", "NCN", "PTY"]


def load_catalogs(raw_dir: Path = RAW_DIR) -> dict:
    """Carga los catálogos de países, regiones y continentes."""
    try:
        return {
            "paises": pd.read_csv(raw_dir / "tc_pais.csv"),
            "regiones": pd.read_csv(raw_dir / "tc_region.csv"),
            "continentes": pd.read_csv(raw_dir / "tc_continente.csv"),
        }
    except FileNotFoundError as e:
        print(f"Advertencia: No se pudieron cargar los catálogos: {e}")
        return {"paises": pd.DataFrame(), "regiones": pd.DataFrame(), "continentes": pd.DataFrame()}


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Carga y concatena los 3 CSVs del INEGI."""
    files = sorted(raw_dir.glob("bcmm_paises_bien_mensual_tr_cifra_*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSVs en {raw_dir}")
    dfs = [pd.read_csv(f).assign(ARCHIVO_ORIGEN=f.stem) for f in files]
    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame, catalogs: dict) -> pd.DataFrame:
    """
    Limpieza:
      - Filtra al nivel país real (descarta agregados).
      - Elimina filas confidenciales usando el ESTATUS oficial.
      - Crea columna `periodo` (datetime mensual).
      - Enriquecer con catálogos (nombres de países, regiones, continentes).
    """
    # 1) Filtrar nivel país real
    df = df[df["PAIS_O_D"].notna() & ~df["PAIS_O_D"].isin(AGGREGATES)].copy()

    # 2) Eliminar confidenciales basándonos en el estatus oficial
    df = df[df["ESTATUS_CIFRA"] != "Confidencial"].copy()

    # 3) Periodo datetime
    df["periodo"] = pd.to_datetime(
        dict(year=df["ANIO"].astype(int), month=df["MES"].astype(int), day=1)
    )

    # 4) Flag periodo ajustado (cifras revisadas 2023-2025)
    df["es_periodo_ajustado"] = df["periodo"].between("2023-01-01", "2025-12-31").astype(int)

    # 5) Tipos
    df["ANIO"] = df["ANIO"].astype(int)
    df["MES"] = df["MES"].astype(int)

    # 6) Enriquecimiento con catálogos (merge)
    if not catalogs["paises"].empty:
        df = df.merge(catalogs["paises"], on="PAIS_O_D", how="left")
    if not catalogs["regiones"].empty:
        df = df.merge(catalogs["regiones"], on="CLAVE_REGION", how="left")
    if not catalogs["continentes"].empty:
        df = df.merge(catalogs["continentes"], on="CLAVE_CONTINENTE", how="left")

    return df.reset_index(drop=True)


def load_and_clean(save: bool = True) -> pd.DataFrame:
    """Pipeline completo: carga, limpia, enriquece y (opcionalmente) guarda en processed/."""
    catalogs = load_catalogs()
    df = clean(load_raw(), catalogs)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / "bcmm_pais_clean.parquet"
        df.to_parquet(out, index=False)
        print(f"Datos procesados guardados en: {out} ({len(df):,} filas)")

    return df


if __name__ == "__main__":
    df = load_and_clean()
    print("\nMuestra de datos enriquecidos:")
    columnas = ["ANIO", "MES", "PAIS_O_D", "PAIS_INEGI", "REGION", "VAL_USD"]
    print(df[columnas].head())
    print(f"\nDimensiones del dataset: {df.shape}")
    print(f"Rango temporal: {df['periodo'].min().strftime('%Y-%m')} a {df['periodo'].max().strftime('%Y-%m')}")