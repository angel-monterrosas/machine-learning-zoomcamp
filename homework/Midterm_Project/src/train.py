"""Entrenamiento de modelos de árboles y tuning para BCMM."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBRegressor

from src.data import load_and_clean
from src.features import (
    build_features, temporal_split,
    CATEGORICAL_COLS, NUMERIC_COLS, TARGET,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Versión del formato del bundle. Si cambia la estructura del dict,
# incrementar para invalidar bundles viejos.
BUNDLE_FORMAT_VERSION = 2


# ---------- Pipeline base ----------
def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("pais_te", TargetEncoder(target_type="continuous", smooth="auto"),
             ["PAIS_O_D"]),
            ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",  # Ignora columnas de catálogos (PAIS_INEGI, CONTINENTE, etc)
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline([("pre", make_preprocessor()), ("model", model)])


# ---------- Evaluación ----------
def evaluate(name: str, pipe, X, y_log) -> dict:
    y_pred_log = pipe.predict(X)
    y_true = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "model": name,
        "R2":   r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
    }


# ---------- Modelos candidatos ----------
def candidate_models() -> dict:
    return {
        "ridge":  Ridge(alpha=1.0),
        "rf":     RandomForestRegressor(
                      n_estimators=200, max_depth=None,
                      min_samples_leaf=5, n_jobs=-1, random_state=42),
        "hgb":    HistGradientBoostingRegressor(
                      max_iter=300, learning_rate=0.05,
                      max_depth=None, random_state=42),
        "xgb":    XGBRegressor(
                      n_estimators=300, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8,
                      n_jobs=-1, random_state=42, tree_method="hist"),
    }


# ---------- Búsqueda de hiperparámetros ----------
PARAM_GRIDS = {
    "rf": {
        "model__n_estimators":      [200, 400, 600, 800],
        "model__max_depth":         [None, 10, 15, 20, 25],
        "model__min_samples_leaf":  [1, 3, 5, 10, 20],
        "model__max_features":      ["sqrt", 0.3, 0.5],
    },
    "hgb": {
        "model__max_iter":          [200, 400, 600],
        "model__learning_rate":     [0.03, 0.05, 0.1],
        "model__max_depth":         [None, 6, 10, 15],
        "model__min_samples_leaf":  [10, 20, 50],
        "model__l2_regularization": [0.0, 1.0, 5.0],
    },
    "xgb": {
        "model__n_estimators":      [200, 400, 600],
        "model__learning_rate":     [0.03, 0.05, 0.1],
        "model__max_depth":         [4, 6, 8, 12],
        "model__subsample":         [0.6, 0.8, 1.0],
        "model__colsample_bytree":  [0.6, 0.8, 1.0],
    },
}


def tune_model(name: str, model, X_train, y_train, n_iter: int = 20) -> Pipeline:
    """RandomizedSearchCV con CV de 3 folds."""
    pipe = make_pipeline(model)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_GRIDS[name],
        n_iter=n_iter,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )
    search.fit(X_train, y_train)
    print(f"Mejores parámetros ({name}): {search.best_params_}")
    print(f"Mejor R2 CV ({name}): {search.best_score_:.4f}")
    return search.best_estimator_


# ---------- Construcción del lookup enriquecido ----------
def build_pais_lookup(df: pd.DataFrame) -> dict:
    """
    Construye el lookup país → metadatos, usando solo las columnas necesarias.

    Rellena nulos de región con 'R00' / 'Sin región asignada' porque el INEGI
    no asigna región a ciertos países (CHN, JPN, KOR, IND, etc.). Solo valida
    que los campos críticos (nombre del país y continente) no estén nulos.
    """
    required = ["PAIS_O_D", "PAIS_INEGI", "CLAVE_CONTINENTE", "CLAVE_REGION",
                "CONTINENTE", "REGION"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas en el DataFrame: {missing}. "
            "¿Ejecutaste load_and_clean() con los catálogos?"
        )

    # DataFrame auxiliar: 1 fila por país con solo las columnas necesarias
    lookup_df = (
        df[required]
        .drop_duplicates(subset=["PAIS_O_D"])
        .set_index("PAIS_O_D")
    )

    # Rellenar nulos de región (INEGI no asigna región a 24 países)
    lookup_df["CLAVE_REGION"] = lookup_df["CLAVE_REGION"].fillna("R00")
    lookup_df["REGION"] = lookup_df["REGION"].fillna("Sin región asignada")

    # Validar solo los campos que sí deberían estar siempre
    critical = ["PAIS_INEGI", "CLAVE_CONTINENTE", "CONTINENTE"]
    null_counts = lookup_df[critical].isna().sum()
    if null_counts.any():
        raise ValueError(
            f"Campos críticos nulos en el lookup de países:\n{null_counts[null_counts > 0]}\n"
            "Verifica que los catálogos tc_pais/tc_continente se cargaron bien."
        )

    # Convertir a dict con el contrato que espera Predictor
    return {
        pais: {
            "nombre": info["PAIS_INEGI"],
            "clave_continente": info["CLAVE_CONTINENTE"],
            "clave_region": info["CLAVE_REGION"],
            "nombre_continente": info["CONTINENTE"],
            "nombre_region": info["REGION"],
        }
        for pais, info in lookup_df.to_dict(orient="index").items()
    }


def main():
    print("Cargando y enriqueciendo datos...")
    df = build_features(load_and_clean())
    train, val, test = temporal_split(df)
    print(f"Split temporal - train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")

    X_train, y_train = train, np.log1p(train[TARGET])
    X_val,   y_val   = val,   np.log1p(val[TARGET])
    X_test,  y_test  = test,  np.log1p(test[TARGET])

    # ---------- Fase 1: comparación rápida ----------
    print("\n" + "-" * 60)
    print("FASE 1 - Comparación de modelos con parámetros por defecto")
    print("-" * 60)

    results = []
    for name, model in candidate_models().items():
        print(f"Entrenando {name}...")
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        results.append(evaluate(f"{name}_val",  pipe, X_val,  y_val))
        results.append(evaluate(f"{name}_test", pipe, X_test, y_test))

    df_res = pd.DataFrame(results).set_index("model").round(4)
    print("\nResultados preliminares:")
    print(df_res.to_string())

    # ---------- Fase 2: tuning ----------
    print("\n" + "-" * 60)
    print("FASE 2 - Tuning de hiperparámetros")
    print("-" * 60)

    best_models = {}
    for name in ["rf", "hgb", "xgb"]:
        print(f"\nAjustando hiperparámetros para {name}...")
        best = tune_model(name, candidate_models()[name], X_train, y_train, n_iter=15)
        best_models[name] = best
        res_val  = evaluate(f"{name}_val_tuned",  best, X_val,  y_val)
        res_test = evaluate(f"{name}_test_tuned", best, X_test, y_test)
        print(pd.DataFrame([res_val, res_test]).set_index("model").round(4).to_string())

    # ---------- Fase 3: guardar bundle ----------
    print("\n" + "-" * 60)
    print("FASE 3 - Evaluación final y guardado del bundle")
    print("-" * 60)

    scores = {
        name: evaluate(name, model, X_val, y_val)["R2"]
        for name, model in best_models.items()
    }

    # Lookup enriquecido (con validación de nulos)
    pais_lookup = build_pais_lookup(df)
    print(f"\nLookup de países: {len(pais_lookup)} (filtrado y enriquecido)")
    print(f"Ejemplo: USA → {pais_lookup.get('USA')}")

    # Forzar XGB para producción
    winner_name = "xgb"
    winner = best_models[winner_name]

    print(f"\nModelo de producción seleccionado: {winner_name} (R2 val = {scores[winner_name]:.4f})")

    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "pipeline": winner,
        "pais_lookup": pais_lookup,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "target": TARGET,
        "winner_name": winner_name,
        "metrics_val": {
            k: {kk: round(float(vv), 4) for kk, vv in evaluate(k, m, X_val, y_val).items()
                if kk != "model"}
            for k, m in best_models.items()
        },
        "trained_at": pd.Timestamp.now().isoformat(),
        "version": "1.2.0",
    }

    bundle_path = MODEL_DIR / "bundle.pkl"
    joblib.dump(bundle, bundle_path)

    print(f"Bundle guardado en: {bundle_path}")
    print(f"Total de países en bundle: {len(pais_lookup)}")
    print(f"Formato del bundle: v{BUNDLE_FORMAT_VERSION}")


if __name__ == "__main__":
    main()