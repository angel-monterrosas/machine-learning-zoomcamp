"""Tests de integración para la API FastAPI."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """
    Cliente de test con lifespan activado.
    El bloque 'with' es obligatorio en versiones recientes de Starlette
    para que se dispare el lifespan (carga del modelo en memoria).
    """
    with TestClient(app) as c:
        yield c


# ---------- Health & metadata ----------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200, f"Error: {r.text}"
    data = r.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "version" in data


def test_metadata(client):
    r = client.get("/metadata")
    assert r.status_code == 200, f"Error: {r.text}"
    data = r.json()
    assert data["model"] == "xgb"
    assert data["version"] == "1.2.0"
    # Ajustado al número real de países filtrados (sin agregados continentales)
    assert data["n_paises"] == 233
    assert "xgb" in data["metrics_val"]
    assert "rf" in data["metrics_val"]
    assert "hgb" in data["metrics_val"]


# ---------- Países ----------
def test_list_paises(client):
    r = client.get("/paises")
    assert r.status_code == 200, f"Error: {r.text}"
    paises = r.json()
    assert isinstance(paises, list), f"Se esperaba lista, recibió: {type(paises)}"
    # Ajustado al número real
    assert len(paises) == 233

    iso_codes = {p["pais_o_d"] for p in paises}

    # Socios principales deben estar presentes
    for iso in ("USA", "CHN", "JPN", "KOR", "IND", "AUS", "DEU", "BRA", "CAN"):
        assert iso in iso_codes, f"Falta {iso}"

    # Los agregados continentales NO deben estar presentes
    for iso in ("AME", "ASI", "EUR", "ZZZ", "EMU", "YUG"):
        assert iso not in iso_codes, f"Agregado {iso} no debería estar en el catálogo"


def test_paises_tienen_nombres_enriquecidos(client):
    """Todos los países deben tener nombre, continente y región (con fallback)."""
    r = client.get("/paises")
    assert r.status_code == 200, f"Error: {r.text}"
    paises = r.json()
    assert isinstance(paises, list), f"Se esperaba lista, recibió: {type(paises)}"

    for p in paises:
        assert p["nombre"] is not None and p["nombre"] != "", f"Falta nombre para {p['pais_o_d']}"
        assert p["nombre_continente"] is not None, f"Falta continente para {p['pais_o_d']}"
        assert p["nombre_region"] is not None, f"Falta región para {p['pais_o_d']}"


def test_pais_sin_region_usa_fallback(client):
    """Países como CHN no tienen región oficial; deben tener fallback R00."""
    r = client.get("/paises")
    assert r.status_code == 200, f"Error: {r.text}"
    paises = r.json()
    assert isinstance(paises, list), f"Se esperaba lista, recibió: {type(paises)}"

    chn = next(p for p in paises if p["pais_o_d"] == "CHN")
    assert chn["clave_continente"] == "C03"
    assert chn["nombre_continente"] == "Asia"
    assert chn["clave_region"] == "R00"
    assert chn["nombre_region"] == "Sin región asignada"


def test_pais_con_region_completa(client):
    """Países como USA sí tienen región real."""
    r = client.get("/paises")
    assert r.status_code == 200, f"Error: {r.text}"
    paises = r.json()
    assert isinstance(paises, list), f"Se esperaba lista, recibió: {type(paises)}"

    usa = next(p for p in paises if p["pais_o_d"] == "USA")
    assert usa["nombre"] == "Estados Unidos"
    assert usa["clave_continente"] == "C01"
    assert usa["nombre_continente"] == "América"
    assert usa["clave_region"] == "R01"
    assert usa["nombre_region"] == "América del Norte"


# ---------- Predict válido ----------
def test_predict_usa_2024(client):
    payload = {
        "pais_o_d": "USA",
        "tipo": "Exportaciones",
        "tipo_bien": "Intermedio",
        "anio": 2024,
        "mes": 6,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"Error: {r.text}"
    data = r.json()
    assert data["predicted_val_usd_miles"] > 1_000_000
    assert data["pais_o_d"] == "USA"
    assert data["nombre"] == "Estados Unidos"
    assert data["nombre_continente"] == "América"
    assert data["nombre_region"] == "América del Norte"
    assert data["model"] == "xgb"
    assert data["version"] == "1.2.0"
    assert data["warning"] is None


def test_predict_chn_2024(client):
    """País sin región oficial debe predecir igual y usar fallback R00."""
    payload = {
        "pais_o_d": "CHN",
        "tipo": "Importaciones",
        "tipo_bien": "Intermedio",
        "anio": 2024,
        "mes": 6,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"Error: {r.text}"
    data = r.json()
    assert data["predicted_val_usd_miles"] > 100_000
    assert data["clave_continente"] == "C03"
    assert data["clave_region"] == "R00"
    assert data["nombre"] == "China"
    assert data["nombre_region"] == "Sin región asignada"


def test_predict_2026_warning(client):
    """Predicción para 2026 debe incluir warning."""
    payload = {
        "pais_o_d": "USA",
        "tipo": "Exportaciones",
        "tipo_bien": "Intermedio",
        "anio": 2026,
        "mes": 3,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"Error: {r.text}"
    data = r.json()
    assert data["warning"] is not None
    assert "2026" in data["warning"]


def test_predict_usa_2024_y_2025_mismo_nivel(client):
    """
    Documenta la limitación conocida: el modelo no captura tendencia interanual
    dentro del periodo 2023-2025. Ambas predicciones deben ser iguales.
    """
    base = {
        "pais_o_d": "USA",
        "tipo": "Exportaciones",
        "tipo_bien": "Intermedio",
        "mes": 6,
    }
    r24 = client.post("/predict", json={**base, "anio": 2024})
    r25 = client.post("/predict", json={**base, "anio": 2025})

    assert r24.status_code == 200, f"Error 2024: {r24.text}"
    assert r25.status_code == 200, f"Error 2025: {r25.text}"
    assert r24.json()["predicted_val_usd_miles"] == r25.json()["predicted_val_usd_miles"]


# ---------- Errores esperados ----------
def test_predict_invalid_pais(client):
    payload = {
        "pais_o_d": "XXX",
        "tipo": "Exportaciones",
        "tipo_bien": "Consumo",
        "anio": 2024,
        "mes": 6,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 400, f"Error: {r.text}"
    assert "XXX" in r.json()["detail"]


def test_predict_agregado_rechazado(client):
    """Los agregados continentales ya no están en el catálogo, deben dar 400."""
    payload = {
        "pais_o_d": "AME",
        "tipo": "Exportaciones",
        "tipo_bien": "Consumo",
        "anio": 2024,
        "mes": 6,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 400, f"Error: {r.text}"
    assert "AME" in r.json()["detail"]


def test_predict_invalid_mes(client):
    payload = {
        "pais_o_d": "USA",
        "tipo": "Exportaciones",
        "tipo_bien": "Consumo",
        "anio": 2024,
        "mes": 13,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_anio(client):
    payload = {
        "pais_o_d": "USA",
        "tipo": "Exportaciones",
        "tipo_bien": "Consumo",
        "anio": 2010,
        "mes": 6,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_tipo(client):
    payload = {
        "pais_o_d": "USA",
        "tipo": "Tránsito",
        "tipo_bien": "Consumo",
        "anio": 2024,
        "mes": 6,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422