# Machine Learning Zoomcamp — Homework 5
# Machine Learning Zoomcamp — Tarea 5

**Results Log / Registro de Resultados**

---

## Summary / Resumen

| # | Topic / Tema | Key Result / Resultado clave |
|---|--------------|------------------------------|
| 1 | Environment check / Comprobación del entorno | `uv 0.13.5` |
| 2 | Pinned Scikit-Learn version / Versión de Scikit-Learn | `1.7.2` |
| 3 | Load model & predict / Cargar modelo y predecir | `0.532994` |
| 4 | Serve model via API / Servir modelo vía API | `0.769799` (`conversion: True`) |
| 5 | Container base image / Imagen base del contenedor | `python:3.11.15-slim-bookworm@sha256:d29f48a3…` |
| 6 | Docker build & smoke test / Build Docker y prueba | `0.769799` · `/health` → `ok` |

---

## Q1 — Environment Check
## Pregunta 1 — Comprobación del entorno

**EN:** Ran `uv --version` to validate the environment and dependency-management tool.

**ES:** Ejecución de `uv --version` para validar la herramienta de gestión de entornos y dependencias.

**Result / Resultado:**

```text
uv 0.13.5 (46776a95d 2026-10-21 x86_64-pc-windows-msvc)
```

---

## Q2 — Pinned Dependency
## Pregunta 2 — Dependencia cerrada

**EN:** Verified the Scikit-Learn version declared in the reference environment.

**ES:** Verificación de la versión de Scikit-Learn declarada en el entorno de referencia.

**Command / Comando:**

```bash
uv run python -c "import sklearn; print(sklearn.__version__)"
```

**Result / Resultado:** `1.7.2`

---

## Q3 — Load the Model
## Pregunta 3 — Cargar el modelo

**EN:** Created and ran a Python script to load `pipeline.bin` with pickle, normalize the test lead record, and compute the conversion probability.

**ES:** Creación y ejecución de un script de Python para cargar `pipeline.bin` con pickle, normalizar el registro del lead de prueba y calcular la probabilidad de conversión.

**Script executed / Script ejecutado (`q3_test.py`):**

```python
import pickle
from model import normalize_record

with open('pipeline.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

record = normalize_record(lead)
probability = float(pipeline.predict_proba([record])[0, 1])
print(f"Probabilidad de conversión: {probability:.6f}")
```

**Result / Resultado:** `0.532994`

---

## Q4 — Serve the Model
## Pregunta 4 — Servir el modelo

**EN:** Started the API with Uvicorn and sent a POST request with the second lead's data to evaluate the prediction endpoint.

**ES:** Inicio de la API con Uvicorn y envío de una petición POST con los datos del segundo lead para evaluar el endpoint de predicción.

**Request / Petición (PowerShell):**

```powershell
$body = @{
    lead_source              = "organic_search"
    industry                 = "technology"
    employment_status        = "employed"
    location                 = "europe"
    number_of_courses_viewed = 4
    annual_income            = 80304.0
    interaction_count        = 7
    lead_score               = 0.74
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:9696/predict" `
                  -Method Post `
                  -Body $body `
                  -ContentType "application/json"
```

**Result / Resultado:** `0.769799` (with field `conversion: True` / con campo `conversion: True`)

---

## Q5 — Inspect the Container Configuration
## Pregunta 5 — Mire la configuración del contenedor

**EN:** Inspected the `Dockerfile` to identify the exact base image tag declared for Python.

**ES:** Inspección del archivo `Dockerfile` para identificar la etiqueta exacta de la imagen base de Python declarada.

**Command / Comando:**

```powershell
Get-Content Dockerfile
```

**Result / Resultado:**

```text
python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
```

---

## Q6 — Make the Container Work
## Pregunta 6 — Hacer funcionar el contenedor

**EN:** Built the Docker image in a WSL (Ubuntu) environment, ran the container, tested the smoke script `q6_test.py`, and verified the health route (`/health`).

**ES:** Construcción de la imagen Docker en un entorno WSL (Ubuntu), ejecución del contenedor, prueba del script de humo `q6_test.py` y verificación de la ruta de salud (`/health`).

**Commands executed in WSL / Comandos ejecutados en WSL:**

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python3 q6_test.py
curl -s http://localhost:9696/health
```

**Results / Resultados:**

- **Inference with test script / Inferencia con script de prueba (`q6_test.py`):**

```python
{'conversion_probability': 0.769799, 'conversion': True}
```

- **Health endpoint check / Verificación del endpoint `/health`:**

```json
{
  "status": "ok",
  "model_sha256": "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1",
  "feature_defaults_sha256": "1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde"
}
```

---

