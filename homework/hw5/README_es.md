# 🚀 ML Zoomcamp 2026 - Módulo 5: Despliegue de Modelos

Este repositorio contiene la solución para la tarea del **Módulo 5 (Model Deployment)** de la cohorte 2026 de [Machine Learning Zoomcamp](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05). El proyecto implementa una API de predicción utilizando **FastAPI**, empaquetada en un contenedor **Docker** y gestionada con el gestor de dependencias **uv**.

---

## 📂 Estructura del Proyecto
```text
hw5/
├── Dockerfile              # Configuración para la contenedorización
├── pyproject.toml          # Configuración del proyecto y dependencias
├── uv.lock                 # Bloqueo de versiones con uv
├── .python-version         # Versión de Python especificada (3.11)
├── model.py                # Lógica de carga y procesamiento del modelo
├── predict.py              # Aplicación FastAPI y endpoints de predicción
├── pipeline.bin            # Modelo de ML serializado
├── model_metadata.json     # Metadatos y hashes de verificación del modelo
├── feature_defaults.json   # Valores por defecto para las características
└── q6_test.py              # Script de prueba para validar el contenedor Docker
```
---

## 🛠️ Tecnologías Utilizadas

* **Python 3.11**
* **FastAPI & Uvicorn** (Framework web y servidor ASGI)
* **Scikit-Learn** (Modelo de Machine Learning)
* **Docker** (Contenedorización)
* **uv** (Gestor ultra-rápido de dependencias de Python)

---

## 🚀 Cómo Ejecutar el Proyecto con Docker

Asegúrate de tener **Docker** instalado y corriendo en tu sistema.

### 1. Construir la imagen de Docker
Abre tu terminal en la carpeta del proyecto y ejecuta el siguiente comando:
docker build -t zoomcamp-model:2026-hw5 .

### 2. Ejecutar el contenedor
Levanta el contenedor mapeando el puerto `9696`:
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5

### 3. Probar el servicio
Abre otra pestaña en tu terminal y realiza las comprobaciones:

* **Verificar estado de salud (`/health`):**
  curl -s http://localhost:9696/health
* **Ejecutar la prueba de predicción:**
  python q6_test.py

---

## 📝 Endpoints Principales

* `GET /health`: Devuelve el estado de salud del servicio y los metadatos de verificación del modelo.
* `POST /predict`: Recibe los datos de entrada en formato JSON y devuelve la probabilidad de conversión predicha por el modelo.