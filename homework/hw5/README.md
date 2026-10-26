# 🚀 ML Zoomcamp 2026 - Module 5: Model Deployment

This repository contains the solution for the **Module 5 (Model Deployment)** homework of the 2026 cohort of [Machine Learning Zoomcamp](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05). The project implements a prediction API using **FastAPI**, containerized with **Docker**, and managed with the **uv** dependency manager.

---

## 📂 Project Structure
```text
hw5/
├── Dockerfile              # Configuration for containerization
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Version locking with uv
├── .python-version         # Specified Python version (3.11)
├── model.py                # Model loading and processing logic
├── predict.py              # FastAPI application and prediction endpoints
├── pipeline.bin            # Serialized ML model
├── model_metadata.json     # Model metadata and verification hashes
├── feature_defaults.json   # Default values for features
└── q6_test.py              # Test script to validate the Docker container
```

---

## 🛠️ Technologies Used

* **Python 3.11**
* **FastAPI & Uvicorn** (Web framework and ASGI server)
* **Scikit-Learn** (Machine Learning model)
* **Docker** (Containerization)
* **uv** (Ultra-fast Python dependency manager)

---

## 🚀 How to Run the Project with Docker

Make sure you have **Docker** installed and running on your system.

### 1. Build the Docker image
Open your terminal in the project folder and run the following command:
docker build -t zoomcamp-model:2026-hw5 .

### 2. Run the container
Start the container mapping port `9696`:
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5

### 3. Test the service
Open another tab in your terminal and run the checks:

* **Check health status (`/health`):**
  curl -s http://localhost:9696/health
* **Run the prediction test:**
  python q6_test.py

---

## 📝 Main Endpoints

* `GET /health`: Returns the health status of the service and model verification metadata.
* `POST /predict`: Receives input data in JSON format and returns the predicted conversion probability.
