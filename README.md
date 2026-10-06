# Machine Learning Zoomcamp 2026

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Solutions, notes, and code for the **Machine Learning Zoomcamp 2026** offered by [DataTalks.Club](https://datatalks.club/).

Soluciones, notas y código para el **Machine Learning Zoomcamp 2026** impartido por [DataTalks.Club](https://datatalks.club/).

---

## 📂 Project Structure / Estructura del Proyecto

```text
machine-learning-zoomcamp/
├── .gitignore               # Ignored files (virtualenvs, .idea, checkpoints)
├── 00_environment_test.ipynb# Environment verification notebook
├── environment.yml          # Conda environment definition
├── README.md                # Main repository documentation
├── README_ENTORNO.md        # Detailed environment setup guide
├── requirements.txt         # Pip dependencies
├── datasets/                # Raw & processed datasets
├── docs/                    # Documentation and additional guides
├── homework/                # Homework assignments by week
│   ├── hw1/                 # Week 1: Introduction to Machine Learning
│   └── hw2/                 # Week 2: Machine Learning for Regression
├── models/                  # Trained models (.bin, .pkl, .onnx)
├── notebooks/               # Experimental notebooks
└── notes/                   # Personal study notes and summaries
```

---

## 📊 Progress / Progreso

| Module | Topic | Homework | Status       |
|---|---|---|--------------|
| **01** | Introduction to ML | [HW 01](./homework/hw1/) | ✅ Completed  |
| **02** | Machine Learning for Regression | [HW 02](./homework/hw2/) | ✅  Completed |
| **03** | Machine Learning for Classification | HW 03 | ⏳ Pending    |
| **04** | Evaluation Metrics for Classification | HW 04 | ⏳ Pending    |
| **05** | Deploying Machine Learning Models | HW 05 | ⏳ Pending    |
| **06** | Decision Trees & Ensemble Learning | HW 06 | ⏳ Pending    |
| **07** | Neural Networks & Deep Learning | HW 07 | ⏳ Pending    |
| **08** | Serverless Deep Learning | HW 08 | ⏳ Pending    |
| **09** | Kubernetes & TensorFlow Serving | HW 09 | ⏳ Pending    |
| **10** | Capstone Projects | Midterm / Capstone | ⏳ Pending    |

---

## 🛠️️ Tech Stack / Stack Tecnológico

* **Language:** Python 3.12
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Environments & Tools:** Conda, JupyterLab, PyCharm, Git


---

## 🚀 How to Run / Cómo Ejecutar

### Option 1: Conda Environment (Recommended / Recomendado)

Consulta la guía detallada paso a paso en [README_ENTORNO.md](./README_ENTORNO.md) o ejecuta los siguientes comandos:

```bash
# 1. Clone the repository / Clonar repositorio
git clone https://github.com/angel-monterrosas/machine-learning-zoomcamp.git
cd machine-learning-zoomcamp

# 2. Create and activate Conda environment / Crear y activar entorno Conda
conda create -n mlzoomcamp python=3.12 -y
conda activate mlzoomcamp

# 3. Install dependencies / Instalar dependencias
pip install -r requirements.txt

# 4. Register Jupyter Kernel / Registrar el kernel en Jupyter
python -m ipykernel install --user --name mlzoomcamp --display-name "Python (mlzoomcamp)"

# 5. Launch JupyterLab / Iniciar JupyterLab
jupyter lab
```

### Option 2: Python `venv`

```bash
# Clone the repository
git clone https://github.com/angel-monterrosas/machine-learning-zoomcamp.git
cd machine-learning-zoomcamp

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies and start Jupyter
pip install -r requirements.txt
jupyter notebook
```