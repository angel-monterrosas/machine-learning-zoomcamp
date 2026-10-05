# Configuración de Entorno de Desarrollo

## Machine Learning Zoomcamp 2026

Este documento detalla los pasos e instrucciones necesarios para preparar el entorno de trabajo utilizado durante el **DataTalks.Club Machine Learning Zoomcamp 2026**.

---

## 🛠️ Herramientas Requeridas

Asegúrate de contar con las siguientes herramientas e instalaciones principales:

* **Git** (control de versiones)
* **Miniconda3** (gestor de entornos virtuales y paquetes)
* **Python 3.12**
* **PyCharm** (IDE de desarrollo)
* **Jupyter Notebook / JupyterLab**

---

## 🚀 Guía de Configuración Paso a Paso

Las siguientes instrucciones fueron probadas en **Windows 10(64-bit)** utilizando **Anaconda Prompt**.

---

### 1. Creación del Entorno Virtual

Abre **Anaconda Prompt** e ingresa el siguiente comando para crear un entorno aislado con Python 3.12:

```bash
conda create -n mlzoomcamp python=3.12 -y
```

---

### 2. Activación del Entorno

Activa el entorno que acabas de crear:

```bash
conda activate mlzoomcamp
```

Verifica que la versión instalada sea la correcta:

```bash
python --version
```

> **Resultado esperado:** `Python 3.12.x`

---

### 3. Instalación de Librerías Principales

Instala los paquetes fundamentales para Data Science y Machine Learning:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

### 4. Instalación de Entornos Jupyter

Instala Jupyter Notebook, JupyterLab y el kernel para conectarlo con el entorno virtual:

```bash
pip install notebook jupyterlab ipykernel
```

Para verificar todas las librerías e identificadores instalados en el entorno:

```bash
pip list
```

---

### 5. Registro del Kernel en Jupyter

Registra tu entorno virtual como un Kernel disponible en Jupyter Notebook/JupyterLab y PyCharm:

```bash
python -m ipykernel install --user --name mlzoomcamp --display-name "Python (mlzoomcamp)"
```

---

## 💡 Configuración del Intérprete en PyCharm

1. Abre **PyCharm** y navega a tu proyecto del Zoomcamp.
2. Ve a **File > Settings > Project: <nombre_proyecto> > Python Interpreter**.
3. Haz clic en **Add Interpreter > Add Local Interpreter...**
4. Selecciona **Conda Environment**.
5. En **Use existing environment**, selecciona `mlzoomcamp`.