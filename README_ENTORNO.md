# Configuración de entorno para el zoocamp de DataTalk
## Machine Learning - Zoocamp 2026 

--- 
Herramientas necesarias:
---
##### 
PyCharm
Miniconda
Python 3.12
Git
Jupyter Notebook
JupyterLab

Para configurar el entorno que ocuparan: en mi caso ocupe los siguientes
una vez instalado miniconda3. 

Windows 64-Bit Graphical Installer

Creacion del entorno para agregar interprete y librerias que ocupare:
utilizando Anaconda Prompt: 
seguir la siguiente lista de comoda

`Anaconda Promtp` 
creacion del entorno dedicado para el zoocamp: 

conda create -n mlzoomcamp python=3.12

activar el entorno: 

conda activate mlzoomcamp 

Comprobar versión de Python: 
python --version

en este caso: Python 3.12.14

instalanado librerias utiles: 

pip install pandas numpy matplotlib seaborn scikit-learn

instalar Jupyter: 
pip install notebook jupyterlab ipykernel

verificar paquetes instalados: 
pip list

registrar kernel para jupyter: 
python -m ipykernel install --user --name mlzoomcamp --display

python -m ipykernel install --user --name mlzoomcamp --display-name "Python (mlzoomcamp)"