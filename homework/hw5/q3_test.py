import pickle
from model import normalize_record

# 1. Cargar el pipeline entrenado usando pickle
with open('pipeline.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# 2. Datos del lead indicados en la pregunta
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

# 3. Normalizar el registro con la función auxiliar del curso
record = normalize_record(lead)

# 4. Obtener la probabilidad de conversión usando predict_proba
probability = float(pipeline.predict_proba([record])[0, 1])

print(f"Probabilidad de conversión: {probability:.6f}")