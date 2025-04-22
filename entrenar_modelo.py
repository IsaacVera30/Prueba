import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Cargar el dataset corregido
df = pd.read_csv("registro_sensor_entrenamiento.csv")

# Entradas y salidas
X = df[["hr", "spo2"]]   # Entradas del modelo
y_sys = df["sys"]        # Salida para modelo sistólico
y_dia = df["dia"]        # Salida para modelo diastólico

# Entrenar modelo sistólico
modelo_sys = LinearRegression()
modelo_sys.fit(X, y_sys)
joblib.dump(modelo_sys, "modelo_sys.pkl")

# Entrenar modelo diastólico
modelo_dia = LinearRegression()
modelo_dia.fit(X, y_dia)
joblib.dump(modelo_dia, "modelo_dia.pkl")

print("✅ Modelos entrenados y guardados como 'modelo_sys.pkl' y 'modelo_dia.pkl'")
