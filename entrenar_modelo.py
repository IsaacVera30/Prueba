# entrenar_modelo.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# 1. Cargar los datos desde el CSV generado por tu prototipo
df = pd.read_csv("entrenamiento_ml.csv")

# 2. Seleccionar las características usadas también en app.py
features = [
    'hr_promedio_sensor',
    'spo2_promedio_sensor',
    'ir_mean_filtrado',
    'red_mean_filtrado',
    'ir_std_filtrado',
    'red_std_filtrado'
]

X = df[features]           # Variables de entrada
y_sys = df['sys_ref']      # Variable objetivo para presión sistólica
y_dia = df['dia_ref']      # Variable objetivo para presión diastólica

# 3. Entrenar los modelos con regresión lineal
modelo_sys = LinearRegression()
modelo_sys.fit(X, y_sys)

modelo_dia = LinearRegression()
modelo_dia.fit(X, y_dia)

# 4. Evaluar los modelos con RMSE (Root Mean Squared Error)
# ⚠️ Corregido: reemplazar squared=False por np.sqrt
pred_sys = modelo_sys.predict(X)
pred_dia = modelo_dia.predict(X)

print("SYS RMSE:", np.sqrt(mean_squared_error(y_sys, pred_sys)))
print("DIA RMSE:", np.sqrt(mean_squared_error(y_dia, pred_dia)))

# 5. Guardar los modelos entrenados en formato .pkl
joblib.dump(modelo_sys, "modelo_sys.pkl")
joblib.dump(modelo_dia, "modelo_dia.pkl")
