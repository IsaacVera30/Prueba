# train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# --- 1. Cargar el CSV de entrenamiento ---
df = pd.read_csv("entrenamiento_ml.csv")

# --- 2. Seleccionar las variables de entrada ---
features = [
    "hr_promedio_sensor",
    "spo2_promedio_sensor",
    "ir_mean_filtrado",
    "red_mean_filtrado",
    "ir_std_filtrado",
    "red_std_filtrado"
]

X = df[features].values
y_sys = df["sys_ref"].values
y_dia = df["dia_ref"].values

# --- 3. Escalado de las features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")  # 💾 Guardar el scaler

# --- 4. División de entrenamiento y prueba ---
X_train, X_test, y_sys_train, y_sys_test = train_test_split(X_scaled, y_sys, test_size=0.2, random_state=42)
_, _, y_dia_train, y_dia_test = train_test_split(X_scaled, y_dia, test_size=0.2, random_state=42)

# --- 5. Entrenamiento de modelos ---
modelo_sys = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_sys.fit(X_train, y_sys_train)

modelo_dia = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_dia.fit(X_train, y_dia_train)

# --- 6. Evaluación rápida (prototipo) ---
pred_sys = modelo_sys.predict(X_test)
pred_dia = modelo_dia.predict(X_test)

print("MAE SYS:", round(mean_absolute_error(y_sys_test, pred_sys), 2))
print("MAE DIA:", round(mean_absolute_error(y_dia_test, pred_dia), 2))

# --- 7. Guardar modelos entrenados ---
joblib.dump(modelo_sys, "models/modelo_sys.pkl")
joblib.dump(modelo_dia, "models/modelo_dia.pkl")
print("✅ Modelos y scaler entrenados y guardados correctamente.")
