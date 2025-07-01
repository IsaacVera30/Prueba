import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# --- 1. Configuración ---
CSV_PATH = 'entrenamiento_ml.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 2. Análisis y preparación de datos ---
print(f"\nCargando y analizando datos de '{CSV_PATH}'...")
df = pd.read_csv(CSV_PATH)

# Análisis inicial
print("\n=== Resumen de datos ===")
print(f"- Total de muestras: {len(df)}")
print("- Valores faltantes por columna:")
print(df.isnull().sum())

# Limpieza
df_clean = df.dropna()
if len(df) != len(df_clean):
    print(f"\nSe eliminaron {len(df) - len(df_clean)} filas con valores faltantes")

# --- 3. Preparación de características ---
features = [
    'hr_promedio_sensor',
    'spo2_promedio_sensor',
    'ir_mean_filtrado',
    'red_mean_filtrado',
    'ir_std_filtrado',
    'red_std_filtrado'
]

X = df_clean[features]
y_sys = df_clean['sys_ref']
y_dia = df_clean['dia_ref']

# División en entrenamiento y prueba (80% train, 20% test)
X_train, X_test, y_sys_train, y_sys_test = train_test_split(X, y_sys, test_size=0.2, random_state=42)
_, _, y_dia_train, y_dia_test = train_test_split(X, y_dia, test_size=0.2, random_state=42)

# --- 4. Entrenamiento y evaluación de modelos ---
print("\n=== Entrenamiento de modelos ===")

# Modelo SYS
print("\n[Modelo SYS - Presión Sistólica]")
modelo_sys = LinearRegression()
modelo_sys.fit(X_train, y_sys_train)

# Evaluación
y_sys_pred = modelo_sys.predict(X_test)
mae_sys = mean_absolute_error(y_sys_test, y_sys_pred)
r2_sys = r2_score(y_sys_test, y_sys_pred)

print(f"- MAE (Error Absoluto Medio): {mae_sys:.2f} mmHg")
print(f"- R² (Coeficiente de Determinación): {r2_sys:.3f}")

# Modelo DIA
print("\n[Modelo DIA - Presión Diastólica]")
modelo_dia = LinearRegression()
modelo_dia.fit(X_train, y_dia_train)

# Evaluación
y_dia_pred = modelo_dia.predict(X_test)
mae_dia = mean_absolute_error(y_dia_test, y_dia_pred)
r2_dia = r2_score(y_dia_test, y_dia_pred)

print(f"- MAE (Error Absoluto Medio): {mae_dia:.2f} mmHg")
print(f"- R² (Coeficiente de Determinación): {r2_dia:.3f}")

# --- 5. Guardado de modelos ---
joblib.dump(modelo_sys, os.path.join(MODELS_DIR, 'modelo_sys.pkl'))
joblib.dump(modelo_dia, os.path.join(MODELS_DIR, 'modelo_dia.pkl'))

print("\n=== Modelos guardados ===")
print(f"- modelo_sys.pkl (MAE: {mae_sys:.2f} mmHg, R²: {r2_sys:.3f})")
print(f"- modelo_dia.pkl (MAE: {mae_dia:.2f} mmHg, R²: {r2_dia:.3f})")
print(f"\nGuardados en: {os.path.abspath(MODELS_DIR)}/")