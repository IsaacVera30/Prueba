import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

# --- 1. Configuración ---
CSV_PATH = 'entrenamiento_ml.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True) # Crea la carpeta 'models' si no existe

# --- 2. Carga de Datos ---
print(f"Cargando datos desde '{CSV_PATH}'...")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{CSV_PATH}'. Asegúrate de que exista.")
    exit()

df.dropna(inplace=True) # Elimina filas con datos incompletos
print(f"Datos cargados. Se usarán {len(df)} muestras para el entrenamiento.")

# --- 3. Entrenamiento Directo de los 4 Modelos ---

# Definimos el set completo de características de entrada
features = [
    'hr_promedio_sensor', 
    'spo2_promedio_sensor',
    'ir_mean_filtrado',
    'red_mean_filtrado',
    'ir_std_filtrado',
    'red_std_filtrado'
]
X = df[features]

print("\n--- INICIANDO ENTRENAMIENTO DIRECTO ---")

# --- Modelo para SYS (Regresión Lineal) ---
print("Entrenando modelo para: SYS")
y_sys = df['sys_ref']
modelo_sys = LinearRegression()
modelo_sys.fit(X, y_sys)
joblib.dump(modelo_sys, os.path.join(MODELS_DIR, 'modelo_sys.pkl'))
print(" -> Modelo 'modelo_sys.pkl' guardado.")

# --- Modelo para DIA (Regresión Lineal) ---
print("Entrenando modelo para: DIA")
y_dia = df['dia_ref']
modelo_dia = LinearRegression()
modelo_dia.fit(X, y_dia)
joblib.dump(modelo_dia, os.path.join(MODELS_DIR, 'modelo_dia.pkl'))
print(" -> Modelo 'modelo_dia.pkl' guardado.")

<<<<<<< HEAD
# --- Modelo para HR (Regresión Lineal) ---
print("Entrenando modelo para: HR")
y_hr = df['hr_ref']
modelo_hr = LinearRegression()
modelo_hr.fit(X, y_hr)
joblib.dump(modelo_hr, os.path.join(MODELS_DIR, 'modelo_hr.pkl'))
print(" -> Modelo 'modelo_hr.pkl' guardado.")

# --- Modelo para SPO2 (Gradient Boosting) ---
print("Entrenando modelo para: SPO2")
# Para SPO2, quitamos la propia variable SPO2 de las entradas para no hacer trampa
X_spo2 = X.drop(columns=['spo2_promedio_sensor'])
y_spo2 = df['spo2_promedio_sensor']
print(" (Aviso: Se ha quitado 'spo2_promedio_sensor' de las entradas para este modelo)")
modelo_spo2 = GradientBoostingRegressor(n_estimators=100, random_state=42)
modelo_spo2.fit(X_spo2, y_spo2)
joblib.dump(modelo_spo2, os.path.join(MODELS_DIR, 'modelo_spo2.pkl'))
print(" -> Modelo 'modelo_spo2.pkl' guardado.")


print("\n¡Proceso completado! Tus 4 modelos han sido entrenados y guardados en la carpeta 'models'.")
=======
print("Modelos entrenados y guardados como 'modelo_sys.pkl' y 'modelo_dia.pkl'")
>>>>>>> 77deff23507c89e9bdf8ec41ce24a5d65cab89a8
