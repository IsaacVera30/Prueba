import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import os, time, csv, requests
import numpy as np
import joblib, pandas as pd
from datetime import datetime
import mysql.connector
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
socketio = SocketIO(app)

# --- Carga de Modelos ---
try:
    modelo_sys = joblib.load('modelo_sys.pkl')
    modelo_dia = joblib.load('modelo_dia.pkl')
    print("✅ Modelos de ML cargados correctamente.")
except Exception as e:
    print(f"⚠️  Advertencia: No se pudieron cargar modelos de ML: {e}")
    modelo_sys, modelo_dia = None, None

# --- Variables y Constantes ---
buffer_datos_entrenamiento = []
last_db_save_time = 0
CSV_FILENAME = "registro_sensor_entrenamiento.csv"
LOCK_FILE = "capture.lock"
DB_CONFIG = {key.replace("MYSQL_", "").lower(): val for key, val in os.environ.items() if key.startswith("MYSQL")}
DB_CONFIG['port'] = int(DB_CONFIG.get('port', 3306))
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
CALLMEBOT_API_KEY = os.environ.get('CALLMEBOT_API_KEY')
CALLMEBOT_PHONE_NUMBER = os.environ.get('CALLMEBOT_PHONE_NUMBER')

# --- Funciones Auxiliares ---
def conectar_db():
    try: return mysql.connector.connect(**DB_CONFIG)
    except Exception as e: print(f"❌ Error DB: {e}"); return None

def guardar_medicion_mysql(data):
    # ... (El código de esta función y las demás auxiliares es idéntico al de mi respuesta anterior)

def procesar_buffer_y_guardar(ref_data):
    global buffer_datos_entrenamiento
    if not buffer_datos_entrenamiento: return
    # Extraer características de la ventana de datos capturada
    features = {
        "hr_promedio_sensor": np.mean([float(d.get("hr_promedio", 0)) for d in buffer_datos_entrenamiento]),
        "spo2_promedio_sensor": np.mean([float(d.get("spo2_sensor", 0)) for d in buffer_datos_entrenamiento]),
        "ir_mean_filtrado": np.mean([float(d.get("ir", 0)) for d in buffer_datos_entrenamiento]),
        "red_mean_filtrado": np.mean([float(d.get("red", 0)) for d in buffer_datos_entrenamiento]),
        "ir_std_filtrado": np.std([float(d.get("ir", 0)) for d in buffer_datos_entrenamiento]),
        "red_std_filtrado": np.std([float(d.get("red", 0)) for d in buffer_datos_entrenamiento])
    }
    # Crear una única fila con las características y los datos de referencia
    final_row = {**features, **ref_data, "timestamp_captura": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    # ... (el resto del guardado en CSV y subida a Drive es idéntico)

# ... (El código de clasificar_nivel_presion y otras funciones auxiliares es idéntico)

### --- RUTAS DE LA API --- ###

@app.route("/")
def home(): return render_template("index.html")

@app.route("/api/data", methods=["POST"])
def recibir_datos():
    global last_db_save_time, buffer_datos_entrenamiento
    data = request.get_json()
    if not data: return jsonify({"error": "No JSON data"}), 400

    # REGLA 1: Siempre enviar datos al panel para visualización en tiempo real.
    socketio.emit('update_data', data)

    # REGLA 2: Si el modo entrenamiento está activo, guardar en buffer.
    if os.path.exists(LOCK_FILE):
        buffer_datos_entrenamiento.append(data)
        socketio.emit('capture_count_update', {'count': len(buffer_datos_entrenamiento)})
    
    # REGLA 3: Si NO estamos en modo entrenamiento, aplicar la lógica de predicción.
    else:
        # La condición para predecir es que haya un dedo puesto.
        if float(data.get("ir", 0)) > 50000:
            if modelo_sys and modelo_dia:
                try:
                    # Crear DataFrame para la predicción
                    input_df = pd.DataFrame([{"hr_promedio_sensor": float(data.get("hr_promedio", 0)), "spo2_promedio_sensor": float(data.get("spo2_sensor", 0)), "ir_mean_filtrado": float(data.get("ir", 0)), "red_mean_filtrado": float(data.get("red", 0)), "ir_std_filtrado": 0, "red_std_filtrado": 0}])
                    # Ejecutar predicción
                    data['sys_ml'] = modelo_sys.predict(input_df)[0]
                    data['dia_ml'] = modelo_dia.predict(input_df)[0]
                    data['estado'] = clasificar_nivel_presion(data['sys_ml'], data['dia_ml'])
                except Exception as e:
                    print(f"❌ Error durante la predicción ML: {e}")
                    data['sys_ml'], data['dia_ml'], data['estado'] = 0, 0, "Error Pred."
            
            # Guardar en Base de Datos cada 5 segundos
            if (time.time() - last_db_save_time) >= 5:
                guardar_medicion_mysql(data)
                socketio.emit('new_record_saved')
                last_db_save_time = time.time()

            # Enviar Alerta si es necesario
            if data.get("estado") == "HT Crisis":
                enviar_alerta_whatsapp(data.get("estado"), data.get("sys_ml"), data.get("dia_ml"))

    # Responder siempre al ESP32 para que actualice su LCD
    response_for_esp = {"sys": data.get("sys_ml", 0), "dia": data.get("dia_ml", 0), "hr": data.get("hr_promedio", 0), "spo2": data.get("spo2_sensor", 0), "nivel": data.get("estado", "Normal")}
    return jsonify(response_for_esp)

# ... (El código de las rutas /api/start_capture, /api/stop_capture, 
#      /api/save_training_data, /api/ultimas_mediciones y /api/test_alert 
#      es idéntico al de mi respuesta anterior)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
