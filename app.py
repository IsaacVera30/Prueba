import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import os
import mysql.connector
import requests
import numpy as np
import csv
from datetime import datetime
import traceback
import time

# Módulos para Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
socketio = SocketIO(app)

# --- Carga de Modelos ---
# (Esta sección es necesaria si el servidor hace la predicción.
# Si el ESP32 ya envía sys/dia calculados, se puede eliminar joblib, pandas y sklearn)
try:
    modelo_sys = joblib.load('modelo_sys.pkl')
    modelo_dia = joblib.load('modelo_dia.pkl')
    print("✅ Modelos de ML cargados correctamente.")
except Exception as e:
    print(f"⚠️  Advertencia: No se pudieron cargar modelos de ML: {e}")
    modelo_sys, modelo_dia = None, None

# --- Variables Globales ---
capturando_entrenamiento = False
buffer_datos_entrenamiento = []
last_db_save_time = 0
CSV_FILENAME = "registro_sensor_entrenamiento_alta_calidad.csv"

# --- Configuración ---
DB_CONFIG = {
    'host': os.environ.get("MYSQLHOST"), 'user': os.environ.get("MYSQLUSER"),
    'password': os.environ.get("MYSQLPASSWORD"), 'database': os.environ.get("MYSQLDATABASE"),
    'port': int(os.environ.get("MYSQLPORT", "3306"))
}
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
CALLMEBOT_API_KEY = os.environ.get('CALLMEBOT_API_KEY')
CALLMEBOT_PHONE_NUMBER = os.environ.get('CALLMEBOT_PHONE_NUMBER')

# --- Funciones Auxiliares ---
def conectar_db():
    try: return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Error DB: {e}"); return None

def guardar_medicion_mysql(data):
    conn = conectar_db()
    if not conn: return
    cursor = conn.cursor()
    query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s, %s, %s, %s)"
    try:
        cursor.execute(query, (
            data.get("id_paciente"), data.get("sys_ml"),
            data.get("dia_ml"), data.get("estado")
        ))
        conn.commit()
    finally:
        if conn.is_connected(): conn.close()

def enviar_alerta_whatsapp(nivel, sys, dia):
    if not CALLMEBOT_API_KEY or not CALLMEBOT_PHONE_NUMBER: return
    mensaje = f"¡Alerta de Salud! Nivel: {nivel} (SYS: {sys}, DIA: {dia})".replace(" ", "%20")
    url = f"https://api.callmebot.com/whatsapp.php?phone={CALLMEBOT_PHONE_NUMBER}&text={mensaje}&apikey={CALLMEBOT_API_KEY}"
    try:
        requests.get(url, timeout=10)
        print(f"✅ Alerta de WhatsApp enviada.")
    except Exception as e:
        print(f"❌ Excepción al enviar alerta: {e}")

def get_google_drive_service():
    try:
        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        creds = service_account.Credentials.from_service_account_file('service_account.json', scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"❌ Error autenticando con Google Drive: {e}"); return None

def subir_csv_a_drive():
    if not FOLDER_ID: return
    service = get_google_drive_service()
    if not service or not os.path.exists(CSV_FILENAME): return
    try:
        file_metadata = {'name': CSV_FILENAME, 'parents': [FOLDER_ID]}
        media = MediaFileUpload(CSV_FILENAME, mimetype='text/csv', resumable=True)
        query = f"name='{CSV_FILENAME}' and '{FOLDER_ID}' in parents and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        if response.get('files'):
            service.files().update(fileId=response.get('files')[0].get('id'), media_body=media).execute()
        else:
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"✅ Archivo CSV sincronizado con Google Drive.")
    except Exception as e:
        print(f"❌ Error al subir archivo a Drive: {e}")

def procesar_buffer_y_guardar(ref_data):
    global buffer_datos_entrenamiento
    if not buffer_datos_entrenamiento: return
    
    features = {
        "hr_promedio_sensor": np.mean([float(d.get("hr_promedio", 0)) for d in buffer_datos_entrenamiento]),
        "spo2_promedio_sensor": np.mean([float(d.get("spo2_sensor", 0)) for d in buffer_datos_entrenamiento]),
        "ir_mean_filtrado": np.mean([float(d.get("ir", 0)) for d in buffer_datos_entrenamiento]),
        "red_mean_filtrado": np.mean([float(d.get("red", 0)) for d in buffer_datos_entrenamiento]),
        "ir_std_filtrado": np.std([float(d.get("ir", 0)) for d in buffer_datos_entrenamiento]),
        "red_std_filtrado": np.std([float(d.get("red", 0)) for d in buffer_datos_entrenamiento])
    }
    final_row = {**features, **ref_data, "timestamp_captura": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    file_exists = os.path.exists(CSV_FILENAME)
    with open(CSV_FILENAME, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_row.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(final_row)
    
    print(f"✅ Fila de entrenamiento guardada en {CSV_FILENAME}")
    buffer_datos_entrenamiento = []
    subir_csv_a_drive()

def clasificar_nivel_presion(pas, pad):
    if pas is None or pad is None: return "N/A"
    if pas > 180 or pad > 120: return "HT Crisis"
    # ... (otras clasificaciones)
    return "Normal"

### --- RUTAS DE LA API --- ###

@app.route("/")
def home(): return render_template("index.html")

@app.route("/api/data", methods=["POST"])
def recibir_datos():
    global last_db_save_time
    data = request.get_json()
    if not data: return jsonify({"error": "No JSON data"}), 400

    # --- Tarea 1: Manejar modo entrenamiento ---
    if capturando_entrenamiento:
        buffer_datos_entrenamiento.append(data)
        socketio.emit('capture_count_update', {'count': len(buffer_datos_entrenamiento)})

    # --- Tarea 2: Emitir datos al panel ---
    socketio.emit('update_data', data)

    # --- Tarea 3: Lógica de Guardado en DB ---
    if not capturando_entrenamiento:
        try:
            ir_value = float(data.get("ir", 0))
            red_value = float(data.get("red", 0))
            
            if ir_value > 50000 and red_value > 50000: # Condición de dedo presente
                current_time = time.time()
                if (current_time - last_db_save_time) >= 5: # Condición de 5 segundos
                    guardar_medicion_mysql(data)
                    socketio.emit('new_record_saved')
                    last_db_save_time = current_time
                    print("✅ Datos guardados en BD (Dedo detectado y 5s cumplidos).")
        except (ValueError, TypeError):
            pass

    # --- Tarea 4: Lógica de Alertas ---
    if data.get("estado") == "HT Crisis":
        enviar_alerta_whatsapp(data.get("estado"), data.get("sys_ml"), data.get("dia_ml"))
    
    return jsonify({"status": "ok"})

### --- ENDPOINTS PARA ENTRENAMIENTO Y DATOS HISTÓRICOS --- ###

@app.route("/api/start_capture", methods=["POST"])
def start_capture():
    global capturando_entrenamiento, buffer_datos_entrenamiento
    capturando_entrenamiento = True
    buffer_datos_entrenamiento = []
    return jsonify({"status": "captura iniciada"})

@app.route("/api/stop_capture", methods=["POST"])
def stop_capture():
    global capturando_entrenamiento
    capturando_entrenamiento = False
    return jsonify({"status": "captura detenida", "muestras": len(buffer_datos_entrenamiento)})

@app.route("/api/save_training_data", methods=["POST"])
def save_training_data():
    if capturando_entrenamiento:
        return jsonify({"error": "Detén la captura antes de guardar."}), 400
    ref_data = request.get_json()
    procesar_buffer_y_guardar(ref_data)
    return jsonify({"status": "muestra de entrenamiento guardada"})

@app.route("/api/ultimas_mediciones")
def get_ultimas_mediciones_db():
    conn = conectar_db()
    if not conn: return jsonify([])
    cursor = conn.cursor(dictionary=True)
    query = "SELECT id, id_paciente, sys, dia, nivel FROM mediciones ORDER BY id DESC LIMIT 20"
    try:
        cursor.execute(query)
        records = cursor.fetchall()
        for rec in records:
            for key in rec: rec[key] = str(rec[key])
        conn.close()
        return jsonify(records)
    except Exception as e:
        if conn.is_connected(): conn.close()
        return jsonify([])

### --- PUNTO DE ENTRADA --- ###
if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
