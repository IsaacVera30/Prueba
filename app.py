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

# Módulos para Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
socketio = SocketIO(app)

# --- Variables Globales ---
autorizado_db = False # Para guardar en la base de datos
capturando_entrenamiento = False
buffer_datos_entrenamiento = []
CSV_FILENAME = "registro_sensor_entrenamiento_alta_calidad.csv"

# --- Configuración ---
DB_CONFIG = {
    'host': os.environ.get("MYSQLHOST"), 'user': os.environ.get("MYSQLUSER"),
    'password': os.environ.get("MYSQLPASSWORD"), 'database': os.environ.get("MYSQLDATABASE"),
    'port': int(os.environ.get("MYSQLPORT", "3306"))
}
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')

# --- Funciones Auxiliares ---

def conectar_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Error DB: {e}")
        return None

def guardar_medicion_mysql(data):
    # Asume que 'data' tiene las claves necesarias
    conn = conectar_db()
    if not conn: return
    cursor = conn.cursor()
    query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel, hr, spo2) VALUES (%s, %s, %s, %s, %s, %s)"
    try:
        cursor.execute(query, (
            data.get("id_paciente"), data.get("sys_ml"), data.get("dia_ml"),
            data.get("estado"), data.get("hr_ml"), data.get("spo2_ml")
        ))
        conn.commit()
        print(f"✅ Datos guardados en MySQL para paciente {data.get('id_paciente')}")
    except Exception as e:
        print(f"❌ Error al guardar en MySQL: {e}")
    finally:
        if conn.is_connected(): conn.close()

def get_google_drive_service():
    try:
        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        # Asume que el service_account.json está como Secret File en Render
        creds = service_account.Credentials.from_service_account_file('service_account.json', scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        print("✅ Servicio de Google Drive autenticado.")
        return service
    except Exception as e:
        print(f"❌ Error autenticando con Google Drive: {e}")
        return None

def subir_csv_a_drive():
    if not FOLDER_ID:
        print("⚠️ Advertencia: GOOGLE_DRIVE_FOLDER_ID no configurado.")
        return
    service = get_google_drive_service()
    if not service or not os.path.exists(CSV_FILENAME):
        return
    try:
        file_metadata = {'name': CSV_FILENAME, 'parents': [FOLDER_ID]}
        media = MediaFileUpload(CSV_FILENAME, mimetype='text/csv', resumable=True)
        query = f"name='{CSV_FILENAME}' and '{FOLDER_ID}' in parents and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        if response.get('files'):
            service.files().update(fileId=response.get('files')[0].get('id'), media_body=media).execute()
            print(f"✅ Archivo CSV actualizado en Google Drive.")
        else:
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"✅ Archivo CSV creado en Google Drive.")
    except Exception as e:
        print(f"❌ Error al subir archivo a Drive: {e}")

def procesar_buffer_y_guardar(ref_data):
    global buffer_datos_entrenamiento
    if not buffer_datos_entrenamiento: return

    hr_list = [float(d.get("hr_promedio", 0)) for d in buffer_datos_entrenamiento]
    spo2_list = [float(d.get("spo2_sensor", 0)) for d in buffer_datos_entrenamiento]
    ir_list = [float(d.get("ir", 0)) for d in buffer_datos_entrenamiento]
    red_list = [float(d.get("red", 0)) for d in buffer_datos_entrenamiento]

    features = {
        "hr_promedio_sensor": np.mean(hr_list), "spo2_promedio_sensor": np.mean(spo2_list),
        "ir_mean_filtrado": np.mean(ir_list), "red_mean_filtrado": np.mean(red_list),
        "ir_std_filtrado": np.std(ir_list), "red_std_filtrado": np.std(red_list)
    }
    final_row = {
        **features,
        "hr_referencia": ref_data["hr_ref"], "sys_referencia": ref_data["sys_ref"],
        "dia_referencia": ref_data["dia_ref"], "timestamp_captura": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    file_exists = os.path.exists(CSV_FILENAME)
    with open(CSV_FILENAME, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(final_row)
    
    print(f"✅ Fila de entrenamiento guardada en {CSV_FILENAME}")
    buffer_datos_entrenamiento = []
    subir_csv_a_drive()

### --- RUTAS DE LA API --- ###

@app.route("/")
def home(): return render_template("index.html")

# Este endpoint debe coincidir con el que usa tu ESP32
@app.route("/api/data", methods=["POST"])
def recibir_datos():
    data = request.get_json()
    if capturando_entrenamiento:
        buffer_datos_entrenamiento.append(data)
        socketio.emit('capture_count_update', {'count': len(buffer_datos_entrenamiento)})
    
    socketio.emit('update_data', data)
    
    if autorizado_db:
        guardar_medicion_mysql(data)
        socketio.emit('new_record_saved')
    
    if data.get("estado") == "HT Crisis":
        # Aquí puedes llamar a una función de alerta si la defines
        pass
    
    return jsonify({"status": "ok"})

### --- ENDPOINTS PARA CONTROL DE ENTRENAMIENTO --- ###

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

### --- ENDPOINTS DE ESTADO Y DATOS HISTÓRICOS --- ###

@app.route("/api/autorizacion", methods=["GET", "POST"])
def api_control_autorizacion_db():
    global autorizado_db
    if request.method == "POST":
        autorizado_db = request.json.get("autorizado", False)
        socketio.emit('status_update_db', {"autorizado": autorizado_db})
    return jsonify({"autorizado": autorizado_db})

@app.route("/api/ultimas_mediciones")
def get_ultimas_mediciones_db():
    conn = conectar_db()
    if not conn: return jsonify([])
    cursor = conn.cursor(dictionary=True)
    query = "SELECT id, id_paciente, sys, dia, hr, spo2, nivel FROM mediciones ORDER BY id DESC LIMIT 20"
    try:
        cursor.execute(query)
        records = cursor.fetchall()
        for rec in records:
            for key in rec: rec[key] = str(rec[key])
        conn.close()
        return jsonify(records)
    except Exception as e:
        print(f"❌ Error al leer historial: {e}")
        if conn.is_connected(): conn.close()
        return jsonify([])

### --- PUNTO DE ENTRADA --- ###
if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
