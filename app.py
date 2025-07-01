import eventlet

# Parcheo necesario para compatibilidad con Flask + SocketIO
eventlet.monkey_patch()

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import os, time, csv, io, requests
import numpy as np
import joblib, pandas as pd
from datetime import datetime
import mysql.connector
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

app = Flask(__name__)
socketio = SocketIO(app)

# Carga de modelos entrenados con nuevas variables filtradas
try:
    modelo_sys = joblib.load('modelo_sys.pkl')  # Modelo de presión sistólica
    modelo_dia = joblib.load('modelo_dia.pkl')  # Modelo de presión diastólica
    print("Modelos de ML cargados correctamente.")
except Exception as e:
    print(f"Error al cargar modelos: {e}")
    modelo_sys, modelo_dia = None, None

# Variables Globales
buffer_datos_entrenamiento = []
last_db_save_time = 0
LOCK_FILE = "capture.lock"
DRIVE_CSV_FILENAME = "entrenamiento_ml.csv"
FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
CALLMEBOT_API_KEY = os.environ.get("CALLMEBOT_API_KEY")
CALLMEBOT_PHONE_NUMBER = os.environ.get("CALLMEBOT_PHONE_NUMBER")
DB_CONFIG = {
    'host': os.environ.get("MYSQLHOST"),
    'user': os.environ.get("MYSQLUSER"),
    'password': os.environ.get("MYSQLPASSWORD"),
    'database': os.environ.get("MYSQLDATABASE"),
    'port': int(os.environ.get("MYSQLPORT", 3306))
}

# Google Drive API

def get_google_drive_service():
    try:
        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        creds = service_account.Credentials.from_service_account_file('service_account.json', scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Error autenticando con Google Drive: {e}")
        return None

# Guardar fila en Google Drive CSV

def append_row_to_drive_csv(row_dict):
    service = get_google_drive_service()
    if not service:
        return
    try:
        query = f"name='{DRIVE_CSV_FILENAME}' and '{FOLDER_ID}' in parents and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])

        fieldnames = list(row_dict.keys())
        string_io = io.StringIO()
        writer = csv.DictWriter(string_io, fieldnames=fieldnames)

        if files:
            file_id = files[0].get('id')
            request_file = service.files().get_media(fileId=file_id)
            downloader = io.BytesIO()
            downloader.write(request_file.execute())
            existing_content = downloader.getvalue().decode('utf-8')
            string_io.write(existing_content)
            if not existing_content.strip().endswith('\n'):
                string_io.write('\n')
            string_io.seek(0, io.SEEK_END)
            csv.writer(string_io).writerow(row_dict.values())
            media = MediaIoBaseUpload(io.BytesIO(string_io.getvalue().encode('utf-8')), mimetype='text/csv', resumable=True)
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            writer.writeheader()
            writer.writerow(row_dict)
            file_metadata = {'name': DRIVE_CSV_FILENAME, 'parents': [FOLDER_ID]}
            media = MediaIoBaseUpload(io.BytesIO(string_io.getvalue().encode('utf-8')), mimetype='text/csv', resumable=True)
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()

        print(f"Fila guardada en Drive: {DRIVE_CSV_FILENAME}")
    except Exception as e:
        print(f"Error al escribir en Drive: {e}")

# Procesar datos del buffer y guardar en Drive

def procesar_buffer_y_guardar(ref_data):
    global buffer_datos_entrenamiento
    if not buffer_datos_entrenamiento:
        return
    try:
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        features = {
            "hr_promedio_sensor": np.mean([float(d.get("hr_promedio", 0)) for d in buffer_datos_entrenamiento]),
            "spo2_promedio_sensor": np.mean([float(d.get("spo2_sensor", 0)) for d in buffer_datos_entrenamiento]),
            "ir_mean_filtrado": np.mean([float(d.get("ir", 0)) for d in buffer_datos_entrenamiento]),
            "red_mean_filtrado": np.mean([float(d.get("red", 0)) for d in buffer_datos_entrenamiento]),
            "ir_std_filtrado": np.std([float(d.get("ir", 0)) for d in buffer_datos_entrenamiento]),
            "red_std_filtrado": np.std([float(d.get("red", 0)) for d in buffer_datos_entrenamiento])
        }

        final_row = {**features,
                     'sys_ref': ref_data.get('sys_ref'),
                     'dia_ref': ref_data.get('dia_ref'),
                     'hr_ref': ref_data.get('hr_ref'),
                     'timestamp_captura': timestamp_str}

        append_row_to_drive_csv(final_row)
    except Exception as e:
        print(f"Error procesando buffer: {e}")
    finally:
        buffer_datos_entrenamiento = []
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

# Predicción usando modelos actualizados con variables filtradas

@app.route("/api/data", methods=["POST"])
def recibir_datos():
    global last_db_save_time, buffer_datos_entrenamiento
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400

    # Simulación de prueba
    if data.get("id_paciente") == 999:
        response = {"sys": 185, "dia": 125, "hr": 99, "spo2": 99, "nivel": "HT Crisis"}
        socketio.emit('update_data', data)
        return jsonify(response)

    data['sys_ml'], data['dia_ml'], data['hr_ml'], data['spo2_ml'], data['estado'] = 0, 0, 0, 0, "---"

    if os.path.exists(LOCK_FILE):
        buffer_datos_entrenamiento.append(data)
        socketio.emit('capture_count_update', {'count': len(buffer_datos_entrenamiento)})
    else:
        if float(data.get("ir", 0)) > 50000:
            if modelo_sys and modelo_dia:
                try:
                    input_data = [[
                        float(data.get("hr_promedio", 0)),
                        float(data.get("spo2_sensor", 0)),
                        float(data.get("ir", 0)),
                        float(data.get("red", 0)),
                        0,  # Si no hay std calculado aún, puedes ajustar
                        0
                    ]]
                    data['sys_ml'] = modelo_sys.predict(input_data)[0]
                    data['dia_ml'] = modelo_dia.predict(input_data)[0]
                    data['hr_ml'] = float(data.get("hr_promedio", 0))
                    data['spo2_ml'] = float(data.get("spo2_sensor", 0))
                    data['estado'] = clasificar_nivel_presion(data['sys_ml'], data['dia_ml'])
                except Exception as e:
                    print(f"Error de predicción: {e}")
                    data['estado'] = "Error Pred."

    socketio.emit('update_data', data)

    return jsonify({
        "sys": round(data.get("sys_ml", 0), 2),
        "dia": round(data.get("dia_ml", 0), 2),
        "hr": round(data.get("hr_ml", 0), 2),
        "spo2": round(data.get("spo2_ml", 0), 2),
        "nivel": data.get("estado", "Normal")
    })


@app.route("/api/start_capture", methods=["POST"])
def start_capture():
    # ...(Esta función no cambia)...
    global buffer_datos_entrenamiento
    if os.path.exists(DRIVE_CSV_FILENAME):
        # Esta lógica puede cambiar si quieres un solo archivo maestro
        pass 
    with open(LOCK_FILE, "w") as f: f.write("capturing")
    buffer_datos_entrenamiento = []
    return jsonify({"status": "captura iniciada"})

@app.route("/api/stop_capture", methods=["POST"])
def stop_capture():
    return jsonify({"status": "captura detenida", "muestras": len(buffer_datos_entrenamiento)})

@app.route("/api/save_training_data", methods=["POST"])
def save_training_data():
    if not os.path.exists(LOCK_FILE): return jsonify({"error": "La captura no está en modo 'pausa'."}), 400
    ref_data = request.get_json()
    procesar_buffer_y_guardar(ref_data)
    return jsonify({"status": "muestra de entrenamiento guardada y sistema reseteado"})

@app.route("/api/ultimas_mediciones")
def get_ultimas_mediciones_db():
    conn = conectar_db()
    if not conn: return jsonify([])
    cursor = conn.cursor(dictionary=True)
    # --- CAMBIO: Seleccionar las nuevas columnas ---
    query = "SELECT id, id_paciente, sys, dia, nivel, hr_ml, spo2_ml FROM mediciones ORDER BY id DESC LIMIT 20"
    try:
        cursor.execute(query)
        records = cursor.fetchall()
        for rec in records:
            for key, value in rec.items():
                if value is not None: rec[key] = str(value)
        conn.close()
        return jsonify(records)
    except Exception as e:
        print(f"Error al obtener últimas mediciones: {e}")
        if conn and conn.is_connected(): conn.close()
        return jsonify([])

# --- ENDPOINT DE PRUEBA PARA ALERTAS ---
@app.route("/api/test_alert", methods=['POST'])
def test_alert():
    data = request.get_json()
    if not data or "sys" not in data or "dia" not in data:
        return jsonify({"error": "Por favor envía 'sys' y 'dia' en el JSON."}), 400
    
    sys_val = float(data["sys"])
    dia_val = float(data["dia"])
    nivel = clasificar_nivel_presion(sys_val, dia_val)
    
    print(f"Alerta de prueba recibida. Nivel: {nivel}")
    
    data_to_save = {
        "id_paciente": data.get("id_paciente", 99),
        "sys_ml": sys_val,
        "dia_ml": dia_val,
        "hr_ml": data.get("hr", 0),
        "spo2_ml": data.get("spo2", 0),
        "estado": nivel
    }
    
    guardar_medicion_mysql(data_to_save)
    socketio.emit('new_record_saved')
    
    if nivel == "HT Crisis":
        enviar_alerta_whatsapp(nivel, sys_val, dia_val)
        return jsonify({"status": "Alerta de crisis procesada y guardada.", "data": data_to_save})
    
    return jsonify({"status": "Alerta de prueba guardada.", "data": data_to_save})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
