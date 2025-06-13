import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import pandas as pd
import joblib
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

# --- Carga de Modelos ---
try:
    modelo_sys = joblib.load('modelo_sys.pkl')
    modelo_dia = joblib.load('modelo_dia.pkl')
    print("✅ Modelos de ML cargados correctamente.")
except Exception as e:
    print(f"❌ ERROR: No se pudieron cargar los modelos de ML: {e}")
    modelo_sys, modelo_dia = None, None

# --- Variables Globales ---
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
CALLMEBOT_API_KEY = os.environ.get('CALLMEBOT_API_KEY')
CALLMEBOT_PHONE_NUMBER = os.environ.get('CALLMEBOT_PHONE_NUMBER')

# --- Funciones Auxiliares ---
def conectar_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Error DB: {e}")
        return None

def guardar_medicion_mysql(data):
    conn = conectar_db()
    if not conn: return
    cursor = conn.cursor()
    # CORREGIDO: La consulta SQL solo usa las columnas que existen en tu DB
    query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s, %s, %s, %s)"
    try:
        # Se pasan solo los valores correspondientes
        cursor.execute(query, (
            data.get("id_paciente"),
            data.get("sys_ml"),
            data.get("dia_ml"),
            data.get("estado")
        ))
        conn.commit()
        print(f"✅ Datos guardados en MySQL para paciente {data.get('id_paciente')}")
    except Exception as e:
        print(f"❌ Error al guardar en MySQL: {e}")
    finally:
        if conn.is_connected(): conn.close()

def enviar_alerta_whatsapp(nivel, sys, dia):
    if not CALLMEBOT_API_KEY or not CALLMEBOT_PHONE_NUMBER:
        print("⚠️ Advertencia: Variables de CallMeBot no configuradas.")
        return
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
    final_row = {
        **features,
        "hr_referencia": ref_data["hr_ref"], "sys_referencia": ref_data["sys_ref"],
        "dia_referencia": ref_data["dia_ref"], "timestamp_captura": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
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
    if pas >= 140 or pad >= 90: return "HT2"
    if (pas >= 130 and pas <= 139) or (pad >= 80 and pad <= 89): return "HT1"
    if (pas >= 120 and pas <= 129) and pad < 80: return "Elevada"
    return "Normal"

### --- RUTAS DE LA API --- ###

@app.route("/")
def home(): return render_template("index.html")

# Endpoint que usa tu ESP32
@app.route("/api/presion", methods=["POST"])
def recibir_datos_presion():
    if not modelo_sys or not modelo_dia:
        return jsonify({"error": "Modelos no disponibles"}), 503
    try:
        data = request.get_json()
        hr_promedio = float(data["hr_promedio"])
        spo2_sensor = float(data["spo2_sensor"])
        id_paciente = int(data.get("id_paciente", 1))

        entrada_df = pd.DataFrame([[hr_promedio, spo2_sensor]], columns=['hr', 'spo2'])
        pas_estimada = modelo_sys.predict(entrada_df)[0]
        pad_estimada = modelo_dia.predict(entrada_df)[0]
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)

        datos_para_panel = {
            "hr_crudo": data.get("hr_crudo"), "hr_promedio": f"{hr_promedio:.0f}",
            "spo2_sensor": f"{spo2_sensor:.1f}", "ir": data.get("ir"), "red": data.get("red"),
            "sys_ml": f"{pas_estimada:.2f}", "dia_ml": f"{pad_estimada:.2f}",
            "hr_ml": f"{hr_promedio:.0f}", "spo2_ml": f"{spo2_sensor:.1f}", "estado": nivel_presion
        }
        
        socketio.emit('update_data', datos_para_panel)
        
        # Guardado automático
        guardar_medicion_mysql(datos_para_panel)
        socketio.emit('new_record_saved')
        
        if nivel_presion == "HT Crisis":
            enviar_alerta_whatsapp(nivel_presion, pas_estimada, pad_estimada)
        
        # Respuesta para el dispositivo
        return jsonify({ "sys": pas_estimada, "dia": pad_estimada, "hr": hr_promedio, "spo2": spo2_sensor, "nivel": nivel_presion }), 200
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

@app.route("/api/data", methods=["POST"])
def recibir_datos_entrenamiento():
    data = request.get_json()
    if capturando_entrenamiento:
        buffer_datos_entrenamiento.append(data)
        socketio.emit('capture_count_update', {'count': len(buffer_datos_entrenamiento)})
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
    # CORREGIDO: La consulta solo pide las columnas que existen
    query = "SELECT id, id_paciente, sys, dia, nivel FROM mediciones ORDER BY id DESC LIMIT 20"
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

### --- ENDPOINT DE PRUEBA PARA ALERTAS (NUEVO) --- ###

@app.route("/api/test_alerta", methods=['POST'])
def test_alerta_whatsapp():
    """
    Este endpoint es solo para pruebas. Permite enviar una alerta
    manualmente desde Postman sin necesidad de un dispositivo.
    Espera un JSON como: {"nivel": "HT Crisis", "sys": 185, "dia": 125}
    """
    data = request.get_json()
    if not data or 'nivel' not in data or 'sys' not in data or 'dia' not in data:
        return jsonify({"error": "Se requiere 'nivel', 'sys' y 'dia' en el JSON."}), 400

    nivel = data['nivel']
    sys = data['sys']
    dia = data['dia']

    print(f"⚠️  Recibida petición de prueba para alerta: Nivel={nivel}, SYS={sys}, DIA={dia}")
    
    # Llama directamente a la función que envía el mensaje
    enviar_alerta_whatsapp(nivel, sys, dia)

    return jsonify({"mensaje": f"Prueba de alerta para '{nivel}' enviada."}), 200

### --- PUNTO DE ENTRADA --- ###
if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
