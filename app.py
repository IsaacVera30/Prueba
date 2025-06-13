# app.py final: combinado con funciones nuevas y estructura original estable

from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np 
import os
import csv
from datetime import datetime
import mysql.connector
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from scipy.signal import butter, filtfilt

app = Flask(__name__)

# --- Carga de Modelos de Machine Learning ---
MODEL_SYS_PATH = "modelo_sys.pkl"
MODEL_DIA_PATH = "modelo_dia.pkl"
modelo_sys = joblib.load(MODEL_SYS_PATH) if os.path.exists(MODEL_SYS_PATH) else None
modelo_dia = joblib.load(MODEL_DIA_PATH) if os.path.exists(MODEL_DIA_PATH) else None

# --- Variables Globales ---
autorizado = False
capturando_entrenamiento = False
buffer_datos_entrenamiento = []
ultima_estimacion = {"sys": "---", "dia": "---", "spo2": "---", "hr": "---", "nivel": "---", "timestamp": "---"}
ventana_ir, ventana_red = [], []
MUESTRAS_SPO2 = 10

# --- Config DB y Drive ---
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', os.environ.get("MYSQLHOST")),
    'user': os.environ.get('DB_USER', os.environ.get("MYSQLUSER")),
    'password': os.environ.get('DB_PASSWORD', os.environ.get("MYSQLPASSWORD")),
    'database': os.environ.get('DB_NAME', os.environ.get("MYSQLDATABASE")),
    'port': int(os.environ.get('DB_PORT', os.environ.get("MYSQLPORT", "3306")))
}

FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', "")
CSV_FILENAME = "registro_sensor_entrenamiento.csv"
KEY_FILE_LOCATION = 'service_account.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# --- Funciones Auxiliares ---
def filtrar_senal_ppg(senal):
    b, a = butter(4, [0.5 / 25, 5.0 / 25], btype='band')
    return filtfilt(b, a, senal)

def extraer_caracteristicas_ppg(segmento_ir, segmento_red, hr_promedio, spo2_promedio):
    ir = filtrar_senal_ppg(segmento_ir)
    red = filtrar_senal_ppg(segmento_red)
    return {
        "hr_promedio_sensor": round(hr_promedio, 2),
        "spo2_promedio_sensor": round(spo2_promedio, 2),
        "ir_mean_filtrado": round(np.mean(ir), 2),
        "red_mean_filtrado": round(np.mean(red), 2),
        "ir_std_filtrado": round(np.std(ir), 2),
        "red_std_filtrado": round(np.std(red), 2)
    }

def calcular_spo2(ir_list, red_list):
    if len(ir_list) < MUESTRAS_SPO2: return 0.0
    ir, red = np.array(ir_list[-MUESTRAS_SPO2:]), np.array(red_list[-MUESTRAS_SPO2:])
    try:
        ratio = (np.mean(np.abs(red - np.mean(red))) / np.mean(red)) / \
                (np.mean(np.abs(ir - np.mean(ir))) / np.mean(ir))
        return round(max(70.0, min(100.0, 110 - 25 * ratio)), 1)
    except: return 0.0

def clasificar_nivel(sys, dia):
    if sys > 180 or dia > 120: return "HT Crisis"
    elif sys >= 140 or dia >= 90: return "HT2"
    elif sys >= 130 or dia >= 80: return "HT1"
    elif sys >= 120 and dia < 80: return "Elevada"
    elif sys < 120 and dia < 80: return "Normal"
    return "Revisar"

def conectar_db():
    return mysql.connector.connect(**DB_CONFIG)

def guardar_mysql(id_paciente, sys, dia, nivel):
    conn = conectar_db(); cur = conn.cursor()
    cur.execute("INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s,%s,%s,%s)", (id_paciente, sys, dia, nivel))
    conn.commit(); cur.close(); conn.close()

def subir_a_drive(ruta, nombre):
    creds = service_account.Credentials.from_service_account_file(KEY_FILE_LOCATION, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    file_metadata = {'name': nombre, 'parents': [FOLDER_ID]}
    media = MediaFileUpload(ruta, mimetype='text/csv')
    service.files().create(body=file_metadata, media_body=media, fields='id').execute()

# --- Rutas ---
@app.route("/")
def index():
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def presion():
    global ultima_estimacion, ventana_ir, ventana_red
    d = request.get_json()
    hr, ir, red = int(d['hr']), int(d['ir']), int(d['red'])
    ventana_ir.append(ir); ventana_red.append(red)
    if len(ventana_ir) > MUESTRAS_SPO2: ventana_ir.pop(0); ventana_red.pop(0)
    spo2 = calcular_spo2(ventana_ir, ventana_red)
    sys, dia = modelo_sys.predict([[hr, spo2]])[0], modelo_dia.predict([[hr, spo2]])[0]
    nivel = clasificar_nivel(sys, dia)
    ultima_estimacion = {
        "sys": round(sys, 2), "dia": round(dia, 2), "spo2": spo2,
        "hr": hr, "nivel": nivel, "timestamp": datetime.now().strftime('%H:%M:%S')
    }
    if ir > 20000 and red > 15000: guardar_mysql(1, sys, dia, nivel)
    return jsonify(ultima_estimacion)

@app.route("/api/autorizacion", methods=["POST"])
def autorizar():
    global autorizado
    autorizado = request.get_json().get("autorizado", False)
    return jsonify({"autorizado": autorizado})

@app.route("/api/iniciar_captura_entrenamiento", methods=["POST"])
def iniciar():
    global capturando_entrenamiento, buffer_datos_entrenamiento
    capturando_entrenamiento = True; buffer_datos_entrenamiento = []
    return jsonify({"estado": "iniciado"})

@app.route("/api/detener_captura_entrenamiento", methods=["POST"])
def detener():
    global capturando_entrenamiento
    capturando_entrenamiento = False
    return jsonify({"estado": "detenido"})

@app.route("/api/guardar_muestra_entrenamiento", methods=["POST"])
def guardar_csv():
    global buffer_datos_entrenamiento
    ref = request.get_json()
    if not buffer_datos_entrenamiento: return jsonify({"error": "sin datos"})
    hrs = [r['hr'] for r in buffer_datos_entrenamiento]
    irs = [r['ir'] for r in buffer_datos_entrenamiento]
    reds = [r['red'] for r in buffer_datos_entrenamiento]
    hr_avg = np.mean(hrs); spo2_avg = calcular_spo2(irs, reds)
    feat = extraer_caracteristicas_ppg(irs, reds, hr_avg, spo2_avg)
    with open(CSV_FILENAME, 'a', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(list(feat.keys()) + ["hr_referencia", "pas_referencia", "pad_referencia", "timestamp"])
        w.writerow(list(feat.values()) + [ref['hr_referencia'], ref['pas_referencia'], ref['pad_referencia'], datetime.now()])
    subir_a_drive(CSV_FILENAME, CSV_FILENAME)
    buffer_datos_entrenamiento = []
    return jsonify({"msg": "guardado en csv"})

@app.route("/api/ultimos_valores", methods=["GET"])
def ultimos():
    conn = conectar_db(); cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM mediciones ORDER BY id DESC LIMIT 20")
    r = cur.fetchall(); cur.close(); conn.close()
    return jsonify(r)

@app.route("/api/ultima_estimacion", methods=["GET"])
def estimacion():
    return jsonify(ultima_estimacion)

@app.route("/api/registro_count")
def count():
    return jsonify({"registros": len(buffer_datos_entrenamiento)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
