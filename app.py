from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import mysql.connector
import csv
import os
import joblib
import datetime
import numpy as np
from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd

app = Flask(__name__)
CORS(app)

# Variables globales
lecturas_sensor = []
registro_activo = False
registros_requeridos = 15
contador_registros = 0
datos_referencia = {}
ultimos_datos = []

# Conexión MySQL
def conectar_mysql():
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        database=os.environ.get("DB_NAME")
    )

# Cargar modelos ML
modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Obtener cantidad de registros actuales
@app.route('/api/registro_count')
def registro_count():
    return jsonify({"count": contador_registros, "total": registros_requeridos})

# Activar registro
@app.route('/api/activar_registro', methods=['POST'])
def activar_registro():
    global registro_activo, contador_registros, lecturas_sensor
    registro_activo = True
    contador_registros = 0
    lecturas_sensor = []
    return jsonify({"mensaje": "Registro activado"})

# Guardar datos de referencia
@app.route('/api/referencia', methods=['POST'])
def guardar_referencia():
    global datos_referencia
    datos = request.json
    datos_referencia = datos
    return jsonify({"mensaje": "Datos de referencia guardados"})

# Recibir datos del ESP32
@app.route('/api/presion', methods=['POST'])
def recibir_presion():
    global lecturas_sensor, contador_registros, registro_activo, datos_referencia, ultimos_datos

    data = request.get_json()
    hr_crudo = data.get("HR_CRUDO")
    hr_avg = data.get("HR_PROMEDIO_SENSOR")
    spo2 = data.get("SpO2_SENSOR")
    ir = data.get("IR")
    red = data.get("RED")

    # Predicciones ML
    entrada = np.array([[hr_avg, spo2, ir, red]])
    sys_ml = modelo_sys.predict(entrada)[0]
    dia_ml = modelo_dia.predict(entrada)[0]
    hr_ml = hr_avg
    spo2_ml = spo2
    estado = "Normal" if sys_ml < 130 and dia_ml < 85 else "Alerta"

    # Datos actuales
    datos_actuales = {
        "HR_CRUDO": hr_crudo,
        "HR_PROMEDIO_SENSOR": hr_avg,
        "SpO2_SENSOR": spo2,
        "IR": ir,
        "RED": red,
        "SYS_ML": sys_ml,
        "DIA_ML": dia_ml,
        "HR_ML": hr_ml,
        "SpO2_ML": spo2_ml,
        "ESTADO": estado
    }

    ultimos_datos.append(datos_actuales)

    if registro_activo and contador_registros < registros_requeridos:
        fila = [
            hr_crudo, hr_avg, spo2, ir, red,
            datos_referencia.get("SYS_REFERENCIA"),
            datos_referencia.get("DIA_REFERENCIA"),
            datos_referencia.get("HR_REFERENCIA")
        ]
        lecturas_sensor.append(fila)
        contador_registros += 1

        if contador_registros == registros_requeridos:
            guardar_csv()
            registro_activo = False
            lecturas_sensor = []
            contador_registros = 0

    return jsonify({"mensaje": "Datos procesados correctamente"})

# Guardar CSV local y subir a Google Drive
def guardar_csv():
    fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"registro_entrenamiento_{fecha}.csv"

    with open(nombre_archivo, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HR_CRUDO", "HR_PROMEDIO_SENSOR", "SpO2_SENSOR", "IR", "RED", "SYS_REFERENCIA", "DIA_REFERENCIA", "HR_REFERENCIA"])
        writer.writerows(lecturas_sensor)

    subir_a_drive(nombre_archivo)

# Subir a Google Drive
def subir_a_drive(nombre_archivo):
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    SERVICE_ACCOUNT_FILE = 'credenciales_google.json'

    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': nombre_archivo, 'parents': [os.environ.get("DRIVE_FOLDER_ID")]}
    media = MediaFileUpload(nombre_archivo, mimetype='text/csv')
    service.files().create(body=file_metadata, media_body=media, fields='id').execute()

# Ruta para obtener el último valor en tiempo real
@app.route('/api/ultimos_valores', methods=['GET'])
def obtener_ultimos_valores():
    if ultimos_datos:
        return jsonify(ultimos_datos[-1])
    return jsonify({'mensaje': 'No hay datos disponibles'}), 404

# Ruta para obtener los últimos 20 registros de la base de datos
@app.route('/api/ultimos_datos', methods=['GET'])
def obtener_ultimos_datos():
    try:
        conn = conectar_mysql()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM presion ORDER BY id DESC LIMIT 20")
        resultados = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(resultados)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
