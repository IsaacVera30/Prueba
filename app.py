from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import csv
import os
import pandas as pd
import joblib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import mysql.connector

app = Flask(__name__)
CORS(app)

# Modelos ML
modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

# Variables globales
lecturas_sensor = {
    "hr_crudo": "--",
    "hr_promedio": "--",
    "spo2_sensor": "--",
    "ir": "--",
    "red": "--"
}

predicciones_ml = {
    "sys_ml": "--",
    "dia_ml": "--",
    "hr_ml": "--",
    "spo2_ml": "--",
    "estado": "--"
}

registro_entrenamiento = []
capturando = False
contador_captura = 0

# Conexión MySQL
def obtener_conexion():
    return mysql.connector.connect(
        host="containers-us-west-107.railway.app",
        user="root",
        password="u0UKO7HdUoLmm5rHQPEN",
        database="railway",
        port=5692
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ultimos_valores')
def ultimos_valores():
    return jsonify({**lecturas_sensor, **predicciones_ml})

@app.route('/api/presion', methods=['POST'])
def presion():
    global contador_captura

    data = request.json
    lecturas_sensor.update({
        "hr_crudo": data["hr_crudo"],
        "hr_promedio": data["hr_promedio"],
        "spo2_sensor": data["spo2_sensor"],
        "ir": data["ir"],
        "red": data["red"]
    })

    # Predicciones ML
    x = [[float(data["hr_promedio"]), float(data["spo2_sensor"]), float(data["ir"]), float(data["red"]), float(data["hr_crudo"]), 0, 0, 0]]
    predicciones_ml["sys_ml"] = round(modelo_sys.predict(x)[0], 2)
    predicciones_ml["dia_ml"] = round(modelo_dia.predict(x)[0], 2)
    predicciones_ml["hr_ml"] = data["hr_promedio"]
    predicciones_ml["spo2_ml"] = data["spo2_sensor"]
    predicciones_ml["estado"] = "HT1" if predicciones_ml["sys_ml"] >= 120 else "NORMAL"

    # Captura para entrenamiento
    if capturando and contador_captura < 15:
        registro_entrenamiento.append([
            data["hr_crudo"],
            data["hr_promedio"],
            data["spo2_sensor"],
            data["ir"],
            data["red"]
        ])
        contador_captura += 1

    # Guardar en MySQL
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
        cursor.execute("""
            INSERT INTO mediciones (id_paciente, sys, dia, nivel)
            VALUES (%s, %s, %s, %s)
        """, (1, predicciones_ml["sys_ml"], predicciones_ml["dia_ml"], predicciones_ml["estado"]))
        conexion.commit()
    except Exception as e:
        print("❌ Error conexión MySQL:", e)
    finally:
        cursor.close()
        conexion.close()

    return jsonify({"mensaje": "Datos recibidos"})

@app.route('/api/registro_count')
def registro_count():
    return jsonify({"count": contador_captura})

@app.route('/api/ultimos_datos')
def ultimos_datos():
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT id_paciente, sys, dia, nivel
            FROM mediciones
            ORDER BY id DESC
            LIMIT 20
        """)
        datos = cursor.fetchall()
        return jsonify(datos)
    except Exception as e:
        return jsonify([])
    finally:
        cursor.close()
        conexion.close()

@app.route('/api/autorizar', methods=['POST'])
def autorizar():
    global registro_entrenamiento, contador_captura
    registro_entrenamiento = []
    contador_captura = 0
    return jsonify({"mensaje": "Autorizado"})

@app.route('/api/iniciar_captura', methods=['POST'])
def iniciar_captura():
    global capturando
    capturando = True
    return jsonify({"mensaje": "Captura iniciada"})

@app.route('/api/guardar_csv', methods=['POST'])
def guardar_csv():
    global capturando, registro_entrenamiento
    capturando = False

    ref = request.json
    sys_ref = ref['sys']
    dia_ref = ref['dia']
    hr_ref = ref['hr']

    ruta_csv = "registro_sensor_entrenamiento.csv"
    with open(ruta_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for fila in registro_entrenamiento:
            writer.writerow(fila + [sys_ref, dia_ref, hr_ref])

    subir_a_drive(ruta_csv)
    registro_entrenamiento = []
    return jsonify({"mensaje": "CSV guardado y enviado"})

# Google Drive
def subir_a_drive(ruta_archivo):
    creds = service_account.Credentials.from_service_account_file("credenciales.json")
    service = build('drive', 'v3', credentials=creds)
    file_metadata = {'name': os.path.basename(ruta_archivo)}
    media = MediaFileUpload(ruta_archivo, resumable=True)
    service.files().create(body=file_metadata, media_body=media, fields='id').execute()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
