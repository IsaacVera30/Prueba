from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import mysql.connector
from datetime import datetime
import os
import io

# Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

app = Flask(__name__)

modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

autorizado = False
ultima_estimacion = {"sys": "---", "dia": "---", "nivel": "---"}

ventana_ir = []
ventana_red = []
MUESTRAS = 10

DB_CONFIG = {
    "host": os.environ.get("MYSQLHOST"),
    "user": os.environ.get("MYSQLUSER"),
    "password": os.environ.get("MYSQLPASSWORD"),
    "database": os.environ.get("MYSQLDATABASE"),
    "port": int(os.environ.get("MYSQLPORT", 3306))
}

FOLDER_ID = "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-"

def clasificar_nivel(sys, dia):
    if sys >= 180 or dia >= 110:
        return "Alerta! ACV"
    elif sys >= 160 or dia >= 100:
        return "H3"
    elif sys >= 140 or dia >= 90:
        return "H2"
    elif sys >= 130 or dia >= 80:
        return "H1"
    else:
        return "Normal"

def guardar_en_mysql(id_paciente, sys, dia, nivel):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (id_paciente, sys, dia, nivel))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Error MySQL:", e)

def subir_a_drive(nombre_archivo):
    try:
        credentials = service_account.Credentials.from_service_account_file(
            "/etc/secrets/credentials.json",
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=credentials)

        # Leer el CSV
        with open(nombre_archivo, "rb") as f:
            media = MediaIoBaseUpload(io.BytesIO(f.read()), mimetype="text/csv")

        # Eliminar versión anterior si existe
        query = f"name='{nombre_archivo}' and '{FOLDER_ID}' in parents"
        results = service.files().list(q=query, spaces="drive").execute()
        for item in results.get("files", []):
            service.files().delete(fileId=item["id"]).execute()

        # Subir archivo
        file_metadata = {
            "name": nombre_archivo,
            "parents": [FOLDER_ID]
        }
        service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        print("✅ Archivo CSV subido a Google Drive")

    except Exception as e:
        print("❌ Error al subir a Google Drive:", e)

@app.route("/")
def index():
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def registrar_datos():
    global ultima_estimacion, autorizado

    data = request.get_json()
    hr = int(data.get("hr", 0))
    ir = int(data.get("ir", 0))
    red = int(data.get("red", 0))
    id_paciente = int(data.get("id_paciente", 1))

    ventana_ir.append(ir)
    ventana_red.append(red)
    if len(ventana_ir) > MUESTRAS: ventana_ir.pop(0)
    if len(ventana_red) > MUESTRAS: ventana_red.pop(0)

    spo2 = 0
    if len(ventana_ir) == MUESTRAS:
        dc_ir = np.mean(ventana_ir)
        dc_red = np.mean(ventana_red)
        ac_ir = np.mean(np.abs(np.array(ventana_ir) - dc_ir))
        ac_red = np.mean(np.abs(np.array(ventana_red) - dc_red))
        ratio = (ac_red / dc_red) / (ac_ir / dc_ir) if dc_ir > 0 and dc_red > 0 else 0
        spo2 = max(70, min(100, 110 - 25 * ratio))

    entrada = pd.DataFrame([[hr, spo2]], columns=["hr", "spo2"])
    sys = modelo_sys.predict(entrada)[0]
    dia = modelo_dia.predict(entrada)[0]
    nivel = clasificar_nivel(sys, dia)

    ultima_estimacion = {
        "sys": f"{sys:.2f}",
        "dia": f"{dia:.2f}",
        "nivel": nivel
    }

    if autorizado:
        with open("registro_sensor_entrenamiento.csv", "a") as f:
            f.write(f"{hr},{spo2},{ir},{red},{sys:.2f},{dia:.2f}\n")
        subir_a_drive("registro_sensor_entrenamiento.csv")

    if ir > 20000 and red > 15000:
        guardar_en_mysql(id_paciente, round(sys, 2), round(dia, 2), nivel)

    return jsonify({
        "sys": round(sys, 2),
        "dia": round(dia, 2),
        "spo2": round(spo2, 1),
        "nivel": nivel
    })

@app.route("/api/autorizacion", methods=["GET"])
def estado_autorizacion():
    return jsonify({"autorizado": autorizado})

@app.route("/api/autorizar", methods=["GET"])
def autorizar_registro():
    global autorizado
    autorizado = True
    return jsonify({"mensaje": "✅ Registro autorizado"})

@app.route("/api/detener", methods=["GET"])
def detener_registro():
    global autorizado
    autorizado = False
    return jsonify({"mensaje": "⛔ Registro detenido"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
