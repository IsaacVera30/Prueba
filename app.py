# app.py
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import pandas as pd
import joblib
import numpy as np
import os
import mysql.connector
from datetime import datetime
# (Se mantienen los demás imports de Google Drive si los necesitas para el entrenamiento)

app = Flask(__name__)
socketio = SocketIO(app)

# --- Carga de Modelos de Machine Learning ---
MODEL_SYS_PATH = "modelo_sys.pkl"
MODEL_DIA_PATH = "modelo_dia.pkl"
modelo_sys = joblib.load(MODEL_SYS_PATH) if os.path.exists(MODEL_SYS_PATH) else None
modelo_dia = joblib.load(MODEL_DIA_PATH) if os.path.exists(MODEL_DIA_PATH) else None
if modelo_sys and modelo_dia:
    print("Modelos de ML cargados exitosamente.")
else:
    print("⚠️ Advertencia: Uno o ambos modelos de ML no se encontraron. La predicción no funcionará.")

# --- Variables Globales de Estado ---
autorizado = False
ultima_estimacion = {"sys": "---", "dia": "---", "spo2": "---", "hr": "---", "nivel": "---", "timestamp": "---"}

# --- Configuración DB ---
DB_HOST = os.environ.get("MYSQLHOST")
DB_USER = os.environ.get("MYSQLUSER")
DB_PASSWORD = os.environ.get("MYSQLPASSWORD")
DB_NAME = os.environ.get("MYSQLDATABASE")
DB_PORT = os.environ.get("MYSQLPORT", "3306")
DB_CONFIG = {'host': DB_HOST, 'user': DB_USER, 'password': DB_PASSWORD, 'database': DB_NAME, 'port': int(DB_PORT)}

# --- Funciones Auxiliares ---
def conectar_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Error conexión MySQL: {err}")
        return None

# MODIFICADO: La función ya no acepta hr ni spo2
def guardar_medicion_mysql(id_paciente, sys, dia, nivel):
    conn = conectar_db()
    if conn is None: return False
    cursor = conn.cursor()
    # MODIFICADO: La consulta SQL solo inserta las columnas que existen en tu tabla
    query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s, %s, %s, %s)"
    try:
        # MODIFICADO: La tupla de valores ya no incluye hr ni spo2
        cursor.execute(query, (id_paciente, sys, dia, nivel))
        conn.commit()
        print(f"MySQL: Datos guardados para paciente {id_paciente}")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Error al guardar en MySQL: {err}")
        conn.rollback()
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def clasificar_nivel_presion(pas, pad):
    if pas is None or pad is None or pas <= 0 or pad <= 0: return "---"
    if pas > 180 or pad > 120: return "HT Crisis"
    if pas >= 140 or pad >= 90: return "HT2"
    if (pas >= 130 and pas <= 139) or (pad >= 80 and pad <= 89): return "HT1"
    if (pas >= 120 and pas <= 129) and pad < 80: return "Elevada"
    if pas < 120 and pad < 80: return "Normal"
    return "Revisar"

# --- Endpoints Flask ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global ultima_estimacion
    if not modelo_sys or not modelo_dia:
        return jsonify({"error": "Modelos ML no cargados"}), 500

    data = request.get_json()
    if not data or "hr" not in data or "spo2" not in data:
        return jsonify({"error": "Datos incompletos, se requiere 'hr' y 'spo2'"}), 400

    try:
        hr = float(data["hr"])
        spo2 = float(data["spo2"])
        id_paciente = int(data.get("id_paciente", 1))

        entrada_df = pd.DataFrame([[hr, spo2]], columns=['hr', 'spo2'])
        pas_estimada = modelo_sys.predict(entrada_df)[0]
        pad_estimada = modelo_dia.predict(entrada_df)[0]
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)

        ultima_estimacion = {
            "sys": f"{pas_estimada:.1f}", "dia": f"{pad_estimada:.1f}",
            "spo2": f"{spo2:.1f}", "hr": f"{hr:.0f}", "nivel": nivel_presion,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        }

        socketio.emit('update_data', ultima_estimacion)

        if autorizado:
            # MODIFICADO: Se llama a la función de guardado sin hr y spo2
            guardar_medicion_mysql(id_paciente, pas_estimada, pad_estimada, nivel_presion)
            socketio.emit('new_record_saved')

        return jsonify({"sys": pas_estimada, "dia": pad_estimada, "nivel": nivel_presion}), 200

    except Exception as e:
        print(f"❌ Error en /api/presion: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route("/api/ultimas_mediciones", methods=["GET"])
def get_ultimas_mediciones_db():
    conn = conectar_db()
    if conn is None: return jsonify([]), 500
    cursor = conn.cursor(dictionary=True)
    # MODIFICADO: La consulta SELECT ya no pide las columnas hr y spo2
    query = "SELECT id, id_paciente, sys, dia, nivel FROM mediciones ORDER BY id DESC LIMIT 20"
    try:
        cursor.execute(query)
        mediciones = cursor.fetchall()
        for med in mediciones:
            for key, val in med.items():
                med[key] = str(val)
        return jsonify(mediciones), 200
    except mysql.connector.Error as err:
        print(f"❌ Error MySQL al leer: {err}")
        return jsonify([]), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.route("/api/autorizacion", methods=["GET", "POST"])
def api_control_autorizacion():
    global autorizado
    if request.method == "POST":
        data = request.get_json()
        nuevo_estado = data.get("autorizado", False)
        autorizado = bool(nuevo_estado)
        print(f"Estado de autorización cambiado a: {autorizado}")
        socketio.emit('status_update', {"autorizado": autorizado})
        return jsonify({"autorizado": autorizado}), 200
    else: # GET
        return jsonify({"autorizado": autorizado}), 200

if __name__ == "__main__":
    print("Iniciando servidor Flask con SocketIO...")
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
