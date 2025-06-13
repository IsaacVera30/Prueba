from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import pandas as pd
import joblib
import os
import mysql.connector
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Carga de modelos del repositorio actual
try:
    modelo_sys = joblib.load('modelo_sys.pkl')
    modelo_dia = joblib.load('modelo_dia.pkl')
    print("✅ Modelos de ML cargados correctamente.")
except Exception as e:
    print(f"❌ ERROR: No se pudieron cargar los modelos de ML: {e}")
    modelo_sys, modelo_dia = None, None

# Configuración de Base de Datos
DB_CONFIG = {
    'host': os.environ.get("MYSQLHOST"), 'user': os.environ.get("MYSQLUSER"),
    'password': os.environ.get("MYSQLPASSWORD"), 'database': os.environ.get("MYSQLDATABASE"),
    'port': int(os.environ.get("MYSQLPORT", "3306"))
}
autorizado = False # Estado para controlar el guardado en BD

def conectar_db():
    try: return mysql.connector.connect(**DB_CONFIG)
    except Exception as e: print(f"❌ Error DB: {e}"); return None

def guardar_medicion_mysql(id_paciente, sys, dia, nivel):
    # (Esta función se mantiene como la definimos antes, es correcta)
    conn = conectar_db()
    if not conn: return
    cursor = conn.cursor()
    query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s, %s, %s, %s)"
    try:
        cursor.execute(query, (id_paciente, sys, dia, nivel))
        conn.commit()
    except Exception as e:
        print(f"❌ Error al guardar en MySQL: {e}")
    finally:
        if conn.is_connected(): conn.close()

def clasificar_nivel_presion(pas, pad):
    if pas > 180 or pad > 120: return "HT Crisis"
    if pas >= 140 or pad >= 90: return "HT2"
    if (pas >= 130 and pas <= 139) or (pad >= 80 and pad <= 89): return "HT1"
    if (pas >= 120 and pas <= 129) and pad < 80: return "Elevada"
    return "Normal"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    if not modelo_sys or not modelo_dia:
        return jsonify({"error": "Modelos no disponibles en el servidor"}), 503

    data = request.get_json()
    try:
        # --- 1. Recibir todos los datos del dispositivo ---
        hr_promedio = float(data["hr_promedio"])
        ir_val = float(data["ir"])
        red_val = float(data["red"])
        id_paciente = int(data.get("id_paciente", 1))

        # --- 2. Realizar la predicción ML ---
        # El modelo actual en tu repo usa 'HR', 'IR' y 'RED'
        entrada_df = pd.DataFrame([[hr_promedio, ir_val, red_val]], columns=['HR', 'IR', 'RED'])
        pas_estimada = modelo_sys.predict(entrada_df)[0]
        pad_estimada = modelo_dia.predict(entrada_df)[0]
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)

        # --- 3. Preparar el paquete de datos COMPLETO para el panel web ---
        datos_para_panel = {
            # Sección "Lecturas del Sensor"
            "hr_crudo": data.get("hr_crudo"),
            "hr_promedio": f"{hr_promedio:.0f}",
            "spo2_sensor": data.get("spo2_sensor"),
            "ir": f"{ir_val:.0f}",
            "red": f"{red_val:.0f}",
            # Sección "Predicción ML"
            "sys_ml": f"{pas_estimada:.2f}",
            "dia_ml": f"{pad_estimada:.2f}",
            "hr_ml": f"{hr_promedio:.0f}",  # Se usa el mismo promedio como referencia
            "spo2_ml": data.get("spo2_sensor"), # Se usa el mismo SpO2 como referencia
            "estado": nivel_presion
        }
        socketio.emit('update_data', datos_para_panel)

        # --- 4. Guardar en BD si está autorizado ---
        if autorizado:
            guardar_medicion_mysql(id_paciente, pas_estimada, pad_estimada, nivel_presion)
            socketio.emit('new_record_saved')

        # --- 5. Enviar la respuesta simple de vuelta al dispositivo ---
        respuesta_para_dispositivo = {
            "sys": pas_estimada,
            "dia": pad_estimada,
            "hr": hr_promedio,
            "spo2": float(data.get("spo2_sensor", 0)),
            "nivel": nivel_presion
        }
        return jsonify(respuesta_para_dispositivo), 200

    except Exception as e:
        print(f"❌ ERROR en /api/presion: {e}")
        return jsonify({"error": str(e)}), 500

# Rutas para autorización y tabla de historial (estas ya son correctas)
@app.route("/api/autorizacion", methods=["GET", "POST"])
def api_control_autorizacion():
    global autorizado
    if request.method == "POST":
        autorizado = request.json.get("autorizado", False)
        socketio.emit('status_update', {"autorizado": autorizado})
    return jsonify({"autorizado": autorizado})

@app.route("/api/ultimas_mediciones")
def get_ultimas_mediciones_db():
    conn = conectar_db()
    if not conn: return jsonify([])
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, sys, dia, nivel FROM mediciones ORDER BY id DESC LIMIT 20")
    records = cursor.fetchall()
    conn.close()
    return jsonify(records)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=10000)
