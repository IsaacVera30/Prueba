import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO
import os, time, csv, io
import numpy as np
import joblib, pandas as pd
from datetime import datetime
import mysql.connector
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import threading  # CAMBIO: añadido para usar threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')  # CAMBIO: Se especifica 'eventlet'

# Carga de modelos entrenados
try:
    modelo_sys = joblib.load('models/modelo_sys.pkl')
    modelo_dia = joblib.load('models/modelo_dia.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("Modelos cargados")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    modelo_sys, modelo_dia, scaler = None, None, None

buffer_datos_entrenamiento = []
last_db_save_time = 0

# Config
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

def clasificar_nivel_presion(pas, pad):
    if pas is None or pad is None:
        return "N/A"
    pas, pad = float(pas), float(pad)
    if pas > 180 or pad > 120:
        return "HT Crisis"
    elif pas >= 140 or pad >= 90:
        return "HT2"
    elif pas >= 130 or pad >= 80:
        return "HT1"
    elif pas >= 120 and pad < 80:
        return "Elevada"
    else:
        return "Normal"

def conectar_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error DB: {e}")
        return None

def guardar_medicion_mysql(data):
    def save_async():
        conn = conectar_db()
        if not conn:
            return
        cursor = conn.cursor()
        query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel, hr_ml, spo2_ml) VALUES (%s, %s, %s, %s, %s, %s)"
        try:
            cursor.execute(query, (
                data.get("id_paciente"),
                data.get("sys_ml"),
                data.get("dia_ml"),
                data.get("estado"),
                data.get("hr_ml"),
                data.get("spo2_ml")
            ))
            conn.commit()
            print("Medición guardada en DB")
        except Exception as e:
            print(f"Error DB: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()
    threading.Thread(target=save_async).start()  # CAMBIO: uso de threading

def enviar_alerta_whatsapp(nivel, sys, dia):
    if not CALLMEBOT_API_KEY or not CALLMEBOT_PHONE_NUMBER:
        print("WhatsApp no configurado")
        return
    def send_async():
        try:
            import urllib.request
            mensaje = f"¡Alerta! Nivel: {nivel} (SYS: {sys}, DIA: {dia})"
            url = f"https://api.callmebot.com/whatsapp.php?phone={CALLMEBOT_PHONE_NUMBER}&text={mensaje.replace(' ', '%20')}&apikey={CALLMEBOT_API_KEY}"
            with urllib.request.urlopen(url, timeout=10) as response:
                print("Alerta enviada")
        except Exception as e:
            print(f"Error alerta: {e}")
    threading.Thread(target=send_async).start()  # CAMBIO

def get_google_drive_service():
    try:
        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        creds = service_account.Credentials.from_service_account_file('service_account.json', scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Error autenticando Drive: {e}")
        return None

def append_row_to_drive_csv(row_dict):
    def save_to_drive_async():
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
                existing_file = service.files().get_media(fileId=file_id).execute()
                existing_content = existing_file.decode('utf-8')
                string_io.write(existing_content)
                if not existing_content.strip().endswith('\n'):
                    string_io.write('\n')
                string_io.seek(0, io.SEEK_END)
                csv.writer(string_io).writerow(row_dict.values())
                media = MediaIoBaseUpload(io.BytesIO(string_io.getvalue().encode('utf-8')), mimetype='text/csv')
                service.files().update(fileId=file_id, media_body=media).execute()
            else:
                writer.writeheader()
                writer.writerow(row_dict)
                media = MediaIoBaseUpload(io.BytesIO(string_io.getvalue().encode('utf-8')), mimetype='text/csv')
                file_metadata = {'name': DRIVE_CSV_FILENAME, 'parents': [FOLDER_ID]}
                service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print("Fila guardada en Google Drive")
        except Exception as e:
            print(f"Error escribiendo en Drive: {e}")
    threading.Thread(target=save_to_drive_async).start() 

def procesar_buffer_y_guardar(ref_data):
    """Procesa los datos del buffer y los guarda en Google Drive"""
    global buffer_datos_entrenamiento
    if not buffer_datos_entrenamiento:
        return
    
    def process_async():
        global buffer_datos_entrenamiento
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

            final_row = {
                **features,
                'sys_ref': ref_data.get('sys_ref'),
                'dia_ref': ref_data.get('dia_ref'),
                'hr_ref': ref_data.get('hr_ref'),
                'timestamp_captura': timestamp_str
            }

            append_row_to_drive_csv(final_row)
        except Exception as e:
            print(f"Error procesando buffer: {e}")
        finally:
            buffer_datos_entrenamiento = []
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
            print("Sistema de captura reseteado")
    
    # Ejecutar procesamiento de forma asíncrona
    eventlet.spawn(process_async)

# ========== RUTAS DE LA API ==========

@app.route("/")
def home():
    """Ruta principal que renderiza el panel de control"""
    return render_template("index.html")

@app.route("/api/data", methods=["POST"])
def recibir_datos():
    """Endpoint principal para recibir datos del ESP32"""
    global last_db_save_time, buffer_datos_entrenamiento
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data"}), 400

    # LÓGICA DE PRUEBA PARA EL BUZZER
    if data.get("id_paciente") == 999:
        print("ID de prueba 999 detectado. Forzando respuesta de HT Crisis para probar el buzzer.")
        response = {"sys": 185, "dia": 125, "hr": 99, "spo2": 99, "nivel": "HT Crisis"}
        socketio.emit('update_data', data)
        return jsonify(response)

    # Inicializar valores de predicción
    data['sys_ml'], data['dia_ml'], data['hr_ml'], data['spo2_ml'], data['estado'] = 0, 0, 0, 0, "---"

    if os.path.exists(LOCK_FILE):
        # Modo captura de entrenamiento
        buffer_datos_entrenamiento.append(data)
        socketio.emit('capture_count_update', {'count': len(buffer_datos_entrenamiento)})
    else:
        # Modo predicción normal
        print(f"IR recibido: {data.get('ir')}")
        if float(data.get("ir", 0)) > 50000:
            if modelo_sys and modelo_dia and scaler:
                try:
                    # Crear lista de historial para std filtrado
                    buffer_datos_entrenamiento.append(data)
                    if len(buffer_datos_entrenamiento) > 5:
                        buffer_datos_entrenamiento.pop(0)
                    
                    ir_vals = [float(d.get("ir", 0)) for d in buffer_datos_entrenamiento]
                    red_vals = [float(d.get("red", 0)) for d in buffer_datos_entrenamiento]

                    raw_features = [[
                        float(data.get("hr_promedio", 0)),
                        float(data.get("spo2_sensor", 0)),
                        float(data.get("ir", 0)),
                        float(data.get("red", 0)),
                        float(np.std(ir_vals)) if len(ir_vals) > 1 else 0,
                        float(np.std(red_vals)) if len(red_vals) > 1 else 0
                    ]]

                    input_data = scaler.transform(raw_features)
                    data['sys_ml'] = round(modelo_sys.predict(input_data)[0], 2)
                    data['dia_ml'] = round(modelo_dia.predict(input_data)[0], 2)
                    data['hr_ml'] = float(data.get("hr_promedio", 0))
                    data['spo2_ml'] = float(data.get("spo2_sensor", 0))
                    data['estado'] = clasificar_nivel_presion(data['sys_ml'], data['dia_ml'])

                except Exception as e:
                    print(f"Error de predicción: {e}")
                    data['estado'] = "Error Pred."
            
            # Guardar en base de datos cada 5 segundos
            if (time.time() - last_db_save_time) >= 5:
                guardar_medicion_mysql(data)
                socketio.emit('new_record_saved')
                last_db_save_time = time.time()

            # Enviar alerta si es crisis hipertensiva
            if data.get("estado") == "HT Crisis":
                enviar_alerta_whatsapp(data.get("estado"), data.get("sys_ml"), data.get("dia_ml"))

    # Enviar datos al panel de control
    socketio.emit('update_data', data)

    # Respuesta para el ESP32
    return jsonify({
        "sys": round(data.get("sys_ml", 0), 2),
        "dia": round(data.get("dia_ml", 0), 2),
        "hr": round(data.get("hr_ml", 0), 2),
        "spo2": round(data.get("spo2_ml", 0), 2),
        "nivel": data.get("estado", "Normal")
    })

@app.route("/api/start_capture", methods=["POST"])
def start_capture():
    """Inicia el modo de captura para entrenamiento"""
    global buffer_datos_entrenamiento
    
    def create_lock_async():
        with open(LOCK_FILE, "w") as f:
            f.write("capturing")
    
    eventlet.spawn(create_lock_async)
    buffer_datos_entrenamiento = []
    print("Captura de entrenamiento iniciada")
    return jsonify({"status": "captura iniciada"})

@app.route("/api/stop_capture", methods=["POST"])
def stop_capture():
    """Detiene el modo de captura"""
    return jsonify({"status": "captura detenida", "muestras": len(buffer_datos_entrenamiento)})

@app.route("/api/save_training_data", methods=["POST"])
def save_training_data():
    """Guarda los datos de entrenamiento capturados"""
    if not os.path.exists(LOCK_FILE):
        return jsonify({"error": "La captura no está en modo 'pausa'."}), 400
    
    ref_data = request.get_json()
    procesar_buffer_y_guardar(ref_data)
    return jsonify({"status": "muestra de entrenamiento guardada y sistema reseteado"})

@app.route("/api/ultimas_mediciones")
def get_ultimas_mediciones_db():
    """Obtiene las últimas mediciones de la base de datos"""
    def get_records_async():
        conn = conectar_db()
        if not conn:
            return []
        
        cursor = conn.cursor(dictionary=True)
        query = "SELECT id, id_paciente, sys, dia, nivel, hr_ml, spo2_ml FROM mediciones ORDER BY id DESC LIMIT 20"
        
        try:
            cursor.execute(query)
            records = cursor.fetchall()
            
            # Convertir todos los valores a string para JSON
            for rec in records:
                for key, value in rec.items():
                    if value is not None:
                        rec[key] = str(value)
            
            return records
        except Exception as e:
            print(f"Error al obtener últimas mediciones: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()
    
    # Para operaciones de lectura que pueden tomar tiempo, usar thread pool
    try:
        records = get_records_async()
        return jsonify(records)
    except Exception as e:
        print(f"Error en ultimas_mediciones: {e}")
        return jsonify([])

@app.route("/api/test_alert", methods=['POST'])
def test_alert():
    """Endpoint de prueba para alertas"""
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

# ========== EVENTOS DE SOCKETIO ==========

@socketio.on('connect')
def handle_connect():
    """Maneja la conexión de clientes WebSocket"""
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    """Maneja la desconexión de clientes WebSocket"""
    print('Cliente desconectado')

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), debug=False)
