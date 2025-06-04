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

app = Flask(__name__)

# --- Carga de Modelos de Machine Learning ---
MODEL_SYS_PATH = "modelo_sys.pkl"
MODEL_DIA_PATH = "modelo_dia.pkl"
modelo_sys = None
modelo_dia = None

try:
    if os.path.exists(MODEL_SYS_PATH):
        modelo_sys = joblib.load(MODEL_SYS_PATH)
        print(f"✅ Modelo {MODEL_SYS_PATH} cargado exitosamente.")
    else:
        print(f"⚠️ Advertencia: Archivo {MODEL_SYS_PATH} no encontrado.")

    if os.path.exists(MODEL_DIA_PATH):
        modelo_dia = joblib.load(MODEL_DIA_PATH)
        print(f"✅ Modelo {MODEL_DIA_PATH} cargado exitosamente.")
    else:
        print(f"⚠️ Advertencia: Archivo {MODEL_DIA_PATH} no encontrado.")
        
except Exception as e:
    print(f"❌ Error al cargar los modelos de Machine Learning: {e}")

# --- Variables Globales ---
autorizado = False
ultima_estimacion = {
    "sys": "---", "dia": "---", "spo2": "---", 
    "hr": "---", "nivel": "---", "timestamp": "---",
    "modo_autorizado": False 
}

ventana_ir = []
ventana_red = []
MUESTRAS = 10 

# --- Configuración de la Base de Datos MySQL (desde variables de entorno) ---
DB_HOST = os.environ.get('DB_HOST', os.environ.get("MYSQLHOST")) 
DB_USER = os.environ.get('DB_USER', os.environ.get("MYSQLUSER"))
DB_PASSWORD = os.environ.get('DB_PASSWORD', os.environ.get("MYSQLPASSWORD"))
DB_NAME = os.environ.get('DB_NAME', os.environ.get("MYSQLDATABASE"))
DB_PORT = os.environ.get('DB_PORT', os.environ.get("MYSQLPORT", "3306")) 

DB_CONFIG = {
    'host': DB_HOST,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME,
    'port': int(DB_PORT) if DB_PORT and DB_PORT.isdigit() else 3306
}
print(f"Configuración DB: Host={DB_CONFIG['host']}, User={DB_CONFIG['user']}, DB={DB_CONFIG['database']}, Port={DB_CONFIG['port']}")

# --- Configuración de Google Drive API ---
# Esta es la variable que faltaba definir globalmente:
KEY_FILE_LOCATION_FALLBACK = 'service_account.json' 
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-") 
CSV_FILENAME = "registro_sensor_entrenamiento.csv"

if not FOLDER_ID or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
    print("⚠️ Advertencia: GOOGLE_DRIVE_FOLDER_ID no está configurado correctamente o usa el valor de ejemplo.")
# La verificación del archivo de credenciales se hará dentro de get_google_drive_service


def clasificar_nivel_presion(pas_valor, pad_valor):
    if pas_valor is None or pad_valor is None or pd.isna(pas_valor) or pd.isna(pad_valor) or pas_valor == -1 or pad_valor == -1: 
        return "---"
    if pas_valor > 180 or pad_valor > 120: return "HT Crisis"
    elif pas_valor >= 140 or pad_valor >= 90: return "HT2"        
    elif (pas_valor >= 130 and pas_valor <= 139) or (pad_valor >= 80 and pad_valor <= 89): return "HT1"        
    elif (pas_valor >= 120 and pas_valor <= 129) and pad_valor < 80: return "Elevada"
    elif pas_valor < 120 and pad_valor < 80: return "Normal"
    else:
        if pad_valor >= 80: 
             if pad_valor >= 90: return "HT2"
             else: return "HT1" 
        return "Revisar" 

def conectar_db():
    if not all([DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database']]):
        print("Error: Configuración de base de datos incompleta.")
        return None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print("✅ Conexión a MySQL establecida.")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Error al conectar a MySQL: {err}")
        return None

def guardar_medicion_mysql(id_paciente_recibido, valor_pas_estimada, valor_pad_estimada, valor_nivel_calculado):
    conn = conectar_db() 
    if conn is None:
        print("guardar_medicion_mysql: No se pudo conectar a la DB.")
        return False
    cursor = conn.cursor()
    query = """
    INSERT INTO mediciones (id_paciente, sys, dia, nivel) 
    VALUES (%s, %s, %s, %s)
    """
    datos_a_insertar = (id_paciente_recibido, valor_pas_estimada, valor_pad_estimada, valor_nivel_calculado)
    try:
        print(f"guardar_medicion_mysql (simplificado): Intentando insertar: {datos_a_insertar}")
        cursor.execute(query, datos_a_insertar)
        conn.commit()
        print(f"Datos guardados en MySQL (simplificado): IDP={id_paciente_recibido}, SYS(PAS)={valor_pas_estimada}, DIA(PAD)={valor_pad_estimada}, Nivel={valor_nivel_calculado}")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Error al guardar en MySQL (simplificado): {err}")
        conn.rollback() 
        return False
    finally:
        if conn.is_connected(): 
            cursor.close()
            conn.close()
            print("Conexión a MySQL cerrada (guardar simplificado).")

def get_google_drive_service():
    global KEY_FILE_LOCATION_FALLBACK # Asegurar que acceda a la global si es necesario (aunque ya lo hace por scope)
    effective_key_file_location = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', KEY_FILE_LOCATION_FALLBACK)
    print(f"Intentando cargar credenciales de Google Drive desde: {effective_key_file_location}")
    if not os.path.exists(effective_key_file_location):
        print(f"⚠️ Advertencia: Archivo de credenciales '{effective_key_file_location}' no encontrado. La subida a Drive no funcionará.")
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(effective_key_file_location, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        print("✅ Servicio de Google Drive autenticado.")
        return service
    except Exception as e: print(f"❌ Error al autenticar con Google Drive: {e}"); return None

def subir_archivo_a_drive(file_path, file_name_on_drive):
    if not FOLDER_ID or FOLDER_ID == 'tu_id_de_carpeta_en_google_drive_placeholder' or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
        print("Error: FOLDER_ID de Google Drive no configurado o es placeholder/ejemplo. No se puede subir archivo.")
        return False
    service = get_google_drive_service()
    if not service: return False
    try:
        file_metadata = {'name': file_name_on_drive, 'parents': [FOLDER_ID]}
        media = MediaFileUpload(file_path, mimetype='text/csv', resumable=True)
        query = f"name='{file_name_on_drive}' and '{FOLDER_ID}' in parents and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        existing_files = response.get('files', [])
        if existing_files: 
            file_id = existing_files[0].get('id')
            updated_file = service.files().update(fileId=file_id, media_body=media).execute()
            print(f"Archivo '{file_name_on_drive}' actualizado en Drive. ID: {updated_file.get('id')}")
        else: 
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"Archivo '{file_name_on_drive}' subido a Drive. ID: {file.get('id')}")
        return True
    except Exception as e: print(f"❌ Error al subir/actualizar archivo en Google Drive: {e}"); return False

@app.route("/")
def home():
    global ultima_estimacion, autorizado
    ultima_estimacion["modo_autorizado"] = autorizado 
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global autorizado, ultima_estimacion, modelo_sys, modelo_dia, ventana_ir, ventana_red
    if not modelo_sys or not modelo_dia:
        return jsonify({"error": "Modelos de ML no cargados"}), 500
    try:
        data = request.get_json();
        if not data: return jsonify({"error": "Request debe ser JSON"}), 400
        hr_in = data.get("hr"); ir_in = data.get("ir"); red_in = data.get("red")
        id_paciente_in = data.get("id_paciente", 1) 
        if hr_in is None or ir_in is None or red_in is None:
            return jsonify({"error": "Datos incompletos: 'hr', 'ir', 'red' requeridos"}), 400
        try:
            hr = int(hr_in); ir_val = int(ir_in); red_val = int(red_in) 
        except ValueError: return jsonify({"error": "'hr', 'ir', 'red' deben ser numéricos enteros"}), 400

        ventana_ir.append(ir_val); ventana_red.append(red_val)
        if len(ventana_ir) > MUESTRAS: ventana_ir.pop(0)
        if len(ventana_red) > MUESTRAS: ventana_red.pop(0)

        spo2_estimada = 0.0 
        if len(ventana_ir) == MUESTRAS:
            if ir_val < 10000 or red_val < 10000: spo2_estimada = 0.0 # Si no hay dedo, spo2 = 0
            else:
                try:
                    np_ventana_ir = np.array(ventana_ir); np_ventana_red = np.array(ventana_red)
                    dc_ir = np.mean(np_ventana_ir); dc_red = np.mean(np_ventana_red)
                    ac_ir = np.mean(np.abs(np_ventana_ir - dc_ir)) 
                    ac_red = np.mean(np.abs(np_ventana_red - dc_red))
                    if dc_ir > 0 and dc_red > 0 and ac_ir > 0 and ac_red > 0 : 
                        ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
                        spo2_calc = 110 - 25 * ratio
                        spo2_estimada = max(70.0, min(100.0, spo2_calc))
                    else: print(f"DEBUG: SpO2 (origen) - DC/AC inválido.")
                except Exception as e_spo2: print(f"Error en cálculo SpO2 (origen): {e_spo2}"); spo2_estimada = 0.0             
        
        pas_estimada = -1.0; pad_estimada = -1.0 
        if spo2_estimada > 0 and modelo_sys and modelo_dia: 
            entrada_df = pd.DataFrame([[float(hr), float(spo2_estimada)]], columns=['hr', 'spo2'])
            pas_estimada = round(modelo_sys.predict(entrada_df)[0], 2)
            pad_estimada = round(modelo_dia.predict(entrada_df)[0], 2)
            
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)
        
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ultima_estimacion = {
            "sys": f"{pas_estimada:.2f}" if pas_estimada != -1 else "---",
            "dia": f"{pad_estimada:.2f}" if pad_estimada != -1 else "---",
            "spo2": f"{spo2_estimada:.1f}" if spo2_estimada > 0 else "---",
            "hr": str(hr), "nivel": nivel_presion,
            "timestamp": current_timestamp, 
            "modo_autorizado": autorizado
        }

        if autorizado:
            try:
                with open(CSV_FILENAME, mode='a', newline='') as file_csv:
                    file_csv.write(f"{hr},{spo2_estimada:.1f},{ir_val},{red_val},{pas_estimada:.2f},{pad_estimada:.2f},{current_timestamp}\n")
                print(f"Datos guardados en {CSV_FILENAME} para entrenamiento.")
                if FOLDER_ID and FOLDER_ID != 'tu_id_de_carpeta_en_google_drive_placeholder' and FOLDER_ID != "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
                     subir_archivo_a_drive(CSV_FILENAME, CSV_FILENAME)
            except Exception as e_csv: print(f"Error al guardar o subir CSV: {e_csv}")

        print(f"DEBUG: Antes de guardar en DB - ir_val={ir_val}, red_val={red_val}, pas_estimada={pas_estimada}")
        if ir_val > 20000 and red_val > 15000 and pas_estimada != -1: 
            guardar_medicion_mysql(id_paciente_in, pas_estimada, pad_estimada, nivel_presion)
        else: print("DEBUG: Condición para guardar en DB no cumplida.")

        return jsonify({
            "sys": round(pas_estimada, 2) if pas_estimada != -1 else -1, 
            "dia": round(pad_estimada, 2) if pad_estimada != -1 else -1, 
            "spo2": round(spo2_estimada, 1) if spo2_estimada > 0 else 0, 
            "nivel": nivel_presion
        }), 200
    except Exception as e:
        print(f"❌ Error general en /api/presion: {e}"); import traceback; traceback.print_exc()
        return jsonify({"error": "Error interno del servidor", "detalle": str(e)}), 500

@app.route("/api/ultimas_mediciones", methods=["GET"])
def get_ultimas_mediciones_db():
    conn = conectar_db()
    if conn is None: return jsonify({"error": "No se pudo conectar a la DB"}), 500
    cursor = conn.cursor(dictionary=True) 
    # Asumiendo que tu tabla 'mediciones' tiene 'id' autoincremental.
    # Si tienes un campo 'timestamp' que la DB llena, ordena por ese.
    query = "SELECT id, id_paciente, sys, dia, nivel FROM mediciones ORDER BY id DESC LIMIT 20"
    mediciones = []
    try:
        print("get_ultimas_mediciones_db: Ejecutando query.")
        cursor.execute(query)
        mediciones = cursor.fetchall()
        # Si tuvieras un campo timestamp de la DB y necesitas formatearlo:
        # for med in mediciones:
        #     if 'timestamp' in med and isinstance(med['timestamp'], datetime):
        #         med['timestamp'] = med['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"get_ultimas_mediciones_db: {len(mediciones)} mediciones obtenidas.")
    except mysql.connector.Error as err:
        print(f"❌ Error al obtener últimas mediciones: {err}")
        return jsonify({"error": "Error al consultar DB", "detalle": str(err)}), 500
    finally:
        if conn.is_connected(): cursor.close(); conn.close(); print("Conexión MySQL cerrada (ultimas_mediciones).")
    return jsonify(mediciones), 200

@app.route("/api/autorizacion", methods=["GET", "POST"])
def api_control_autorizacion():
    global autorizado, ultima_estimacion 
    if request.method == "POST":
        try:
            data = request.get_json()
            if data is None or "autorizado" not in data:
                 return jsonify({"error": "Payload JSON esperado"}), 400
            nuevo_estado = data.get("autorizado")
            if isinstance(nuevo_estado, bool):
                autorizado = nuevo_estado
                print(f"Estado de autorización cambiado a: {autorizado}")
                ultima_estimacion["modo_autorizado"] = autorizado 
                return jsonify({"mensaje": f"Autorización cambiada a {autorizado}", "autorizado": autorizado}), 200
            else: return jsonify({"error": "Valor de 'autorizado' debe ser booleano"}), 400
        except Exception as e:
            print(f"❌ Error en POST /api/autorizacion: {e}")
            return jsonify({"error": "Error interno", "detalle": str(e)}), 400
    else: # GET
        return jsonify({"autorizado": autorizado}), 200

@app.route("/api/ultima_estimacion", methods=["GET"])
def get_ultima_medicion():
    global ultima_estimacion
    ultima_estimacion["modo_autorizado"] = autorizado 
    return jsonify(ultima_estimacion)

if __name__ == "__main__":
    print("Iniciando servidor Flask para desarrollo local...")
    if not all([DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database']]):
        print("ADVERTENCIA: Faltan variables de entorno para la base de datos.")
    if not FOLDER_ID or FOLDER_ID == 'tu_id_de_carpeta_en_google_drive_placeholder' or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
        print("ADVERTENCIA: GOOGLE_DRIVE_FOLDER_ID no configurado.")
    
    effective_key_file_location = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', KEY_FILE_LOCATION_FALLBACK)
    if not os.path.exists(effective_key_file_location):
         print(f"ADVERTENCIA: Archivo de credenciales '{effective_key_file_location}' no encontrado.")

    port = int(os.environ.get("PORT", 10000)) 
    app.run(host='0.0.0.0', port=port, debug=True)
