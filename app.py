from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib # Para cargar modelos .pkl de scikit-learn
import numpy as np # Importado para cálculos de SpO2 como en el código origen
import os
import csv
from datetime import datetime
import mysql.connector # Asegúrate de tener PyMySQL o mysql-connector-python instalado
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload # Usado en la versión actual del Canvas

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
    "hr": "---", "nivel": "---", "timestamp": "---"
}

# Ventana de datos para SpO2 (como en el código origen)
ventana_ir = []
ventana_red = []
MUESTRAS = 10 # Coincide con MUESTRAS en el código origen para SpO2

# --- Configuración de la Base de Datos MySQL (desde variables de entorno) ---
DB_HOST = os.environ.get('DB_HOST') # En tu origen era MYSQLHOST
DB_USER = os.environ.get('DB_USER') # En tu origen era MYSQLUSER
DB_PASSWORD = os.environ.get('DB_PASSWORD') # En tu origen era MYSQLPASSWORD
DB_NAME = os.environ.get('DB_NAME') # En tu origen era MYSQLDATABASE
DB_PORT = os.environ.get('DB_PORT', 3306) # En tu origen era MYSQLPORT

DB_CONFIG = {
    'host': DB_HOST,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME,
    'port': int(DB_PORT) if DB_PORT else 3306
}
print(f"Configuración DB: Host={DB_CONFIG['host']}, User={DB_CONFIG['user']}, DB={DB_CONFIG['database']}, Port={DB_CONFIG['port']}")


# --- Configuración de Google Drive API ---
# Usando la estructura del Canvas que es más flexible para el path de credenciales
KEY_FILE_LOCATION = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'service_account.json') # Render puede usar GOOGLE_APPLICATION_CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-") # Usando el ID de tu código origen como default
CSV_FILENAME = "registro_sensor_entrenamiento.csv"

if not FOLDER_ID or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-": # Si es el valor por defecto o no está
    print("⚠️ Advertencia: GOOGLE_DRIVE_FOLDER_ID no está configurado correctamente o usa el valor de ejemplo. La subida a Drive podría no funcionar como se espera.")
if not os.path.exists(KEY_FILE_LOCATION):
    print(f"⚠️ Advertencia: Archivo de credenciales '{KEY_FILE_LOCATION}' no encontrado. La subida a Drive no funcionará.")


# --- Cálculo de SpO2 (adaptado del código origen) ---
def calcular_spo2_desde_origen(current_ir, current_red):
    global ventana_ir, ventana_red # Usando las ventanas globales como en tu código origen
    
    # Asegurar que los valores sean numéricos
    try:
        ir_val = float(current_ir)
        red_val = float(current_red)
    except ValueError:
        print("Error: ir_val o red_val no son numéricos para SpO2 (origen).")
        return 0 # Como en tu código origen

    ventana_ir.append(ir_val)
    ventana_red.append(red_val)

    if len(ventana_ir) > MUESTRAS:
        ventana_ir.pop(0)
    if len(ventana_red) > MUESTRAS:
        ventana_red.pop(0)

    spo2 = 0 # Valor por defecto como en tu código origen
    if len(ventana_ir) == MUESTRAS: # Solo calcular si tenemos suficientes muestras
        # Evitar procesar si los valores son demasiado bajos (sin dedo)
        # Este umbral es del código del Canvas, tu código origen no lo tenía aquí explícitamente
        # pero es una buena práctica. Lo mantendré.
        if ir_val < 10000 or red_val < 10000: 
            print("DEBUG: Valores IR/RED bajos en SpO2 (origen), posible no dedo.")
            return 0 # O -1 para indicar error/no cálculo

        try:
            np_ventana_ir = np.array(ventana_ir)
            np_ventana_red = np.array(ventana_red)

            dc_ir = np.mean(np_ventana_ir)
            dc_red = np.mean(np_ventana_red)
            
            # Cálculo de AC como en tu código origen
            ac_ir = np.mean(np.abs(np_ventana_ir - dc_ir))
            ac_red = np.mean(np.abs(np_ventana_red - dc_red))

            if dc_ir <= 0 or dc_red <= 0 or ac_ir <= 0 or ac_red <= 0:
                print("DEBUG: DC o AC es cero o negativo en SpO2 (origen).")
                return 0 # O -1

            ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
            spo2_calc = 110 - 25 * ratio 
            spo2 = max(70.0, min(100.0, spo2_calc)) # Asegurar rango 70-100
            
        except ZeroDivisionError:
            print("Error: División por cero en cálculo de SpO2 (origen).")
            return 0 # O -1
        except Exception as e:
            print(f"Error inesperado en cálculo de SpO2 (origen): {e}")
            return 0 # O -1
            
    return round(spo2, 1)


def clasificar_nivel_presion(pas_valor, pad_valor):
    """Clasifica la presión arterial según los nuevos niveles."""
    if pas_valor == -1 or pad_valor == -1 or pd.isna(pas_valor) or pd.isna(pad_valor): 
        return "---"
        
    # Criterios actualizados
    if pas_valor > 180 or pad_valor > 120:
        return "HT Crisis"
    elif pas_valor >= 140 or pad_valor >= 90:
        return "HT2"        
    elif (pas_valor >= 130 and pas_valor <= 139) or (pad_valor >= 80 and pad_valor <= 89):
        return "HT1"        
    elif (pas_valor >= 120 and pas_valor <= 129) and pad_valor < 80:
        return "Elevada"
    elif pas_valor < 120 and pad_valor < 80:
        return "Normal"
    else:
        if pad_valor >= 80: 
             if pad_valor >= 90: return "HT2"
             else: return "HT1" 
        return "Revisar" 

def conectar_db():
    # ... (Se mantiene la lógica de conexión del Canvas) ...
    if not all([DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database']]):
        print("Error: Configuración de base de datos incompleta. No se puede conectar.")
        return None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print("✅ Conexión a MySQL establecida.")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Error al conectar a la base de datos MySQL: {err}")
        return None

def guardar_medicion_mysql(id_paciente_recibido, valor_pas_estimada, valor_pad_estimada, valor_spo2_estimada, valor_hr_recibido, valor_nivel_calculado):
    # ... (Se mantiene la lógica de guardado del Canvas, que incluye spo2, hr, timestamp) ...
    # ... (La query ya usa 'sys' y 'dia' como en la corrección anterior) ...
    conn = conectar_db() 
    if conn is None:
        print("guardar_medicion_mysql: No se pudo conectar a la DB.")
        return False
    
    cursor = conn.cursor()
    timestamp_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    query = """
    INSERT INTO mediciones (id_paciente, sys, dia, spo2, hr, nivel, timestamp) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    datos_a_insertar = (id_paciente_recibido, valor_pas_estimada, valor_pad_estimada, valor_spo2_estimada, valor_hr_recibido, valor_nivel_calculado, timestamp_actual)
    
    try:
        print(f"guardar_medicion_mysql: Intentando insertar: {datos_a_insertar}")
        cursor.execute(query, datos_a_insertar)
        conn.commit()
        print(f"Datos guardados en MySQL: IDP={id_paciente_recibido}, SYS(PAS)={valor_pas_estimada}, DIA(PAD)={valor_pad_estimada}, SpO2={valor_spo2_estimada}, HR={valor_hr_recibido}, Nivel={valor_nivel_calculado}")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Error al guardar en MySQL: {err}")
        conn.rollback() 
        return False
    finally:
        if conn.is_connected(): 
            cursor.close()
            conn.close()
            print("Conexión a MySQL cerrada después de guardar/fallar.")

def get_google_drive_service():
    # ... (Se mantiene la lógica de Google Drive del Canvas) ...
    if not os.path.exists(KEY_FILE_LOCATION):
        print(f"⚠️ Advertencia: Archivo de credenciales '{KEY_FILE_LOCATION}' no encontrado. La subida a Drive no funcionará.")
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(
            KEY_FILE_LOCATION, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        print("✅ Servicio de Google Drive autenticado.")
        return service
    except Exception as e:
        print(f"❌ Error al autenticar con Google Drive: {e}")
        return None

def subir_archivo_a_drive(file_path, file_name_on_drive):
    # ... (Se mantiene la lógica de Google Drive del Canvas, que actualiza si existe) ...
    if not FOLDER_ID or FOLDER_ID == 'tu_id_de_carpeta_en_google_drive': 
        print("Error: FOLDER_ID de Google Drive no configurado o es placeholder. No se puede subir archivo.")
        return False
        
    service = get_google_drive_service()
    if not service:
        return False
    
    try:
        file_metadata = {
            'name': file_name_on_drive,
            'parents': [FOLDER_ID] 
        }
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
    except Exception as e:
        print(f"❌ Error al subir/actualizar archivo '{file_name_on_drive}' en Google Drive: {e}")
        return False

@app.route("/")
def home():
    global ultima_estimacion, autorizado
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global autorizado, ultima_estimacion, modelo_sys, modelo_dia, ventana_ir, ventana_red # Añadir ventanas globales
    
    if not modelo_sys or not modelo_dia:
        print("Error crítico: Modelos de ML no están cargados.")
        return jsonify({"error": "Modelos de ML no cargados en el servidor"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request debe ser JSON"}), 400

        # Obtener datos como en tu código origen (int)
        hr_in = data.get("hr") 
        ir_in = data.get("ir")
        red_in = data.get("red")
        id_paciente_in = data.get("id_paciente", 1) 

        if hr_in is None or ir_in is None or red_in is None:
            return jsonify({"error": "Datos incompletos: 'hr', 'ir', 'red' son requeridos"}), 400
        
        try:
            # En tu código origen, hr, ir, red se usan como int para algunas cosas
            # pero para los cálculos de spo2 y predicción se necesitarán floats
            hr = float(hr_in) 
            ir_float = float(ir_in) # Usar para spo2
            red_float = float(red_in) # Usar para spo2
        except ValueError:
            return jsonify({"error": "'hr', 'ir', 'red' deben ser numéricos"}), 400

        # Usar el cálculo de SpO2 adaptado de tu código origen
        spo2_estimada = calcular_spo2_desde_origen(ir_float, red_float)
        
        pas_estimada = -1.0 
        pad_estimada = -1.0 

        if spo2_estimada != 0 and spo2_estimada != -1 and modelo_sys and modelo_dia: # spo2=0 es un valor válido en tu origen, pero no ideal para predicción
            # Asegurar que las columnas coincidan con el entrenamiento
            entrada_df = pd.DataFrame([[hr, spo2_estimada]], columns=['hr', 'spo2'])
            pas_estimada = round(modelo_sys.predict(entrada_df)[0], 2) # .2f como en tu origen
            pad_estimada = round(modelo_dia.predict(entrada_df)[0], 2) # .2f como en tu origen
            
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)
        
        ultima_estimacion["sys"] = f"{pas_estimada:.2f}" if pas_estimada != -1 else "---"
        ultima_estimacion["dia"] = f"{pad_estimada:.2f}" if pad_estimada != -1 else "---"
        ultima_estimacion["spo2"] = f"{spo2_estimada:.1f}" if spo2_estimada != 0 and spo2_estimada != -1 else "---"
        ultima_estimacion["hr"] = str(int(hr)) # hr como int
        ultima_estimacion["nivel"] = nivel_presion
        ultima_estimacion["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if autorizado:
            try:
                # Guardar en CSV como en tu código origen
                with open(CSV_FILENAME, mode='a', newline='') as file_csv:
                    # Formato: hr,spo2,ir,red,sys,dia
                    file_csv.write(f"{int(hr)},{spo2_estimada:.1f},{int(ir_in)},{int(red_in)},{pas_estimada:.2f},{pad_estimada:.2f}\n")
                print(f"Datos guardados en {CSV_FILENAME} para entrenamiento.")
                if FOLDER_ID and FOLDER_ID != 'tu_id_de_carpeta_en_google_drive':
                     subir_archivo_a_drive(CSV_FILENAME, CSV_FILENAME)
            except Exception as e_csv:
                print(f"Error al guardar o subir CSV de entrenamiento: {e_csv}")

        # Condición para guardar en MySQL (ir > 20000 and red > 15000 como en tu código origen)
        print(f"DEBUG: Antes de guardar en DB - ir={ir_in}, red={red_in}, pas_estimada={pas_estimada}")
        if int(ir_in) > 20000 and int(red_in) > 15000 and pas_estimada != -1: 
            guardar_medicion_mysql(id_paciente_in, pas_estimada, pad_estimada, spo2_estimada, hr, nivel_presion)
        else:
            print("DEBUG: Condición para guardar en DB no cumplida (según umbrales origen).")

        return jsonify({
            "sys": round(pas_estimada, 2) if pas_estimada != -1 else -1, 
            "dia": round(pad_estimada, 2) if pad_estimada != -1 else -1, 
            "spo2": round(spo2_estimada, 1) if spo2_estimada != 0 and spo2_estimada != -1 else 0, # Devolver 0 si no se calculó, como en origen
            "nivel": nivel_presion
        }), 200

    except Exception as e:
        print(f"❌ Error general en /api/presion: {e}")
        return jsonify({"error": "Error interno del servidor", "detalle": str(e)}), 500

# Manteniendo la estructura de /api/autorizacion del Canvas (GET y POST)
@app.route("/api/autorizacion", methods=["GET", "POST"])
def api_control_autorizacion():
    global autorizado
    if request.method == "POST":
        try:
            data = request.get_json()
            if data is None or "autorizado" not in data:
                 return jsonify({"error": "Payload JSON esperado con la clave 'autorizado'"}), 400

            nuevo_estado = data.get("autorizado")
            if isinstance(nuevo_estado, bool):
                autorizado = nuevo_estado
                print(f"Estado de autorización cambiado a: {autorizado}")
                ultima_estimacion["modo_autorizado"] = autorizado 
                return jsonify({"mensaje": f"Autorización cambiada a {autorizado}", "autorizado": autorizado}), 200
            else:
                return jsonify({"error": "Valor de 'autorizado' debe ser booleano (true/false)"}), 400
        except Exception as e:
            print(f"❌ Error en POST /api/autorizacion: {e}")
            return jsonify({"error": "Error interno del servidor", "detalle": str(e)}), 400
    else: 
        return jsonify({"autorizado": autorizado}), 200

@app.route("/api/ultima_estimacion", methods=["GET"])
def get_ultima_medicion():
    global ultima_estimacion
    return jsonify(ultima_estimacion)

if __name__ == "__main__":
    print("Iniciando servidor Flask para desarrollo local...")
    # ... (advertencias de configuración como en el Canvas) ...
    if not all([DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database']]):
        print("ADVERTENCIA: Faltan variables de entorno para la base de datos. La conexión a DB podría fallar.")
    if not FOLDER_ID or FOLDER_ID == 'tu_id_de_carpeta_en_google_drive' or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
        print("ADVERTENCIA: GOOGLE_DRIVE_FOLDER_ID no configurado o es placeholder/ejemplo. La subida a Drive no funcionará.")
    if not os.path.exists(KEY_FILE_LOCATION):
         print(f"ADVERTENCIA: Archivo de credenciales '{KEY_FILE_LOCATION}' no encontrado. La subida a Drive no funcionará.")

    port = int(os.environ.get("PORT", 10000)) # Usando el puerto de tu código origen para local
    app.run(host='0.0.0.0', port=port, debug=True)
