from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib # Para cargar modelos .pkl de scikit-learn
import os
import csv
from datetime import datetime
import mysql.connector # Asegúrate de tener PyMySQL o mysql-connector-python instalado
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)

# --- Carga de Modelos de Machine Learning ---
# Los modelos deben haber sido entrenados y guardados previamente.
# Se asume que esperan características como 'hr' y 'spo2'.
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
    # Considerar no iniciar la app si los modelos son cruciales y no se cargan.

# --- Variables Globales ---
autorizado = False  # Controla el modo de registro de datos para entrenamiento
ultima_estimacion = {
    "sys": "---", "dia": "---", "spo2": "---", 
    "hr": "---", "nivel": "---", "timestamp": "---"
}

# --- Configuración de la Base de Datos MySQL (desde variables de entorno) ---
DB_HOST = os.environ.get('DB_HOST')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_NAME = os.environ.get('DB_NAME')
DB_PORT = os.environ.get('DB_PORT', 3306) # Puerto por defecto de MySQL

DB_CONFIG = {
    'host': DB_HOST,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME,
    'port': int(DB_PORT) if DB_PORT else 3306
}
print(f"Configuración DB: Host={DB_CONFIG['host']}, User={DB_CONFIG['user']}, DB={DB_CONFIG['database']}")


# --- Configuración de Google Drive API (desde variables de entorno y archivo de credenciales) ---
KEY_FILE_LOCATION = 'service_account.json'  # Nombre del archivo de credenciales JSON
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID') # ID de la carpeta en Google Drive
CSV_FILENAME = "registro_sensor_entrenamiento.csv"

if not FOLDER_ID:
    print("⚠️ Advertencia: GOOGLE_DRIVE_FOLDER_ID no está configurado. La subida a Drive no funcionará.")

# --- Ventana de datos para SpO2 (cálculo simplificado) ---
# NOTA: Este es un cálculo muy básico de SpO2. Para precisión clínica,
# se requiere un algoritmo de procesamiento de señal PPG mucho más robusto y calibrado.
ventana_ir_spo2 = []
ventana_red_spo2 = []
MUESTRAS_SPO2_CALC = 10 # Número de muestras para el cálculo simple de SpO2

def calcular_spo2_simple(ir_val, red_val):
    global ventana_ir_spo2, ventana_red_spo2
    
    # Asegurar que los valores sean numéricos
    try:
        ir_val = float(ir_val)
        red_val = float(red_val)
    except ValueError:
        print("Error: ir_val o red_val no son numéricos para SpO2.")
        return -1

    ventana_ir_spo2.append(ir_val)
    ventana_red_spo2.append(red_val)

    if len(ventana_ir_spo2) > MUESTRAS_SPO2_CALC:
        ventana_ir_spo2.pop(0)
        ventana_red_spo2.pop(0)

    if len(ventana_ir_spo2) < MUESTRAS_SPO2_CALC:
        return -1 # No hay suficientes datos

    # Evitar procesar si los valores son demasiado bajos (sin dedo)
    if ir_val < 10000 or red_val < 10000: # Umbral simple, ajustar según sea necesario
        return -1

    try:
        dc_ir = sum(ventana_ir_spo2) / len(ventana_ir_spo2)
        ac_ir = max(ventana_ir_spo2) - min(ventana_ir_spo2)
        
        dc_red = sum(ventana_red_spo2) / len(ventana_red_spo2)
        ac_red = max(ventana_red_spo2) - min(ventana_red_spo2)

        if dc_ir <= 0 or dc_red <= 0 or ac_ir <= 0 or ac_red <= 0: # Evitar división por cero o valores no válidos
            return -1

        R = (ac_red / dc_red) / (ac_ir / dc_ir)
        # Fórmula empírica común (puede necesitar calibración): SpO2 = A - B * R
        # Valores A y B típicos: A=110, B=25. Estos pueden variar.
        spo2 = 110 - 25 * R 
        
        if spo2 > 100: spo2 = 100.0
        elif spo2 < 70: spo2 = 70.0 # Límite inferior razonable para este cálculo simple
        
        return round(spo2, 1)
    except ZeroDivisionError:
        print("Error: División por cero en cálculo de SpO2.")
        return -1
    except Exception as e:
        print(f"Error inesperado en cálculo de SpO2: {e}")
        return -1

def clasificar_nivel_presion(pas, pad):
    """Clasifica la presión arterial según los nuevos niveles."""
    if pas == -1 or pad == -1: # Si no hay predicción válida
        return "---"
        
    # Criterios basados en guías comunes (ej. AHA/ACC 2017, adaptados)
    if pas > 180 or pad > 120:
        return "HT Crisis"  # Crisis Hipertensiva
    elif pas >= 140 or pad >= 90: # OJO: algunas guías dicen PAS >= 140 Y PAD >=90, otras usan O. Usaremos O para ser más sensibles.
        return "HT2"        # Hipertensión Grado 2
    elif (pas >= 130 and pas <= 139) or (pad >= 80 and pad <= 89):
        return "HT1"        # Hipertensión Grado 1
    elif (pas >= 120 and pas <= 129) and pad < 80:
        return "Elevada"
    elif pas < 120 and pad < 80:
        return "Normal"
    else:
        # Este caso podría darse si, por ejemplo, PAS es <120 pero PAD es >80 y <90.
        # Según algunas guías, esto podría ser HT1 (si PAD 80-89).
        # La lógica actual prioriza la condición más severa si hay "O".
        # Si PAS es normal pero PAD es HT1, se clasifica como HT1.
        # Si PAS es normal pero PAD es HT2, se clasifica como HT2.
        # Si PAS es normal pero PAD es Crisis, se clasifica como Crisis.
        # Considerar si se necesita una categoría "Indeterminada" o revisar la lógica de "O" vs "Y".
        # Por ahora, si no cae en las anteriores, podría ser un caso límite o necesitar revisión.
        if pad >= 80: # Si PAS es normal pero PAD está elevada
             if pad >= 90: return "HT2"
             else: return "HT1" # PAD 80-89
        return "Revisar" 

def conectar_db():
    """Establece conexión con la base de datos MySQL."""
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

def guardar_medicion_mysql(id_paciente, pas, pad, spo2, hr, nivel):
    """Guarda una medición en la base de datos MySQL."""
    conn = conectar_db()
    if conn is None:
        return False
    
    cursor = conn.cursor()
    timestamp_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Asegúrate que la tabla 'mediciones' exista y tenga estas columnas.
    query = """
    INSERT INTO mediciones (id_paciente, pas, pad, spo2, hr, nivel, timestamp) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(query, (id_paciente, pas, pad, spo2, hr, nivel, timestamp_actual))
        conn.commit()
        print(f"Datos guardados en MySQL: IDP={id_paciente}, PAS={pas}, PAD={pad}, SpO2={spo2}, HR={hr}, Nivel={nivel}")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Error al guardar en MySQL: {err}")
        conn.rollback()
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("Conexión a MySQL cerrada.")

def get_google_drive_service():
    """Autentica y construye el servicio de Google Drive."""
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
    """Sube o actualiza un archivo en Google Drive."""
    if not FOLDER_ID:
        print("Error: FOLDER_ID de Google Drive no configurado. No se puede subir archivo.")
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
        
        # Buscar si el archivo ya existe para actualizarlo
        query = f"name='{file_name_on_drive}' and '{FOLDER_ID}' in parents and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        existing_files = response.get('files', [])

        if existing_files: # Si el archivo existe, actualizarlo
            file_id = existing_files[0].get('id')
            updated_file = service.files().update(fileId=file_id, media_body=media).execute()
            print(f"Archivo '{file_name_on_drive}' actualizado en Drive. ID: {updated_file.get('id')}")
        else: # Si no existe, crearlo
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"Archivo '{file_name_on_drive}' subido a Drive. ID: {file.get('id')}")
        return True
    except Exception as e:
        print(f"❌ Error al subir/actualizar archivo '{file_name_on_drive}' en Google Drive: {e}")
        return False

@app.route("/")
def home():
    """Ruta principal que muestra la interfaz web (index.html)."""
    global ultima_estimacion, autorizado
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    """Endpoint para recibir datos del ESP32, estimar PA y SpO2, y almacenar."""
    global autorizado, ultima_estimacion, modelo_sys, modelo_dia
    
    if not modelo_sys or not modelo_dia:
        print("Error crítico: Modelos de ML no están cargados.")
        return jsonify({"error": "Modelos de ML no cargados en el servidor"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request debe ser JSON"}), 400

        hr_in = data.get("hr")
        ir_in = data.get("ir")
        red_in = data.get("red")
        id_paciente_in = data.get("id_paciente", 1) # ID paciente por defecto 1 si no se envía

        if hr_in is None or ir_in is None or red_in is None:
            return jsonify({"error": "Datos incompletos: 'hr', 'ir', 'red' son requeridos"}), 400
        
        # Convertir a float, manejar posibles errores
        try:
            hr = float(hr_in)
            ir = float(ir_in)
            red = float(red_in)
        except ValueError:
            return jsonify({"error": "'hr', 'ir', 'red' deben ser numéricos"}), 400


        # Cálculo de SpO2 (usando la función simplificada)
        spo2_estimada = calcular_spo2_simple(ir, red)
        
        pas_estimada = -1.0
        pad_estimada = -1.0

        # Solo predecir PA si SpO2 es válida y los modelos están cargados
        if spo2_estimada != -1 and modelo_sys and modelo_dia:
            # Crear DataFrame para la predicción, asegurando que los nombres de columnas
            # coincidan con los usados durante el entrenamiento del modelo.
            # Si tus modelos fueron entrenados con 'hr' y 'spo2':
            entrada_df = pd.DataFrame([[hr, spo2_estimada]], columns=['hr', 'spo2'])
            pas_estimada = round(modelo_sys.predict(entrada_df)[0], 1)
            pad_estimada = round(modelo_dia.predict(entrada_df)[0], 1)
            
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)
        
        # Actualizar la última estimación global
        ultima_estimacion["sys"] = str(pas_estimada) if pas_estimada != -1 else "---"
        ultima_estimacion["dia"] = str(pad_estimada) if pad_estimada != -1 else "---"
        ultima_estimacion["spo2"] = str(spo2_estimada) if spo2_estimada != -1 else "---"
        ultima_estimacion["hr"] = str(hr)
        ultima_estimacion["nivel"] = nivel_presion
        ultima_estimacion["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Lógica de guardado condicional
        if autorizado:
            # Guardar en CSV local para entrenamiento
            # Es importante decidir qué se guarda: ¿solo datos crudos o también estimaciones?
            # Si se guardan estimaciones, ¿son las del modelo actual o se espera una referencia?
            # Para reentrenamiento, idealmente se guardan datos crudos + PA de referencia.
            # Aquí guardaremos lo que tenemos del sensor y las estimaciones actuales.
            try:
                with open(CSV_FILENAME, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Formato: hr, spo2_estimada, ir, red, pas_estimada, pad_estimada (o PAS_ref, PAD_ref si se envían)
                    writer.writerow([hr, spo2_estimada, ir, red, pas_estimada, pad_estimada])
                print(f"Datos guardados en {CSV_FILENAME} para entrenamiento.")
                
                # Subir a Google Drive si está configurado
                if FOLDER_ID and FOLDER_ID != 'tu_id_de_carpeta_en_google_drive':
                     subir_archivo_a_drive(CSV_FILENAME, CSV_FILENAME)
            except Exception as e_csv:
                print(f"Error al guardar o subir CSV de entrenamiento: {e_csv}")


        # Guardar en MySQL si la lectura del sensor es válida (dedo presente) y hay estimación
        # Umbral simple para IR. Ajustar según sea necesario.
        if ir > 10000 and red > 10000 and pas_estimada != -1: 
            guardar_medicion_mysql(id_paciente_in, pas_estimada, pad_estimada, spo2_estimada, hr, nivel_presion)

        return jsonify({
            "sys": pas_estimada, 
            "dia": pad_estimada, 
            "spo2": spo2_estimada,
            "nivel": nivel_presion
        }), 200

    except Exception as e:
        print(f"❌ Error general en /api/presion: {e}")
        return jsonify({"error": "Error interno del servidor", "detalle": str(e)}), 500

@app.route("/api/autorizacion", methods=["GET", "POST"])
def api_control_autorizacion():
    """Endpoint para controlar el modo de autorización (registro de datos)."""
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
                # Actualizar también la última estimación para reflejar el cambio en index.html si es necesario
                ultima_estimacion["modo_autorizado"] = autorizado 
                return jsonify({"mensaje": f"Autorización cambiada a {autorizado}", "autorizado": autorizado}), 200
            else:
                return jsonify({"error": "Valor de 'autorizado' debe ser booleano (true/false)"}), 400
        except Exception as e:
            print(f"❌ Error en POST /api/autorizacion: {e}")
            return jsonify({"error": "Error interno del servidor", "detalle": str(e)}), 400
    else: # GET
        return jsonify({"autorizado": autorizado}), 200

@app.route("/api/ultima_estimacion", methods=["GET"])
def get_ultima_medicion():
    """Endpoint para obtener la última estimación (usado por index.html)."""
    global ultima_estimacion
    return jsonify(ultima_estimacion)

if __name__ == "__main__":
    # Esta sección es para desarrollo local.
    # Render (o Gunicorn) usará el Procfile: web: gunicorn app:app
    # Asegurarse que las variables de entorno para DB y Drive estén configuradas localmente si se prueba así.
    print("Iniciando servidor Flask para desarrollo local...")
    if not all([DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database']]):
        print("ADVERTENCIA: Faltan variables de entorno para la base de datos. La conexión a DB podría fallar.")
    if not FOLDER_ID or FOLDER_ID == 'tu_id_de_carpeta_en_google_drive':
        print("ADVERTENCIA: FOLDER_ID de Google Drive no configurado. La subida a Drive no funcionará.")
    if not os.path.exists(KEY_FILE_LOCATION):
         print(f"ADVERTENCIA: Archivo de credenciales '{KEY_FILE_LOCATION}' no encontrado. La subida a Drive no funcionará.")


    port = int(os.environ.get("PORT", 5001)) # Usar puerto 5001 para local si 5000 está ocupado
    app.run(host='0.0.0.0', port=port, debug=True)
