from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import os
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- Variables Globales ---
autorizado_general = False
capturando_entrenamiento = False
buffer_datos_entrenamiento = []
NUM_MUESTRAS_ENTRENAMIENTO = 15

# Ruta del archivo CSV
csv_file = 'datos_entrenamiento.csv'

# --- Endpoint para recibir datos desde el ESP32 ---
@app.route('/api/presion', methods=['POST'])
def recibir_datos():
    global buffer_datos_entrenamiento

    data = request.get_json()
    hr_prom = data.get('hr')
    ir = data.get('ir')
    red = data.get('red')
    spo2 = data.get('spo2')
    timestamp = datetime.utcnow().isoformat()

    if capturando_entrenamiento:
        buffer_datos_entrenamiento.append({
            'hr_promedio_sensor': hr_prom,
            'spo2_promedio_sensor': spo2,
            'ir_mean_filtrado': ir,
            'red_mean_filtrado': red,
            'timestamp_captura': timestamp
        })
        print(f"[Captura ML] {len(buffer_datos_entrenamiento)}/15 muestras almacenadas...")

    # Simulamos respuesta del modelo ML
    respuesta = {
        'sys': 125,
        'dia': 82,
        'hr': hr_prom,
        'spo2': spo2,
        'nivel': 'Normal'
    }
    return jsonify(respuesta)

# --- Endpoint para autorización general ---
@app.route('/api/autorizacion', methods=['GET', 'POST'])
def autorizacion():
    global autorizado_general
    if request.method == 'GET':
        return jsonify({ 'autorizado': autorizado_general })
    else:
        data = request.get_json()
        autorizado_general = data.get('autorizado', False)
        return jsonify({ 'autorizado': autorizado_general, 'mensaje': 'Estado actualizado.' })

# --- Iniciar / Detener captura de entrenamiento ---
@app.route('/api/iniciar_captura_entrenamiento', methods=['POST'])
def iniciar_captura():
    global capturando_entrenamiento, buffer_datos_entrenamiento
    capturando_entrenamiento = True
    buffer_datos_entrenamiento = []
    return jsonify({ 'capturando': True, 'mensaje': 'Captura iniciada.' })

@app.route('/api/detener_captura_entrenamiento', methods=['POST'])
def detener_captura():
    global capturando_entrenamiento
    capturando_entrenamiento = False
    return jsonify({ 'capturando': False, 'mensaje': 'Captura detenida.' })

# --- Guardar muestras en CSV con referencias ---
@app.route('/api/guardar_muestra_entrenamiento', methods=['POST'])
def guardar_muestra():
    global buffer_datos_entrenamiento

    if not buffer_datos_entrenamiento:
        return jsonify({ 'error': 'No hay datos de entrenamiento en buffer.' })

    data = request.get_json()
    pas_ref = data.get('pas_referencia')
    pad_ref = data.get('pad_referencia')
    hr_ref = data.get('hr_referencia')

    if pas_ref is None or pad_ref is None or hr_ref is None:
        return jsonify({ 'error': 'Faltan valores de referencia.' })

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['hr_promedio_sensor','spo2_promedio_sensor','ir_mean_filtrado','red_mean_filtrado','hr_referencia','pas_referencia','pad_referencia','timestamp_captura'])

        for row in buffer_datos_entrenamiento[:NUM_MUESTRAS_ENTRENAMIENTO]:
            writer.writerow([
                row['hr_promedio_sensor'],
                row['spo2_promedio_sensor'],
                row['ir_mean_filtrado'],
                row['red_mean_filtrado'],
                hr_ref,
                pas_ref,
                pad_ref,
                row['timestamp_captura']
            ])

    buffer_datos_entrenamiento = []
    return jsonify({ 'mensaje': f'{NUM_MUESTRAS_ENTRENAMIENTO} muestras guardadas con éxito en CSV.' })

# --- Última estimación simulada para index.html ---
@app.route('/api/ultima_estimacion', methods=['GET'])
def ultima_estimacion():
    return jsonify({
        'sys': 125,
        'dia': 82,
        'spo2': 97,
        'hr': 75,
        'nivel': 'Normal',
        'timestamp': datetime.utcnow().isoformat(),
        'modo_autorizado': autorizado_general
    })

# --- Mediciones simuladas para tabla Railway ---
@app.route('/api/ultimas_mediciones', methods=['GET'])
def mediciones():
    return jsonify([
        { 'id': 1, 'id_paciente': 1, 'sys': 130, 'dia': 85, 'nivel': 'HT1' },
        { 'id': 2, 'id_paciente': 1, 'sys': 120, 'dia': 78, 'nivel': 'Normal' },
        { 'id': 3, 'id_paciente': 1, 'sys': 140, 'dia': 90, 'nivel': 'HT2' }
    ])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
