# modules/api_nodo_datos.py
# API especializada para recibir y procesar datos del ESP32

from flask import Blueprint, request, jsonify
import time
import numpy as np
from datetime import datetime
import logging

# Crear blueprint para la API de nodos
api_nodo_bp = Blueprint('api_nodo', __name__)

# Logger específico para este módulo
logger = logging.getLogger(__name__)

# Buffer para procesamiento de datos en tiempo real
class DataBuffer:
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.ir_values = []
        self.red_values = []
        self.timestamps = []
    
    def add_sample(self, ir, red, timestamp):
        """Añadir nueva muestra al buffer"""
        self.ir_values.append(ir)
        self.red_values.append(red) 
        self.timestamps.append(timestamp)
        
        # Mantener tamaño máximo
        if len(self.ir_values) > self.max_size:
            self.ir_values.pop(0)
            self.red_values.pop(0)
            self.timestamps.pop(0)
    
    def get_filtered_values(self):
        """Obtener valores filtrados (promedio)"""
        if len(self.ir_values) == 0:
            return 0, 0, 0, 0
            
        ir_mean = np.mean(self.ir_values)
        red_mean = np.mean(self.red_values)
        ir_std = np.std(self.ir_values) if len(self.ir_values) > 1 else 0
        red_std = np.std(self.red_values) if len(self.red_values) > 1 else 0
        
        return ir_mean, red_mean, ir_std, red_std
    
    def is_finger_detected(self, ir_threshold=800, ir_max=100000):
        """Verificar si hay dedo detectado"""
        if len(self.ir_values) == 0:
            return False
        
        current_ir = self.ir_values[-1]
        return ir_threshold < current_ir < ir_max
    
    def get_signal_quality(self):
        """Evaluar calidad de la señal"""
        if len(self.ir_values) < 5:
            return "insufficient_data"
        
        ir_mean, red_mean, ir_std, red_std = self.get_filtered_values()
        
        # Criterios de calidad
        if ir_mean < 800:
            return "no_finger"
        elif ir_mean > 100000:
            return "saturated"
        elif ir_std > ir_mean * 0.3:  # Mucha variabilidad
            return "noisy"
        else:
            return "good"

# Buffer global para cada paciente
patient_buffers = {}

def get_patient_buffer(patient_id):
    """Obtener o crear buffer para un paciente"""
    if patient_id not in patient_buffers:
        patient_buffers[patient_id] = DataBuffer()
    return patient_buffers[patient_id]

# ========== ENDPOINTS DE LA API ==========

@api_nodo_bp.route('/raw_data', methods=['POST'])
def receive_raw_data():
    """
    Endpoint principal para recibir datos crudos del ESP32
    Procesa los datos y devuelve análisis ML
    """
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("Datos JSON vacíos recibidos")
            return jsonify({"error": "No JSON data"}), 400

        # Extraer datos básicos
        patient_id = data.get('id_paciente', 1)
        ir_value = float(data.get('ir', 0))
        red_value = float(data.get('red', 0))
        timestamp = data.get('timestamp', int(time.time() * 1000))

        logger.info(f"Datos ESP32 - Paciente:{patient_id} IR:{ir_value} RED:{red_value}")

        # CASO ESPECIAL: ID 999 para prueba de buzzer
        if patient_id == 999:
            logger.info("Modo prueba buzzer activado")
            response = {
                "sys": 185, "dia": 125, "hr": 99, "spo2": 99, 
                "nivel": "HT Crisis", "calidad": "test_mode"
            }
            # Notificar vía WebSocket
            if hasattr(api_nodo_bp, 'websocket_handler'):
                api_nodo_bp.websocket_handler.emit_update(data)
            return jsonify(response)

        # Obtener buffer del paciente
        buffer = get_patient_buffer(patient_id)
        
        # Añadir nueva muestra
        buffer.add_sample(ir_value, red_value, timestamp)
        
        # Evaluar calidad de señal
        signal_quality = buffer.get_signal_quality()
        finger_detected = buffer.is_finger_detected()
        
        logger.info(f"Calidad señal: {signal_quality}, Dedo: {finger_detected}")

        # Respuesta por defecto
        response = {
            "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
            "nivel": "Sin datos", "calidad": signal_quality
        }

        # Solo procesar con ML si hay buena señal
        if finger_detected and signal_quality == "good":
            response = process_with_ml(buffer, patient_id, data)
            
            # Guardar en base de datos si está configurado
            if hasattr(api_nodo_bp, 'db_manager') and api_nodo_bp.db_manager.is_connected():
                save_measurement_async(response, patient_id)
            
            # Enviar alertas si es necesario
            if hasattr(api_nodo_bp, 'alert_system'):
                api_nodo_bp.alert_system.check_and_send_alert(response)
        
        else:
            # Mapear estado según calidad de señal
            status_map = {
                "no_finger": "Sin dedo",
                "saturated": "Señal saturada", 
                "noisy": "Señal inestable",
                "insufficient_data": "Procesando..."
            }
            response["nivel"] = status_map.get(signal_quality, "Estado desconocido")

        # Notificar vía WebSocket
        if hasattr(api_nodo_bp, 'websocket_handler'):
            update_data = {**data, **response, "signal_quality": signal_quality}
            api_nodo_bp.websocket_handler.emit_update(update_data)

        logger.info(f"Respuesta: SYS:{response['sys']} DIA:{response['dia']} Nivel:{response['nivel']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error procesando datos: {e}")
        return jsonify({
            "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
            "nivel": "Error servidor", "calidad": "error"
        }), 500

def process_with_ml(buffer, patient_id, original_data):
    """Procesar datos con ML y calcular HR/SpO2"""
    try:
        # Obtener valores filtrados
        ir_mean, red_mean, ir_std, red_std = buffer.get_filtered_values()
        
        # Calcular HR usando análisis de picos
        hr_calculated = calculate_heart_rate(buffer)
        
        # Calcular SpO2 usando ratio R/IR
        spo2_calculated = calculate_spo2(buffer)
        
        logger.info(f"HR calculado: {hr_calculated}, SpO2: {spo2_calculated}")

        # Usar ML para predecir presión arterial si está disponible
        sys_pred, dia_pred = 0, 0
        if hasattr(api_nodo_bp, 'ml_processor') and api_nodo_bp.ml_processor.is_ready():
            sys_pred, dia_pred = api_nodo_bp.ml_processor.predict_pressure(
                hr_calculated, spo2_calculated, ir_mean, red_mean, ir_std, red_std
            )
            logger.info(f"ML: SYS:{sys_pred} DIA:{dia_pred}")

        # Clasificar nivel de presión
        nivel = classify_pressure_level(sys_pred, dia_pred)

        return {
            "sys": round(sys_pred, 1),
            "dia": round(dia_pred, 1), 
            "hr": round(hr_calculated, 0),
            "spo2": round(spo2_calculated, 0),
            "nivel": nivel,
            "calidad": "good"
        }

    except Exception as e:
        logger.error(f"Error en procesamiento ML: {e}")
        return {
            "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
            "nivel": "Error ML", "calidad": "error"
        }

def calculate_heart_rate(buffer):
    """Calcular frecuencia cardíaca usando análisis de picos"""
    if len(buffer.ir_values) < 10:
        return 0
    
    try:
        # Usar valores IR para detectar picos
        ir_values = np.array(buffer.ir_values)
        
        # Encontrar diferencias para detectar picos
        diff = np.diff(ir_values)
        
        # Detectar cambios de negativo a positivo (picos)
        peaks = []
        for i in range(1, len(diff)):
            if diff[i-1] < 0 and diff[i] > 0:
                peaks.append(i)
        
        if len(peaks) < 2:
            return 0
        
        # Calcular intervalos entre picos
        intervals = np.diff(peaks)
        avg_interval = np.mean(intervals)
        
        # Convertir a BPM (asumiendo ~1 Hz de muestreo del ESP32)
        sample_rate = 1.0  # 1 muestra por segundo del ESP32
        hr = 60.0 / (avg_interval * sample_rate)
        
        # Validar rango
        if 40 <= hr <= 200:
            return hr
        else:
            return 0
            
    except Exception as e:
        logger.warning(f"Error calculando HR: {e}")
        return 0

def calculate_spo2(buffer):
    """Calcular SpO2 usando ratio R/IR"""
    if len(buffer.ir_values) < 5:
        return 0
    
    try:
        # Calcular AC y DC para cada canal
        ir_values = np.array(buffer.ir_values)
        red_values = np.array(buffer.red_values)
        
        # Calcular componentes AC (variabilidad) y DC (promedio)
        ir_ac = np.std(ir_values)
        ir_dc = np.mean(ir_values)
        red_ac = np.std(red_values)
        red_dc = np.mean(red_values)
        
        # Evitar división por cero
        if ir_dc == 0 or red_dc == 0 or ir_ac == 0:
            return 0
        
        # Calcular ratio R (fórmula estándar para oximetría)
        r = (red_ac / red_dc) / (ir_ac / ir_dc)
        
        # Ecuación de calibración para MAX30102 (empírica)
        spo2 = 104 - 17 * r
        
        # Limitar a rango válido
        if 70 <= spo2 <= 100:
            return spo2
        else:
            return 0
            
    except Exception as e:
        logger.warning(f"Error calculando SpO2: {e}")
        return 0

def classify_pressure_level(sys_pressure, dia_pressure):
    """Clasificar nivel de presión arterial"""
    if sys_pressure == 0 or dia_pressure == 0:
        return "Sin datos"
    
    if sys_pressure > 180 or dia_pressure > 120:
        return "HT Crisis"
    elif sys_pressure >= 140 or dia_pressure >= 90:
        return "HT2"
    elif sys_pressure >= 130 or dia_pressure >= 80:
        return "HT1"
    elif sys_pressure >= 120 and dia_pressure < 80:
        return "Elevada"
    else:
        return "Normal"

def save_measurement_async(measurement_data, patient_id):
    """Guardar medición en base de datos de forma asíncrona"""
    try:
        if hasattr(api_nodo_bp, 'db_manager'):
            data_to_save = {
                'id_paciente': patient_id,
                'sys_ml': measurement_data['sys'],
                'dia_ml': measurement_data['dia'],
                'hr_ml': measurement_data['hr'],
                'spo2_ml': measurement_data['spo2'],
                'estado': measurement_data['nivel']
            }
            api_nodo_bp.db_manager.save_measurement_async(data_to_save)
            logger.info(f"Medición guardada para paciente {patient_id}")
    except Exception as e:
        logger.error(f"Error guardando medición: {e}")

# ========== ENDPOINTS ADICIONALES ==========

@api_nodo_bp.route('/patient_status/<int:patient_id>', methods=['GET'])
def get_patient_status(patient_id):
    """Obtener estado actual de un paciente específico"""
    try:
        if patient_id in patient_buffers:
            buffer = patient_buffers[patient_id]
            ir_mean, red_mean, ir_std, red_std = buffer.get_filtered_values()
            
            status = {
                "patient_id": patient_id,
                "finger_detected": buffer.is_finger_detected(),
                "signal_quality": buffer.get_signal_quality(),
                "buffer_size": len(buffer.ir_values),
                "last_values": {
                    "ir_mean": round(ir_mean, 2),
                    "red_mean": round(red_mean, 2),
                    "ir_std": round(ir_std, 2),
                    "red_std": round(red_std, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(status)
        else:
            return jsonify({
                "patient_id": patient_id,
                "status": "no_data",
                "message": "No hay datos para este paciente"
            }), 404
            
    except Exception as e:
        logger.error(f"Error obteniendo estado del paciente {patient_id}: {e}")
        return jsonify({"error": "Error interno"}), 500

@api_nodo_bp.route('/reset_buffer/<int:patient_id>', methods=['POST'])
def reset_patient_buffer(patient_id):
    """Resetear buffer de datos de un paciente"""
    try:
        if patient_id in patient_buffers:
            del patient_buffers[patient_id]
            logger.info(f"Buffer reseteado para paciente {patient_id}")
            return jsonify({
                "status": "success", 
                "message": f"Buffer del paciente {patient_id} reseteado"
            })
        else:
            return jsonify({
                "status": "info",
                "message": f"No existía buffer para paciente {patient_id}"
            })
    except Exception as e:
        logger.error(f"Error reseteando buffer: {e}")
        return jsonify({"error": "Error interno"}), 500

@api_nodo_bp.route('/system_metrics', methods=['GET'])
def get_system_metrics():
    """Obtener métricas del sistema de procesamiento de datos"""
    try:
        metrics = {
            "active_patients": len(patient_buffers),
            "total_buffers": sum(len(buf.ir_values) for buf in patient_buffers.values()),
            "patients_with_data": [
                {
                    "id": pid,
                    "buffer_size": len(buf.ir_values),
                    "finger_detected": buf.is_finger_detected(),
                    "signal_quality": buf.get_signal_quality()
                }
                for pid, buf in patient_buffers.items()
            ],
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        return jsonify({"error": "Error interno"}), 500

# ========== MANEJO DE ERRORES ESPECÍFICOS ==========

@api_nodo_bp.errorhandler(ValueError)
def handle_value_error(e):
    logger.error(f"Error de valor en API nodo: {e}")
    return jsonify({"error": "Datos inválidos"}), 400

@api_nodo_bp.errorhandler(KeyError)
def handle_key_error(e):
    logger.error(f"Clave faltante en API nodo: {e}")
    return jsonify({"error": "Datos incompletos"}), 400

# ========== CONFIGURACIÓN DEL MÓDULO ==========

def init_api_module(app):
    """Inicializar el módulo API con la aplicación Flask"""
    logger.info("Inicializando módulo API de nodos")
    
    # Configurar logging específico para este módulo
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[API-NODO] %(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info("Módulo API de nodos inicializado correctamente")

# Inicializar automáticamente cuando se importe
init_api_module(None)
