# modules/api_nodo_datos.py
# API especializada para recibir datos del ESP32 - CORREGIDA PARA BD

from flask import Blueprint, request, jsonify
import time
import numpy as np
from datetime import datetime
import logging

# Crear blueprint para la API de nodos
api_nodo_bp = Blueprint('api_nodo', __name__)

# Logger específico para este módulo
logger = logging.getLogger(__name__)

class MLSampleBuffer:
    """Buffer exclusivo para predicción ML - 50 muestras"""
    
    def __init__(self):
        self.patient_buffers = {}
        self.target_samples = 50
        self.max_inactive_time = 300
    
    def add_sample(self, patient_id, ir, red, timestamp):
        """Añadir muestra al buffer ML"""
        if patient_id not in self.patient_buffers:
            self.patient_buffers[patient_id] = []
        
        sample = {
            'ir': float(ir),
            'red': float(red),
            'timestamp': timestamp,
            'time_added': time.time()
        }
        
        self.patient_buffers[patient_id].append(sample)
        
        # Mantener solo las últimas 50 muestras
        if len(self.patient_buffers[patient_id]) > self.target_samples:
            self.patient_buffers[patient_id] = self.patient_buffers[patient_id][-self.target_samples:]
        
        return len(self.patient_buffers[patient_id])
    
    def get_sample_count(self, patient_id):
        """Obtener número de muestras ML del paciente"""
        return len(self.patient_buffers.get(patient_id, []))
    
    def has_enough_samples(self, patient_id):
        """Verificar si tiene 50 muestras para predicción"""
        return self.get_sample_count(patient_id) >= self.target_samples
    
    def get_samples_for_ml(self, patient_id):
        """Obtener datos procesados para ML"""
        if not self.has_enough_samples(patient_id):
            return None
        
        samples = self.patient_buffers[patient_id][-self.target_samples:]
        
        ir_values = [s['ir'] for s in samples]
        red_values = [s['red'] for s in samples]
        
        try:
            processed_data = {
                'hr_calculated': self._calculate_heart_rate(ir_values),
                'spo2_calculated': self._calculate_spo2(ir_values, red_values),
                'ir_mean': np.mean(ir_values),
                'red_mean': np.mean(red_values),
                'ir_std': np.std(ir_values),
                'red_std': np.std(red_values)
            }
            return processed_data
        except Exception as e:
            logger.error(f"Error procesando muestras ML: {e}")
            return None
    
    def _calculate_heart_rate(self, ir_values):
        """Calcular frecuencia cardíaca usando análisis de picos en señal IR"""
        try:
            ir_array = np.array(ir_values, dtype=float)
            
            if len(ir_array) < 10:
                return 75.0
            
            # Suavizar la señal
            window_size = min(5, len(ir_array) // 10)
            if window_size >= 3:
                smoothed = np.convolve(ir_array, np.ones(window_size)/window_size, mode='valid')
            else:
                smoothed = ir_array
            
            # Encontrar picos
            peaks = []
            if len(smoothed) > 2:
                threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)
                
                for i in range(1, len(smoothed) - 1):
                    if (smoothed[i] > smoothed[i-1] and 
                        smoothed[i] > smoothed[i+1] and 
                        smoothed[i] > threshold):
                        peaks.append(i)
            
            # Calcular HR desde picos
            if len(peaks) >= 2:
                intervals = np.diff(peaks)
                if len(intervals) > 0:
                    avg_interval = np.mean(intervals)
                    sample_rate = 0.5
                    time_per_sample = 1.0 / sample_rate
                    time_between_beats = avg_interval * time_per_sample
                    hr = 60.0 / time_between_beats
                    
                    if 40 <= hr <= 150:
                        return float(hr)
            
            return self._estimate_hr_from_variability(ir_values)
            
        except Exception as e:
            logger.warning(f"Error calculando HR: {e}")
            return 75.0
    
    def _estimate_hr_from_variability(self, ir_values):
        """Método alternativo para estimar HR"""
        try:
            diff = np.diff(ir_values)
            variability = np.std(diff)
            
            if variability > 1500:
                return 85.0
            elif variability > 800:
                return 75.0
            elif variability > 400:
                return 68.0
            else:
                return 62.0
        except:
            return 75.0
    
    def _calculate_spo2(self, ir_values, red_values):
        """Calcular SpO2 usando ratio R/IR estándar"""
        try:
            ir_array = np.array(ir_values, dtype=float)
            red_array = np.array(red_values, dtype=float)
            
            if len(ir_array) < 10 or len(red_array) < 10:
                return 98.0
            
            # Calcular componentes AC y DC
            ir_ac = np.std(ir_array)
            ir_dc = np.mean(ir_array)
            red_ac = np.std(red_array)
            red_dc = np.mean(red_array)
            
            if ir_dc <= 0 or red_dc <= 0 or ir_ac <= 0:
                return 98.0
            
            # Calcular ratio R
            r = (red_ac / red_dc) / (ir_ac / ir_dc)
            
            # Ecuación de calibración
            spo2 = 104 - 17 * r
            
            if 85 <= spo2 <= 100:
                return float(spo2)
            else:
                return 98.0
                
        except Exception as e:
            logger.warning(f"Error calculando SpO2: {e}")
            return 98.0
    
    def clear_patient_buffer(self, patient_id):
        """Limpiar buffer ML del paciente"""
        if patient_id in self.patient_buffers:
            del self.patient_buffers[patient_id]
            logger.info(f"Buffer ML limpiado para paciente {patient_id}")


class TrainingSampleBuffer:
    """Buffer exclusivo para entrenamiento - Independiente del ML"""
    
    def __init__(self):
        self.patient_samples = {}
        self.is_active = False
        self.current_patient = None
        self.sample_count = 0
    
    def start_collection(self, patient_id=1):
        """Iniciar recolección de entrenamiento"""
        self.is_active = True
        self.current_patient = patient_id
        self.patient_samples[patient_id] = []
        self.sample_count = 0
        logger.info(f"Entrenamiento iniciado para paciente {patient_id}")
    
    def add_training_sample(self, patient_id, ir, red, timestamp):
        """Añadir muestra SOLO para entrenamiento"""
        if not self.is_active or patient_id != self.current_patient:
            return 0
        
        sample = {
            'ir': float(ir),
            'red': float(red),
            'timestamp': timestamp,
            'captured_at': time.time()
        }
        
        self.patient_samples[patient_id].append(sample)
        self.sample_count = len(self.patient_samples[patient_id])
        
        logger.debug(f"Muestra entrenamiento añadida: {self.sample_count} para paciente {patient_id}")
        return self.sample_count
    
    def stop_collection(self):
        """Detener recolección y preparar para guardar"""
        if not self.is_active:
            return 0
        
        self.is_active = False
        samples_collected = self.sample_count
        logger.info(f"Entrenamiento detenido. {samples_collected} muestras recolectadas")
        return samples_collected
    
    def get_samples_for_saving(self, patient_id):
        """Obtener muestras procesadas para guardar"""
        if patient_id not in self.patient_samples or not self.patient_samples[patient_id]:
            return None
        
        samples = self.patient_samples[patient_id]
        
        ir_values = [s['ir'] for s in samples]
        red_values = [s['red'] for s in samples]
        
        try:
            processed = {
                'hr_promedio_sensor': self._calculate_average_hr(ir_values),
                'spo2_promedio_sensor': self._calculate_average_spo2(ir_values, red_values),
                'ir_mean_filtrado': np.mean(ir_values),
                'red_mean_filtrado': np.mean(red_values),
                'ir_std_filtrado': np.std(ir_values),
                'red_std_filtrado': np.std(red_values),
                'total_samples': len(samples)
            }
            return processed
        except Exception as e:
            logger.error(f"Error procesando muestras entrenamiento: {e}")
            return None
    
    def clear_current_session(self):
        """Limpiar sesión actual después de guardar"""
        if self.current_patient and self.current_patient in self.patient_samples:
            del self.patient_samples[self.current_patient]
        
        self.is_active = False
        self.current_patient = None
        self.sample_count = 0
        logger.info("Sesión de entrenamiento limpiada")
    
    def get_status(self):
        """Estado del buffer de entrenamiento"""
        return {
            'active': self.is_active,
            'current_patient': self.current_patient,
            'sample_count': self.sample_count,
            'ready_to_save': not self.is_active and self.sample_count > 0
        }
    
    def _calculate_average_hr(self, ir_values):
        """Calcular HR promedio para entrenamiento"""
        if len(ir_values) < 10:
            return 75.0
        
        variability = np.std(ir_values)
        if variability > 1500:
            return 85.0
        elif variability > 800:
            return 75.0
        else:
            return 68.0
    
    def _calculate_average_spo2(self, ir_values, red_values):
        """Calcular SpO2 promedio para entrenamiento"""
        try:
            ir_mean = np.mean(ir_values)
            red_mean = np.mean(red_values)
            
            if ir_mean > 0 and red_mean > 0:
                ratio = red_mean / ir_mean
                spo2 = 110 - 25 * ratio
                return max(85, min(100, spo2))
            else:
                return 98.0
        except:
            return 98.0


# Crear instancias separadas
ml_sample_buffer = MLSampleBuffer()
training_buffer = TrainingSampleBuffer()

def classify_pressure_level(sys_pressure, dia_pressure):
    """Clasificar nivel de presión arterial según guías médicas"""
    try:
        sys_val, dia_val = float(sys_pressure), float(dia_pressure)
    except (ValueError, TypeError):
        return "Error"
    
    if sys_val > 180 or dia_val > 120:
        return "HT Crisis"
    
    if sys_val >= 140 or dia_val >= 90:
        return "HT2"
    
    if (130 <= sys_val <= 139) or (80 <= dia_val <= 89):
        return "HT1"
    
    if 120 <= sys_val <= 129 and dia_val < 80:
        return "Elevada"
    
    if sys_val < 120 and dia_val < 80:
        return "Normal"
    
    return "Revisar"

@api_nodo_bp.route('/raw_data', methods=['POST'])
def receive_raw_data():
    """
    Endpoint principal para recibir datos del ESP32 - CORREGIDO PARA BD
    """
    try:
        data = request.get_json()
        if not data:
            logger.warning("Datos JSON vacíos recibidos")
            return jsonify({"error": "No JSON data"}), 400

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
                "nivel": "HT Crisis", "muestras_recolectadas": 50,
                "calidad": "complete"
            }
            if hasattr(api_nodo_bp, 'websocket_handler'):
                api_nodo_bp.websocket_handler.emit_update(data)
            return jsonify(response)

        # VERIFICAR SI ESTAMOS EN MODO ENTRENAMIENTO
        training_status = training_buffer.get_status()
        
        if training_status['active'] and training_status['current_patient'] == patient_id:
            # MODO ENTRENAMIENTO - Buffer separado
            training_count = training_buffer.add_training_sample(patient_id, ir_value, red_value, timestamp)
            
            logger.info(f"ENTRENAMIENTO: Muestra {training_count} añadida para paciente {patient_id}")
            
            response = {
                "mode": "training",
                "training_samples": training_count,
                "status": "collecting_training_data",
                "patient_id": patient_id,
                "message": "Recolectando datos para entrenamiento"
            }
            
            # Notificar vía WebSocket
            if hasattr(api_nodo_bp, 'websocket_handler'):
                training_update = {
                    "training_active": True,
                    "training_count": training_count,
                    "training_patient": patient_id
                }
                api_nodo_bp.websocket_handler.emit_update(training_update, "training_update")
            
            return jsonify(response)
        
        else:
            # MODO PREDICCIÓN ML - Buffer separado
            ml_count = ml_sample_buffer.add_sample(patient_id, ir_value, red_value, timestamp)
            
            logger.info(f"ML: Muestra {ml_count}/50 añadida para paciente {patient_id}")
            
            response = {
                "sys": 0, 
                "dia": 0, 
                "hr": 0, 
                "spo2": 0,
                "nivel": f"Recolectando datos... {ml_count}/50",
                "muestras_recolectadas": ml_count,
                "muestras_necesarias": 50,
                "calidad": "collecting"
            }

            # PREDICCIÓN ML cuando llegue a 50 muestras
            if ml_count == 50:
                logger.info(f"50 muestras ML alcanzadas para paciente {patient_id}")
                
                ml_data = ml_sample_buffer.get_samples_for_ml(patient_id)
                
                if ml_data and hasattr(api_nodo_bp, 'ml_processor') and api_nodo_bp.ml_processor.is_ready():
                    try:
                        # Realizar predicción ML
                        result = api_nodo_bp.ml_processor.predict_pressure(
                            ml_data['hr_calculated'], 
                            ml_data['spo2_calculated'], 
                            ml_data['ir_mean'], 
                            ml_data['red_mean'], 
                            ml_data['ir_std'], 
                            ml_data['red_std']
                        )
                        
                        if isinstance(result, tuple) and len(result) >= 2:
                            sys_pred = result[0]
                            dia_pred = result[1]
                            hr_final = result[2] if len(result) > 2 else ml_data['hr_calculated']
                            
                            if sys_pred > 0 and dia_pred > 0:
                                response.update({
                                    "sys": round(float(sys_pred), 1),
                                    "dia": round(float(dia_pred), 1),
                                    "hr": round(float(hr_final), 0),
                                    "spo2": round(float(ml_data['spo2_calculated']), 0),
                                    "nivel": classify_pressure_level(sys_pred, dia_pred),
                                    "calidad": "complete"
                                })
                                
                                logger.info(f"Predicción ML exitosa - SYS:{sys_pred} DIA:{dia_pred}")
                                
                                # GUARDAR EN BD CON MANEJO ROBUSTO
                                if hasattr(api_nodo_bp, 'db_manager') and api_nodo_bp.db_manager.is_connected():
                                    try:
                                        measurement_data = {
                                            'id_paciente': patient_id,
                                            'sys': float(sys_pred),
                                            'dia': float(dia_pred),
                                            'hr_ml': float(hr_final),
                                            'spo2_ml': float(ml_data['spo2_calculated']),
                                            'nivel': str(response["nivel"])
                                        }
                                        
                                        # Intentar guardar de forma asíncrona
                                        save_success = api_nodo_bp.db_manager.save_measurement_async(measurement_data)
                                        if save_success:
                                            logger.info(f"Medición guardada en BD exitosamente")
                                        else:
                                            logger.warning(f"No se pudo guardar medición en BD")
                                            
                                    except Exception as db_error:
                                        logger.error(f"Error específico guardando en BD: {db_error}")
                                        # No fallar el endpoint por error de BD
                                else:
                                    logger.warning("BD no disponible - medición no guardada")
                                
                                # Verificar alertas
                                if hasattr(api_nodo_bp, 'alert_system'):
                                    try:
                                        alert_data = {
                                            'patient_id': patient_id,
                                            'nivel': response["nivel"],
                                            'sys': sys_pred,
                                            'dia': dia_pred,
                                            'hr': hr_final,
                                            'spo2': ml_data['spo2_calculated']
                                        }
                                        api_nodo_bp.alert_system.check_and_send_alert(alert_data)
                                    except Exception as alert_error:
                                        logger.error(f"Error procesando alerta: {alert_error}")
                                
                                # Notificar vía WebSocket
                                if hasattr(api_nodo_bp, 'websocket_handler'):
                                    try:
                                        api_nodo_bp.websocket_handler.emit_new_record_saved()
                                    except Exception as ws_error:
                                        logger.error(f"Error WebSocket: {ws_error}")
                            else:
                                logger.error("Predicciones ML inválidas")
                                response["nivel"] = "Error predicción"
                                response["calidad"] = "error"
                        
                        # Limpiar buffer ML después de predicción
                        ml_sample_buffer.clear_patient_buffer(patient_id)
                        logger.info(f"Buffer ML limpiado para paciente {patient_id}")
                        
                    except Exception as e:
                        logger.error(f"Error en predicción ML: {e}")
                        response["nivel"] = "Error ML"
                        response["calidad"] = "error"
                        ml_sample_buffer.clear_patient_buffer(patient_id)
                else:
                    logger.warning("ML processor no disponible")
                    response["nivel"] = "ML no disponible"
                    response["calidad"] = "error"
            
            # Notificar vía WebSocket
            if hasattr(api_nodo_bp, 'websocket_handler'):
                try:
                    update_data = {**data, **response}
                    api_nodo_bp.websocket_handler.emit_update(update_data)
                except Exception as ws_error:
                    logger.error(f"Error WebSocket update: {ws_error}")

            return jsonify(response)

    except Exception as e:
        logger.error(f"Error procesando datos del ESP32: {e}")
        return jsonify({
            "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
            "nivel": "Error servidor", 
            "muestras_recolectadas": 0,
            "calidad": "error"
        }), 500

# Resto de endpoints permanecen igual...

@api_nodo_bp.route('/training/start', methods=['POST'])
def start_training_collection():
    """Iniciar recolección de entrenamiento"""
    try:
        data = request.get_json() or {}
        patient_id = data.get('patient_id', 1)
        
        training_buffer.start_collection(patient_id)
        
        return jsonify({
            "success": True,
            "message": "Entrenamiento iniciado",
            "patient_id": patient_id,
            "status": training_buffer.get_status()
        })
    except Exception as e:
        logger.error(f"Error iniciando entrenamiento: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_nodo_bp.route('/training/stop', methods=['POST'])
def stop_training_collection():
    """Detener recolección de entrenamiento"""
    try:
        samples_collected = training_buffer.stop_collection()
        
        return jsonify({
            "success": True,
            "message": "Entrenamiento detenido",
            "samples_collected": samples_collected,
            "status": training_buffer.get_status()
        })
    except Exception as e:
        logger.error(f"Error deteniendo entrenamiento: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_nodo_bp.route('/training/save', methods=['POST'])
def save_training_data():
    """Guardar datos de entrenamiento con referencias"""
    try:
        ref_data = request.get_json()
        if not ref_data:
            return jsonify({"success": False, "error": "Datos de referencia requeridos"}), 400
        
        status = training_buffer.get_status()
        if status['active']:
            return jsonify({"success": False, "error": "Debe detener la captura primero"}), 400
        
        if status['sample_count'] == 0:
            return jsonify({"success": False, "error": "No hay muestras para guardar"}), 400
        
        # Obtener muestras procesadas
        processed_data = training_buffer.get_samples_for_saving(status['current_patient'])
        if not processed_data:
            return jsonify({"success": False, "error": "Error procesando muestras"}), 400
        
        # Preparar datos finales
        final_data = {
            **processed_data,
            'sys_ref': float(ref_data['sys_ref']),
            'dia_ref': float(ref_data['dia_ref']),
            'hr_ref': float(ref_data['hr_ref']),
            'timestamp_captura': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Guardar usando data collector
        if hasattr(api_nodo_bp, 'data_collector'):
            result = api_nodo_bp.data_collector.save_training_sample(final_data)
            
            if result.get('success'):
                # Limpiar sesión después de guardar exitosamente
                training_buffer.clear_current_session()
                
                return jsonify({
                    "success": True,
                    "message": "Datos de entrenamiento guardados exitosamente",
                    "data_saved": final_data
                })
            else:
                return jsonify({"success": False, "error": result.get('error', 'Error guardando')})
        else:
            return jsonify({"success": False, "error": "Data collector no disponible"}), 500
            
    except Exception as e:
        logger.error(f"Error guardando entrenamiento: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def init_api_module(app):
    """Inicializar el módulo API con la aplicación Flask"""
    logger.info("Inicializando módulo API con buffers separados")
    logger.info("Módulo API inicializado - ML Buffer y Training Buffer creados")

# Inicializar automáticamente cuando se importe
init_api_module(None)
