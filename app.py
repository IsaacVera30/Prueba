import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO
import os
import time
import logging
from datetime import datetime
import threading
import sys
import warnings

warnings.filterwarnings("ignore")
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING) 
logging.getLogger('eventlet').setLevel(logging.WARNING)

from modules.ml_processor import MLProcessor
from modules.database_manager import DatabaseManager
from modules.alert_system import AlertSystem
from modules.data_collector import DataCollector
from modules.websocket_handler import WebSocketHandler
from modules.api_nodo_datos import api_nodo_bp

logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MedicalMonitorApp:
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
        
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*", 
            async_mode='eventlet',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=100000,
            always_connect=False,
            transports=['websocket', 'polling'],
            allow_upgrades=True,
            cookie=None
        )
        
        self.ml_processor = MLProcessor()
        self.db_manager = DatabaseManager()
        self.alert_system = AlertSystem()
        self.data_collector = DataCollector()
        self.websocket_handler = WebSocketHandler(self.socketio)
        
        self.system_start_time = time.time()
        self.last_db_save_time = 0
        self.save_interval = 5 
        self._setup_module_connections()
        self._register_routes()
        self._register_socketio_events()
        
        logger.info("Sistema de monitoreo médico inicializado")
    
    def _setup_module_connections(self):
        api_nodo_bp.ml_processor = self.ml_processor
        api_nodo_bp.db_manager = self.db_manager
        api_nodo_bp.alert_system = self.alert_system
        api_nodo_bp.websocket_handler = self.websocket_handler
        api_nodo_bp.data_collector = self.data_collector
        
        self.app.register_blueprint(api_nodo_bp, url_prefix='/api')
        self.websocket_handler.register_event_handlers()
        
        logger.info("Conexiones entre módulos configuradas")
    
    def _register_routes(self):
        
        @self.app.route("/")
        def home():
            return render_template("index.html")
               
        @self.app.route("/api/training/start", methods=["POST"])
        def start_training_session():
            try:
                data = request.get_json() or {}
                patient_id = data.get('patient_id', 1)
                
                from modules.api_nodo_datos import training_buffer
                training_buffer.start_collection(patient_id)
                
                result = {
                    "success": True,
                    "message": "Entrenamiento iniciado",
                    "patient_id": patient_id,
                    "status": training_buffer.get_status()
                }
                
                logger.info("Sesión de entrenamiento iniciada")
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error iniciando entrenamiento: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/training/stop", methods=["POST"])
        def stop_training_session():
            try:
                from modules.api_nodo_datos import training_buffer
                samples_collected = training_buffer.stop_collection()
                
                result = {
                    "success": True,
                    "message": "Entrenamiento detenido",
                    "samples_collected": samples_collected,
                    "status": training_buffer.get_status()
                }
                
                logger.info("Sesión de entrenamiento detenida")
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error deteniendo entrenamiento: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/training/save", methods=["POST"])
        def save_training_data():
            try:
                ref_data = request.get_json()
                if not ref_data:
                    return jsonify({"success": False, "error": "Datos de referencia requeridos"}), 400
                
                from modules.api_nodo_datos import training_buffer
                status = training_buffer.get_status()
                
                if status['active']:
                    return jsonify({"success": False, "error": "Debe detener la captura primero"}), 400
                
                if status['sample_count'] == 0:
                    return jsonify({"success": False, "error": "No hay muestras para guardar"}), 400
                
                processed_data = training_buffer.get_samples_for_saving(status['current_patient'])
                if not processed_data:
                    return jsonify({"success": False, "error": "Error procesando muestras"}), 400
                
                final_data = {
                    **processed_data,
                    'sys_ref': float(ref_data['sys_ref']),
                    'dia_ref': float(ref_data['dia_ref']),
                    'hr_ref': float(ref_data['hr_ref']),
                    'timestamp_captura': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                result = self.data_collector.save_training_sample(final_data)
                
                if result.get('success'):
                    training_buffer.clear_current_session()
                    
                    logger.info("Datos de entrenamiento guardados exitosamente")
                    return jsonify({
                        "success": True,
                        "message": "Datos de entrenamiento guardados exitosamente",
                        "data_saved": final_data
                    })
                else:
                    logger.error(f"Error guardando entrenamiento: {result.get('error')}")
                    return jsonify({"success": False, "error": result.get('error', 'Error guardando')})
                
            except Exception as e:
                logger.error(f"Error guardando datos entrenamiento: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/training/status", methods=["GET"])
        def get_training_status():
            try:
                from modules.api_nodo_datos import training_buffer
                status = training_buffer.get_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error obteniendo estado entrenamiento: {e}")
                return jsonify({"active": False, "error": str(e)}), 500

        @self.app.route("/api/data", methods=["POST"])
        def recibir_datos():
            return self._handle_legacy_esp32_data()
        
        @self.app.route("/api/start_capture", methods=["POST"])
        def start_capture():
            try:
                from modules.api_nodo_datos import training_buffer
                training_buffer.start_collection(1)
                
                result = {
                    "success": True,
                    "status": "success",
                    "message": "Captura iniciada"
                }
                
                logger.info("Captura de entrenamiento iniciada (legacy)")
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error iniciando captura: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/stop_capture", methods=["POST"])
        def stop_capture():
            try:
                from modules.api_nodo_datos import training_buffer
                samples_collected = training_buffer.stop_collection()
                
                result = {
                    "success": True,
                    "status": "success",
                    "muestras_en_buffer": samples_collected,
                    "session_summary": {
                        "total_samples": samples_collected
                    }
                }
                
                logger.info(f"Captura detenida. {samples_collected} muestras")
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error deteniendo captura: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/save_training_data", methods=["POST"])
        def save_training_data_legacy():
            try:
                ref_data = request.get_json()
                if not ref_data:
                    return jsonify({"error": "Datos de referencia requeridos"}), 400
                
                from modules.api_nodo_datos import training_buffer
                status = training_buffer.get_status()
                
                if status['sample_count'] == 0:
                    return jsonify({"error": "No hay datos en el buffer"}), 400
                
                processed_data = training_buffer.get_samples_for_saving(status['current_patient'])
                if not processed_data:
                    return jsonify({"error": "Error procesando muestras"}), 400
                
                final_data = {
                    **processed_data,
                    'sys_ref': float(ref_data['sys_ref']),
                    'dia_ref': float(ref_data['dia_ref']),
                    'hr_ref': float(ref_data['hr_ref']),
                    'timestamp_captura': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                result = self.data_collector.save_training_sample(final_data)
                
                if result.get('success'):
                    training_buffer.clear_current_session()
                    
                    return jsonify({
                        "status": "success",
                        "message": "Datos de entrenamiento guardados",
                        "samples_processed": status['sample_count']
                    })
                else:
                    return jsonify({"error": result.get('error', 'Error guardando')})
            
            except Exception as e:
                logger.error(f"Error guardando datos entrenamiento: {e}")
                return jsonify({"error": str(e)}), 500
                
        @self.app.route("/api/ultimas_mediciones")
        def get_ultimas_mediciones():
            try:
                records = self.db_manager.get_latest_measurements(limit=20)
                return jsonify(records)
            except Exception as e:
                logger.error(f"Error obteniendo mediciones: {e}")
                return jsonify([])
        
        @self.app.route("/api/mediciones_recientes")
        def get_mediciones_recientes():
            try:
                limit = request.args.get('limit', 20, type=int)
                records = self.db_manager.get_latest_measurements(limit=limit)
                
                mediciones = []
                for record in records:
                    mediciones.append({
                        'id': record.get('id'),
                        'patient_id': record.get('id_paciente'),
                        'sys': float(record.get('sys', 0)),
                        'dia': float(record.get('dia', 0)),
                        'hr': float(record.get('hr_ml', 0)),
                        'spo2': float(record.get('spo2_ml', 0)),
                        'nivel': record.get('nivel', '---'),
                        'timestamp': str(record.get('timestamp_medicion', '')),
                        'fecha_formateada': str(record.get('timestamp_medicion', ''))[:19]
                    })
                
                return jsonify({
                    'success': True,
                    'mediciones': mediciones,
                    'count': len(mediciones)
                })
                
            except Exception as e:
                logger.error(f"Error obteniendo mediciones recientes: {e}")
                return jsonify({
                    'success': False, 
                    'error': str(e),
                    'mediciones': []
                })
        
        @self.app.route("/api/test_alert", methods=['POST'])
        def test_alert():
            try:
                data = request.get_json()
                if not data or "sys" not in data or "dia" not in data:
                    return jsonify({"error": "Datos incompletos"}), 400
                
                test_data = {
                    "patient_id": data.get("id_paciente", 99),
                    "sys": float(data["sys"]),
                    "dia": float(data["dia"]),
                    "hr": data.get("hr", 0),
                    "spo2": data.get("spo2", 0),
                    "nivel": self._classify_pressure_level(float(data["sys"]), float(data["dia"]))
                }
                
                self.db_manager.save_measurement_async(test_data)
                self.alert_system.check_and_send_alert(test_data)
                self.websocket_handler.emit_new_record_saved()
                
                return jsonify({
                    "status": "success",
                    "message": "Alerta de prueba procesada",
                    "data": test_data
                })
            
            except Exception as e:
                logger.error(f"Error en test de alerta: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/system_status")
        def get_system_status():
            try:
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "uptime_hours": (time.time() - self.system_start_time) / 3600,
                    "modules": {
                        "ml_processor": self.ml_processor.get_status(),
                        "database": self.db_manager.get_system_health(),
                        "alerts": self.alert_system.get_status(),
                        "websocket": self.websocket_handler.get_status(),
                        "data_collector": self.data_collector.get_status()
                    }
                }
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error obteniendo estado del sistema: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/ml_status")
        def get_ml_status():
            try:
                return jsonify(self.ml_processor.get_status())
            except Exception as e:
                logger.error(f"Error obteniendo estado ML: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/db_status")
        def get_db_status():
            try:
                return jsonify(self.db_manager.get_system_health())
            except Exception as e:
                logger.error(f"Error obteniendo estado BD: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/alert_status")
        def get_alert_status():
            try:
                return jsonify(self.alert_system.get_status())
            except Exception as e:
                logger.error(f"Error obteniendo estado alertas: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/drive_status")
        def check_drive_status():
            try:
                status = self.data_collector.get_status()
                file_info = self.data_collector.get_file_info()
                test_connection = self.data_collector.test_drive_connection()
                
                return jsonify({
                    "drive_status": status,
                    "file_info": file_info,
                    "connection_test": test_connection,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        logger.info("Rutas principales registradas")
    
    def _register_socketio_events(self):
        
        @self.socketio.on('connect')
        def handle_connect():
            pass
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            pass
        
        @self.socketio.on('request_system_status')
        def handle_status_request():
            try:
                from modules.api_nodo_datos import training_buffer
                training_status = training_buffer.get_status()
                
                status = {
                    "ml_ready": self.ml_processor.is_ready(),
                    "db_connected": self.db_manager.is_connected(),
                    "alerts_configured": self.alert_system.is_configured(),
                    "training_active": training_status['active'],
                    "training_samples": training_status['sample_count'],
                    "connected_clients": self.websocket_handler.get_connected_clients_count()
                }
                self.socketio.emit('system_status', status)
            except:
                pass
        
        logger.info("Eventos SocketIO registrados")
    
    def _handle_legacy_esp32_data(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data"}), 400
            
            return self._process_esp32_data_legacy(data)
        
        except Exception as e:
            logger.error(f"Error procesando datos legacy: {e}")
            return jsonify({
                "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
                "nivel": "Error servidor"
            }), 500
    
    def _process_esp32_data_legacy(self, data):
        patient_id = data.get('id_paciente', 1)
        
        if patient_id == 999:
            response = {"sys": 185, "dia": 125, "hr": 99, "spo2": 99, "nivel": "HT Crisis"}
            self.websocket_handler.emit_update(data)
            return jsonify(response)
        
        from modules.api_nodo_datos import training_buffer
        training_status = training_buffer.get_status()
        
        if training_status['active']:
            ir_value = float(data.get("ir", 0))
            red_value = float(data.get("red", 0))
            timestamp = data.get('timestamp', int(time.time() * 1000))
            
            training_count = training_buffer.add_training_sample(patient_id, ir_value, red_value, timestamp)
            
            self.websocket_handler.emit_update({
                "training_count": training_count,
                "training_active": True
            }, "training_update")
            
            return jsonify({
                "status": "capturando", 
                "muestras": training_count,
                "mode": "training"
            })
        
        response = {"sys": 0, "dia": 0, "hr": 0, "spo2": 0, "nivel": "Sin datos"}
        
        ir_value = float(data.get("ir", 0))
        if ir_value > 50000:
            if self.ml_processor.is_ready():
                try:
                    hr = float(data.get("hr_promedio", 0))
                    spo2 = float(data.get("spo2_sensor", 0))
                    ir_mean = float(data.get("ir", 0))
                    red_mean = float(data.get("red", 0))
                    
                    ir_std = ir_mean * 0.02
                    red_std = red_mean * 0.02
                    
                    sys_pred, dia_pred, hr_final = self.ml_processor.predict_pressure(
                        hr, spo2, ir_mean, red_mean, ir_std, red_std
                    )
                    
                    response.update({
                        "sys": round(sys_pred, 2),
                        "dia": round(dia_pred, 2),
                        "hr": round(hr_final, 2),
                        "spo2": round(spo2, 2),
                        "nivel": self._classify_pressure_level(sys_pred, dia_pred)
                    })
                    
                    if (time.time() - self.last_db_save_time) >= self.save_interval:
                        measurement_data = {
                            'id_paciente': patient_id,
                            'sys': sys_pred,
                            'dia': dia_pred,
                            'hr_ml': hr_final,
                            'spo2_ml': spo2,
                            'nivel': response["nivel"]
                        }
                        self.db_manager.save_measurement_async(measurement_data)
                        self.websocket_handler.emit_new_record_saved()
                        self.last_db_save_time = time.time()
                    
                    alert_data = {
                        'patient_id': patient_id,
                        'nivel': response["nivel"],
                        'sys': sys_pred,
                        'dia': dia_pred,
                        'hr': hr_final,
                        'spo2': spo2
                    }
                    self.alert_system.check_and_send_alert(alert_data)
                
                except Exception as e:
                    logger.error(f"Error en predicción ML: {e}")
                    response["nivel"] = "Error ML"
        
        self.websocket_handler.emit_update({**data, **response})
        
        return jsonify(response)
    
    def _classify_pressure_level(self, sys_pressure, dia_pressure):
        if sys_pressure is None or dia_pressure is None:
            return "N/A"
        
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
    
    def run(self, host='0.0.0.0', port=None, debug=False):
        if port is None:
            port = int(os.environ.get("PORT", 10000))
        
        logger.info(f"Iniciando servidor en {host}:{port}")
        logger.info(f"Estado módulos - ML: {self.ml_processor.is_ready()}, "
                   f"BD: {self.db_manager.is_connected()}, "
                   f"Alertas: {self.alert_system.is_configured()}")
        
        if self.db_manager.is_connected():
            self.db_manager.create_tables_if_not_exist()
        
        try:
            import socket
            socket.setdefaulttimeout(30)
            
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                log_output=False
            )
        except Exception as e:
            logger.error(f"Error ejecutando servidor: {e}")
            self.shutdown()
    
    def shutdown(self):
        try:
            logger.info("Iniciando apagado del sistema...")
            
            self.websocket_handler.shutdown()
            self.alert_system.shutdown()
            self.db_manager.close_connections()
            
            logger.info("Sistema apagado correctamente")
        except Exception:
            pass

medical_app = MedicalMonitorApp()
app = medical_app.app
socketio = medical_app.socketio

def create_app():
    return medical_app.app

class QuietStderr:
    def write(self, s):
        if "Bad file descriptor" not in s and "socket shutdown error" not in s:
            sys.__stderr__.write(s)
    
    def flush(self):
        sys.__stderr__.flush()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.stderr = QuietStderr()
    
    try:
        medical_app.run(debug=False)
    except KeyboardInterrupt:
        medical_app.shutdown()
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        medical_app.shutdown()
