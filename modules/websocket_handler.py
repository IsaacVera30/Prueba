# modules/websocket_handler.py
# Manejador WebSocket CORREGIDO - Compatible con Buffers Separados

import logging
import threading
import time
from datetime import datetime
from collections import defaultdict
import queue

class WebSocketHandler:
    """Manejador WebSocket mejorado para sistema de buffers separados"""
    
    def __init__(self, socketio_instance=None):
        self.logger = logging.getLogger(__name__)
        self.socketio = socketio_instance
        
        # Estados básicos
        self.connected_clients = set()
        self.messages_sent = 0
        self.messages_failed = 0
        self.last_message_time = 0
        
        # Cola de mensajes
        self.message_queue = queue.Queue()
        self.max_queue_size = 200
        
        # Worker thread
        self.worker_thread = None
        self.should_stop = False
        
        # Rate limiting
        self.rate_limit_enabled = True
        self.max_messages_per_second = 10
        self.last_emit_time = defaultdict(float)
        
        # Métricas específicas para buffers
        self.training_messages_sent = 0
        self.ml_messages_sent = 0
        self.alert_messages_sent = 0
        
        self.logger.info("WebSocket Handler inicializado con soporte para buffers separados")
        
        if self.socketio:
            self._start_worker()
    
    def _start_worker(self):
        """Iniciar worker thread"""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Worker thread WebSocket iniciado")
    
    def _worker_loop(self):
        """Loop principal del worker"""
        while not self.should_stop:
            try:
                message_data = self.message_queue.get(timeout=1)
                self._process_message(message_data)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker WebSocket: {e}")
                self.messages_failed += 1
    
    def _process_message(self, message_data):
        """Procesar mensaje desde la cola"""
        try:
            if not self.socketio:
                return False
            
            event_name = message_data.get('event', 'update_data')
            data = message_data.get('data', {})
            room = message_data.get('room', None)
            
            # Rate limiting
            if self.rate_limit_enabled:
                current_time = time.time()
                if current_time - self.last_emit_time[event_name] < (1.0 / self.max_messages_per_second):
                    return False
                self.last_emit_time[event_name] = current_time
            
            # Limpiar datos
            clean_data = self._clean_data(data)
            
            # Enviar mensaje
            if room:
                self.socketio.emit(event_name, clean_data, room=room)
            else:
                self.socketio.emit(event_name, clean_data)
            
            self.messages_sent += 1
            self.last_message_time = time.time()
            
            # Actualizar métricas específicas
            if 'training' in event_name:
                self.training_messages_sent += 1
            elif 'alert' in event_name:
                self.alert_messages_sent += 1
            else:
                self.ml_messages_sent += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando mensaje WebSocket: {e}")
            self.messages_failed += 1
            return False
    
    def _clean_data(self, data):
        """Limpiar datos para envío WebSocket"""
        if isinstance(data, dict):
            clean = {}
            for k, v in data.items():
                if v is not None:
                    if isinstance(v, (int, float, str, bool, list, dict)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
            return clean
        return data
    
    def emit_update(self, data, event_name='update_data'):
        """Emitir actualización general"""
        return self._queue_message(event_name, data)
    
    def emit_training_update(self, training_data):
        """Emitir actualización específica de entrenamiento"""
        try:
            # Preparar datos específicos de entrenamiento
            clean_training_data = {
                'training_active': training_data.get('training_active', False),
                'training_count': training_data.get('training_count', 0),
                'training_patient': training_data.get('training_patient', 1),
                'training_phase': training_data.get('training_phase', 'idle'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log para debugging
            self.logger.debug(f"Enviando actualización entrenamiento: {clean_training_data}")
            
            return self._queue_message('training_update', clean_training_data)
            
        except Exception as e:
            self.logger.error(f"Error emitiendo actualización entrenamiento: {e}")
            return False
    
    def emit_ml_prediction(self, prediction_data):
        """Emitir predicción ML"""
        try:
            # Preparar datos de predicción ML
            clean_prediction_data = {
                'sys': prediction_data.get('sys', 0),
                'dia': prediction_data.get('dia', 0),
                'hr': prediction_data.get('hr', 0),
                'spo2': prediction_data.get('spo2', 0),
                'nivel': prediction_data.get('nivel', '---'),
                'muestras_recolectadas': prediction_data.get('muestras_recolectadas', 0),
                'calidad': prediction_data.get('calidad', 'collecting'),
                'patient_id': prediction_data.get('patient_id', 1),
                'timestamp': datetime.now().isoformat()
            }
            
            return self._queue_message('ml_prediction', clean_prediction_data)
            
        except Exception as e:
            self.logger.error(f"Error emitiendo predicción ML: {e}")
            return False
    
    def emit_sensor_data(self, sensor_data):
        """Emitir datos del sensor"""
        try:
            clean_sensor_data = {
                'ir': sensor_data.get('ir', 0),
                'red': sensor_data.get('red', 0),
                'finger_detected': sensor_data.get('finger_detected', False),
                'timestamp': datetime.now().isoformat()
            }
            
            return self._queue_message('sensor_data', clean_sensor_data)
            
        except Exception as e:
            self.logger.error(f"Error emitiendo datos sensor: {e}")
            return False
    
    def emit_capture_count(self, count):
        """Emitir contador de muestras (legacy compatibility)"""
        return self.emit_update({'count': count}, 'capture_count_update')
    
    def emit_new_record_saved(self):
        """Notificar registro guardado"""
        return self.emit_update({
            'message': 'Nuevo registro guardado',
            'timestamp': datetime.now().isoformat()
        }, 'new_record_saved')
    
    def emit_alert(self, alert_data):
        """Emitir alerta médica"""
        try:
            alert_message = {
                'type': 'medical_alert',
                'level': alert_data.get('level', 'info'),
                'patient_id': alert_data.get('patient_id'),
                'sys': alert_data.get('sys', 0),
                'dia': alert_data.get('dia', 0),
                'hr': alert_data.get('hr', 0),
                'spo2': alert_data.get('spo2', 0),
                'nivel': alert_data.get('nivel', '---'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Enviar con prioridad alta (sin rate limiting)
            return self._queue_message('medical_alert', alert_message, priority=True)
            
        except Exception as e:
            self.logger.error(f"Error emitiendo alerta: {e}")
            return False
    
    def emit_system_status(self, status_data):
        """Emitir estado del sistema"""
        try:
            status_message = {
                'timestamp': datetime.now().isoformat(),
                'status': status_data,
                'websocket_metrics': self.get_basic_metrics()
            }
            
            return self._queue_message('system_status', status_message)
            
        except Exception as e:
            self.logger.error(f"Error emitiendo estado sistema: {e}")
            return False
    
    def emit_error(self, error_message, error_type='general'):
        """Emitir mensaje de error"""
        try:
            error_data = {
                'type': error_type,
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            }
            
            return self._queue_message('error_message', error_data, priority=True)
            
        except Exception as e:
            self.logger.error(f"Error emitiendo error: {e}")
            return False
    
    def emit_training_completed(self, result_data):
        """Emitir notificación de entrenamiento completado"""
        try:
            completion_data = {
                'type': 'training_completed',
                'success': result_data.get('success', False),
                'message': result_data.get('message', ''),
                'samples_processed': result_data.get('samples_processed', 0),
                'file_saved': result_data.get('file_saved', False),
                'timestamp': datetime.now().isoformat()
            }
            
            return self._queue_message('training_completed', completion_data, priority=True)
            
        except Exception as e:
            self.logger.error(f"Error emitiendo finalización entrenamiento: {e}")
            return False
    
    def emit_buffer_status(self, buffer_status):
        """Emitir estado de los buffers"""
        try:
            buffer_data = {
                'ml_buffer': buffer_status.get('ml_buffer', {}),
                'training_buffer': buffer_status.get('training_buffer', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            return self._queue_message('buffer_status', buffer_data)
            
        except Exception as e:
            self.logger.error(f"Error emitiendo estado buffers: {e}")
            return False
    
    def _queue_message(self, event_name, data, room=None, priority=False):
        """Añadir mensaje a la cola de envío"""
        try:
            if self.message_queue.qsize() >= self.max_queue_size:
                if not priority:
                    self.logger.warning("Cola WebSocket llena, descartando mensaje")
                    return False
                else:
                    # Para mensajes prioritarios, remover mensaje más antiguo
                    try:
                        self.message_queue.get_nowait()
                    except queue.Empty:
                        pass
            
            message = {
                'event': event_name,
                'data': data,
                'room': room,
                'priority': priority,
                'queued_at': time.time()
            }
            
            self.message_queue.put(message)
            return True
            
        except Exception as e:
            self.logger.error(f"Error añadiendo mensaje a cola: {e}")
            return False
    
    def handle_client_connect(self, client_id, client_info=None):
        """Manejar conexión de cliente"""
        try:
            self.connected_clients.add(client_id)
            self.logger.info(f"Cliente WebSocket conectado: {client_id}")
            
            # Enviar estado inicial al cliente recién conectado
            initial_status = {
                'connected': True,
                'server_time': datetime.now().isoformat(),
                'websocket_version': '2.0'
            }
            
            self._queue_message('connection_status', initial_status, room=client_id)
            
        except Exception as e:
            self.logger.error(f"Error manejando conexión cliente: {e}")
    
    def handle_client_disconnect(self, client_id):
        """Manejar desconexión de cliente"""
        try:
            self.connected_clients.discard(client_id)
            self.logger.info(f"Cliente WebSocket desconectado: {client_id}")
            
        except Exception as e:
            self.logger.error(f"Error manejando desconexión: {e}")
    
    def get_connected_clients_count(self):
        """Obtener número de clientes conectados"""
        try:
            return len(self.connected_clients)
        except:
            return 0
    
    def get_basic_metrics(self):
        """Obtener métricas básicas"""
        return {
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed,
            'connected_clients': len(self.connected_clients),
            'queue_size': self.message_queue.qsize()
        }
    
    def get_status(self):
        """Obtener estado completo del WebSocket handler"""
        try:
            return {
                'socketio_configured': self.socketio is not None,
                'connected_clients': len(self.connected_clients),
                'messages_sent': self.messages_sent,
                'messages_failed': self.messages_failed,
                'training_messages': self.training_messages_sent,
                'ml_messages': self.ml_messages_sent,
                'alert_messages': self.alert_messages_sent,
                'queue_size': self.message_queue.qsize(),
                'max_queue_size': self.max_queue_size,
                'rate_limit_enabled': self.rate_limit_enabled,
                'worker_active': self.worker_thread.is_alive() if self.worker_thread else False,
                'last_message': datetime.fromtimestamp(self.last_message_time).isoformat() if self.last_message_time else None
            }
        except Exception as e:
            self.logger.error(f"Error obteniendo estado WebSocket: {e}")
            return {
                'socketio_configured': False,
                'connected_clients': 0,
                'messages_sent': 0,
                'messages_failed': 0,
                'error': str(e)
            }
    
    def get_performance_metrics(self):
        """Obtener métricas detalladas de rendimiento"""
        try:
            total_messages = self.messages_sent + self.messages_failed
            success_rate = (self.messages_sent / total_messages * 100) if total_messages > 0 else 0
            
            return {
                'total_messages': total_messages,
                'success_rate': round(success_rate, 2),
                'messages_per_type': {
                    'training': self.training_messages_sent,
                    'ml_predictions': self.ml_messages_sent,
                    'alerts': self.alert_messages_sent,
                    'other': self.messages_sent - (self.training_messages_sent + self.ml_messages_sent + self.alert_messages_sent)
                },
                'queue_utilization': (self.message_queue.qsize() / self.max_queue_size * 100) if self.max_queue_size > 0 else 0,
                'connected_clients': len(self.connected_clients),
                'worker_status': 'active' if (self.worker_thread and self.worker_thread.is_alive()) else 'inactive'
            }
        except Exception as e:
            self.logger.error(f"Error obteniendo métricas WebSocket: {e}")
            return {'error': str(e)}
    
    def broadcast_system_message(self, message, message_type='info'):
        """Enviar mensaje del sistema a todos los clientes"""
        try:
            system_message = {
                'type': 'system_message',
                'level': message_type,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            
            return self._queue_message('system_message', system_message, priority=(message_type == 'error'))
            
        except Exception as e:
            self.logger.error(f"Error enviando mensaje sistema: {e}")
            return False
    
    def register_event_handlers(self):
        """Registrar manejadores de eventos SocketIO"""
        if not self.socketio:
            self.logger.warning("SocketIO no configurado, no se pueden registrar handlers")
            return
        
        try:
            @self.socketio.on('connect')
            def handle_connect():
                client_id = request.sid if 'request' in globals() else 'unknown'
                self.handle_client_connect(client_id)
            
            @self.socketio.on('disconnect')
            def handle_disconnect():
                client_id = request.sid if 'request' in globals() else 'unknown'
                self.handle_client_disconnect(client_id)
            
            @self.socketio.on('request_status')
            def handle_status_request():
                status = self.get_status()
                self._queue_message('status_response', status)
            
            @self.socketio.on('ping')
            def handle_ping():
                pong_data = {
                    'timestamp': datetime.now().isoformat(),
                    'server_time': time.time()
                }
                self._queue_message('pong', pong_data)
            
            self.logger.info("Event handlers WebSocket registrados")
            
        except Exception as e:
            self.logger.error(f"Error registrando event handlers: {e}")
    
    def configure_rate_limiting(self, enabled=True, max_messages_per_second=10):
        """Configurar rate limiting"""
        self.rate_limit_enabled = enabled
        self.max_messages_per_second = max_messages_per_second
        self.logger.info(f"Rate limiting configurado: {enabled}, {max_messages_per_second} msg/s")
    
    def clear_message_queue(self):
        """Limpiar cola de mensajes"""
        try:
            cleared = 0
            while not self.message_queue.empty():
                self.message_queue.get_nowait()
                cleared += 1
            
            self.logger.info(f"Cola WebSocket limpiada: {cleared} mensajes eliminados")
            return cleared
            
        except Exception as e:
            self.logger.error(f"Error limpiando cola: {e}")
            return 0
    
    def send_heartbeat(self):
        """Enviar heartbeat a clientes conectados"""
        heartbeat_data = {
            'status': 'alive',
            'timestamp': datetime.now().isoformat(),
            'connected_clients': len(self.connected_clients)
        }
        
        return self._queue_message('heartbeat', heartbeat_data)
    
    def set_socketio(self, socketio_instance):
        """Configurar instancia SocketIO"""
        self.socketio = socketio_instance
        if not self.worker_thread or not self.worker_thread.is_alive():
            self._start_worker()
        self.logger.info("SocketIO configurado")
    
    def shutdown(self):
        """Apagar WebSocket handler de forma segura"""
        try:
            self.logger.info("Cerrando WebSocket Handler...")
            
            # Detener worker
            self.should_stop = True
            
            # Enviar mensaje de despedida
            try:
                self.broadcast_system_message("Servidor desconectándose", "warning")
                time.sleep(0.5)  # Dar tiempo para enviar mensaje
            except:
                pass
            
            # Limpiar cola
            self.clear_message_queue()
            
            # Limpiar clientes
            self.connected_clients.clear()
            
            self.logger.info("WebSocket Handler cerrado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error cerrando WebSocket Handler: {e}")
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        try:
            if hasattr(self, 'should_stop'):
                self.should_stop = True
        except:
            pass
