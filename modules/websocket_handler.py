# modules/websocket_handler.py
# Manejador especializado para comunicaciones WebSocket

import logging
import threading
import time
import json
from datetime import datetime
from collections import defaultdict
import queue

class WebSocketHandler:
    """Manejador especializado para comunicaciones WebSocket en tiempo real"""
    
    def __init__(self, socketio_instance=None):
        self.logger = logging.getLogger(__name__)
        self.socketio = socketio_instance
        
        # Estado de conexiones
        self.connected_clients = set()
        self.client_info = {}
        
        # Métricas de comunicación
        self.messages_sent = 0
        self.messages_failed = 0
        self.last_message_time = 0
        
        # Buffer para mensajes pendientes
        self.message_queue = queue.Queue()
        self.max_queue_size = 1000
        
        # Control de rate limiting
        self.rate_limits = defaultdict(list)
        self.max_messages_per_minute = 60
        
        # Lock para thread safety
        self.handler_lock = threading.Lock()
        
        # Worker thread para mensajes asíncronos
        self.worker_thread = None
        self.should_stop = False
        
        # Configurar logging
        self.logger.info("WebSocket Handler inicializado")
        
        # Inicializar worker si hay socketio disponible
        if self.socketio:
            self._start_worker_thread()
    
    def set_socketio(self, socketio_instance):
        """Configurar instancia de SocketIO"""
        self.socketio = socketio_instance
        if not self.worker_thread:
            self._start_worker_thread()
        self.logger.info("SocketIO configurado en WebSocket Handler")
    
    def _start_worker_thread(self):
        """Iniciar hilo trabajador para mensajes asíncronos"""
        if self.socketio:
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            self.logger.info("Worker thread WebSocket iniciado")
    
    def _worker_loop(self):
        """Loop principal del worker thread"""
        while not self.should_stop:
            try:
                # Obtener mensaje de la cola
                message_data = self.message_queue.get(timeout=1)
                
                # Procesar mensaje
                self._process_queued_message(message_data)
                
                # Marcar como completado
                self.message_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker WebSocket: {e}")
    
    def _process_queued_message(self, message_data):
        """Procesar mensaje de la cola"""
        try:
            event_name = message_data.get('event')
            data = message_data.get('data')
            room = message_data.get('room')
            
            if self.socketio:
                if room:
                    self.socketio.emit(event_name, data, room=room)
                else:
                    self.socketio.emit(event_name, data)
                
                self.messages_sent += 1
                self.last_message_time = time.time()
                self.logger.debug(f"Mensaje WebSocket enviado: {event_name}")
            
        except Exception as e:
            self.logger.error(f"Error enviando mensaje WebSocket: {e}")
            self.messages_failed += 1
    
    def emit_update(self, data, event_name='update_data'):
        """Emitir actualización de datos a todos los clientes"""
        if not self.socketio:
            self.logger.warning("SocketIO no configurado")
            return False
        
        try:
            # Validar y preparar datos
            clean_data = self._prepare_data_for_emit(data)
            
            # Verificar rate limiting
            if self._check_rate_limit('global'):
                # Enviar inmediatamente si la cola está vacía
                if self.message_queue.empty():
                    self.socketio.emit(event_name, clean_data)
                    self.messages_sent += 1
                    self.last_message_time = time.time()
                else:
                    # Añadir a cola si hay backlog
                    self._queue_message(event_name, clean_data)
                
                self.logger.debug(f"Datos actualizados vía WebSocket: {event_name}")
                return True
            else:
                self.logger.warning("Rate limit alcanzado para emisiones")
                return False
                
        except Exception as e:
            self.logger.error(f"Error emitiendo actualización: {e}")
            self.messages_failed += 1
            return False
    
    def emit_capture_count(self, count):
        """Emitir contador de muestras capturadas"""
        return self.emit_update({'count': count}, 'capture_count_update')
    
    def emit_new_record_saved(self):
        """Notificar que se guardó un nuevo registro"""
        return self.emit_update({'timestamp': datetime.now().isoformat()}, 'new_record_saved')
    
    def emit_alert(self, alert_data):
        """Emitir alerta médica"""
        alert_message = {
            'type': 'medical_alert',
            'level': alert_data.get('level', 'info'),
            'patient_id': alert_data.get('patient_id'),
            'message': alert_data.get('message', ''),
            'timestamp': datetime.now().isoformat(),
            'data': alert_data
        }
        return self.emit_update(alert_message, 'medical_alert')
    
    def emit_system_status(self, status_data):
        """Emitir estado del sistema"""
        status_message = {
            'timestamp': datetime.now().isoformat(),
            'status': status_data
        }
        return self.emit_update(status_message, 'system_status')
    
    def emit_error(self, error_message, error_type='general'):
        """Emitir mensaje de error"""
        error_data = {
            'type': error_type,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }
        return self.emit_update(error_data, 'error_message')
    
    def emit_to_room(self, room, event_name, data):
        """Emitir mensaje a una sala específica"""
        if not self.socketio:
            return False
        
        try:
            clean_data = self._prepare_data_for_emit(data)
            self._queue_message(event_name, clean_data, room)
            return True
        except Exception as e:
            self.logger.error(f"Error emitiendo a sala {room}: {e}")
            return False
    
    def _queue_message(self, event_name, data, room=None):
        """Añadir mensaje a la cola"""
        if self.message_queue.qsize() >= self.max_queue_size:
            self.logger.warning("Cola de mensajes WebSocket llena, descartando mensaje más antiguo")
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                pass
        
        message_data = {
            'event': event_name,
            'data': data,
            'room': room,
            'timestamp': time.time()
        }
        
        self.message_queue.put(message_data)
    
    def _prepare_data_for_emit(self, data):
        """Preparar datos para emisión (limpiar y validar)"""
        if isinstance(data, dict):
            # Limpiar valores None y convertir tipos problemáticos
            clean_data = {}
            for key, value in data.items():
                if value is not None:
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        clean_data[key] = value
                    else:
                        clean_data[key] = str(value)
            return clean_data
        elif isinstance(data, (list, str, int, float, bool)):
            return data
        else:
            return str(data)
    
    def _check_rate_limit(self, client_id):
        """Verificar rate limiting para evitar spam"""
        current_time = time.time()
        
        with self.handler_lock:
            # Limpiar entradas antiguas
            cutoff_time = current_time - 60  # 1 minuto
            self.rate_limits[client_id] = [
                t for t in self.rate_limits[client_id] 
                if t > cutoff_time
            ]
            
            # Verificar límite
            if len(self.rate_limits[client_id]) >= self.max_messages_per_minute:
                return False
            
            # Registrar nuevo mensaje
            self.rate_limits[client_id].append(current_time)
            return True
    
    def handle_client_connect(self, client_id, client_info=None):
        """Manejar conexión de cliente"""
        with self.handler_lock:
            self.connected_clients.add(client_id)
            if client_info:
                self.client_info[client_id] = {
                    **client_info,
                    'connected_at': datetime.now().isoformat(),
                    'messages_sent': 0
                }
        
        self.logger.info(f"Cliente WebSocket conectado: {client_id}")
        
        # Enviar estado inicial al cliente
        self.emit_system_status({
            'message': 'Conectado al sistema de monitoreo',
            'client_id': client_id,
            'server_time': datetime.now().isoformat()
        })
    
    def handle_client_disconnect(self, client_id):
        """Manejar desconexión de cliente"""
        with self.handler_lock:
            self.connected_clients.discard(client_id)
            if client_id in self.client_info:
                connection_duration = time.time() - time.mktime(
                    datetime.fromisoformat(self.client_info[client_id]['connected_at']).timetuple()
                )
                self.logger.info(f"Cliente {client_id} desconectado después de {connection_duration:.1f}s")
                del self.client_info[client_id]
            
            # Limpiar rate limits
            if client_id in self.rate_limits:
                del self.rate_limits[client_id]
        
        self.logger.info(f"Cliente WebSocket desconectado: {client_id}")
    
    def get_connected_clients_count(self):
        """Obtener número de clientes conectados"""
        return len(self.connected_clients)
    
    def get_status(self):
        """Obtener estado del manejador WebSocket"""
        return {
            'socketio_configured': self.socketio is not None,
            'connected_clients': len(self.connected_clients),
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed,
            'success_rate': self.messages_sent / max(self.messages_sent + self.messages_failed, 1),
            'queue_size': self.message_queue.qsize(),
            'last_message': datetime.fromtimestamp(self.last_message_time).isoformat() if self.last_message_time else None,
            'worker_active': self.worker_thread is not None and self.worker_thread.is_alive()
        }
    
    def get_performance_metrics(self):
        """Obtener métricas detalladas de rendimiento"""
        current_time = time.time()
        
        # Calcular mensajes por minuto
        recent_messages = []
        for timestamps in self.rate_limits.values():
            recent_messages.extend([t for t in timestamps if current_time - t < 3600])  # última hora
        
        messages_last_hour = len(recent_messages)
        
        return {
            'total_messages_sent': self.messages_sent,
            'total_messages_failed': self.messages_failed,
            'messages_last_hour': messages_last_hour,
            'avg_messages_per_minute': messages_last_hour / 60,
            'queue_utilization': self.message_queue.qsize() / self.max_queue_size,
            'active_rate_limits': len(self.rate_limits),
            'client_stats': {
                'total_connected': len(self.connected_clients),
                'client_details': [
                    {
                        'id': client_id,
                        'connected_duration_minutes': (current_time - time.mktime(
                            datetime.fromisoformat(info['connected_at']).timetuple()
                        )) / 60,
                        'messages_sent': info.get('messages_sent', 0)
                    }
                    for client_id, info in self.client_info.items()
                ]
            }
        }
    
    def broadcast_system_message(self, message, message_type='info'):
        """Enviar mensaje del sistema a todos los clientes"""
        system_message = {
            'type': 'system_message',
            'level': message_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        return self.emit_update(system_message, 'system_message')
    
    def send_heartbeat(self):
        """Enviar heartbeat a clientes conectados"""
        heartbeat_data = {
            'timestamp': datetime.now().isoformat(),
            'server_status': 'online',
            'connected_clients': len(self.connected_clients)
        }
        return self.emit_update(heartbeat_data, 'heartbeat')
    
    def clear_message_queue(self):
        """Limpiar cola de mensajes"""
        cleared_count = self.message_queue.qsize()
        
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info(f"Cola de mensajes WebSocket limpiada: {cleared_count} mensajes")
        return cleared_count
    
    def update_client_info(self, client_id, info_update):
        """Actualizar información de un cliente"""
        if client_id in self.client_info:
            self.client_info[client_id].update(info_update)
            return True
        return False
    
    def get_client_info(self, client_id):
        """Obtener información de un cliente específico"""
        return self.client_info.get(client_id, {})
    
    def configure_rate_limiting(self, max_messages_per_minute=None, max_queue_size=None):
        """Configurar parámetros de rate limiting"""
        if max_messages_per_minute is not None:
            self.max_messages_per_minute = max_messages_per_minute
            self.logger.info(f"Rate limit actualizado: {max_messages_per_minute} msg/min")
        
        if max_queue_size is not None:
            self.max_queue_size = max_queue_size
            self.logger.info(f"Tamaño máximo de cola actualizado: {max_queue_size}")
    
    def emit_data_batch(self, data_list, event_name='batch_update', batch_size=10):
        """Emitir datos en lotes para evitar saturar la conexión"""
        if not data_list:
            return True
        
        try:
            total_batches = len(data_list) // batch_size + (1 if len(data_list) % batch_size else 0)
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                batch_data = {
                    'batch_number': i // batch_size + 1,
                    'total_batches': total_batches,
                    'data': batch,
                    'timestamp': datetime.now().isoformat()
                }
                
                if not self.emit_update(batch_data, event_name):
                    return False
                
                # Pequeña pausa entre lotes
                time.sleep(0.1)
            
            self.logger.info(f"Datos enviados en {total_batches} lotes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando datos en lotes: {e}")
            return False
    
    def register_event_handlers(self):
        """Registrar manejadores de eventos WebSocket"""
        if not self.socketio:
            self.logger.warning("No se puede registrar handlers sin SocketIO")
            return
        
        @self.socketio.on('connect')
        def handle_connect():
            client_id = request.sid if 'request' in globals() else 'unknown'
            self.handle_client_connect(client_id)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid if 'request' in globals() else 'unknown'
            self.handle_client_disconnect(client_id)
        
        @self.socketio.on('client_info')
        def handle_client_info(data):
            client_id = request.sid if 'request' in globals() else 'unknown'
            self.update_client_info(client_id, data)
        
        @self.socketio.on('request_status')
        def handle_status_request():
            status = self.get_status()
            self.emit_system_status(status)
        
        self.logger.info("Event handlers WebSocket registrados")
    
    def shutdown(self):
        """Apagar manejador WebSocket de forma segura"""
        self.logger.info("Iniciando apagado del WebSocket Handler...")
        
        # Detener worker thread
        self.should_stop = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        # Enviar mensaje de despedida a clientes
        if self.socketio and self.connected_clients:
            self.broadcast_system_message("Servidor cerrando conexión", "warning")
            time.sleep(1)  # Dar tiempo para enviar el mensaje
        
        # Limpiar datos
        with self.handler_lock:
            self.connected_clients.clear()
            self.client_info.clear()
            self.rate_limits.clear()
        
        # Limpiar cola
        cleared = self.clear_message_queue()
        
        self.logger.info(f"WebSocket Handler cerrado. {cleared} mensajes pendientes procesados")
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'should_stop'):
            self.should_stop = True
