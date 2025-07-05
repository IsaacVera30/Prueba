# modules/database_manager.py
# Gestor especializado para operaciones de base de datos

import mysql.connector
from mysql.connector import pooling
import threading
import queue
import time
import logging
import os
from datetime import datetime
from contextlib import contextmanager

class DatabaseManager:
    """Gestor especializado para operaciones de base de datos con pool de conexiones"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuración de BD
        self.db_config = {
            'host': os.environ.get("MYSQLHOST"),
            'user': os.environ.get("MYSQLUSER"),
            'password': os.environ.get("MYSQLPASSWORD"),
            'database': os.environ.get("MYSQLDATABASE"),
            'port': int(os.environ.get("MYSQLPORT", 22614)),
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci',
            'autocommit': True
        }
        
        # Pool de conexiones
        self.connection_pool = None
        self.pool_size = 5
        
        # Cola para operaciones asíncronas
        self.operation_queue = queue.Queue()
        self.worker_thread = None
        self.should_stop = False
        
        # Métricas
        self.operation_count = 0
        self.error_count = 0
        self.last_operation_time = 0
        
        # Lock para thread safety
        self.operation_lock = threading.Lock()
        
        # Inicializar conexiones
        self._initialize_pool()
        self._start_worker_thread()
    
    def _initialize_pool(self):
        """Inicializar pool de conexiones"""
        try:
            # Verificar configuración
            if not all([self.db_config['host'], self.db_config['user'], 
                       self.db_config['password'], self.db_config['database']]):
                self.logger.warning("Configuración de BD incompleta, funcionando sin BD")
                return
            
            # Crear pool de conexiones
            pool_config = {
                **self.db_config,
                'pool_name': 'medical_monitor_pool',
                'pool_size': self.pool_size,
                'pool_reset_session': True,
                'pool_pre_ping': True,
                'connect_timeout': 10,
                'autocommit': True
            }
            
            self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)
            self.logger.info(f"Pool de conexiones BD inicializado ({self.pool_size} conexiones)")
            
            # Verificar conexión
            self._test_connection()
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error inicializando pool BD: {e}")
            self.connection_pool = None
        except Exception as e:
            self.logger.error(f"Error inesperado inicializando BD: {e}")
            self.connection_pool = None
    
    def _test_connection(self):
        """Probar conexión a la base de datos"""
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result[0] == 1:
                    self.logger.info("Conexión BD verificada exitosamente")
                cursor.close()
        except Exception as e:
            self.logger.error(f"Error probando conexión BD: {e}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager para obtener conexión del pool"""
        connection = None
        try:
            if self.connection_pool:
                connection = self.connection_pool.get_connection()
                yield connection
            else:
                raise Exception("Pool de conexiones no disponible")
        except mysql.connector.Error as e:
            self.logger.error(f"Error obteniendo conexión BD: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def _start_worker_thread(self):
        """Iniciar hilo trabajador para operaciones asíncronas"""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Hilo trabajador BD iniciado")
    
    def _worker_loop(self):
        """Loop principal del hilo trabajador"""
        while not self.should_stop:
            try:
                # Obtener operación de la cola (timeout para poder verificar should_stop)
                operation = self.operation_queue.get(timeout=1)
                
                # Ejecutar operación
                self._execute_operation(operation)
                
                # Marcar tarea como completada
                self.operation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker BD: {e}")
                self.error_count += 1
    
    def _execute_operation(self, operation):
        """Ejecutar una operación de BD"""
        operation_type = operation.get('type')
        data = operation.get('data')
        
        try:
            with self.operation_lock:
                if operation_type == 'save_measurement':
                    self._save_measurement_sync(data)
                elif operation_type == 'save_training_data':
                    self._save_training_data_sync(data)
                elif operation_type == 'cleanup_old_data':
                    self._cleanup_old_data_sync(data)
                else:
                    self.logger.warning(f"Tipo de operación desconocido: {operation_type}")
                
                self.operation_count += 1
                self.last_operation_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Error ejecutando operación {operation_type}: {e}")
            self.error_count += 1
    
    def save_measurement_async(self, measurement_data):
        """Guardar medición de forma asíncrona"""
        if not self.is_connected():
            self.logger.warning("BD no disponible, medición no guardada")
            return
        
        operation = {
            'type': 'save_measurement',
            'data': measurement_data
        }
        
        self.operation_queue.put(operation)
        self.logger.debug(f"Medición añadida a cola BD (queue size: {self.operation_queue.qsize()})")
    
    def _save_measurement_sync(self, data):
        """Guardar medición sincrónicamente - ACTUALIZADO para tu estructura"""
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Query actualizada para tu estructura de tabla
                query = """
                INSERT INTO mediciones (id_paciente, sys, dia, nivel, hr_ml, spo2_ml, timestamp_medicion)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                
                # Valores mapeados a tu estructura
                values = (
                    data.get('id_paciente', 1),           # id_paciente
                    data.get('sys_ml', 0),                # sys (valor ML calibrado)
                    data.get('dia_ml', 0),                # dia (valor ML calibrado)
                    data.get('estado', 'Sin datos'),      # nivel (clasificación médica)
                    data.get('hr_ml', 0),                 # hr_ml (frecuencia cardíaca ML)
                    data.get('spo2_ml', 0),               # spo2_ml (saturación ML)
                    datetime.now()                        # timestamp_medicion
                )
                
                cursor.execute(query, values)
                cursor.close()
                
                self.logger.debug(f"Medición guardada: Paciente {data.get('id_paciente')}, SYS: {data.get('sys_ml')}, DIA: {data.get('dia_ml')}")
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error guardando medición: {e}")
            raise
    
    def get_latest_measurements(self, limit=20, patient_id=None):
        """Obtener últimas mediciones - ACTUALIZADO para tu estructura"""
        if not self.is_connected():
            return []
        
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                # Query actualizada para tu estructura
                if patient_id:
                    query = """
                    SELECT id, id_paciente, sys, dia, nivel, hr_ml, spo2_ml, timestamp_medicion
                    FROM mediciones 
                    WHERE id_paciente = %s
                    ORDER BY id DESC 
                    LIMIT %s
                    """
                    cursor.execute(query, (patient_id, limit))
                else:
                    query = """
                    SELECT id, id_paciente, sys, dia, nivel, hr_ml, spo2_ml, timestamp_medicion
                    FROM mediciones 
                    ORDER BY id DESC 
                    LIMIT %s
                    """
                    cursor.execute(query, (limit,))
                
                records = cursor.fetchall()
                cursor.close()
                
                # Convertir a string para JSON y manejar tipos
                for record in records:
                    for key, value in record.items():
                        if value is not None:
                            if key in ['sys', 'dia', 'hr_ml', 'spo2_ml']:
                                # Mantener números como float/int
                                record[key] = float(value) if '.' in str(value) else int(value)
                            else:
                                record[key] = str(value)
                
                self.logger.debug(f"Obtenidas {len(records)} mediciones")
                return records
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error obteniendo mediciones: {e}")
            return []
    
    def get_patient_statistics(self, patient_id, days=7):
        """Obtener estadísticas de un paciente"""
        if not self.is_connected():
            return {}
        
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                query = """
                SELECT 
                    COUNT(*) as total_mediciones,
                    AVG(sys) as sys_promedio,
                    AVG(dia) as dia_promedio,
                    AVG(hr_ml) as hr_promedio,
                    AVG(spo2_ml) as spo2_promedio,
                    MIN(sys) as sys_min,
                    MAX(sys) as sys_max,
                    MIN(dia) as dia_min,
                    MAX(dia) as dia_max,
                    nivel
                FROM mediciones 
                WHERE id_paciente = %s 
                AND timestamp_medicion >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY nivel
                ORDER BY total_mediciones DESC
                """
                
                cursor.execute(query, (patient_id, days))
                stats = cursor.fetchall()
                cursor.close()
                
                # Convertir a formato más útil
                result = {
                    'patient_id': patient_id,
                    'period_days': days,
                    'levels_stats': stats,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.logger.debug(f"Estadísticas generadas para paciente {patient_id}")
                return result
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def save_training_data_async(self, training_data):
        """Guardar datos de entrenamiento de forma asíncrona"""
        if not self.is_connected():
            self.logger.warning("BD no disponible, datos entrenamiento no guardados")
            return
        
        operation = {
            'type': 'save_training_data',
            'data': training_data
        }
        
        self.operation_queue.put(operation)
        self.logger.debug("Datos entrenamiento añadidos a cola BD")
    
    def _save_training_data_sync(self, data):
        """Guardar datos de entrenamiento sincrónicamente"""
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Crear tabla de entrenamiento si no existe
                create_table_query = """
                CREATE TABLE IF NOT EXISTS datos_entrenamiento (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    hr_promedio_sensor DECIMAL(8,2),
                    spo2_promedio_sensor DECIMAL(5,2),
                    ir_mean_filtrado DECIMAL(10,2),
                    red_mean_filtrado DECIMAL(10,2),
                    ir_std_filtrado DECIMAL(10,2),
                    red_std_filtrado DECIMAL(10,2),
                    sys_ref DECIMAL(5,2),
                    dia_ref DECIMAL(5,2),
                    hr_ref DECIMAL(5,2),
                    timestamp_captura TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                cursor.execute(create_table_query)
                
                # Insertar datos
                insert_query = """
                INSERT INTO datos_entrenamiento 
                (hr_promedio_sensor, spo2_promedio_sensor, ir_mean_filtrado, red_mean_filtrado,
                 ir_std_filtrado, red_std_filtrado, sys_ref, dia_ref, hr_ref, timestamp_captura)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                values = (
                    data.get('hr_promedio_sensor'),
                    data.get('spo2_promedio_sensor'),
                    data.get('ir_mean_filtrado'),
                    data.get('red_mean_filtrado'),
                    data.get('ir_std_filtrado'),
                    data.get('red_std_filtrado'),
                    data.get('sys_ref'),
                    data.get('dia_ref'),
                    data.get('hr_ref'),
                    data.get('timestamp_captura')
                )
                
                cursor.execute(insert_query, values)
                cursor.close()
                
                self.logger.debug("Datos entrenamiento guardados")
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error guardando datos entrenamiento: {e}")
            raise
    
    def cleanup_old_data_async(self, days_to_keep=30):
        """Limpiar datos antiguos de forma asíncrona"""
        operation = {
            'type': 'cleanup_old_data',
            'data': {'days_to_keep': days_to_keep}
        }
        
        self.operation_queue.put(operation)
        self.logger.info(f"Limpieza datos antiguos programada ({days_to_keep} días)")
    
    def _cleanup_old_data_sync(self, data):
        """Limpiar datos antiguos sincrónicamente"""
        try:
            days_to_keep = data.get('days_to_keep', 30)
            
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Limpiar mediciones antiguas
                query = """
                DELETE FROM mediciones 
                WHERE timestamp_medicion < DATE_SUB(NOW(), INTERVAL %s DAY)
                """
                cursor.execute(query, (days_to_keep,))
                deleted_measurements = cursor.rowcount
                
                # Limpiar datos de entrenamiento antiguos (mantener más tiempo)
                query = """
                DELETE FROM datos_entrenamiento 
                WHERE timestamp_captura < DATE_SUB(NOW(), INTERVAL %s DAY)
                """
                cursor.execute(query, (days_to_keep * 3,))  # Mantener 3x más tiempo
                deleted_training = cursor.rowcount
                
                cursor.close()
                
                self.logger.info(f"Limpieza completada: {deleted_measurements} mediciones, {deleted_training} datos entrenamiento")
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error en limpieza: {e}")
            raise
    
    def get_system_health(self):
        """Obtener información de salud del sistema BD"""
        if not self.is_connected():
            return {"status": "disconnected"}
        
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                # Información del pool
                pool_info = {
                    "pool_size": self.connection_pool.pool_size if self.connection_pool else 0,
                    "connections_in_use": self.pool_size - len(self.connection_pool._cnx_queue._queue) if self.connection_pool else 0
                }
                
                # Estadísticas de operaciones
                stats = {
                    "operation_count": self.operation_count,
                    "error_count": self.error_count,
                    "queue_size": self.operation_queue.qsize(),
                    "last_operation": datetime.fromtimestamp(self.last_operation_time).isoformat() if self.last_operation_time else None
                }
                
                # Información de tablas
                cursor.execute("SELECT COUNT(*) as total FROM mediciones")
                measurements_count = cursor.fetchone()['total']
                
                # Verificar si existe tabla datos_entrenamiento
                cursor.execute("SHOW TABLES LIKE 'datos_entrenamiento'")
                training_table_exists = cursor.fetchone() is not None
                
                if training_table_exists:
                    cursor.execute("SELECT COUNT(*) as total FROM datos_entrenamiento")
                    training_count = cursor.fetchone()['total']
                else:
                    training_count = 0
                
                cursor.close()
                
                return {
                    "status": "connected",
                    "pool_info": pool_info,
                    "operation_stats": stats,
                    "data_counts": {
                        "measurements": measurements_count,
                        "training_data": training_count
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error obteniendo salud BD: {e}")
            return {"status": "error", "error": str(e)}
    
    def is_connected(self):
        """Verificar si la BD está conectada"""
        return self.connection_pool is not None
    
    def execute_custom_query(self, query, params=None, fetch_results=True):
        """Ejecutar query personalizado de forma segura"""
        if not self.is_connected():
            return None
        
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch_results:
                    results = cursor.fetchall()
                    cursor.close()
                    return results
                else:
                    affected_rows = cursor.rowcount
                    cursor.close()
                    return affected_rows
                    
        except mysql.connector.Error as e:
            self.logger.error(f"Error ejecutando query personalizado: {e}")
            return None
    
    def create_tables_if_not_exist(self):
        """Crear tablas necesarias si no existen - ACTUALIZADO"""
        if not self.is_connected():
            return False
        
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Tabla mediciones - ESTRUCTURA ACTUALIZADA para coincidir con tu tabla
                mediciones_table = """
                CREATE TABLE IF NOT EXISTS mediciones (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    id_paciente INT NOT NULL,
                    sys DECIMAL(5,2) NOT NULL,
                    dia DECIMAL(5,2) NOT NULL,
                    nivel VARCHAR(50) NOT NULL,
                    hr_ml DECIMAL(5,2) NOT NULL,
                    spo2_ml DECIMAL(5,2) NOT NULL,
                    timestamp_medicion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_paciente (id_paciente),
                    INDEX idx_timestamp (timestamp_medicion),
                    INDEX idx_nivel (nivel)
                )
                """
                
                # Tabla datos_entrenamiento
                entrenamiento_table = """
                CREATE TABLE IF NOT EXISTS datos_entrenamiento (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    hr_promedio_sensor DECIMAL(8,2),
                    spo2_promedio_sensor DECIMAL(5,2),
                    ir_mean_filtrado DECIMAL(10,2),
                    red_mean_filtrado DECIMAL(10,2),
                    ir_std_filtrado DECIMAL(10,2),
                    red_std_filtrado DECIMAL(10,2),
                    sys_ref DECIMAL(5,2),
                    dia_ref DECIMAL(5,2),
                    hr_ref DECIMAL(5,2),
                    timestamp_captura TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                
                cursor.execute(mediciones_table)
                cursor.execute(entrenamiento_table)
                cursor.close()
                
                self.logger.info("Tablas verificadas/creadas exitosamente")
                return True
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error creando tablas: {e}")
            return False
    
    def test_insert_sample_data(self):
        """Insertar datos de prueba para verificar funcionamiento"""
        if not self.is_connected():
            return False
        
        try:
            sample_data = {
                'id_paciente': 999,
                'sys_ml': 120.5,
                'dia_ml': 80.2,
                'hr_ml': 75.0,
                'spo2_ml': 98.0,
                'estado': 'Normal'
            }
            
            self.save_measurement_async(sample_data)
            self.logger.info("Datos de prueba insertados")
            return True
            
        except Exception as e:
            self.logger.error(f"Error insertando datos de prueba: {e}")
            return False
    
    def get_connection_info(self):
        """Obtener información de conexión (sin contraseña)"""
        return {
            'host': self.db_config.get('host', 'No configurado'),
            'database': self.db_config.get('database', 'No configurado'),
            'user': self.db_config.get('user', 'No configurado'),
            'port': self.db_config.get('port', 'No configurado'),
            'connected': self.is_connected(),
            'pool_size': self.pool_size
        }
    
    def close_connections(self):
        """Cerrar todas las conexiones de forma segura"""
        self.logger.info("Cerrando conexiones BD...")
        
        # Detener worker thread
        self.should_stop = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        # Esperar a que se vacíe la cola
        if not self.operation_queue.empty():
            self.logger.info("Esperando operaciones pendientes...")
            try:
                self.operation_queue.join()
            except:
                pass
        
        # Cerrar pool
        if self.connection_pool:
            try:
                # Cerrar todas las conexiones del pool
                while True:
                    try:
                        conn = self.connection_pool.get_connection()
                        conn.close()
                    except:
                        break
                self.logger.info("Conexiones BD cerradas")
            except Exception as e:
                self.logger.error(f"Error cerrando conexiones: {e}")
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'should_stop'):
            self.should_stop = True
        if hasattr(self, 'connection_pool') and self.connection_pool:
            try:
                self.close_connections()
            except:
                pass
