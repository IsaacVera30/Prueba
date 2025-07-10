# modules/database_manager.py
# Gestor especializado para operaciones de base de datos - CORREGIDO PARA RENDER

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
    """Gestor especializado para operaciones de base de datos con pool de conexiones mejorado"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuración de BD con manejo robusto para Render
        self.db_config = self._build_db_config()
        
        # Pool de conexiones
        self.connection_pool = None
        self.pool_size = 3  # Reducido para Render
        
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
        
        # Estado de conexión
        self.last_connection_test = 0
        self.connection_test_interval = 30  # Probar cada 30 segundos
        
        # Inicializar conexiones
        self._initialize_pool()
        if self.is_connected():
            self._start_worker_thread()
    
    def _build_db_config(self):
        """Construir configuración de BD robusta para diferentes entornos"""
        host = os.environ.get("MYSQLHOST")
        user = os.environ.get("MYSQLUSER") 
        password = os.environ.get("MYSQLPASSWORD")
        database = os.environ.get("MYSQLDATABASE")
        port = int(os.environ.get("MYSQLPORT", 3306))
        
        if not all([host, user, password, database]):
            self.logger.warning("Configuración de BD incompleta - Variables de entorno faltantes")
            return None
        
        # Configuración base
        config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'port': port,
            'charset': 'utf8mb4',
            'connect_timeout': 30,
            'raise_on_warnings': False,
            'autocommit': True,
            'use_unicode': True
        }
        
        # Configuraciones SSL específicas para diferentes proveedores
        if 'railway' in host.lower() or 'planetscale' in host.lower():
            # Configuración para Railway/PlanetScale
            config.update({
                'ssl_disabled': False,
                'ssl_verify_cert': True,
                'ssl_verify_identity': True
            })
        elif 'aiven' in host.lower() or 'digitalocean' in host.lower():
            # Configuración para Aiven/DigitalOcean
            config.update({
                'ssl_disabled': False,
                'ssl_verify_cert': False,
                'ssl_verify_identity': False
            })
        else:
            # Configuración genérica - intentar sin SSL primero
            config.update({
                'ssl_disabled': True
            })
        
        self.logger.info(f"Configuración BD: {host}:{port}/{database} ({user})")
        return config
    
    def _initialize_pool(self):
        """Inicializar pool de conexiones con múltiples estrategias"""
        if not self.db_config:
            self.logger.warning("No se puede inicializar BD sin configuración")
            return
        
        # Estrategias de conexión ordenadas por probabilidad de éxito
        strategies = [
            # 1. SSL deshabilitado
            {**self.db_config, 'ssl_disabled': True},
            # 2. SSL habilitado sin verificación
            {**self.db_config, 'ssl_disabled': False, 'ssl_verify_cert': False, 'ssl_verify_identity': False},
            # 3. SSL habilitado con verificación
            {**self.db_config, 'ssl_disabled': False, 'ssl_verify_cert': True, 'ssl_verify_identity': True},
            # 4. Configuración mínima
            {k: v for k, v in self.db_config.items() if k in ['host', 'user', 'password', 'database', 'port', 'charset']}
        ]
        
        for i, config in enumerate(strategies):
            try:
                self.logger.info(f"Intentando conexión BD - Estrategia {i+1}")
                
                # Intentar conexión simple primero
                test_conn = mysql.connector.connect(**config)
                test_conn.close()
                self.logger.info(f"Conexión simple exitosa - Estrategia {i+1}")
                
                # Crear pool de conexiones
                pool_config = {
                    **config,
                    'pool_name': 'medical_monitor_pool',
                    'pool_size': self.pool_size,
                    'pool_reset_session': True
                }
                
                self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)
                self.logger.info(f"Pool BD inicializado - Estrategia {i+1} ({self.pool_size} conexiones)")
                
                # Verificar pool
                self._test_pool_connection()
                return  # Éxito, salir del loop
                
            except mysql.connector.Error as e:
                self.logger.warning(f"Estrategia {i+1} falló: {e}")
                self.connection_pool = None
                continue
            except Exception as e:
                self.logger.warning(f"Error inesperado estrategia {i+1}: {e}")
                self.connection_pool = None
                continue
        
        # Si llegamos aquí, todas las estrategias fallaron
        self.logger.error("TODAS las estrategias de conexión BD fallaron")
        self.logger.error("Sistema funcionará sin base de datos")
    
    def _test_pool_connection(self):
        """Probar que el pool de conexiones funcione"""
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                cursor.close()
                if result and result[0] == 1:
                    self.logger.info("Pool de conexiones verificado exitosamente")
                    return True
                else:
                    raise Exception("Query de prueba falló")
        except Exception as e:
            self.logger.error(f"Error probando pool: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Context manager para obtener conexión del pool con manejo robusto"""
        connection = None
        try:
            if not self.connection_pool:
                raise Exception("Pool de conexiones no disponible")
            
            connection = self.connection_pool.get_connection()
            if not connection.is_connected():
                connection.reconnect()
            
            yield connection
            
        except mysql.connector.PoolError as e:
            self.logger.error(f"Error del pool de conexiones: {e}")
            # Intentar reinicializar pool
            self._reinitialize_pool()
            raise
        except mysql.connector.Error as e:
            self.logger.error(f"Error de conexión BD: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error inesperado obteniendo conexión: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                try:
                    connection.close()
                except:
                    pass
    
    def _reinitialize_pool(self):
        """Reinicializar pool de conexiones"""
        try:
            self.logger.info("Reinicializando pool de conexiones...")
            old_pool = self.connection_pool
            self.connection_pool = None
            
            # Cerrar pool anterior si existe
            if old_pool:
                try:
                    # No hay método directo para cerrar pool, solo ignorar
                    pass
                except:
                    pass
            
            # Reinicializar
            self._initialize_pool()
            
        except Exception as e:
            self.logger.error(f"Error reinicializando pool: {e}")
    
    def _start_worker_thread(self):
        """Iniciar hilo trabajador para operaciones asíncronas"""
        if not self.is_connected():
            self.logger.warning("No se puede iniciar worker sin conexión BD")
            return
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Hilo trabajador BD iniciado")
    
    def _worker_loop(self):
        """Loop principal del hilo trabajador con manejo robusto"""
        while not self.should_stop:
            try:
                # Obtener operación de la cola
                operation = self.operation_queue.get(timeout=1)
                
                # Verificar conexión periódicamente
                current_time = time.time()
                if current_time - self.last_connection_test > self.connection_test_interval:
                    self._periodic_connection_test()
                    self.last_connection_test = current_time
                
                # Ejecutar operación
                if self.is_connected():
                    self._execute_operation(operation)
                else:
                    self.logger.warning("Operación BD descartada - Sin conexión")
                    self.error_count += 1
                
                # Marcar tarea como completada
                self.operation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker BD: {e}")
                self.error_count += 1
    
    def _periodic_connection_test(self):
        """Prueba periódica de conexión"""
        try:
            if self.connection_pool:
                with self._get_connection() as connection:
                    cursor = connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    cursor.close()
        except Exception as e:
            self.logger.warning(f"Prueba periódica de conexión falló: {e}")
            # Intentar reinicializar
            self._reinitialize_pool()
    
    def _execute_operation(self, operation):
        """Ejecutar una operación de BD con manejo de errores"""
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
                    return
                
                self.operation_count += 1
                self.last_operation_time = time.time()
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error MySQL ejecutando {operation_type}: {e}")
            self.error_count += 1
            
            # Si es error de conexión, intentar reinicializar
            if e.errno in [2003, 2006, 2013]:  # Connection errors
                self._reinitialize_pool()
                
        except Exception as e:
            self.logger.error(f"Error general ejecutando {operation_type}: {e}")
            self.error_count += 1
    
    def save_measurement_async(self, measurement_data):
        """Guardar medición de forma asíncrona con validación"""
        if not self.is_connected():
            self.logger.warning("BD no disponible, medición no guardada")
            return False
        
        # Validar datos mínimos
        if not self._validate_measurement_data(measurement_data):
            self.logger.error("Datos de medición inválidos")
            return False
        
        operation = {
            'type': 'save_measurement',
            'data': measurement_data
        }
        
        try:
            self.operation_queue.put(operation)
            self.logger.debug(f"Medición añadida a cola BD (queue size: {self.operation_queue.qsize()})")
            return True
        except Exception as e:
            self.logger.error(f"Error añadiendo medición a cola: {e}")
            return False
    
    def _validate_measurement_data(self, data):
        """Validar datos de medición"""
        required_fields = ['id_paciente', 'sys', 'dia', 'nivel']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Campo requerido faltante: {field}")
                return False
        
        # Validar rangos
        try:
            sys_val = float(data['sys'])
            dia_val = float(data['dia'])
            
            if not (50 <= sys_val <= 300):
                self.logger.error(f"SYS fuera de rango: {sys_val}")
                return False
            
            if not (30 <= dia_val <= 200):
                self.logger.error(f"DIA fuera de rango: {dia_val}")
                return False
            
            if dia_val >= sys_val:
                self.logger.error(f"DIA >= SYS: {dia_val} >= {sys_val}")
                return False
            
            return True
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error validando valores numéricos: {e}")
            return False
    
    def _save_measurement_sync(self, data):
        """Guardar medición sincrónicamente"""
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Query para estructura: id, id_paciente, sys, dia, nivel, hr_ml, spo2_ml
                query = """
                INSERT INTO mediciones (id_paciente, sys, dia, nivel, hr_ml, spo2_ml)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                values = (
                    int(data.get('id_paciente', 1)),
                    float(data.get('sys', 0)),
                    float(data.get('dia', 0)),
                    str(data.get('nivel', 'Sin datos')),
                    float(data.get('hr_ml', 0)),
                    float(data.get('spo2_ml', 0))
                )
                
                cursor.execute(query, values)
                cursor.close()
                
                self.logger.debug(f"Medición guardada: Paciente {data.get('id_paciente')}, "
                                f"SYS: {data.get('sys')}, DIA: {data.get('dia')}")
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error MySQL guardando medición: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error general guardando medición: {e}")
            raise
    
    def get_latest_measurements(self, limit=20, patient_id=None):
        """Obtener últimas mediciones con manejo robusto"""
        if not self.is_connected():
            self.logger.warning("BD no disponible para obtener mediciones")
            return []
        
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                if patient_id:
                    query = """
                    SELECT id, id_paciente, sys, dia, nivel, hr_ml, spo2_ml
                    FROM mediciones 
                    WHERE id_paciente = %s
                    ORDER BY id DESC 
                    LIMIT %s
                    """
                    cursor.execute(query, (patient_id, limit))
                else:
                    query = """
                    SELECT id, id_paciente, sys, dia, nivel, hr_ml, spo2_ml
                    FROM mediciones 
                    ORDER BY id DESC 
                    LIMIT %s
                    """
                    cursor.execute(query, (limit,))
                
                records = cursor.fetchall()
                cursor.close()
                
                # Convertir tipos para JSON
                for record in records:
                    for key, value in record.items():
                        if value is not None:
                            if key in ['sys', 'dia', 'hr_ml', 'spo2_ml']:
                                record[key] = float(value) if '.' in str(value) else int(value)
                            else:
                                record[key] = str(value)
                
                self.logger.debug(f"Obtenidas {len(records)} mediciones")
                return records
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error MySQL obteniendo mediciones: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error general obteniendo mediciones: {e}")
            return []
    
    def is_connected(self):
        """Verificar si la BD está conectada y funcionando"""
        try:
            if not self.connection_pool:
                return False
            
            # Prueba rápida de conexión
            with self._get_connection() as connection:
                if connection.is_connected():
                    return True
                return False
                
        except Exception:
            return False
    
    def get_system_health(self):
        """Obtener información de salud del sistema BD"""
        try:
            if not self.is_connected():
                return {
                    "status": "disconnected",
                    "error": "Pool no disponible",
                    "operation_stats": {
                        "operation_count": self.operation_count,
                        "error_count": self.error_count,
                        "queue_size": self.operation_queue.qsize()
                    }
                }
            
            with self._get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                # Información del pool
                pool_info = {
                    "pool_size": self.pool_size,
                    "pool_available": self.connection_pool is not None
                }
                
                # Estadísticas de operaciones
                stats = {
                    "operation_count": self.operation_count,
                    "error_count": self.error_count,
                    "queue_size": self.operation_queue.qsize(),
                    "last_operation": datetime.fromtimestamp(self.last_operation_time).isoformat() if self.last_operation_time else None,
                    "success_rate": (self.operation_count / max(self.operation_count + self.error_count, 1)) * 100
                }
                
                # Información de tablas
                try:
                    cursor.execute("SELECT COUNT(*) as total FROM mediciones")
                    measurements_count = cursor.fetchone()['total']
                except:
                    measurements_count = 0
                
                cursor.close()
                
                return {
                    "status": "connected",
                    "pool_info": pool_info,
                    "operation_stats": stats,
                    "data_counts": {
                        "measurements": measurements_count
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "operation_stats": {
                    "operation_count": self.operation_count,
                    "error_count": self.error_count,
                    "queue_size": self.operation_queue.qsize()
                }
            }
    
    def get_connection_info(self):
        """Obtener información de conexión sin datos sensibles"""
        if not self.db_config:
            return {"error": "Configuración no disponible"}
        
        return {
            'host': self.db_config.get('host', 'No configurado'),
            'database': self.db_config.get('database', 'No configurado'),
            'user': self.db_config.get('user', 'No configurado'),
            'port': self.db_config.get('port', 'No configurado'),
            'connected': self.is_connected(),
            'pool_size': self.pool_size,
            'ssl_disabled': self.db_config.get('ssl_disabled', 'No especificado')
        }
    
    def create_tables_if_not_exist(self):
        """Crear tablas necesarias si no existen"""
        if not self.is_connected():
            self.logger.warning("No se pueden crear tablas sin conexión BD")
            return False
        
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Tabla mediciones con estructura exacta
                mediciones_table = """
                CREATE TABLE IF NOT EXISTS mediciones (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    id_paciente INT NOT NULL DEFAULT 1,
                    sys DECIMAL(5,2) NOT NULL DEFAULT 0,
                    dia DECIMAL(5,2) NOT NULL DEFAULT 0,
                    nivel VARCHAR(50) NOT NULL DEFAULT 'Normal',
                    hr_ml DECIMAL(5,2) DEFAULT 0,
                    spo2_ml DECIMAL(5,2) DEFAULT 0,
                    timestamp_medicion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_paciente (id_paciente),
                    INDEX idx_nivel (nivel),
                    INDEX idx_timestamp (timestamp_medicion)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
                
                cursor.execute(mediciones_table)
                cursor.close()
                
                self.logger.info("Tablas verificadas/creadas exitosamente")
                return True
                
        except mysql.connector.Error as e:
            self.logger.error(f"Error MySQL creando tablas: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error general creando tablas: {e}")
            return False
    
    def close_connections(self):
        """Cerrar todas las conexiones de forma segura"""
        self.logger.info("Cerrando conexiones BD...")
        
        # Detener worker thread
        self.should_stop = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        # Procesar operaciones pendientes críticas
        pending_operations = 0
        while not self.operation_queue.empty() and pending_operations < 10:
            try:
                operation = self.operation_queue.get_nowait()
                if operation.get('type') == 'save_measurement':
                    try:
                        self._execute_operation(operation)
                        pending_operations += 1
                    except:
                        pass
            except queue.Empty:
                break
            except:
                break
        
        # Limpiar pool
        if self.connection_pool:
            self.connection_pool = None
        
        self.logger.info(f"Conexiones BD cerradas ({pending_operations} operaciones pendientes procesadas)")
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'should_stop'):
            self.should_stop = True
        if hasattr(self, 'connection_pool') and self.connection_pool:
            try:
                self.close_connections()
            except:
                pass
