# Inicialización del paquete de módulos del sistema médico

"""
Sistema Modular de Monitoreo Médico
====================================

Módulos especializados:
- api_nodo_datos: API para ESP32
- ml_processor: Procesador ML
- database_manager: Gestor BD
- alert_system: Sistema alertas
- data_collector: Recolector datos
- websocket_handler: WebSockets
"""

# Configuración de logging para todos los módulos
import logging

def setup_module_logging():
    """Configurar logging para todos los módulos"""
    formatter = logging.Formatter(
        '[%(name)s] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('modules')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

# Configurar logging al importar
module_logger = setup_module_logging()
module_logger.info("Paquete modules inicializado")

# Importar clases principales
try:
    from .ml_processor import MLProcessor
    from .database_manager import DatabaseManager  
    from .alert_system import AlertSystem
    
    __all__ = [
        'MLProcessor',
        'DatabaseManager', 
        'AlertSystem',
        'setup_module_logging'
    ]
    
    module_logger.info("Clases principales importadas")
    
except ImportError as e:
    module_logger.warning(f"Algunas importaciones fallaron: {e}")
    __all__ = ['setup_module_logging']
