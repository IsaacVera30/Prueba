# modules/ml_processor.py
# Procesador ML especializado CORREGIDO - CÓDIGO COMPLETO

import joblib
import numpy as np
import logging
from pathlib import Path
import threading
import time
from datetime import datetime

class MLProcessor:
    """Procesador de Machine Learning CORREGIDO para predicciones médicas"""
    
    def __init__(self, models_path='models'):
        self.models_path = Path(models_path)
        self.logger = logging.getLogger(__name__)
        
        # Modelos ML
        self.modelo_sys = None
        self.modelo_dia = None
        self.scaler = None
        
        # Estado del procesador
        self.is_initialized = False
        self.last_prediction_time = 0
        self.prediction_count = 0
        
        # Cache de predicciones recientes
        self.prediction_cache = {}
        self.cache_timeout = 5  # segundos
        
        # Métricas de rendimiento
        self.prediction_times = []
        self.error_count = 0
        
        # Lock para thread safety
        self.prediction_lock = threading.Lock()
        
        # Calibración 
        self.calibration_enabled = True
        self.calibration_factors = {
            'sys_global': 0.87,  # CAMBIO: Reducido de 1.25 a 1.10
            'dia_global': 0.89,  # Mantener para reducir DIA
        }
        
        # Inicializar modelos
        self._load_models()
        
    def _load_models(self):
        """Cargar modelos ML desde archivos con validación robusta"""
        try:
            self.logger.info("Cargando modelos ML...")
            
            # Verificar archivos
            model_files = {
                'sys': self.models_path / 'modelo_sys.pkl',
                'dia': self.models_path / 'modelo_dia.pkl', 
                'scaler': self.models_path / 'scaler.pkl'
            }
            
            missing_files = [name for name, path in model_files.items() if not path.exists()]
            if missing_files:
                self.logger.warning(f"Archivos faltantes: {missing_files}")
                self.logger.warning("Funcionando en modo degradado sin ML")
                return False
            
            # Cargar modelos con manejo de errores
            try:
                self.modelo_sys = joblib.load(model_files['sys'])
                self.logger.info("Modelo SYS cargado")
            except Exception as e:
                self.logger.error(f"Error cargando modelo SYS: {e}")
                return False
                
            try:
                self.modelo_dia = joblib.load(model_files['dia'])
                self.logger.info("Modelo DIA cargado")
            except Exception as e:
                self.logger.error(f"Error cargando modelo DIA: {e}")
                return False
                
            try:
                self.scaler = joblib.load(model_files['scaler'])
                self.logger.info("Scaler cargado")
            except Exception as e:
                self.logger.error(f"Error cargando scaler: {e}")
                return False
            
            self.is_initialized = True
            self.logger.info("Modelos ML cargados correctamente")
            
            # Verificar funcionamiento
            return self._validate_models()
            
        except Exception as e:
            self.logger.error(f"Error general cargando modelos ML: {e}")
            self.is_initialized = False
            return False
    
    def _validate_models(self):
        """Validar que los modelos funcionen correctamente"""
        try:
            # Test con datos típicos
            test_features = np.array([[75, 98, 1250, 890, 15, 12]], dtype=float)
            
            # Verificar scaler
            scaled_features = self.scaler.transform(test_features)
            self.logger.info(f"Test scaler OK - shape: {scaled_features.shape}")
            
            # Verificar modelos
            sys_pred = self.modelo_sys.predict(scaled_features)[0]
            dia_pred = self.modelo_dia.predict(scaled_features)[0]
            
            self.logger.info(f"Test predicción - SYS: {sys_pred:.1f}, DIA: {dia_pred:.1f}")
            
            # Validar que las predicciones sean números
            if not (isinstance(sys_pred, (int, float, np.number)) and 
                    isinstance(dia_pred, (int, float, np.number))):
                raise ValueError("Predicciones no son números válidos")
            
            # Validar rangos básicos
            if not (50 <= sys_pred <= 250 and 30 <= dia_pred <= 150):
                self.logger.warning(f"Predicciones test fuera de rango: SYS={sys_pred}, DIA={dia_pred}")
            
            self.logger.info("Validación ML exitosa")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando modelos: {e}")
            self.is_initialized = False
            return False
    
    def predict_pressure(self, hr, spo2, ir_mean, red_mean, ir_std, red_std):
        """
        Predecir presión arterial CORREGIDO con manejo robusto de errores
        
        Returns:
            tuple: (presión_sistólica, presión_diastólica, hr_corregido)
        """
        if not self.is_ready():
            self.logger.warning("Modelos ML no disponibles")
            return 0.0, 0.0, 75.0
        
        start_time = time.time()
        
        try:
            with self.prediction_lock:
                # Validar y limpiar entradas
                try:
                    hr = float(hr) if hr is not None else 75.0
                    spo2 = float(spo2) if spo2 is not None else 98.0
                    ir_mean = float(ir_mean) if ir_mean is not None else 1000.0
                    red_mean = float(red_mean) if red_mean is not None else 800.0
                    ir_std = float(ir_std) if ir_std is not None else 10.0
                    red_std = float(red_std) if red_std is not None else 8.0
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Error convirtiendo entradas: {e}")
                    return 0.0, 0.0, 75.0
                
                # Corregir HR si es necesario
                hr_corregido = self._calcular_hr_corregido(ir_mean, red_mean, hr)
                
                # Preparar features en el orden correcto
                features = np.array([[
                    hr_corregido,   # hr_promedio_sensor
                    spo2,           # spo2_promedio_sensor
                    ir_mean,        # ir_mean_filtrado
                    red_mean,       # red_mean_filtrado
                    ir_std,         # ir_std_filtrado
                    red_std         # red_std_filtrado
                ]], dtype=float)
                
                self.logger.debug(f"Features ML: HR:{hr_corregido:.1f} SpO2:{spo2:.1f} IR:{ir_mean:.0f} RED:{red_mean:.0f}")
                
                # Validar features
                if not self._validate_features(features[0]):
                    self.logger.warning("Features fuera de rango válido")
                    return 0.0, 0.0, hr_corregido
                
                # Escalar features
                try:
                    scaled_features = self.scaler.transform(features)
                except Exception as e:
                    self.logger.error(f"Error escalando features: {e}")
                    return 0.0, 0.0, hr_corregido
                
                # Predicción ML
                try:
                    sys_pred_original = float(self.modelo_sys.predict(scaled_features)[0])
                    dia_pred_original = float(self.modelo_dia.predict(scaled_features)[0])
                except Exception as e:
                    self.logger.error(f"Error en predicción ML: {e}")
                    return 0.0, 0.0, hr_corregido
                
                self.logger.info(f"Predicción ML original: SYS:{sys_pred_original:.1f} DIA:{dia_pred_original:.1f}")
                
                # Aplicar calibración
                if self.calibration_enabled:
                    sys_pred = sys_pred_original * self.calibration_factors['sys_global']
                    dia_pred = dia_pred_original * self.calibration_factors['dia_global']
                    self.logger.info(f"Predicción calibrada: SYS:{sys_pred:.1f} DIA:{dia_pred:.1f}")
                else:
                    sys_pred, dia_pred = sys_pred_original, dia_pred_original
                
                # Validar y corregir predicciones finales
                sys_pred, dia_pred = self._validate_predictions(sys_pred, dia_pred)
                
                # Actualizar métricas
                prediction_time = time.time() - start_time
                self.prediction_times.append(prediction_time)
                if len(self.prediction_times) > 100:
                    self.prediction_times.pop(0)
                
                self.prediction_count += 1
                self.last_prediction_time = time.time()
                
                self.logger.info(f"Predicción FINAL: SYS:{sys_pred:.1f} DIA:{dia_pred:.1f} HR:{hr_corregido:.0f} ({prediction_time*1000:.1f}ms)")
                
                return sys_pred, dia_pred, hr_corregido
                
        except Exception as e:
            self.logger.error(f"Error grave en predicción ML: {e}")
            self.error_count += 1
            return 0.0, 0.0, 75.0
    
    def _calcular_hr_corregido(self, ir_mean, red_mean, hr_original):
        """Calcular HR corregido usando señales y valor original"""
        try:
            # Si el HR original es razonable, usarlo como base
            if 50 <= hr_original <= 120:
                hr_base = hr_original
            else:
                hr_base = 75.0
            
            # Ajustar basado en la relación de señales
            signal_ratio = ir_mean / max(red_mean, 1.0)
            
            if signal_ratio > 2.0:
                # Señal IR dominante = HR posiblemente alta
                hr_ajustado = hr_base + min(10, (signal_ratio - 2.0) * 5)
            elif signal_ratio < 1.2:
                # Señal IR baja = HR posiblemente baja
                hr_ajustado = hr_base - min(8, (1.2 - signal_ratio) * 8)
            else:
                # Ratio normal
                hr_ajustado = hr_base
            
            # Limitar a rango fisiológico
            hr_final = max(50, min(130, hr_ajustado))
            
            self.logger.debug(f"HR corregido: {hr_original:.1f} -> {hr_final:.1f} (ratio: {signal_ratio:.2f})")
            
            return hr_final
            
        except Exception as e:
            self.logger.warning(f"Error calculando HR corregido: {e}")
            return 75.0
    
    def _validate_features(self, features):
        """Validar que las features estén en rangos razonables"""
        try:
            hr, spo2, ir_mean, red_mean, ir_std, red_std = features
            
            # Rangos válidos más amplios
            valid_ranges = {
                'hr': (40, 150),
                'spo2': (70, 100),
                'ir_mean': (100, 300000),
                'red_mean': (50, 200000),
                'ir_std': (0, 100000),
                'red_std': (0, 80000)
            }
            
            values = [hr, spo2, ir_mean, red_mean, ir_std, red_std]
            names = ['hr', 'spo2', 'ir_mean', 'red_mean', 'ir_std', 'red_std']
            
            for value, name in zip(values, names):
                min_val, max_val = valid_ranges[name]
                if not (min_val <= value <= max_val):
                    self.logger.warning(f"{name} fuera de rango: {value} (válido: {min_val}-{max_val})")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando features: {e}")
            return False
    
    def _validate_predictions(self, sys_pred, dia_pred):
        """Validar y ajustar predicciones a rangos médicamente válidos"""
        try:
            # Convertir a float si no lo son
            sys_pred = float(sys_pred)
            dia_pred = float(dia_pred)
            
            # Rangos médicamente válidos
            sys_min, sys_max = 70, 220
            dia_min, dia_max = 40, 130
            
            # Ajustar si están fuera de rango
            if sys_pred < sys_min:
                sys_pred = sys_min
                self.logger.debug(f"SYS ajustado al mínimo: {sys_min}")
            elif sys_pred > sys_max:
                sys_pred = sys_max
                self.logger.debug(f"SYS ajustado al máximo: {sys_max}")
                
            if dia_pred < dia_min:
                dia_pred = dia_min
                self.logger.debug(f"DIA ajustado al mínimo: {dia_min}")
            elif dia_pred > dia_max:
                dia_pred = dia_max
                self.logger.debug(f"DIA ajustado al máximo: {dia_max}")
            
            # Validar relación sistólica > diastólica
            if dia_pred >= sys_pred:
                dia_pred = sys_pred - 15
                self.logger.debug(f"DIA ajustado para mantener SYS > DIA: {dia_pred}")
            
            # Verificar que no sean NaN o infinito
            if not (np.isfinite(sys_pred) and np.isfinite(dia_pred)):
                self.logger.error("Predicciones no finitas detectadas")
                return 120.0, 80.0  # Valores por defecto
            
            return round(float(sys_pred), 1), round(float(dia_pred), 1)
            
        except Exception as e:
            self.logger.error(f"Error validando predicciones: {e}")
            return 120.0, 80.0  # Valores por defecto seguros
    
    def is_ready(self):
        """Verificar si el procesador ML está listo"""
        return (self.is_initialized and 
                self.modelo_sys is not None and 
                self.modelo_dia is not None and 
                self.scaler is not None)
    
    def get_status(self):
        """Obtener estado actual del procesador ML"""
        return {
            "initialized": self.is_initialized,
            "models_loaded": self.is_ready(),
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "cache_size": len(self.prediction_cache),
            "avg_prediction_time_ms": np.mean(self.prediction_times) * 1000 if self.prediction_times else 0,
            "last_prediction": datetime.fromtimestamp(self.last_prediction_time).isoformat() if self.last_prediction_time else None,
            "calibration_enabled": self.calibration_enabled,
            "calibration_factors": self.calibration_factors
        }
    
    def test_prediction(self):
        """Realizar predicción de prueba con valores típicos"""
        try:
            # Valores típicos
            test_hr = 75
            test_spo2 = 98
            test_ir_mean = 1250
            test_red_mean = 890
            test_ir_std = 15
            test_red_std = 12
            
            sys_pred, dia_pred, hr_final = self.predict_pressure(
                test_hr, test_spo2, test_ir_mean, test_red_mean, test_ir_std, test_red_std
            )
            
            return {
                "success": True,
                "input": {
                    "hr": test_hr,
                    "spo2": test_spo2,
                    "ir_mean": test_ir_mean,
                    "red_mean": test_red_mean,
                    "ir_std": test_ir_std,
                    "red_std": test_red_std
                },
                "prediction": {
                    "sys": sys_pred,
                    "dia": dia_pred,
                    "hr_corrected": hr_final
                },
                "valid_range": 80 <= sys_pred <= 160 and 50 <= dia_pred <= 100,
                "calibration_applied": self.calibration_enabled
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def reload_models(self):
        """Recargar modelos desde archivos"""
        self.logger.info("Recargando modelos ML...")
        self.is_initialized = False
        self.modelo_sys = None
        self.modelo_dia = None
        self.scaler = None
        
        return self._load_models()
    
    def clear_cache(self):
        """Limpiar cache de predicciones"""
        with self.prediction_lock:
            self.prediction_cache.clear()
            self.logger.info("Cache de predicciones limpiado")
    
    def update_calibration_factors(self, new_factors):
        """Actualizar factores de calibración"""
        try:
            if 'sys_global' in new_factors:
                self.calibration_factors['sys_global'] = float(new_factors['sys_global'])
            
            if 'dia_global' in new_factors:
                self.calibration_factors['dia_global'] = float(new_factors['dia_global'])
            
            self.logger.info("Factores de calibración actualizados")
            self.clear_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error actualizando calibración: {e}")
            return False
    
    def enable_calibration(self, enabled=True):
        """Habilitar/deshabilitar calibración"""
        self.calibration_enabled = enabled
        self.clear_cache()
        
        status = "habilitada" if enabled else "deshabilitada"
        self.logger.info(f"Calibración {status}")
    
    def get_calibration_info(self):
        """Obtener información actual de calibración"""
        return {
            "enabled": self.calibration_enabled,
            "factors": self.calibration_factors,
            "last_update": datetime.now().isoformat()
        }
    
    def get_performance_metrics(self):
        """Obtener métricas detalladas de rendimiento"""
        return {
            "total_predictions": self.prediction_count,
            "total_errors": self.error_count,
            "success_rate": (self.prediction_count / max(self.prediction_count + self.error_count, 1)) * 100,
            "avg_prediction_time_ms": np.mean(self.prediction_times) * 1000 if self.prediction_times else 0,
            "max_prediction_time_ms": max(self.prediction_times) * 1000 if self.prediction_times else 0,
            "min_prediction_time_ms": min(self.prediction_times) * 1000 if self.prediction_times else 0,
            "cache_size": len(self.prediction_cache),
            "models_ready": self.is_ready(),
            "calibration_active": self.calibration_enabled
        }
    
    def reset_metrics(self):
        """Resetear métricas de rendimiento"""
        self.prediction_count = 0
        self.error_count = 0
        self.prediction_times = []
        self.last_prediction_time = 0
        self.logger.info("Métricas de ML reseteadas")
    
    def validate_model_files(self):
        """Validar que los archivos de modelos sean correctos"""
        try:
            model_files = {
                'sys': self.models_path / 'modelo_sys.pkl',
                'dia': self.models_path / 'modelo_dia.pkl', 
                'scaler': self.models_path / 'scaler.pkl'
            }
            
            results = {}
            
            for name, path in model_files.items():
                results[name] = {
                    "exists": path.exists(),
                    "size_mb": path.stat().st_size / (1024 * 1024) if path.exists() else 0,
                    "readable": False
                }
                
                if path.exists():
                    try:
                        if name == 'scaler':
                            test_obj = joblib.load(path)
                            # Test básico del scaler
                            test_data = np.array([[75, 98, 1250, 890, 15, 12]])
                            test_obj.transform(test_data)
                        else:
                            test_obj = joblib.load(path)
                            # Test básico del modelo
                            test_data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
                            test_obj.predict(test_data)
                        
                        results[name]["readable"] = True
                        
                    except Exception as e:
                        results[name]["error"] = str(e)
            
            return {
                "validation_success": all(r["readable"] for r in results.values()),
                "files": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "validation_success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def shutdown(self):
        """Apagar procesador ML de forma segura"""
        try:
            self.logger.info("Apagando procesador ML...")
            self.clear_cache()
            
            # Limpiar modelos de memoria
            self.modelo_sys = None
            self.modelo_dia = None
            self.scaler = None
            self.is_initialized = False
            
            self.logger.info("Procesador ML apagado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error apagando ML processor: {e}")
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        try:
            if hasattr(self, 'is_initialized') and self.is_initialized:
                self.shutdown()
        except:
            pass
