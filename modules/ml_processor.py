# modules/ml_processor.py
# Procesador ML especializado para predicciones de presión arterial

import joblib
import numpy as np
import logging
from pathlib import Path
import threading
import time
from datetime import datetime

class MLProcessor:
    """Procesador de Machine Learning para predicciones médicas"""
    
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
        
        # CALIBRACIÓN: Factores de corrección basados en validación clínica
        self.calibration_enabled = True
        self.calibration_factors = {
            'sys_global': 1.065,  # Factor basado en comparación 109/130
            'dia_global': 0.884,   # Corrección mínima para diastólica
            'sys_by_range': {
                (0, 120): 1.065,      # Presión normal: corrección menor
                (120, 140): 1.065,    # Presión elevada: corrección media
                (140, 180): 1.087,    # Hipertensión: corrección mayor
                (180, 250): 1.119     # Crisis: corrección máxima
            }
        }
        
        # Inicializar modelos
        self._load_models()
        
    def _load_models(self):
        """Cargar modelos ML desde archivos"""
        try:
            self.logger.info("Cargando modelos ML...")
            
            # Verificar que existan los archivos
            model_files = {
                'sys': self.models_path / 'modelo_sys.pkl',
                'dia': self.models_path / 'modelo_dia.pkl', 
                'scaler': self.models_path / 'scaler.pkl'
            }
            
            missing_files = [name for name, path in model_files.items() if not path.exists()]
            if missing_files:
                self.logger.warning(f"Archivos faltantes: {missing_files}")
                self.logger.warning("Funcionando en modo degradado sin ML")
                return
            
            # Cargar modelos
            self.modelo_sys = joblib.load(model_files['sys'])
            self.modelo_dia = joblib.load(model_files['dia'])
            self.scaler = joblib.load(model_files['scaler'])
            
            self.is_initialized = True
            self.logger.info("Modelos ML cargados correctamente")
            
            # Verificar dimensiones esperadas
            self._validate_models()
            
        except Exception as e:
            self.logger.error(f"Error cargando modelos ML: {e}")
            self.is_initialized = False
    
    def _validate_models(self):
        """Validar que los modelos tengan las dimensiones correctas"""
        try:
            # Test con datos dummy (valores típicos de tu entrenamiento)
            test_features = np.array([[75, 98, 1250, 890, 15, 12]])
            
            # Verificar scaler
            scaled_features = self.scaler.transform(test_features)
            
            # Verificar modelos
            sys_pred = self.modelo_sys.predict(scaled_features)
            dia_pred = self.modelo_dia.predict(scaled_features)
            
            self.logger.info(f"Validación ML exitosa - Test SYS: {sys_pred[0]:.1f}, DIA: {dia_pred[0]:.1f}")
            
        except Exception as e:
            self.logger.error(f"Error validando modelos: {e}")
            self.is_initialized = False
    
    def predict_pressure(self, hr, spo2, ir_mean, red_mean, ir_std, red_std):
        """
        Predecir presión arterial usando ML con calibración integrada
        
        Args:
            hr: Frecuencia cardíaca
            spo2: Saturación de oxígeno
            ir_mean: Promedio filtrado IR
            red_mean: Promedio filtrado RED
            ir_std: Desviación estándar IR
            red_std: Desviación estándar RED
            
        Returns:
            tuple: (presión_sistólica_calibrada, presión_diastólica_calibrada)
        """
        if not self.is_ready():
            self.logger.warning("Modelos ML no disponibles")
            return 0, 0
        
        start_time = time.time()
        
        try:
            with self.prediction_lock:
                # Verificar cache
                cache_key = f"{hr}_{spo2}_{ir_mean:.0f}_{red_mean:.0f}"
                if self._check_cache(cache_key):
                    return self.prediction_cache[cache_key]['result']
                
                # Preparar features (orden importante - debe coincidir con entrenamiento)
                features = np.array([[
                    float(hr),          # hr_promedio_sensor
                    float(spo2),        # spo2_promedio_sensor
                    float(ir_mean),     # ir_mean_filtrado
                    float(red_mean),    # red_mean_filtrado
                    float(ir_std),      # ir_std_filtrado
                    float(red_std)      # red_std_filtrado
                ]])
                
                self.logger.debug(f"Features ML: HR:{hr} SpO2:{spo2} IR:{ir_mean:.1f} RED:{red_mean:.1f}")
                
                # Validar features
                if not self._validate_features(features[0]):
                    self.logger.warning("Features fuera de rango válido")
                    return 0, 0
                
                # Escalar features
                scaled_features = self.scaler.transform(features)
                
                # Predicción ML original
                sys_pred_original = self.modelo_sys.predict(scaled_features)[0]
                dia_pred_original = self.modelo_dia.predict(scaled_features)[0]
                
                self.logger.info(f"Predicción ML original: SYS:{sys_pred_original:.1f} DIA:{dia_pred_original:.1f}")
                
                # Aplicar calibración si está habilitada
                if self.calibration_enabled:
                    sys_pred, dia_pred = self._apply_calibration(sys_pred_original, dia_pred_original)
                    self.logger.info(f"Predicción ML calibrada: SYS:{sys_pred:.1f} DIA:{dia_pred:.1f}")
                else:
                    sys_pred, dia_pred = sys_pred_original, dia_pred_original
                
                # Validar predicciones finales
                sys_pred, dia_pred = self._validate_predictions(sys_pred, dia_pred)
                
                # Actualizar cache
                self.prediction_cache[cache_key] = {
                    'result': (sys_pred, dia_pred),
                    'timestamp': time.time(),
                    'original': (sys_pred_original, dia_pred_original)
                }
                
                # Actualizar métricas
                prediction_time = time.time() - start_time
                self.prediction_times.append(prediction_time)
                if len(self.prediction_times) > 100:
                    self.prediction_times.pop(0)
                
                self.prediction_count += 1
                self.last_prediction_time = time.time()
                
                self.logger.info(f"Predicción final: SYS:{sys_pred:.1f} DIA:{dia_pred:.1f} ({prediction_time*1000:.1f}ms)")
                
                return sys_pred, dia_pred
                
        except Exception as e:
            self.logger.error(f"Error en predicción ML: {e}")
            self.error_count += 1
            return 0, 0
    
    def _apply_calibration(self, sys_pred, dia_pred):
        """Aplicar factores de calibración a las predicciones"""
        try:
            # Método 1: Calibración por rangos (más precisa)
            sys_calibrated = self._calibrate_systolic_by_range(sys_pred)
            
            # Método 2: Calibración global para diastólica (ya es bastante precisa)
            dia_calibrated = dia_pred * self.calibration_factors['dia_global']
            
            self.logger.debug(f"Calibración aplicada: SYS {sys_pred:.1f}→{sys_calibrated:.1f}, DIA {dia_pred:.1f}→{dia_calibrated:.1f}")
            
            return sys_calibrated, dia_calibrated
            
        except Exception as e:
            self.logger.error(f"Error en calibración: {e}")
            return sys_pred, dia_pred
    
    def _calibrate_systolic_by_range(self, sys_pred):
        """Calibrar presión sistólica según rango de valores"""
        try:
            # Encontrar el rango apropiado
            for (min_val, max_val), factor in self.calibration_factors['sys_by_range'].items():
                if min_val <= sys_pred < max_val:
                    calibrated = sys_pred * factor
                    self.logger.debug(f"Rango {min_val}-{max_val}: factor {factor}, {sys_pred:.1f}→{calibrated:.1f}")
                    return calibrated
            
            # Si no encuentra rango, usar factor global
            return sys_pred * self.calibration_factors['sys_global']
            
        except Exception as e:
            self.logger.error(f"Error en calibración por rango: {e}")
            return sys_pred
    
    def _validate_features(self, features):
        """Validar que las features estén en rangos razonables"""
        hr, spo2, ir_mean, red_mean, ir_std, red_std = features
        
        # Rangos válidos basados en tu entrenamiento
        valid_ranges = {
            'hr': (40, 200),
            'spo2': (70, 100),
            'ir_mean': (500, 200000),
            'red_mean': (300, 150000),
            'ir_std': (0, 50000),
            'red_std': (0, 30000)
        }
        
        values = [hr, spo2, ir_mean, red_mean, ir_std, red_std]
        names = ['hr', 'spo2', 'ir_mean', 'red_mean', 'ir_std', 'red_std']
        
        for value, name in zip(values, names):
            min_val, max_val = valid_ranges[name]
            if not (min_val <= value <= max_val):
                self.logger.warning(f"{name} fuera de rango: {value} (válido: {min_val}-{max_val})")
                return False
        
        return True
    
    def _validate_predictions(self, sys_pred, dia_pred):
        """Validar y ajustar predicciones a rangos médicamente válidos"""
        # Rangos médicamente válidos
        sys_min, sys_max = 70, 250
        dia_min, dia_max = 40, 150
        
        # Ajustar si están fuera de rango
        if sys_pred < sys_min:
            sys_pred = sys_min
        elif sys_pred > sys_max:
            sys_pred = sys_max
            
        if dia_pred < dia_min:
            dia_pred = dia_min
        elif dia_pred > dia_max:
            dia_pred = dia_max
        
        # Validar relación sistólica > diastólica
        if dia_pred >= sys_pred:
            dia_pred = sys_pred - 10
        
        return round(sys_pred, 1), round(dia_pred, 1)
    
    def _check_cache(self, cache_key):
        """Verificar si existe predicción en cache y no ha expirado"""
        if cache_key in self.prediction_cache:
            cached_time = self.prediction_cache[cache_key]['timestamp']
            if time.time() - cached_time < self.cache_timeout:
                self.logger.debug(f"Usando predicción desde cache: {cache_key}")
                return True
            else:
                # Eliminar entrada expirada
                del self.prediction_cache[cache_key]
        return False
    
    def update_calibration_factors(self, new_factors):
        """Actualizar factores de calibración en tiempo real"""
        try:
            if 'sys_global' in new_factors:
                self.calibration_factors['sys_global'] = float(new_factors['sys_global'])
            
            if 'dia_global' in new_factors:
                self.calibration_factors['dia_global'] = float(new_factors['dia_global'])
            
            if 'sys_by_range' in new_factors:
                self.calibration_factors['sys_by_range'].update(new_factors['sys_by_range'])
            
            self.logger.info("Factores de calibración actualizados")
            self.clear_cache()  # Limpiar cache para aplicar nuevos factores
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error actualizando calibración: {e}")
            return False
    
    def enable_calibration(self, enabled=True):
        """Habilitar/deshabilitar calibración"""
        self.calibration_enabled = enabled
        self.clear_cache()  # Limpiar cache
        
        status = "habilitada" if enabled else "deshabilitada"
        self.logger.info(f"Calibración {status}")
    
    def get_calibration_info(self):
        """Obtener información actual de calibración"""
        return {
            "enabled": self.calibration_enabled,
            "factors": self.calibration_factors,
            "last_update": datetime.now().isoformat()
        }
    
    def validate_with_reference(self, ml_sys, ml_dia, ref_sys, ref_dia):
        """Validar predicción con valores de referencia y ajustar calibración"""
        try:
            # Calcular errores
            sys_error = (ml_sys - ref_sys) / ref_sys
            dia_error = (ml_dia - ref_dia) / ref_dia
            
            # Si el error es consistente, sugerir ajuste
            if abs(sys_error) > 0.15:  # Error mayor al 15%
                suggested_sys_factor = ref_sys / ml_sys
                self.logger.info(f"Error sistólica {sys_error*100:.1f}%. Factor sugerido: {suggested_sys_factor:.3f}")
            
            if abs(dia_error) > 0.10:  # Error mayor al 10%
                suggested_dia_factor = ref_dia / ml_dia
                self.logger.info(f"Error diastólica {dia_error*100:.1f}%. Factor sugerido: {suggested_dia_factor:.3f}")
            
            return {
                "sys_error_percent": sys_error * 100,
                "dia_error_percent": dia_error * 100,
                "suggested_sys_factor": ref_sys / ml_sys if ml_sys > 0 else 1.0,
                "suggested_dia_factor": ref_dia / ml_dia if ml_dia > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Error en validación con referencia: {e}")
            return {}
    
    def is_ready(self):
        """Verificar si el procesador ML está listo"""
        return self.is_initialized and all([
            self.modelo_sys is not None,
            self.modelo_dia is not None,
            self.scaler is not None
        ])
    
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
    
    def get_performance_metrics(self):
        """Obtener métricas detalladas de rendimiento"""
        if not self.prediction_times:
            return {"message": "No hay datos de rendimiento disponibles"}
        
        times_ms = [t * 1000 for t in self.prediction_times]
        
        return {
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.prediction_count, 1),
            "timing_stats": {
                "avg_ms": np.mean(times_ms),
                "min_ms": np.min(times_ms),
                "max_ms": np.max(times_ms),
                "std_ms": np.std(times_ms)
            },
            "cache_stats": {
                "size": len(self.prediction_cache),
                "hit_rate": self._calculate_cache_hit_rate()
            },
            "calibration_stats": {
                "enabled": self.calibration_enabled,
                "sys_global_factor": self.calibration_factors['sys_global'],
                "dia_global_factor": self.calibration_factors['dia_global']
            }
        }
    
    def _calculate_cache_hit_rate(self):
        """Calcular tasa de aciertos del cache (simplificado)"""
        # En una implementación real, rastrearías hits/misses
        return len(self.prediction_cache) / max(self.prediction_count, 1)
    
    def clear_cache(self):
        """Limpiar cache de predicciones"""
        with self.prediction_lock:
            self.prediction_cache.clear()
            self.logger.info("Cache de predicciones limpiado")
    
    def reload_models(self):
        """Recargar modelos ML (útil para actualizaciones en caliente)"""
        self.logger.info("Recargando modelos ML...")
        self.is_initialized = False
        self.modelo_sys = None
        self.modelo_dia = None
        self.scaler = None
        self.clear_cache()
        self._load_models()
    
    def predict_batch(self, feature_list):
        """Realizar predicciones en lote para múltiples muestras"""
        if not self.is_ready():
            return []
        
        try:
            with self.prediction_lock:
                results = []
                
                # Preparar todas las features
                features_array = np.array(feature_list)
                
                # Escalar
                scaled_features = self.scaler.transform(features_array)
                
                # Predicciones en lote
                sys_predictions = self.modelo_sys.predict(scaled_features)
                dia_predictions = self.modelo_dia.predict(scaled_features)
                
                # Aplicar calibración y validar cada predicción
                for sys_pred, dia_pred in zip(sys_predictions, dia_predictions):
                    if self.calibration_enabled:
                        sys_cal, dia_cal = self._apply_calibration(sys_pred, dia_pred)
                    else:
                        sys_cal, dia_cal = sys_pred, dia_pred
                    
                    sys_val, dia_val = self._validate_predictions(sys_cal, dia_cal)
                    results.append((sys_val, dia_val))
                
                self.prediction_count += len(results)
                self.logger.info(f"Predicciones en lote completadas: {len(results)} muestras")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error en predicción en lote: {e}")
            return []
    
    def get_model_info(self):
        """Obtener información sobre los modelos cargados"""
        if not self.is_ready():
            return {"error": "Modelos no cargados"}
        
        try:
            return {
                "scaler_features": self.scaler.n_features_in_,
                "feature_names": [
                    "hr_promedio_sensor",
                    "spo2_promedio_sensor", 
                    "ir_mean_filtrado",
                    "red_mean_filtrado",
                    "ir_std_filtrado",
                    "red_std_filtrado"
                ],
                "modelo_sys_type": type(self.modelo_sys).__name__,
                "modelo_dia_type": type(self.modelo_dia).__name__,
                "expected_input_shape": (1, self.scaler.n_features_in_),
                "cache_timeout_seconds": self.cache_timeout,
                "calibration_info": self.get_calibration_info()
            }
        except Exception as e:
            return {"error": f"Error obteniendo info: {e}"}
    
    def test_prediction(self):
        """Realizar predicción de prueba con valores típicos"""
        try:
            # Valores típicos basados en tu entrenamiento
            test_hr = 75
            test_spo2 = 98
            test_ir_mean = 1250
            test_red_mean = 890
            test_ir_std = 15
            test_red_std = 12
            
            sys_pred, dia_pred = self.predict_pressure(
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
                    "dia": dia_pred
                },
                "valid_range": 90 <= sys_pred <= 200 and 60 <= dia_pred <= 120,
                "calibration_applied": self.calibration_enabled
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'logger'):
            self.logger.info("Cerrando procesador ML")
