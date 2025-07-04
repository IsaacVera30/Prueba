# train.py - VERSIÓN SIMPLIFICADA
# Script de entrenamiento sin dependencias extra

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def setup_directories():
    """Crear directorios necesarios"""
    Path("models").mkdir(exist_ok=True)
    print("Directorios verificados")

def load_and_validate_data(csv_path):
    """Cargar y validar datos de entrenamiento"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Datos cargados: {len(df)} muestras")
        
        # Verificar columnas requeridas
        required_columns = [
            "hr_promedio_sensor",
            "spo2_promedio_sensor", 
            "ir_mean_filtrado",     
            "red_mean_filtrado",     
            "ir_std_filtrado",
            "red_std_filtrado",
            "sys_ref",
            "dia_ref"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes: {missing_columns}")
        
        print("Todas las columnas requeridas están presentes")
        
        # Mostrar estadísticas básicas
        print("\nEstadísticas de los datos:")
        print(f"HR promedio: {df['hr_promedio_sensor'].mean():.1f} ± {df['hr_promedio_sensor'].std():.1f}")
        print(f"SpO2 promedio: {df['spo2_promedio_sensor'].mean():.1f} ± {df['spo2_promedio_sensor'].std():.1f}")
        print(f"SYS promedio: {df['sys_ref'].mean():.1f} ± {df['sys_ref'].std():.1f}")
        print(f"DIA promedio: {df['dia_ref'].mean():.1f} ± {df['dia_ref'].std():.1f}")
        
        return df, required_columns
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")
    except Exception as e:
        raise Exception(f"Error cargando datos: {e}")

def preprocess_data(df, feature_columns):
    """Preprocesar datos para entrenamiento"""
    # Eliminar filas con valores nulos
    df_clean = df.dropna(subset=feature_columns + ["sys_ref", "dia_ref"])
    print(f"Datos después de limpiar NaN: {len(df_clean)} muestras")
    
    # Eliminar outliers extremos
    for col in ["sys_ref", "dia_ref"]:
        Q1 = df_clean[col].quantile(0.01)
        Q3 = df_clean[col].quantile(0.99)
        df_clean = df_clean[(df_clean[col] >= Q1) & (df_clean[col] <= Q3)]
    
    print(f"Datos después de eliminar outliers: {len(df_clean)} muestras")
    
    # Extraer features y targets
    feature_names = [
        "hr_promedio_sensor",
        "spo2_promedio_sensor",
        "ir_mean_filtrado",     
        "red_mean_filtrado",       
        "ir_std_filtrado",
        "red_std_filtrado"
    ]
    
    X = df_clean[feature_names].values
    y_sys = df_clean["sys_ref"].values
    y_dia = df_clean["dia_ref"].values
    
    print("Features utilizadas:")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")
    
    return X, y_sys, y_dia, feature_names

def train_models(X, y_sys, y_dia):
    """Entrenar modelos de ML"""
    print("\nIniciando entrenamiento de modelos...")
    
    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # División train/test
    X_train, X_test, y_sys_train, y_sys_test = train_test_split(
        X_scaled, y_sys, test_size=0.2, random_state=42
    )
    _, _, y_dia_train, y_dia_test = train_test_split(
        X_scaled, y_dia, test_size=0.2, random_state=42
    )
    
    print(f"Datos entrenamiento: {len(X_train)} muestras")
    print(f"Datos prueba: {len(X_test)} muestras")
    
    # Entrenar modelo sistólica
    modelo_sys = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    modelo_sys.fit(X_train, y_sys_train)
    
    # Entrenar modelo diastólica
    modelo_dia = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    modelo_dia.fit(X_train, y_dia_train)
    
    print("Modelos entrenados exitosamente")
    
    return modelo_sys, modelo_dia, scaler, X_test, y_sys_test, y_dia_test

def evaluate_models(modelo_sys, modelo_dia, scaler, X_test, y_sys_test, y_dia_test):
    """Evaluar rendimiento de los modelos"""
    print("\nEvaluando modelos...")
    
    # Predicciones
    pred_sys = modelo_sys.predict(X_test)
    pred_dia = modelo_dia.predict(X_test)
    
    # Métricas sistólica
    mae_sys = mean_absolute_error(y_sys_test, pred_sys)
    r2_sys = r2_score(y_sys_test, pred_sys)
    
    # Métricas diastólica  
    mae_dia = mean_absolute_error(y_dia_test, pred_dia)
    r2_dia = r2_score(y_dia_test, pred_dia)
    
    print(f"SISTÓLICA - MAE: {mae_sys:.2f} mmHg, R²: {r2_sys:.3f}")
    print(f"DIASTÓLICA - MAE: {mae_dia:.2f} mmHg, R²: {r2_dia:.3f}")
    
    # Verificar rango de predicciones
    print(f"\nRango predicciones SYS: {pred_sys.min():.1f} - {pred_sys.max():.1f}")
    print(f"Rango predicciones DIA: {pred_dia.min():.1f} - {pred_dia.max():.1f}")
    
    return {
        'mae_sys': mae_sys,
        'mae_dia': mae_dia,
        'r2_sys': r2_sys,
        'r2_dia': r2_dia
    }

def save_models(modelo_sys, modelo_dia, scaler):
    """Guardar modelos entrenados"""
    print("\nGuardando modelos...")
    
    try:
        joblib.dump(modelo_sys, "models/modelo_sys.pkl")
        joblib.dump(modelo_dia, "models/modelo_dia.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        
        print("Modelos guardados exitosamente:")
        print("  - models/modelo_sys.pkl")
        print("  - models/modelo_dia.pkl") 
        print("  - models/scaler.pkl")
        
        # Verificar tamaños de archivos
        for filename in ["modelo_sys.pkl", "modelo_dia.pkl", "scaler.pkl"]:
            filepath = f"models/{filename}"
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename}: {size_mb:.2f} MB")
            
    except Exception as e:
        raise Exception(f"Error guardando modelos: {e}")

def test_models_integration():
    """Probar que los modelos funcionen correctamente"""
    print("\nProbando integración de modelos...")
    
    try:
        # Cargar modelos
        modelo_sys = joblib.load("models/modelo_sys.pkl")
        modelo_dia = joblib.load("models/modelo_dia.pkl")
        scaler = joblib.load("models/scaler.pkl")
        
        # Datos de prueba (valores típicos)
        test_features = np.array([[
            75,    # hr_promedio_sensor
            98,    # spo2_promedio_sensor
            1250,  # ir_mean_filtrado
            890,   # red_mean_filtrado
            15,    # ir_std_filtrado
            12     # red_std_filtrado
        ]])
        
        # Escalar y predecir
        test_scaled = scaler.transform(test_features)
        sys_pred = modelo_sys.predict(test_scaled)[0]
        dia_pred = modelo_dia.predict(test_scaled)[0]
        
        print(f"Prueba con datos típicos:")
        print(f"  HR: 75, SpO2: 98, IR: 1250, RED: 890")
        print(f"  Predicción → SYS: {sys_pred:.1f}, DIA: {dia_pred:.1f}")
        
        # Validar que las predicciones sean razonables
        if 90 <= sys_pred <= 200 and 60 <= dia_pred <= 120:
            print("Predicciones en rango normal")
        else:
            print("Predicciones fuera de rango esperado")
            
        return True
        
    except Exception as e:
        print(f"Error en test de integración: {e}")
        return False

def generate_feature_importance(modelo_sys, modelo_dia, feature_names):
    """Generar análisis de importancia de features"""
    print("\nAnalizando importancia de features...")
    
    # Importancia sistólica
    importance_sys = modelo_sys.feature_importances_
    importance_dia = modelo_dia.feature_importances_
    
    print("\nImportancia features para SISTÓLICA:")
    for name, importance in zip(feature_names, importance_sys):
        print(f"  {name}: {importance:.3f}")
    
    print("\nImportancia features para DIASTÓLICA:")
    for name, importance in zip(feature_names, importance_dia):
        print(f"  {name}: {importance:.3f}")

def main():
    """Función principal"""
    print("ENTRENAMIENTO DE MODELOS ML - VERSIÓN CORREGIDA")
    print("=" * 60)
    
    try:
        # 1. Configurar directorios
        setup_directories()
        
        # 2. Cargar datos (buscar en ubicaciones posibles)
        csv_paths = [
            "data/entrenamiento_ml.csv",
            "entrenamiento_ml.csv",
            "../entrenamiento_ml.csv"
        ]
        
        df = None
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                print(f"Usando archivo: {csv_path}")
                df, required_columns = load_and_validate_data(csv_path)
                break
        
        if df is None:
            raise FileNotFoundError("No se encontró entrenamiento_ml.csv en ninguna ubicación")
        
        # 3. Preprocesar
        X, y_sys, y_dia, feature_names = preprocess_data(df, required_columns)
        
        # 4. Entrenar
        modelo_sys, modelo_dia, scaler, X_test, y_sys_test, y_dia_test = train_models(X, y_sys, y_dia)
        
        # 5. Evaluar
        metrics = evaluate_models(modelo_sys, modelo_dia, scaler, X_test, y_sys_test, y_dia_test)
        
        # 6. Análisis de features
        generate_feature_importance(modelo_sys, modelo_dia, feature_names)
        
        # 7. Guardar
        save_models(modelo_sys, modelo_dia, scaler)
        
        # 8. Test de integración
        if test_models_integration():
            print("\nENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            print("Los modelos están listos para usar en producción")
        else:
            print("\nFALLÓ EL TEST DE INTEGRACIÓN")
            
        print(f"\nRESUMEN:")
        print(f"  • Muestras procesadas: {len(X)}")
        print(f"  • MAE Sistólica: {metrics['mae_sys']:.2f} mmHg")
        print(f"  • MAE Diastólica: {metrics['mae_dia']:.2f} mmHg")
        print(f"  • R² Sistólica: {metrics['r2_sys']:.3f}")
        print(f"  • R² Diastólica: {metrics['r2_dia']:.3f}")
        
    except Exception as e:
        print(f"\nERROR DURANTE EL ENTRENAMIENTO: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)