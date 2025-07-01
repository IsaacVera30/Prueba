import pandas as pd

# Cargar el archivo sin encabezado
df = pd.read_csv("registro_sensor_entrenamiento.csv", header=None)

# Verificar si hay una sola columna con todos los datos separados por comas
if df.shape[1] == 1:
    # Separar la única columna en varias columnas por coma
    df_split = df[0].str.split(",", expand=True)
else:
    df_split = df  # Ya está dividido correctamente

# Eliminar filas que no tengan 6 columnas completas
df_split = df_split[df_split.apply(lambda row: row.count() == 6, axis=1)]

# Renombrar columnas
df_split.columns = ["hr", "spo2", "ir", "red", "sys", "dia"]

# Guardar archivo limpio
df_split.to_csv("registro_sensor_entrenamiento.csv", index=False)

print("✅ Archivo corregido exitosamente.")
