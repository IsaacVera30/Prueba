from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

# Cargar modelos entrenados
modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

# App Flask
app = Flask(__name__)
autorizado = False
ultima_estimacion = {"sys": "---", "dia": "---"}  # Inicialmente vacíos

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def registrar_datos():
    global autorizado, ultima_estimacion
    try:
        data = request.get_json()
        hr = int(data.get("hr", 0))
        spo2 = int(data.get("spo2", 0))
        ir = int(data.get("ir", 0))
        red = int(data.get("red", 0))

        entrada = pd.DataFrame([[hr, spo2]], columns=["hr", "spo2"])
        sys = modelo_sys.predict(entrada)[0]
        dia = modelo_dia.predict(entrada)[0]

        ultima_estimacion["sys"] = round(sys, 2)
        ultima_estimacion["dia"] = round(dia, 2)

        if autorizado:
            with open("registro_sensor_entrenamiento.csv", "a") as f:
                f.write(f"{hr},{spo2},{ir},{red},{sys:.2f},{dia:.2f}\n")

        return jsonify({
            "mensaje": "✅ Estimación generada",
            "sys": round(sys, 2),
            "dia": round(dia, 2)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route("/api/autorizacion", methods=["GET"])
def obtener_autorizacion():
    return jsonify({"autorizado": autorizado}), 200

@app.route("/api/autorizar", methods=["GET"])
def autorizar_registro():
    global autorizado
    autorizado = True
    return jsonify({"mensaje": "✅ Registro autorizado"}), 200

@app.route("/api/detener", methods=["GET"])
def detener_registro():
    global autorizado
    autorizado = False
    return jsonify({"mensaje": "⛔ Registro detenido"}), 200

@app.route("/api/estimacion", methods=["GET"])
def obtener_estimacion():
    return jsonify(ultima_estimacion), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
