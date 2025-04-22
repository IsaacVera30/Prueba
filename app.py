from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

autorizado = False
registrando = False
ultima_estimacion = {"sys": "---", "dia": "---"}

@app.route("/")
def index():
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def registrar_datos():
    global autorizado, ultima_estimacion, registrando
    data = request.get_json()
    hr = int(data.get("hr", 0))
    spo2 = int(data.get("spo2", 0))
    ir = int(data.get("ir", 0))
    red = int(data.get("red", 0))

    entrada = pd.DataFrame([[hr, spo2]], columns=["hr", "spo2"])
    sys = modelo_sys.predict(entrada)[0]
    dia = modelo_dia.predict(entrada)[0]

    ultima_estimacion = {"sys": f"{sys:.2f}", "dia": f"{dia:.2f}"}

    if autorizado:
        with open("registro_sensor_entrenamiento.csv", "a") as f:
            f.write(f"{hr},{spo2},{ir},{red},{sys:.2f},{dia:.2f}\n")

    return jsonify({"sys": round(sys, 2), "dia": round(dia, 2)})

@app.route("/api/autorizacion", methods=["GET"])
def estado_autorizacion():
    return jsonify({"autorizado": autorizado})

@app.route("/api/autorizar", methods=["GET"])
def autorizar_registro():
    global autorizado
    autorizado = True
    return jsonify({"mensaje": "✅ Registro autorizado"})

@app.route("/api/detener", methods=["GET"])
def detener_registro():
    global autorizado
    autorizado = False
    return jsonify({"mensaje": "⛔ Registro detenido"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
