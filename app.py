from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

autorizado = False
ultima_estimacion = {"sys": "---", "dia": "---"}

ventana_ir = []
ventana_red = []
MUESTRAS = 10

@app.route("/")
def index():
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def registrar_datos():
    global ultima_estimacion, autorizado

    data = request.get_json()
    hr = int(data.get("hr", 0))
    ir = int(data.get("ir", 0))
    red = int(data.get("red", 0))

    ventana_ir.append(ir)
    ventana_red.append(red)

    if len(ventana_ir) > MUESTRAS:
        ventana_ir.pop(0)
    if len(ventana_red) > MUESTRAS:
        ventana_red.pop(0)

    spo2 = 0
    if len(ventana_ir) == MUESTRAS and len(ventana_red) == MUESTRAS:
        dc_ir = np.mean(ventana_ir)
        dc_red = np.mean(ventana_red)
        ac_ir = np.mean(np.abs(np.array(ventana_ir) - dc_ir))
        ac_red = np.mean(np.abs(np.array(ventana_red) - dc_red))

        if ac_ir > 0 and ac_red > 0:
            ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
            spo2 = max(70, min(100, 110 - 25 * ratio))

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
