import socket
import time 
import requests
import threading
import os
#from flask import Flask

MASTER_URL = os.getenv('MASTER_URL', 'http://master-service:5000')

#app = Flask(__name__)
iteration = 0

#@app.route("/status")
def status():
    return f"Worker corriendo en {socket.gethostname()}, iteración {iteration}"

def run_worker():
    global iteration
    worker_id = socket.gethostname()
    while True:
        iteration += 1

        metrica = {
            "worker_id": worker_id,
            "iteration": iteration,
            "loss": 1.0 / (iteration + 1)
        }
        
        try:
            response = requests.post(f"{MASTER_URL}/report", json=metrica)
        except Exception as e:
            print(f"Error enviando metrica: {e}")

        print(f"\tIteración {iteration} - Simulando entrenamiento...")
        time.sleep(5)

if __name__ == "__main__":
    hostname = socket.gethostname()
    print(f"Worker iniciado en {hostname}", flush=True)

    #worker_thread = threading.Thread(target=run_worker, args=(hostname,), daemon=True)
    #worker_thread.start()
    
    #print("Iniciando servidor Flask en puerto 5000...", flush=True)
    #app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    run_worker()