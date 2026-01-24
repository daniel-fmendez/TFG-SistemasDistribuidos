from flask import Flask, request
import datetime

app = Flask(__name__)
metrics = []

@app.route('/report', methods =['POST'])
def recive_metrics():
    data = request.json
    data['timestamp'] = datetime.datetime.now().isoformat()

    metrics.append(data)

    print(f"- Recibido de {data.get('worker_id')}: {data}")
    return {"status": "ok"}

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return {"metrics": metrics}

if __name__ == '__main__':
    # Solo un master
    app.run(host='0.0.0.0', port=5000)