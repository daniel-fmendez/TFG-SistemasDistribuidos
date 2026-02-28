FROM python:3.11-slim

WORKDIR /app
COPY worker/requirements.txt . 
RUN pip install -r requirements.txt
ENV PYTHONPATH="/app:/app/proto"

COPY shared/ .
COPY worker/ .

CMD ["python", "worker.py"]