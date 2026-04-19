FROM python:3.11-slim

WORKDIR /app
COPY worker/requirements.txt . 
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app:/app/proto"

COPY shared/ .
COPY worker/ .

CMD ["python", "worker.py"]