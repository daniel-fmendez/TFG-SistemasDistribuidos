FROM python:3.11-slim

WORKDIR /app
COPY worker.py .

RUN pip install requests

CMD ["python", "worker.py"]