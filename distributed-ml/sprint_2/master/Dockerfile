FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install grpcio grpcio-tools

CMD ["python", "master.py"]