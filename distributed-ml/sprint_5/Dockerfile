FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir grpcio grpcio-tools transformers datasets kubernetes
COPY . .

RUN mkdir -p /root/.cache/huggingface

CMD ["python", "master.py"]