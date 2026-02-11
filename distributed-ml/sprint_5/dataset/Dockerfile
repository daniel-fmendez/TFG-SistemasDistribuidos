FROM python:3.11-slim

WORKDIR /app
RUN pip install datasets transformers
COPY . .

CMD ["python", "dataset_loader.py"]