FROM python:3.11-slim

WORKDIR /app
COPY dataset/requirements.txt . 
RUN pip install -r requirements.txt

COPY shared/ .
COPY dataset/ .


CMD ["python", "dataset_loader.py"]