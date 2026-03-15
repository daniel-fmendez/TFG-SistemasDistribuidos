FROM python:3.11-slim

WORKDIR /app
COPY deployer/requirements.txt . 
RUN pip install -r requirements.txt

COPY shared/ .
COPY deployer/ .


CMD ["python", "deployer.py"]