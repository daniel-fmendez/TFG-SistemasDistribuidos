FROM python:3.11-slim

WORKDIR /app
COPY deployer/requirements.txt . 
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY shared/ .
COPY deployer/ .


CMD ["python", "deployer.py"]