FROM python:3.11-slim

WORKDIR /app
COPY master/requirements.txt . 
RUN pip install -r requirements.txt
ENV PYTHONPATH="/app:/app/proto"

COPY shared/ .
COPY master/ .

CMD ["python", "master.py"]