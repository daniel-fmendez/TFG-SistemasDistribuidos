FROM python:3.11-slim

WORKDIR /app
COPY master.py .

RUN pip install flask

EXPOSE 5000

CMD ["python", "master.py"]