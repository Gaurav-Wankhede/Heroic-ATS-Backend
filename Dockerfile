FROM python:3.11-slim

WORKDIR /app/backend

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn main:app --port=8000 --host=0.0.0.0
