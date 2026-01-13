FROM python:3.8-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl -y

RUN mkdir -p ~/.paddleocr

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]