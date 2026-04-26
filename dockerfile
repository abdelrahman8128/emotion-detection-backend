FROM python:3.11-slim

RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY app/ app/
COPY static/ static/

COPY models/results/tflite_mlp.tflite          models/results/
COPY models/results/emotion_detection_dual.tflite models/results/
COPY models/results/tflite_model3.tflite        models/results/
COPY models/results/scaler.pkl                  models/results/
COPY models/results/scaler_dual.pkl             models/results/
COPY models/results/label_encoder.pkl           models/results/

RUN mkdir -p uploads

ENV MODEL_DIR=./models/results

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
