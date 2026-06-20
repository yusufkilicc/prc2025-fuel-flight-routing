# Self-contained web app (bundled demo data + models).
#   docker build -t prc-fuel-routing .
#   docker run -p 8600:8600 prc-fuel-routing
FROM python:3.11-slim

WORKDIR /app

# Only the web app + engine deps are needed at runtime (not the training stack).
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        pandas==2.3.3 numpy==2.3.4 pyarrow==16.0.0 \
        lightgbm==4.6.0 openap==2.5.0 \
        fastapi==0.136.0 uvicorn==0.44.0 airportsdata==20260315

COPY src/ ./src/
COPY web/ ./web/
COPY models/ ./models/
COPY data/apt.parquet data/flightlist_train.parquet ./data/

ENV HOST=0.0.0.0 PORT=8600
EXPOSE 8600
CMD ["python", "web/web_app.py"]
