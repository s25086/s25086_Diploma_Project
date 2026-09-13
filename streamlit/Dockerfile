# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# curl — wymagany przez healthcheck w docker-compose.yaml
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit==1.60.0 \
    pandas==2.1.4 \
    numpy==1.26.4 \
    lightgbm==4.6.0 \
    scikit-learn==1.5.2 \
    shap==0.46.0 \
    matplotlib==3.10.9

# Copy your Streamlit app code (place app.py etc. in this streamlit/ folder)
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
