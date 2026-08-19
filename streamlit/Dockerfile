# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Own cache mount, separate from the Airflow image's pip cache above
RUN --mount=type=cache,target=/root/.cache/pip,id=streamlit-pip-cache \
    pip install streamlit==1.60.0

# Copy your Streamlit app code (place app.py etc. in this streamlit/ folder)
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
