# syntax=docker/dockerfile:1
FROM apache/airflow:3.1.6-python3.12

USER root

# Installation of java (required for spark)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         default-jdk \
         libgomp1 \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# Installation of python libraries
RUN pip install --no-cache-dir \
    pyspark==4.1.1 \
    pandas==2.1.4 \
    scikit-learn==1.5.2 \
    xgboost==3.2.0 \
    catboost==1.2.10 \
    lightgbm==4.6.0 \
    matplotlib==3.10.9 \
    seaborn==0.13.2 \
    pillow==12.2.0