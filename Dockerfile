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

# Installation of python libraries.
# Each library has its own RUN layer, so adding/changing one doesn't invalidate the rest.
# Each also mounts a persistent pip cache (survives even when a layer must rebuild),
# so re-downloading from the internet only happens for genuinely new/changed packages.
RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install pyspark==4.1.1

RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install pandas==2.1.4

RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install scikit-learn==1.5.2

RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install xgboost==3.2.0

RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install catboost==1.2.10

RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install lightgbm==4.6.0

RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install matplotlib==3.10.9

RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install seaborn==0.13.2

RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install pillow==12.2.0

# Optuna is used inside the training tasks (hyperparameter tuning), so it belongs in this
# same image as airflow-worker. Installed against Airflow's official constraints file so
# it can't silently upgrade/downgrade a dependency that Airflow itself relies on.
RUN --mount=type=cache,target=/home/airflow/.cache/pip,id=pip-cache,uid=50000,gid=0 \
    pip install --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.6/constraints-3.12.txt" \
    optuna==4.9.0