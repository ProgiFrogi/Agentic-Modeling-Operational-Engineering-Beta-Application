# Dockerfile
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python библиотеки
RUN pip install --no-cache-dir \
    pandas==2.0.3 \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-learn==1.3.0 \
    scipy==1.11.1 \
    xgboost==1.7.6 \
    lightgbm==4.0.0 \
    catboost==1.2.0 \
    requests==2.31.0 \
    jinja2==3.1.2

# Устанавливаем рабочую директорию
WORKDIR /workspace

# Создаем скрипт для выполнения кода
RUN echo '#!/bin/bash\npython -c "$1"' > /usr/local/bin/run-python && \
    chmod +x /usr/local/bin/run-python

CMD ["/bin/bash"]