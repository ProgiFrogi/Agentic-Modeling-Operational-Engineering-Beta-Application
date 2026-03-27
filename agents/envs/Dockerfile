FROM python:3.10-slim

# Устанавливаем необходимые библиотеки
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    tqdm \
    jupyter

WORKDIR /workspace
CMD ["python"]