ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

WORKDIR /app

ARG INSTALL_MODEL_RUNTIMES=false
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_MODEL_RUNTIMES" = "true" ]; then \
        pip install --no-cache-dir --index-url "$PYTORCH_INDEX_URL" torch torchvision \
        && pip install --no-cache-dir ultralytics rfdetr; \
    fi

COPY src ./src

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
