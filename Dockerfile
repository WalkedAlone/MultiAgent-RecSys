# Backend-only image for AgentRec.
# Data, checkpoints, indexes and logs are mounted at runtime; they are not
# copied into the image.
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# libgomp is required by the Faiss CPU wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /bin/bash app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch==2.5.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app pyproject.toml README.md .env.example ./

RUN mkdir -p /app/data /app/checkpoints /app/indexes /app/logs \
    /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache

USER app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
