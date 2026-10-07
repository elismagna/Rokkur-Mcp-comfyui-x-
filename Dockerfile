FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY alembic.ini ./
COPY migrations ./migrations
COPY config ./config
COPY workflows ./workflows

RUN useradd --create-home --uid 1000 studio && mkdir -p /app/data && chown studio /app/data
USER studio

ENV PYTHONUNBUFFERED=1 STUDIO_CONFIG_DIR=/app/config STUDIO_WORKFLOWS_DIR=/app/workflows
EXPOSE 8400
CMD ["rokkur-studio", "api", "--host", "0.0.0.0", "--port", "8400"]
