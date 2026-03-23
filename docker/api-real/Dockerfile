FROM python:3.12-slim

WORKDIR /app

# Install tobira with serving + llm (Ollama) dependencies
COPY python/ /tmp/tobira-python/
RUN pip install --no-cache-dir "/tmp/tobira-python[serving,llm]" && \
    rm -rf /tmp/tobira-python

COPY docker/api-real/config.toml /app/config.toml

EXPOSE 8000

CMD ["tobira", "serve", "--config", "/app/config.toml", "--host", "0.0.0.0", "--port", "8000"]
