FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY mock_server.py .

EXPOSE 8000

CMD ["uvicorn", "mock_server:app", "--host", "0.0.0.0", "--port", "8000"]
