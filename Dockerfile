FROM python:3.12.2-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src
COPY app ./app

RUN python -m pip install --no-cache-dir -e ".[api]"

COPY data/models ./data/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]