FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .[dev]

COPY . .

# Default: run the FastAPI gateway.
# Override with: docker run ... pytest tests/ to run the test suite.
CMD ["uvicorn", "adaroute.api:app", "--host", "0.0.0.0", "--port", "7000"]
