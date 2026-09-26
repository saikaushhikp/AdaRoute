FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .[dev]

COPY . .

CMD ["pytest", "tests/"]
