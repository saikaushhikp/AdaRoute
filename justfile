build:
    docker build -t adaroute .

test:
    docker run --rm adaroute pytest -v tests/ adaroute/experiments/

lint:
    docker run --rm adaroute ruff check .

format:
    docker run --rm -v "{{justfile_directory()}}:/app" -w /app adaroute ruff format .

dev:
    pip install -e .[dev]
