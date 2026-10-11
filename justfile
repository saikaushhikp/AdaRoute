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

benchmark:
    docker run --rm -v "{{justfile_directory()}}/results:/app/results" adaroute python -m adaroute.experiments.runner

ui-build:
    docker build -t adaroute-ui ./ui

ui-test:
    cd ui && npm run test

up:
    docker compose up --build -d

down:
    docker compose down

cli:
    python3 ui/cli/main.py
