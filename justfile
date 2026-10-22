build:
    docker build -t adaroute .

test:
    docker run --rm --add-host=host.docker.internal:host-gateway -e UI_BASE_URL=http://host.docker.internal:8080 -v "{{justfile_directory()}}:/app" -w /app adaroute pytest -v tests/ adaroute/experiments/test_integration.py

test-local:
    pytest -v tests/ adaroute/experiments/test_integration.py

lint:
    docker run --rm -v "{{justfile_directory()}}:/app" -w /app adaroute ruff check .

lint-local:
    ruff check .

format:
    docker run --rm -v "{{justfile_directory()}}:/app" -w /app adaroute ruff format .

format-local:
    ruff format .

dev:
    pip install -e .[dev]

benchmark:
    docker run --rm -v "{{justfile_directory()}}:/app" -w /app adaroute python -m adaroute.experiments.runner

benchmark-event:
    docker run --rm -v "{{justfile_directory()}}:/app" -w /app adaroute python -m adaroute.experiments.event_runner

benchmark-event-local:
    python -m adaroute.experiments.event_runner

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
