FROM python:3.12

WORKDIR /app/vedana

RUN --mount=type=cache,target=/root/.cache pip install uv

# Copy jims packages
COPY jims/packages/ /app/jims/packages/
COPY vedana/packages/ ./packages/
COPY vedana/data/ ./data/

# Copy dependency files, startup and migration scripts
COPY vedana/pyproject.toml vedana/uv.lock ./

RUN --mount=type=cache,target=/root/.cache uv sync --all-packages
