FROM python:3.12

ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache pip install uv

COPY jims/packages/ /app/jims/packages/
COPY vedana/packages/ /app/vedana/packages/

WORKDIR /app/vedana

# Copy dependency files, startup and migration scripts
COPY vedana/pyproject.toml vedana/uv.lock ./
RUN --mount=type=cache,target=/root/.cache uv sync --all-packages

ENV DATAPIPE_PIPELINE=vedana_etl.app
WORKDIR /app/vedana/packages/vedana-etl