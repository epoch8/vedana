FROM python:3.12

ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache pip install uv

COPY pyproject.toml uv.lock /app/
COPY libs/ /app/libs/

WORKDIR /app/projects/vedana

# Copy dependency files, startup and migration scripts
COPY projects/vedana/pyproject.toml projects/vedana/uv.lock ./
RUN --mount=type=cache,target=/root/.cache uv sync

ENV DATAPIPE_PIPELINE=vedana_etl.app
ENV JIMS_APP=vedana_core.app

COPY projects/vedana/ ./
