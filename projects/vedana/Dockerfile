FROM python:3.12 AS base

ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache pip install uv

RUN apt-get update && apt-get install -y \
    caddy \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/
COPY libs/ /app/libs/

WORKDIR /app/projects/vedana

# Copy dependency files, startup and migration scripts
COPY projects/vedana/pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache uv sync

ENV DATAPIPE_PIPELINE=vedana_etl.app
ENV JIMS_APP=vedana_core.app
ENV PORT=80

COPY projects/vedana/ ./

#########################################
FROM base AS build

RUN cp .env.ci-cd .env 
# Builds to .web/build/client/
RUN reflex export --loglevel debug --frontend-only --no-zip

#########################################
FROM base AS runtime

COPY --from=build /app/projects/vedana/.web/build/client/ /srv
