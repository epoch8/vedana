FROM python:3.12 AS base

ARG PROJECT_SRC=projects/vedana
ARG JIMS_APP=vedana_core.app
ARG DATAPIPE_PIPELINE=vedana_etl.app

# Convert ARGs to ENV for inheritance to child stages
ENV PROJECT_SRC=${PROJECT_SRC}
ENV JIMS_APP=${JIMS_APP}
ENV DATAPIPE_PIPELINE=${DATAPIPE_PIPELINE}

ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache pip install uv

RUN apt-get update && apt-get install -y \
    caddy \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/
COPY libs/ /app/libs/

WORKDIR /app/${PROJECT_SRC}

# Copy dependency files, startup and migration scripts
COPY ${PROJECT_SRC}/ ./
RUN --mount=type=cache,target=/root/.cache uv sync

#########################################
FROM base AS build

RUN cp .env.ci-cd .env 
# Builds to .web/build/client/
RUN reflex export --loglevel debug --frontend-only --no-zip

#########################################
FROM base AS runtime

COPY --from=build /app/${PROJECT_SRC}/.web/build/client/ /srv
