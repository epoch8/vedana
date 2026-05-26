FROM python:3.12 AS base

ARG PROJECT_SRC=apps/jims-demo
ARG JIMS_APP=jims_demo.app:app
ARG VERSION=0.0.0

ENV PROJECT_SRC=${PROJECT_SRC}
ENV JIMS_APP=${JIMS_APP}
ENV UV_DYNAMIC_VERSIONING_BYPASS=${VERSION}

ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache pip install uv

RUN apt-get update && apt-get install -y \
    caddy \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/
COPY libs/ /app/libs/

WORKDIR /app/${PROJECT_SRC}

COPY ${PROJECT_SRC}/ ./
RUN cp -r /app/libs/jims-backoffice/assets ./assets
RUN --mount=type=cache,target=/root/.cache uv sync

#########################################
FROM base AS build

RUN cp .env.ci-cd .env
RUN reflex export --loglevel debug --frontend-only --no-zip

#########################################
FROM base AS runtime

COPY --from=build /app/${PROJECT_SRC}/.web/build/client/ /srv
