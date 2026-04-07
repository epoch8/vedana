FROM python:3.12

ARG PROJECT_SRC=apps/stell
ARG JIMS_APP=stell_core.app
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
RUN --mount=type=cache,target=/root/.cache uv sync
