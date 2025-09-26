FROM python:3.12 AS base

ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache pip install uv

ADD --chmod=755 https://github.com/just-containers/s6-overlay/releases/download/v3.1.6.2/s6-overlay-noarch.tar.xz /tmp
ADD --chmod=755 https://github.com/just-containers/s6-overlay/releases/download/v3.1.6.2/s6-overlay-x86_64.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz \
    && tar -C / -Jxpf /tmp/s6-overlay-x86_64.tar.xz \
    && rm -f /tmp/s6-overlay-*.tar.xz

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
COPY libs/vedana-backoffice/deploy/single-port-container/services.d/ /etc/services.d/
COPY libs/vedana-backoffice/deploy/single-port-container/Caddyfile /etc/caddy/Caddyfile

# Make service scripts executable
RUN chmod +x /etc/services.d/*/run
