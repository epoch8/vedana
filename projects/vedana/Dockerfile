FROM python:3.12

WORKDIR /app/vedana

RUN --mount=type=cache,target=/root/.cache pip install uv

# Copy jims packages
COPY jims/packages/ /app/jims/packages/

# Copy dependency files, startup and migration scripts
COPY vedana/pyproject.toml vedana/uv.lock ./

# Install dependencies using uv
# ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN --mount=type=cache,target=/root/.cache uv sync --no-install-project

# Copy application code
COPY vedana/packages/vedana-core/ ./packages/vedana-core/
COPY vedana/packages/vedana-gradio/ ./packages/vedana-gradio/
COPY vedana/packages/vedana-tg/ ./packages/vedana-tg/
COPY vedana/data/ ./data/

RUN --mount=type=cache,target=/root/.cache uv sync --all-packages
