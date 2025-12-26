# Vedana

## Setup
1. `uv sync`
2. Для запуска скриптов - `uv run python app/update_graph_db.py` и т.д.

## Backoffice
...

## Kubernetes Job Execution
### Prerequisites
- kind (Kubernetes in Docker) installed

### Setup

```bash
# Build Vedana image for cluster
docker-compose -f projects/vedana/docker-compose.yml build

# Create local K8s cluster
cd projects/vedana
projects/vedana/scripts/setup-kind.sh

docker-compose -f projects/vedana/docker-compose.yml -f up --build -d vedana-backoffice
```

### Test ETL Job Execution

1. Open backoffice UI
2. Navigate to ETL page
3. Select steps
4. Select Execution mode "k8s"
5. Click "Run Selected"
6. Watch logs stream from Job in real-time or open logs from Jobs card

