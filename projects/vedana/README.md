# Graph RAG на Memgraph

## Настройка
1. `uv sync`
2. Для запуска скриптов - `uv run python app/update_graph_db.py` и т.д.

### Environment variables
[.env](.env) файл (расположить здесь же):
```dotenv
OPENAI_API_KEY
GRIST_SERVER
GRIST_API_KEY
GRIST_DATA_MODEL_DOC_ID
GRIST_DATA_DOC_ID
```

## Запуск
1. `docker-compose up` чтобы запустить Gradio-приложение, memgraph и memgraph-lab
2. Если memgraph-lab говорит "Memgraph not detected" - нужно добавить соединение вручную, `host=memgraph, port=7687`
3. Gradio UI будет доступен по адресу [http://localhost:7860/](http://localhost:7860/)

### Обновление данных / графа
Данные в граф загружаются в [update_graph_db](app/update_graph_db.py). 
О сновной источник - Grist (в вебе). Для работы нужно прописать:
- env GRIST_API_KEY
- адрес Grist'а (дефолтный `https://api.getgrist.com`)
- workspace id (можно найти в Settings -> API -> Document ID) - отдельные для датамодели и для данных.

Также возможна загрузка из локальных csv-файлов (по аналогии с GrandCypher-версией демостенда). 
Для этого нужно загрузить csv в директорию `csv_path` (прописать её в [settings.py](settings.py)) и подредачить [update_graph_db](app/update_graph_db.py) - раскомментировать там блок с загрузкой csv.

Другой простой способ загрузки данных, если они уже были в графе - через Import в Memgraph-lab
