# 0.3.1

* Splitting ETL pipelines into tabs by new pipeline label 
* Improve data views: server-side pagination, custom styling
* Add stats to ETL step / data table cards - last run time, row changes, row counts
* Improve ETL Graph rendering logic
* Refactor backoffice state.py - split into smaller states per-page 

# 0.3.0 - 2025.11.20

* ETL - added monitoring dashboard on the main page
* General - UI improvements: layouts, tags filtering in JIMS thread viewer
* Chat - added total chat cost counter
* ETL - added row count and last run changes in DataTable view

* Add link to Telegram bot if `TELEGRAM_BOT_TOKEN` env is provided

# 0.2.0 - 2025.10.29

* JIMS thread viewer - add feedback processing flow, UI updates

# 0.1.0

Initial commit:
* ETL page - view and select steps / data tables and run Datapipe pipelines
* Chat page - basic chat UI
* JIMS thread viewer - view previous conversations, add feedback
