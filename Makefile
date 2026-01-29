
build_jims_backoffice:
	uv build libs/jims-backoffice

build_jims_core:
	uv build libs/jims-core

build_jims_telegram:
	uv build libs/jims-telegram

build_jims_tui:
	uv build libs/jims-tui

build_vedana_core:
	uv build libs/vedana-core

build_vedana_etl:
	uv build libs/vedana-etl

build_vedana_backoffice:
	uv build libs/vedana-backoffice

build: build_jims_backoffice build_jims_core build_jims_telegram build_jims_tui build_vedana_core build_vedana_etl build_vedana_backoffice

build-vedana-project:
	cd apps/vedana && make build

publish:
	UV_PUBLISH_USERNAME="oauth2accesstoken" UV_PUBLISH_PASSWORD="$$(gcloud auth print-access-token)" uv publish

clean:
	rm -rf dist/
