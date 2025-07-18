REPO=ghcr.io/epoch8/ai-assistants/vedana
VERSION=$(shell uv version --short | sed 's/\+/-/')

IMAGE=${REPO}:${VERSION}

build:
	echo "{\"repository\": \"${REPO}\", \"tag\": \"${VERSION}\"}" > deploy/yc-vedana-app/current_version.json
	docker build -t ${IMAGE} -f ../vedana/Dockerfile ..

upload:
	docker push ${IMAGE}
