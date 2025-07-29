REPO=ghcr.io/epoch8/ai-assistants/vedana
VERSION=$(shell uv version --short | sed 's/\+/-/')

IMAGE=${REPO}:${VERSION}

TAG="vedana/${VERSION}"

build:
	@git diff-index --quiet HEAD || (echo "Error: There are uncommitted changes" && exit 1)

	echo "{\"repository\": \"${REPO}\", \"tag\": \"${VERSION}\"}" > deploy/yc-vedana-app/current_version.json
	git add deploy/yc-vedana-app/current_version.json
	git commit -m "Update vedana version to ${VERSION}"

	docker build -t ${IMAGE} -f ../vedana/Dockerfile ..
	git tag "${TAG}"

upload:
	git push
	git push origin "${TAG}"
	docker push ${IMAGE}
