BASE_VERSION=$(shell uv version --short | sed 's/[\+\/]/-/g')

BRANCH=$(shell git rev-parse --abbrev-ref HEAD | sed 's/[\+\/]/-/g')
GIT_HASH=$(shell git rev-parse --short HEAD)

REPO=cr.yandex/crp0qk8lt84ipugi4pe0/vedana
BASE_IMAGE=${REPO}:$(VERSION)

ifeq ($(BRANCH),master)
	VERSION=$(BASE_VERSION)
else
	VERSION=$(BASE_VERSION)-$(BRANCH)-$(GIT_HASH)
endif

IMAGE=$(REPO):$(VERSION)

build:
	echo "{\"repository\": \"${REPO}\", \"tag\": \"${VERSION}\"}" > deploy/yc-vedana-app/current_version.json
	git add deploy/yc-vedana-app/current_version.json

	docker build -t ${IMAGE} -f Dockerfile ../..

CURRENT_VERSION=$(shell yq -r .tag ./deploy/yc-vedana-app/current_version.json)
CURRENT_IMAGE=$(REPO):$(CURRENT_VERSION)

docker-run-it:
	docker run --rm -it ${CURRENT_IMAGE} bash

git-tag:
	git tag -f "vedana-${CURRENT_VERSION}"
	git push -f origin "vedana-${CURRENT_VERSION}"

upload:
	docker push ${CURRENT_IMAGE}
