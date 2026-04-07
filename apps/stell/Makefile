BASE_VERSION=$(shell uv version --short | sed 's/[\+\/]/-/g')

BRANCH=$(shell git rev-parse --abbrev-ref HEAD | sed 's/[\+\/]/-/g')
GIT_HASH=$(shell git rev-parse --short HEAD)

REPO=ghcr.io/epoch8/vedana/stell

ifeq ($(BRANCH),master)
	VERSION=$(BASE_VERSION)
else
	VERSION=$(BASE_VERSION)-$(BRANCH)-$(GIT_HASH)
endif

IMAGE=$(REPO):$(VERSION)

build:
	docker build --platform linux/amd64 -t ${IMAGE} -f Dockerfile ../..

docker-run-it:
	docker run --rm -it ${IMAGE} bash

upload:
	docker push ${IMAGE}
