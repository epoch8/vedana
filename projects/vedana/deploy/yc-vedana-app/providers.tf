terraform {
  required_providers {
    helm       = {}
    kubernetes = {}
    yandex = {
      source = "yandex-cloud/yandex"
    }
  }
}
