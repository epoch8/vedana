terraform {
  required_providers {
    helm       = {}
    kubernetes = {}
    yandex = {
      source = "yandex-cloud/yandex"
    }
    authentik = {
      source = "goauthentik/authentik"
      version = ">=2025.6.0"
    }
  }
}
