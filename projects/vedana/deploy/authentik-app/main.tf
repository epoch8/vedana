terraform {
  required_providers {
    authentik = {
      source = "goauthentik/authentik"
      version = ">=2025.6.0"
    }
  }
}

###

variable "slug" {
  type = string
}

variable "base_domain" {
  type = string
}

variable "authentik_host" {
  type    = string
  default = "https://authentik.chatbot.epoch8.co"
}

variable "authentik_group_ids" {
  type = list(string)
  default = []
}

variable "authentik_service_connection_kubernetes_name" {
  type    = string
  default = "Local Kubernetes Cluster"
}

variable "authentik_authorization_flow_slug" {
  type    = string
  default = "default-provider-authorization-implicit-consent"
}

variable "authentik_invalidation_flow_slug" {
  type    = string
  default = "default-provider-invalidation-flow"
}

###

data "authentik_flow" "default_authorization_flow" {
  slug = var.authentik_authorization_flow_slug
}

data "authentik_flow" "default_invalidation_flow" {
  slug = var.authentik_invalidation_flow_slug
}

resource "authentik_group" "project" {
  name  = "vedana-${var.slug}"
  users = []

  lifecycle {
    ignore_changes = [
      users,
    ]
  }
}

resource "authentik_provider_proxy" "project" {
  name = "${var.slug}-proxy"

  mode = "forward_domain"

  external_host = "https://auth.${var.base_domain}"
  cookie_domain = "${var.base_domain}"

  authorization_flow = data.authentik_flow.default_authorization_flow.id
  invalidation_flow  = data.authentik_flow.default_invalidation_flow.id
}

resource "authentik_application" "project" {
  name = var.slug
  slug = var.slug

  group = var.slug

  protocol_provider = authentik_provider_proxy.project.id
}

resource "authentik_policy_binding" "project" {
  target = authentik_application.project.uuid
  group  = authentik_group.project.id
  order  = 0
}

data "authentik_group" "internal" {
  name = "Epoch8 AI Team"
}

resource "authentik_policy_binding" "project_groups" {
  for_each = toset(var.authentik_group_ids)

  target = authentik_application.project.uuid
  group  = each.value
  order  = 0
}

resource "authentik_policy_binding" "internal" {
  target = authentik_application.project.uuid
  group  = data.authentik_group.internal.id
  order  = 0
}

data "authentik_service_connection_kubernetes" "default" {
  name = var.authentik_service_connection_kubernetes_name
}

resource "authentik_outpost" "project" {
  name = var.slug

  type = "proxy"

  service_connection = data.authentik_service_connection_kubernetes.default.id

  protocol_providers = [
    authentik_provider_proxy.project.id
  ]

  config = jsonencode({
    authentik_host                = var.authentik_host
    kubernetes_namespace          = "authentik"
    kubernetes_ingress_class_name = "nginx"
    kubernetes_ingress_annotations = {
      "cert-manager.io/cluster-issuer" = "letsencrypt"
    }
    kubernetes_ingress_secret_name = "${var.slug}-outpost-tls"
  })
}

output "ingress_annotations" {
  value = {
    "nginx.ingress.kubernetes.io/auth-signin"           = "https://auth.${var.base_domain}/outpost.goauthentik.io/start?rd=$scheme://$http_host$escaped_request_uri"
    "nginx.ingress.kubernetes.io/auth-url"              = "http://ak-outpost-${authentik_outpost.project.name}.authentik.svc.cluster.local:9000/outpost.goauthentik.io/auth/nginx"
    "nginx.ingress.kubernetes.io/auth-response-headers" = "Set-Cookie,X-authentik-username,X-authentik-groups,X-authentik-entitlements,X-authentik-email,X-authentik-name,X-authentik-uid"
    "nginx.ingress.kubernetes.io/auth-snippet"          = "proxy_set_header X-Forwarded-Host $http_host;"
  }
}
