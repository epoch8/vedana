data "authentik_flow" "default_authorization_flow" {
  slug = "default-provider-authorization-implicit-consent"
}

data "authentik_flow" "default_invalidation_flow" {
  slug = "default-provider-invalidation-flow"
}

resource "authentik_group" "project" {
  count = var.authentik_enabled ? 1 : 0

  name  = "vedana-${var.project}-${var.environment}"
  users = []

  lifecycle {
    ignore_changes = [
      users,
    ]
  }
}

resource "authentik_provider_proxy" "project" {
  count = var.authentik_enabled ? 1 : 0

  name = "${local.slug}-proxy"

  mode = "forward_domain"

  external_host = "https://auth.${local.slug}.${var.base_domain}"
  cookie_domain = "${local.slug}.${var.base_domain}"

  authorization_flow = data.authentik_flow.default_authorization_flow.id
  invalidation_flow  = data.authentik_flow.default_invalidation_flow.id
}

resource "authentik_application" "project" {
  count = var.authentik_enabled ? 1 : 0

  name = local.slug
  slug = local.slug

  group = local.slug

  protocol_provider = authentik_provider_proxy.project[0].id
}

resource "authentik_policy_binding" "project" {
  count = var.authentik_enabled ? 1 : 0

  target = authentik_application.project[0].uuid
  group  = authentik_group.project[0].id
  order  = 0
}

data "authentik_group" "internal" {
  count = var.authentik_enabled ? 1 : 0

  name = "Epoch8 AI Team"
}

resource "authentik_policy_binding" "internal" {
  count = var.authentik_enabled ? 1 : 0

  target = authentik_application.project[0].uuid
  group  = data.authentik_group.internal[0].id
  order  = 0
}

data "authentik_service_connection_kubernetes" "default" {
  count = var.authentik_enabled ? 1 : 0

  name = "Local Kubernetes Cluster"
}

resource "authentik_outpost" "project" {
  count = var.authentik_enabled ? 1 : 0

  name = local.slug

  type = "proxy"

  service_connection = data.authentik_service_connection_kubernetes.default[0].id

  protocol_providers = [
    authentik_provider_proxy.project[0].id
  ]

  config = jsonencode({
    authentik_host                = "https://authentik.chatbot.epoch8.co"
    kubernetes_namespace          = "authentik"
    kubernetes_ingress_class_name = "nginx"
    kubernetes_ingress_annotations = {
      "cert-manager.io/cluster-issuer" = "letsencrypt"
    }
    kubernetes_ingress_secret_name = "${local.slug}-outpost-tls"
  })
}

locals {
  authentik_ingress_annotations = var.authentik_enabled ? {
    "nginx.ingress.kubernetes.io/auth-signin"           = "https://auth.${local.slug}.${var.base_domain}/outpost.goauthentik.io/start?rd=$scheme://$http_host$escaped_request_uri"
    "nginx.ingress.kubernetes.io/auth-url"              = "http://ak-outpost-${authentik_outpost.project[0].name}.authentik.svc.cluster.local:9000/outpost.goauthentik.io/auth/nginx"
    "nginx.ingress.kubernetes.io/auth-response-headers" = "Set-Cookie,X-authentik-username,X-authentik-groups,X-authentik-entitlements,X-authentik-email,X-authentik-name,X-authentik-uid"
    "nginx.ingress.kubernetes.io/auth-snippet"          = "proxy_set_header X-Forwarded-Host $http_host;"
  } : {}
}
