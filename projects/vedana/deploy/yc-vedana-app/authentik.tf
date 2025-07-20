data "authentik_group" "internal" {
  name = "Epoch8 AI Team"
}

resource "authentik_group" "project" {
  name  = "vedana-${var.project}-${var.environment}"
  users = []

  lifecycle {
    ignore_changes = [
      users,
    ]
  }
}

module "authentik_domain_auth" {
  source = "../authentik-app"

  slug        = "${var.project}-${var.environment}"
  base_domain = "${var.project}-${var.environment}.${var.base_domain}"

  authentik_host                               = var.authentik_host
  authentik_service_connection_kubernetes_name = var.authentik_service_connection_kubernetes_name
  authentik_authorization_flow_slug            = var.authentik_authorization_flow_slug
  authentik_invalidation_flow_slug             = var.authentik_invalidation_flow_slug

  authentik_group_ids = [
    authentik_group.project.id,
    data.authentik_group.internal.id,
  ]
}

locals {
  authentik_ingress_annotations = module.authentik_domain_auth.ingress_annotations
}
