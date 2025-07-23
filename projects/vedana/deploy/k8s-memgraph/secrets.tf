resource "kubernetes_secret_v1" "memgraph_secrets" {
  metadata {
    name      = "${var.name}-secrets"
    namespace = var.kubernetes_namespace
  }

  data = {
    USER     = var.user
    PASSWORD = var.password
  }
  type = "Opaque"

  lifecycle {
    ignore_changes = [
      metadata[0].namespace,
    ]
  }
}
