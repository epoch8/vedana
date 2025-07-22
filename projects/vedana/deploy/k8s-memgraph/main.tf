locals {
  values_file = var.values_file != "" ? var.values_file : "${path.module}/values.yaml"
}

resource "helm_release" "memgraph" {
  name       = var.name
  namespace  = var.kubernetes_namespace
  repository = "https://memgraph.github.io/helm-charts"
  chart      = "memgraph"
  version    = "0.2.3"
  values = [
    templatefile(
      local.values_file,
      {
        secrets_name    = kubernetes_secret_v1.memgraph_secrets.metadata[0].name
        memgraph_config = var.memgraph_config
        size            = var.size
        resources       = var.resources
      }
    )
  ]

  lifecycle {
    ignore_changes = [
      name,
      namespace,
    ]
  }
}

output "config" {
  value = {
    local_uri = "bolt://${var.name}.${helm_release.memgraph.metadata[0].namespace}.svc.cluster.local:7687"
    user      = var.user
    password  = var.password
  }
}
