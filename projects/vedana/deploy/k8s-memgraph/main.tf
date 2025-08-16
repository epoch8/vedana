resource "helm_release" "memgraph" {
  name       = var.name
  namespace  = var.kubernetes_namespace
  repository = "https://memgraph.github.io/helm-charts"
  chart      = "memgraph"
  version    = "0.2.3"
  values = [
    <<EOT
    persistentVolumeClaim:
      storageSize: ${var.size}

    memgraphConfig:
    - "--also-log-to-stderr=true"
    %{ for value in var.memgraph_config}
    - "${value}"
    %{ endfor }

    secrets:
      enabled: true
      name: ${kubernetes_secret_v1.memgraph_secrets.metadata[0].name}

    resources:
      requests:
        cpu: ${var.resources.requests.cpu}
        memory: ${var.resources.requests.memory}
      limits:
        cpu: ${var.resources.limits.cpu}
        memory: ${var.resources.limits.memory}
    EOT

    # templatefile(
    #   local.values_file,
    #   {
    #     secrets_name    = kubernetes_secret_v1.memgraph_secrets.metadata[0].name
    #     memgraph_config = var.memgraph_config
    #     size            = var.size
    #     resources       = var.resources
    #   }
    # )
  ]

  lifecycle {
    ignore_changes = [
      name,
      namespace,
    ]
  }
}

output "local_host" {
  value = "${helm_release.memgraph.name}.${helm_release.memgraph.metadata[0].namespace}.svc.cluster.local"
}

output "local_port" {
  value = "7687"
}

output "config" {
  value = {
    local_uri  = "bolt://${helm_release.memgraph.name}.${helm_release.memgraph.metadata[0].namespace}.svc.cluster.local:7687"
    user       = var.user
    password   = var.password
  }
}
