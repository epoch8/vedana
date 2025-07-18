resource "random_password" "memgraph_password" {
  length  = 16
  special = false
}

module "memgraph" {
  source = "../k8s-memgraph"

  kubernetes_namespace = var.k8s_namespace
  name                 = "${var.project}-memgraph"
  user                 = var.project
  password             = random_password.memgraph_password.result
  resources = {
    requests = { cpu = "0.1", memory = "2Gi" }
    limits   = { cpu = "1", memory = "3Gi" }
  }
}
