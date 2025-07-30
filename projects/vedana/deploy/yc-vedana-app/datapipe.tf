resource "random_string" "etl_db_password" {
  length  = 16
  special = false
}

resource "yandex_mdb_postgresql_user" "etl" {
  cluster_id = var.yc_mdb_cluster_id
  name       = "${local.project_underscore}_etl"
  password   = random_string.etl_db_password.result
  conn_limit = 10

  lifecycle {
    ignore_changes = [
      name,
      password,
    ]
  }
}

resource "yandex_mdb_postgresql_database" "etl" {
  cluster_id = var.yc_mdb_cluster_id
  name       = "${local.project_underscore}_etl"
  owner      = yandex_mdb_postgresql_user.etl.name

  lifecycle {
    ignore_changes = [
      name,
    ]
  }
}

resource "helm_release" "datapipe" {
  name      = "${local.slug}-datapipe"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "datapipe"
  version    = "0.3.0"
  # chart = "${path.module}/../../../../../helm-charts/charts/datapipe/"


  values = [
    jsonencode({
      image = {
        repository = local.etl_image_repository
        tag        = local.etl_image_tag
      }
      imagePullSecrets = [{ name = var.image_pull_secrets }]

      initJob : {
        enabled : true
        # workingDir: "/app/vedana/packages/vedana-core"
        command : [
          "uv",
          "run",
          "alembic",
          "upgrade",
          "head"
        ]
      }

      loops = [
        {
          name     = "regular"
          schedule = "0 0 * * *"
          labels = "flow=regular"
          resources = {
            requests = {
              cpu    = "200m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
          }
        }
      ]
      env = concat([
        for key, value in local.vedana_env : {
          name  = key
          value = value
        }
      ], [
        {
          name  = "DB_CONN_URI"
          value = "postgresql://${yandex_mdb_postgresql_user.etl.name}:${random_string.etl_db_password.result}@${data.yandex_mdb_postgresql_cluster.db_cluster.host.0.fqdn}:6432/${yandex_mdb_postgresql_database.etl.name}"
        }
      ])
    }),
  ]
}
