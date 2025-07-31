locals {
  datapipe_domain = "datapipe.${var.base_domain}"

  datapipe_env = concat([
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
}

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

resource "helm_release" "datapipe_api" {
  name      = "${local.slug}-datapipe-api"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-app"
  version    = "0.17.1"

  values = [
    jsonencode({
      env=local.datapipe_env
    }),
    <<EOF
    image:
      repository: ${local.app_image_repository}
      tag: ${local.app_image_tag}

    %{~if var.image_pull_secrets != null~}
    imagePullSecrets:
      - name: ${var.image_pull_secrets}
    %{~endif~}

    command:
      - datapipe
      - api

    resources:
      requests:
        cpu: 0.1
        memory: 256Mi
      limits:
        cpu: 1
        memory: 1Gi

    port: 8000

    probe:
      path: "/"

    initJob:
      enabled: true
      command:
        - alembic
        - upgrade
        - head

    domain: "${local.datapipe_domain}"
    ingress:
      enabled: true
      nginx: true
      annotations:
        cert-manager.io/cluster-issuer: letsencrypt
        nginx.ingress.kubernetes.io/proxy-body-size: "0"
        nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
        nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
        %{~for key, value in var.authentik_ingress_annotations~}
        ${key}: ${value}
        %{~endfor~}
      tls:
        - secretName: ${local.datapipe_domain}-tls
          hosts:
            - ${local.datapipe_domain}
    EOF
  ]
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

      loops = [
        {
          name     = "regular"
          schedule = "0 0 * * *"
          labels = "flow=regular"
          resources = {
            requests = {
              cpu    = "1"
              memory = "512Mi"
            }
            limits = {
              cpu    = "1"
              memory = "1Gi"
            }
          }
        }
      ]
      env = local.datapipe_env
    }),
  ]
}
