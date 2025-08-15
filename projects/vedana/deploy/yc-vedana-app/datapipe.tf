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
  name       = "${local.project_underscore}_${var.environment}_etl"
  password   = random_string.etl_db_password.result
  conn_limit = 5

  lifecycle {
    ignore_changes = [
      name,
      password,
    ]
  }
}

resource "yandex_mdb_postgresql_database" "etl" {
  cluster_id = var.yc_mdb_cluster_id
  name       = "${local.project_underscore}_${var.environment}_etl"
  owner      = yandex_mdb_postgresql_user.etl.name

  lifecycle {
    ignore_changes = [
      name,
    ]
  }
}

module "authentik_datapipe_auth" {
  source = "../../../../../chatbot-terraform/modules/yc-k8s-authentik-app-auth/"

  project = var.project

  app_slug   = "${local.slug}-datapipe"
  app_name   = "Datapipe ${var.environment} ${var.project}"
  app_domain = local.datapipe_domain

  authentik_group_ids = var.authentik_group_ids
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
      repository: ${local.image_repository}
      tag: ${local.image_tag}

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
        %{~for key, value in module.authentik_datapipe_auth.ingress_annotations~}
        ${key}: ${value}
        %{~endfor~}
      tls:
        - secretName: ${local.datapipe_domain}-tls
          hosts:
            - ${local.datapipe_domain}
    EOF
  ]
}

resource "helm_release" "datapipe_all" {
  name      = "${local.slug}-datapipe-all"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-cronjob"
  version    = "0.1.3"
  # chart = "${path.module}/../../../../../helm-charts/charts/simple-cronjob/"


  values = [
    jsonencode({
      image = {
        repository = local.image_repository
        tag        = local.image_tag
      }
      imagePullSecrets = [{ name = var.image_pull_secrets }]

      schedule = "0 0 * * *"

      volumes = local.llm_volumes

      command = [
        "datapipe",
        "step",
        "run",
      ]
      resources = {
        requests = {
          cpu    = "1"
          memory = "1Gi"
        }
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
      }

      env = local.datapipe_env
    }),
  ]
}
