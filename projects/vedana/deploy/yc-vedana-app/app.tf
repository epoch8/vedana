locals {
  project_underscore = replace(var.project, "-", "_")

  current_version = jsondecode(file("${path.module}/current_version.json"))

  image_repository = var.image_repository != null ? var.image_repository : local.current_version.repository
  image_tag        = var.image_tag != null ? var.image_tag : local.current_version.tag
}


resource "random_string" "demo_password" {
  length  = 16
  special = false
}


locals {
  demo_domain       = "demo.${var.project}.${var.base_domain}"
  backoffice_domain = "backoffice.${var.project}.${var.base_domain}"

  vedana_env = {
    APP_USER = "admin"
    APP_PWD  = random_string.demo_password.result

    JIMS_DB_CONN_URI = "postgresql://${yandex_mdb_postgresql_user.jims.name}:${random_string.db.result}@${data.yandex_mdb_postgresql_cluster.db_cluster.host.0.fqdn}:6432/${yandex_mdb_postgresql_database.jims.name}"

    MEMGRAPH_URI  = module.memgraph.config.local_uri
    MEMGRAPH_USER = module.memgraph.config.user
    MEMGRAPH_PWD  = module.memgraph.config.password

    SENTRY_DSN         = var.sentry_dsn
    SENTRY_ENVIRONMENT = "${var.project}-${var.environment}"

    OPENAI_BASE_URL = "https://oai-proxy-hzkr3iwwhq-ew.a.run.app/v1/"
    OPENAI_API_KEY  = "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"
    EMBEDDINGS_DIM  = 1024

    GRIST_SERVER_URL        = var.grist.server_url
    GRIST_API_KEY           = var.grist.api_key
    GRIST_DATA_MODEL_DOC_ID = var.grist.data_model_doc_id
    GRIST_DATA_DOC_ID       = var.grist.data_doc_id

    TELEGRAM_BOT_TOKEN = var.telegram_bot_token
    MODEL              = "gpt-4.1-mini"
  }

  common_values = <<EOF
  image:
    repository: ${local.image_repository}
    tag: ${local.image_tag}

  %{~if var.image_pull_secrets != null~}
  imagePullSecrets:
    - name: ${var.image_pull_secrets}
  %{~endif~}

  env:
  %{~for key, value in local.vedana_env~}
    - name: ${key}
      value: "${value}"
  %{~endfor~}

  livenessProbe:
    httpGet:
      path: /healthz
      port: http
      scheme: HTTP
    initialDelaySeconds: 10
    timeoutSeconds: 1
    periodSeconds: 10
    successThreshold: 1
    failureThreshold: 3

  readinessProbe:
    httpGet:
      path: /healthz
      port: http
      scheme: HTTP
    initialDelaySeconds: 10
    timeoutSeconds: 1
    periodSeconds: 10
    successThreshold: 1
    failureThreshold: 3
  EOF
}

resource "helm_release" "demo" {
  name      = "${var.project}-demo"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-app"
  version    = "0.16.1"

  values = [
    local.common_values,
    <<EOF
    command:
      - uv
      - run
      - python
      - -m
      - vedana.gradio_app

    resources:
      requests:
        cpu: "200m"
        memory: "256Mi"
      limits:
        cpu: "500m"
        memory: "512Mi"

    port: 7860

    probe:
      path: "/healthz"

    initJob:
      enabled: true
      command:
        - uv
        - run
        - alembic
        - upgrade
        - head

    domain: "${local.demo_domain}"
    ingress:
      enabled: true
      nginx: true
      annotations:
        cert-manager.io/cluster-issuer: letsencrypt
        nginx.ingress.kubernetes.io/proxy-body-size: "0"
        nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
        nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
      tls:
        - secretName: ${local.demo_domain}-tls
          hosts:
            - ${local.demo_domain}
    EOF
  ]

  lifecycle {
    ignore_changes = [
      name,
    ]
  }
}

resource "helm_release" "backoffice" {
  name      = "${var.project}-backoffice"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-app"
  version    = "0.16.1"

  values = [
    local.common_values,
    <<EOF
    command:
      - uv
      - run
      - jims-backoffice

    resources:
      requests:
        cpu: "200m"
        memory: "256Mi"
      limits:
        cpu: "500m"
        memory: "512Mi"

    port: 8000

    probe:
      path: "/"

    domain: "${local.backoffice_domain}"
    ingress:
      enabled: true
      nginx: true
      annotations:
        cert-manager.io/cluster-issuer: letsencrypt
        nginx.ingress.kubernetes.io/proxy-body-size: "0"
        nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
        nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
      tls:
        - secretName: ${local.backoffice_domain}-tls
          hosts:
            - ${local.backoffice_domain}
    EOF
  ]

  lifecycle {
    ignore_changes = [
      name,
    ]
  }
}

resource "helm_release" "tg" {
  name      = "${var.project}-tg"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-app"
  version    = "0.16.1"

  values = [
    local.common_values,
    <<EOF
    command:
      - uv
      - run
      - python
      - -m
      - vedana.tg_app

    resources:
      requests:
        cpu: "200m"
        memory: "256Mi"
      limits:
        cpu: "500m"
        memory: "512Mi"

    port: 8000

    probe:
      path: "/healthz"
    EOF
  ]

  lifecycle {
    ignore_changes = [
      name,
    ]
  }
}

# resource "helm_release" "rag_tg_jims" {
#   count = var.tg_jims.enabled ? 1 : 0

#   name      = "${var.name}-tg-jims"
#   namespace = var.kubernetes_namespace

#   repository = "https://epoch8.github.io/helm-charts/"
#   chart      = "simple-app"
#   version    = "0.15.2"

#   values = [
#     templatefile(
#       "${path.module}/tg_jims_values.yaml",
#       {
#         image              = var.rag_image
#         version            = var.rag_version
#         image_pull_secrets = var.image_pull_secrets
#         resources          = var.tg_jims.resources
#         envs               = var.envs
#         size               = var.size
#         host               = "tg-jims-${var.host}"
#       }
#     )
#   ]
# }


output "config" {
  value = {
    demo = {
      uri      = "http://${local.demo_domain}"
      user     = "admin"
      password = random_string.demo_password.result
    }
  }
}
