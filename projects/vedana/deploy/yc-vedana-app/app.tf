locals {
  project_underscore = replace(var.project, "-", "_")

  current_version = jsondecode(file("${path.module}/current_version.json"))

  image_repository = var.image_repository != null ? var.image_repository : local.current_version.repository
  image_tag        = var.image_tag != null ? var.image_tag : local.current_version.tag

  slug = "${var.project}-${var.environment}"

  demo_domain       = "demo.${var.base_domain}"
  backoffice_domain = "backoffice.${var.base_domain}"

  tg_enabled = var.telegram_bot_token != null && var.telegram_bot_token != ""

  vedana_env = merge(
    {
      PYTHONUNBUFFERED = "1"

      JIMS_DB_CONN_URI = local.db_conn_uri
      DB_CONN_URI      = local.db_conn_uri

      MEMGRAPH_URI  = module.memgraph.config.local_uri
      MEMGRAPH_USER = module.memgraph.config.user
      MEMGRAPH_PWD  = module.memgraph.config.password

      SENTRY_DSN         = var.sentry_dsn
      SENTRY_ENVIRONMENT = local.slug
      SENTRY_RELEASE     = local.image_tag

      VERSION = local.image_tag

      GRIST_SERVER_URL        = var.grist.server_url
      GRIST_API_KEY           = var.grist.api_key
      GRIST_DATA_MODEL_DOC_ID = var.grist.data_model_doc_id
      GRIST_DATA_DOC_ID       = var.grist.data_doc_id
    },
    var.grist.test_set_doc_id != null ? {
      TEST_ENVIRONMENT      = local.slug
      GRIST_TEST_SET_DOC_ID = var.grist.test_set_doc_id
    } : {},
    local.tg_enabled ? {
      TELEGRAM_BOT_TOKEN = var.telegram_bot_token
    } : {},
    var.enable_api ? {
      API_KEY = random_string.api_key[0].result
    } : {},
    local.llm_env,
    local.gdrive_env,
    var.extra_env,
  )

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

  volumes:
  %{~if local.llm_volumes != []~}
  %{~for volume in local.llm_volumes~}
    - ${jsonencode(volume)}
  %{~endfor~}
  %{~endif~}
  %{~if local.gdrive_volumes != []~}
  %{~for volume in local.gdrive_volumes~}
    - ${jsonencode(volume)}
  %{~endfor~}
  %{~endif~}

  livenessProbe:
    httpGet:
      path: /healthz
      port: http
      scheme: HTTP
    initialDelaySeconds: 10
    timeoutSeconds: 1
    periodSeconds: 10
    successThreshold: 1
    failureThreshold: 10

  readinessProbe:
    httpGet:
      path: /healthz
      port: http
      scheme: HTTP
    initialDelaySeconds: 10
    timeoutSeconds: 1
    periodSeconds: 10
    successThreshold: 1
    failureThreshold: 10
  EOF
}

resource "random_string" "api_key" {
  count = var.enable_api ? 1 : 0

  length  = 32
  special = false
}

###

module "authentik_demo_auth" {
  source = "../../../../../chatbot-terraform/modules/yc-k8s-authentik-app-auth/"

  project = var.project

  app_slug   = "${local.slug}-demo"
  app_name   = "Demo ${var.environment} ${var.project}"
  app_domain = local.demo_domain

  authentik_group_ids = var.authentik_group_ids
}

resource "helm_release" "demo" {
  name      = "${local.slug}-demo"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-app"
  version    = "0.17.1"
  # chart = "${path.module}/../../../../../helm-charts/charts/simple-app/"

  values = [
    local.common_values,
    <<EOF
    command:
      - python
      - -m
      - vedana_gradio.gradio_app

    resources:
      requests:
        cpu: "200m"
        memory: "512Mi"
      limits:
        cpu: "500m"
        memory: "1Gi"

    port: 7860

    probe:
      path: "/healthz"

    initJob:
      enabled: true
      command:
        - alembic
        - upgrade
        - heads

    domain: "${local.demo_domain}"
    ingress:
      enabled: true
      nginx: true
      annotations:
        cert-manager.io/cluster-issuer: letsencrypt
        nginx.ingress.kubernetes.io/proxy-body-size: "0"
        nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
        nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
        %{~for key, value in module.authentik_demo_auth.ingress_annotations~}
        ${key}: ${value}
        %{~endfor~}
      tls:
        - secretName: ${local.demo_domain}-tls
          hosts:
            - ${local.demo_domain}
    EOF
  ]

  lifecycle {
    ignore_changes = [
      name,
      namespace,
    ]
  }
}

module "authentik_backoffice_auth" {
  source = "../../../../../chatbot-terraform/modules/yc-k8s-authentik-app-auth/"

  project = var.project

  app_slug   = "${local.slug}-backoffice"
  app_name   = "Backoffice ${var.environment} ${var.project}"
  app_domain = local.backoffice_domain

  authentik_group_ids = var.authentik_group_ids
}

resource "helm_release" "backoffice" {
  name      = "${local.slug}-backoffice"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-app"
  version    = "0.17.1"

  values = [
    local.common_values,
    <<EOF
    command:
      - vedana-backoffice-with-caddy

    resources:
      requests:
        cpu: "200m"
        memory: "256Mi"
      limits:
        cpu: "500m"
        memory: "512Mi"

    port: 9000

    probe:
      path: "/ping"

    domain: "${local.backoffice_domain}"
    ingress:
      enabled: true
      nginx: true
      annotations:
        cert-manager.io/cluster-issuer: letsencrypt
        nginx.ingress.kubernetes.io/proxy-body-size: "0"
        nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
        nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
        %{~for key, value in module.authentik_backoffice_auth.ingress_annotations~}
        ${key}: ${value}
        %{~endfor~}
      tls:
        - secretName: ${local.backoffice_domain}-tls
          hosts:
            - ${local.backoffice_domain}
    EOF
  ]

  lifecycle {
    ignore_changes = [
      name,
      namespace,
    ]
  }
}

###

resource "helm_release" "tg" {
  count = local.tg_enabled ? 1 : 0

  name      = "${local.slug}-tg"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-app"
  version    = "0.17.1"

  values = [
    local.common_values,
    <<EOF
    command:
      - jims-telegram
      - --enable-sentry
      - --enable-healthcheck
      - --verbose

    resources:
      requests:
        cpu: "200m"
        memory: "512Mi"
      limits:
        cpu: "500m"
        memory: "1Gi"

    port: 8000

    probe:
      path: "/healthz"
    EOF
  ]

  lifecycle {
    ignore_changes = [
      name,
      namespace,
    ]
  }
}

###

resource "helm_release" "api" {
  count = var.enable_api ? 1 : 0

  name      = "${local.slug}-api"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-app"
  version    = "0.17.1"

  values = [
    local.common_values,
    jsonencode({
      port      = 8080
      command   = var.api_command
      resources = var.api_resources
      probe = {
        path = "/healthz"
      }

      domain = "api.${var.base_domain}"

      ingress = {
        enabled = true
        nginx   = true

        annotations = {
          "cert-manager.io/cluster-issuer"                 = "letsencrypt"
          "nginx.ingress.kubernetes.io/proxy-body-size"    = "0"
          "nginx.ingress.kubernetes.io/proxy-read-timeout" = "600"
          "nginx.ingress.kubernetes.io/proxy-send-timeout" = "600"
        }

        tls = [
          {
            secretName = "api.${var.base_domain}-tls"
            hosts      = ["api.${var.base_domain}"]
          }
        ]
      }
    }),
  ]

  lifecycle {
    ignore_changes = [
      name,
      namespace,
    ]
  }
}

###

output "config" {
  value = {
    demo       = "https://${local.demo_domain}"
    backoffice = "https://${local.backoffice_domain}"
    image      = "${local.image_repository}:${local.image_tag}"
  }
}
