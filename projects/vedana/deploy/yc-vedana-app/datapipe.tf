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
      value = local.db_conn_uri
    }
  ])
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
      env = local.datapipe_env
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

resource "helm_release" "datapipe_regular" {
  name      = "${local.slug}-datapipe-regular"
  namespace = var.k8s_namespace

  repository = "https://epoch8.github.io/helm-charts/"
  chart      = "simple-cronjob"
  version    = "0.1.3"
  # chart = "${path.module}/../../../../../helm-charts/charts/simple-cronjob/"


  values = [
    jsonencode(merge({
      image = {
        repository = local.image_repository
        tag        = local.image_tag
      }

      schedule = "0 0 * * *"

      volumes = concat(local.llm_volumes, local.gdrive_volumes)

      command = [
        "datapipe",
        "step",
        "--labels=flow=regular",
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
      },
      var.image_pull_secrets != null ? {
        imagePullSecrets = [{ name = var.image_pull_secrets }]
      } : {},
    ))
  ]
}
