locals {
  gdrive_enabled = var.gdrive_config != null

  gdrive_env = local.gdrive_enabled ? {
    GDRIVE_FOLDER_ID=var.gdrive_config.folder_id
    GDRIVE_PATH_TO_SA_JSON="/secrets/${local.slug}-gdrive-sa/sa_creds.json"
  } : {}
  gdrive_volumes = local.gdrive_enabled ? [{
    name = "gdrive-sa"
    secret = {
      secretName = kubernetes_secret_v1.gdrive_sa[0].metadata[0].name
    }
    mountPath = "/secrets/${local.slug}-gdrive-sa"
  }] : []
}

resource "kubernetes_secret_v1" "gdrive_sa" {
  count = local.gdrive_enabled ? 1 : 0

  metadata {
    name      = "${local.slug}-gdrive-gcp-sa"
    namespace = var.k8s_namespace
  }

  data = {
    "sa_creds.json" = var.gdrive_config.gcp_sa_json
  }
}