locals {
  gcp_sa_enabled = var.llm_config.gcp_sa_json != null

  gcp_llm_env = {
    GOOGLE_APPLICATION_CREDENTIALS = "/secrets/${local.slug}-gcp-sa/gcp-sa.json"
  }
  gcp_llm_volume = local.gcp_sa_enabled ? [{
    name = "gcp-sa"
    secret = {
      secretName = kubernetes_secret_v1.gcp_sa[0].metadata[0].name
    }
    mountPath = "/secrets/${local.slug}-gcp-sa"
  }] : []

  openai_env = merge(
    var.llm_config.openai_base_url != null ? {
      OPENAI_BASE_URL = var.llm_config.openai_base_url
    } : {},
    var.llm_config.openai_api_key != null ? {
      OPENAI_API_KEY = var.llm_config.openai_api_key
    } : {}
  )

  llm_env = merge(
    local.openai_env,
    local.gcp_sa_enabled ? local.gcp_llm_env : {},
    var.llm_config.model != null ? {
      MODEL = var.llm_config.model
    } : {},
    var.llm_config.embeddings_model != null ? {
      EMBEDDINGS_MODEL = var.llm_config.embeddings_model
    } : {},
    var.llm_config.embeddings_dim != null ? {
      EMBEDDINGS_DIM = var.llm_config.embeddings_dim
    } : {}
  )
  llm_volumes = local.gcp_sa_enabled ? local.gcp_llm_volume : []
}

resource "kubernetes_secret_v1" "gcp_sa" {
  count = local.gcp_sa_enabled ? 1 : 0

  metadata {
    name      = "${local.slug}-gcp-sa"
    namespace = var.k8s_namespace
  }

  data = {
    "gcp-sa.json" = var.llm_config.gcp_sa_json
  }
}
