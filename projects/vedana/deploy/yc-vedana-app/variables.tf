variable "project" {
  type = string
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "sentry_dsn" {
  type    = string
  default = null
}

variable "k8s_namespace" {
  type = string
}

variable "base_domain" {
  type = string
}

variable "yc_mdb_cluster_id" {
  type = string
}

variable "memgraph_resources" {
  type = object({
    requests : map(string)
    limits : map(string)
  })
  default = {
    requests = { cpu = "0.1", memory = "2Gi" }
    limits   = { cpu = "1", memory = "3Gi" }
  }
}

variable "datapipe_resources" {
  type = object({
    requests : map(string)
    limits : map(string)
  })
  default = {
    requests = { cpu = "1", memory = "1Gi" }
    limits   = { cpu = "1", memory = "2Gi" }
  }
}

variable "image_repository" {
  type    = string
  default = null
}

variable "image_tag" {
  type    = string
  default = null
}

variable "image_pull_secrets" {
  type    = string
  default = null
}

variable "telegram_bot_token" {
  type    = string
  default = null
}

variable "grist" {
  type = object({
    server_url = string
    api_key    = string

    data_model_doc_id = string
    data_doc_id       = string
    test_set_doc_id   = optional(string)
  })
}

variable "llm_config" {
  type = object({
    openai_base_url  = optional(string)
    openai_api_key   = optional(string)
    gcp_sa_json      = optional(string)
    model            = optional(string)
    embeddings_model = optional(string)
    embeddings_dim   = optional(string)
    # gcp_project_id = string
  })
  default = {
    openai_base_url  = "https://oai-proxy-hzkr3iwwhq-ew.a.run.app/v1/"
    openai_api_key   = "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"
    model            = "gpt-4.1-mini"
    embeddings_model = "text-embedding-3-large"
    embeddings_dim   = "1024"
    gcp_sa_json      = null
  }
}

###

variable "enable_api" {
  type    = bool
  default = false
}

variable "api_command" {
  type    = list(string)
  default = null
}

variable "api_resources" {
  type = object({
    requests = map(string)
    limits   = map(string)
  })
  default = {
    requests = { cpu = "200m", memory = "512Mi" }
    limits   = { cpu = "500m", memory = "1Gi" }
  }
}

###

variable "authentik_group_ids" {
  type    = list(string)
  default = []
}

variable "gdrive_config" {
  type = object({
    folder_id   = string
    gcp_sa_json = string
  })
  default = null
}

variable "extra_env" {
  type    = map(string)
  default = {}
}
