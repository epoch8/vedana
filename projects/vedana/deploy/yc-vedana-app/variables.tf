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
    limits   : map(string)
  })
  default = {
    requests = { cpu = "0.1", memory = "2Gi" }
    limits   = { cpu = "1", memory = "3Gi" }
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
    data_model_doc_id = string
    data_doc_id = string
    api_key = string
  })
}

