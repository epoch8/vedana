variable "kubernetes_namespace" {
  type = string
}
variable "values_file" {
  type    = string
  default = ""
}
variable "name" {
  type    = string
  default = "memgraph"
}
variable "memgraph_config" {
  type = list(string)
  default = ["--telemetry-enabled=false", "--storage-mode=IN_MEMORY_ANALYTICAL"]
}
variable "user" {
  type = string
}
variable "password" {
  type      = string
  sensitive = true
}
variable "size" {
  type    = string
  default = "10Gi"
}
variable "resources" {
  type = object({
    requests : object({ cpu : string, memory : string }),
    limits : object({ cpu : string, memory : string })
  })
  default = {
    requests : { cpu : "1", memory : "6Gi" },
    limits : { cpu : "2", memory : "8Gi" },
  }
}