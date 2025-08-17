data "yandex_mdb_postgresql_cluster" "db_cluster" {
  cluster_id = var.yc_mdb_cluster_id
}

###

resource "random_string" "jims_db_password" {
  length  = 16
  special = false
}

resource "yandex_mdb_postgresql_user" "jims" {
  cluster_id = var.yc_mdb_cluster_id
  name       = "${local.project_underscore}_vedana"
  password   = random_string.jims_db_password.result
  conn_limit = 5

  lifecycle {
    ignore_changes = [
      name,
      password,
    ]
  }
}

resource "yandex_mdb_postgresql_database" "jims" {
  cluster_id = var.yc_mdb_cluster_id
  name       = "${local.project_underscore}_vedana"
  owner      = yandex_mdb_postgresql_user.jims.name

  lifecycle {
    ignore_changes = [
      name,
    ]
  }
}

###

resource "random_string" "etl_db_password" {
  length  = 16
  special = false
}

resource "yandex_mdb_postgresql_user" "etl" {
  cluster_id = var.yc_mdb_cluster_id
  name       = "${local.project_underscore}_${var.environment}_etl"
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
  name       = "${local.project_underscore}_${var.environment}_etl"
  owner      = yandex_mdb_postgresql_user.etl.name

  lifecycle {
    ignore_changes = [
      name,
    ]
  }
}

###

locals {
  db_conn_uri = "postgresql://${yandex_mdb_postgresql_user.etl.name}:${random_string.etl_db_password.result}@${data.yandex_mdb_postgresql_cluster.db_cluster.host.0.fqdn}:6432/${yandex_mdb_postgresql_database.etl.name}"
}