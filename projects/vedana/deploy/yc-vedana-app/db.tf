data "yandex_mdb_postgresql_cluster" "db_cluster" {
  cluster_id = var.yc_mdb_cluster_id
}

moved {
  from = random_string.db
  to   = random_string.jims_db_password
}

resource "random_string" "jims_db_password" {
  length  = 16
  special = false
}

resource "yandex_mdb_postgresql_user" "jims" {
  cluster_id = var.yc_mdb_cluster_id
  name       = "${local.project_underscore}_vedana"
  password   = random_string.jims_db_password.result
  conn_limit = 10

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
