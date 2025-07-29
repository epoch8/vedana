data "yandex_mdb_postgresql_cluster" "db_cluster" {
  cluster_id = var.yc_mdb_cluster_id
}

