from collect_operations_snapshot import SOURCE_HOSTS


def test_aps_data_metric_uses_dedicated_data_filesystem() -> None:
    assert SOURCE_HOSTS["host_aps_data"]["path_default"] == "/home/aurora/data"
    assert SOURCE_HOSTS["host_aps_data"]["path_default"] != SOURCE_HOSTS["host_aps_root"]["path_default"]
