from deep_alpha.config import load_yaml


def test_data_configuration_loads() -> None:
    config = load_yaml("configs/data.yaml")
    data = config["data"]

    assert data["interval"] == "5m"
    assert data["start_month"] == "2025-07"
    assert data["end_month"] == "2026-06"
    assert len(data["symbols"]) == 20
