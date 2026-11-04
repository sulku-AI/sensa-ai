from sensa_ai import config


def test_every_param_has_status_and_source() -> None:
    for section, items in config.load_raw().items():
        for key, v in items.items():
            assert set(v) >= {"value", "status", "source"}, f"{section}.{key}"
            assert v["status"] in config.STATUSES, f"{section}.{key}: {v['status']}"


def test_lookahead_is_user_decision_30s() -> None:
    assert config.load()["lookahead"]["window_s"] == 30.0


def test_prompts_have_proxy_and_negatives() -> None:
    p = config.load_prompts()
    assert p["positive"] and p["negative"] and p["proxy"]
