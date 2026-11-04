from sensa_ai import config


def _leaves(d: dict, prefix: str = ""):  # type: ignore[no-untyped-def]
    for k, v in d.items():
        if isinstance(v, dict) and "value" in v:
            yield prefix + k, v
        elif isinstance(v, dict):
            yield from _leaves(v, prefix + k + ".")


def test_every_param_has_status_and_source() -> None:
    for key, v in _leaves(config.load_raw()):
        assert set(v) >= {"value", "status", "source"}, key
        assert v["status"] in config.STATUSES, f"{key}: {v['status']}"


def test_profiles_override_and_validate() -> None:
    base = config.load()
    strong = config.load(profile="guclu")
    assert strong["blur"]["sigma_px"] > base["blur"]["sigma_px"]
    assert strong["gate"]["threshold"] < base["gate"]["threshold"]
    assert config.load(profile="orta")["blur"] == base["blur"]
    for name in config.load_raw()["profiles"]:
        config.load(profile=name)  # her profil anahtarı gerçekten var olmalı
    try:
        config.set_profile("guclu")
        assert config.load()["blur"]["dilate_px"] == strong["blur"]["dilate_px"]
    finally:
        config.set_profile(None)


def test_lookahead_is_user_decision_30s() -> None:
    assert config.load()["lookahead"]["window_s"] == 30.0


def test_prompts_have_proxy_and_negatives() -> None:
    p = config.load_prompts()
    assert p["positive"] and p["negative"] and p["proxy"]
