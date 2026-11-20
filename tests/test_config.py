from pathlib import Path

import pytest

from kittra_sim.config import DEFAULT_CONFIG, deep_merge, load_config, validate_config
from kittra_sim.constants import CONFIGS_DIR

ALL_ENV_CONFIGS = sorted((CONFIGS_DIR / "env").rglob("*.yaml"))


def test_defaults_are_nominal():
    cfg = load_config(None)
    for section in ("physics", "visual", "sensor", "actuator"):
        assert cfg[section]["enabled"] is False


@pytest.mark.parametrize("path", ALL_ENV_CONFIGS, ids=lambda p: p.stem)
def test_every_config_loads_and_validates(path: Path):
    cfg = load_config(path)
    assert cfg["name"] == path.stem
    validate_config(cfg)


def test_load_by_name_and_inherit():
    rnd = load_config("randomized")
    ood = load_config("ood")
    assert rnd["physics"]["enabled"] and ood["physics"]["enabled"]   # inherited flag
    # OOD ranges strictly contain the training ranges
    for key in ("friction", "mass", "damping", "kp"):
        assert ood["physics"][key][0] < rnd["physics"][key][0]
        assert ood["physics"][key][1] > rnd["physics"][key][1]
    assert ood["actuator"]["max_delay_steps"] > rnd["actuator"]["max_delay_steps"]
    assert ood["visual"]["cam_pos_offset"] > rnd["visual"]["cam_pos_offset"]


def test_ablations_change_exactly_one_section():
    nominal = load_config("nominal")
    for path in sorted((CONFIGS_DIR / "env" / "ablation").glob("*.yaml")):
        cfg = load_config(path)
        enabled = [s for s in ("physics", "visual", "sensor", "actuator") if cfg[s]["enabled"]]
        assert len(enabled) == 1, f"{path.stem} enables {enabled}"
        assert cfg["task"] == nominal["task"]


def test_validation_rejects_bad_values():
    bad = deep_merge(DEFAULT_CONFIG, {"physics": {"friction": [1.5, 0.5]}})
    with pytest.raises(ValueError):
        validate_config(bad)
    bad = deep_merge(DEFAULT_CONFIG, {"actuator": {"min_delay_steps": 3, "max_delay_steps": 1}})
    with pytest.raises(ValueError):
        validate_config(bad)
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")
