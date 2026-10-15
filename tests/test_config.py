import glob
import os

import pytest

from ska_simulations.config import ConfigError, load_config, noise_file_paths, parse_overrides
from ska_simulations.utils import run_root_name

EXAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "examples", "*.yaml")))


def test_defaults_load():
    cfg = load_config()
    assert cfg["observation"]["array"] == "AA2"


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_examples_load(path):
    load_config(path)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("observation:\n  nchans: 10\n")
    with pytest.raises(ConfigError, match="observation.nchans"):
        load_config(str(p))


def test_overrides():
    cfg = load_config(None, parse_overrides(["observation.nchan=7", "sky.model=cal"]))
    assert cfg["observation"]["nchan"] == 7 and cfg["sky"]["model"] == "cal"


def test_root_name_matches_old_convention():
    # runsimjob.py: f'{AA}_{LENGTH_s/60:.1f}min_{nCHAN}channels_{dTIME}s_{dFREQ}kHz_{FREQ0:.1f}MHz'
    assert run_root_name("AA2", 7.5 / 60, 460, 6.8, 43.4, 120.0) == \
        "AA2_7.5min_460channels_6.8s_43.4kHz_120.0MHz"
    assert run_root_name("AA2", 10.0 / 60, 27777, 0.85, 5.4, 120) == \
        "AA2_10.0min_27777channels_0.85s_5.4kHz_120.0MHz"


def test_noise_file_names():
    cfg = load_config(None, {"observation": {"time_step_s": 3.4, "freq_step_khz": 21.7}})
    f, r = noise_file_paths(cfg)
    assert f.endswith("oskar_noise_freqs_3.4s_21.7kHz.txt")
    assert r.endswith("oskar_noise_rms_3.4s_21.7kHz.txt")
