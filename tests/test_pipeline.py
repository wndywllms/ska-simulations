import os

from astropy.io import fits
from astropy.table import Table

from ska_simulations.config import load_config
from ska_simulations.oskar_sim import subband_freqs
from ska_simulations.pipeline import SimulationRun, main
from ska_simulations.skymodels import build_sky_table


def test_subbands_cover_band():
    cfg = load_config(None, {"observation": {"nchan": 6912, "nsubbands": 6, "freq_step_khz": 21.7}})
    chunks = subband_freqs(cfg)
    assert sum(len(c) for c in chunks) == 6912 and len(chunks) == 6


def test_sky_models(fake_site, tmp_path):
    _, site = fake_site
    for model, n in (("LoTSSDEEP", None), ("testsky", 2), ("cal", 1), ("COSMOS", 1)):
        cfg = load_config(str(site), {"sky": {"model": model}})
        out = str(tmp_path / f"{model}.fits")
        build_sky_table(cfg, out)
        t = Table.read(out)
        assert set(t.colnames) == {"RA", "DEC", "I", "Q", "U", "V", "Freq", "Spidx", "RM", "Maj", "Min", "PA"}
        if n:
            assert len(t) == n
    assert Table.read(str(tmp_path / "COSMOS.fits"))["Spidx"][0] == -0.7


def test_full_pipeline_with_fakes(fake_site, monkeypatch):
    tmp, site = fake_site
    cwd = os.getcwd()
    try:
        main([str(site), "--overwrite", "yes", "--set", "observation.nchan=20",
              "--set", "simulator.cable_errors=true", "--set", "sky.flux_limit_jy=1e-3"])
    finally:
        os.chdir(cwd)
    run = tmp / "runs" / "AA2_60.0min_20channels_6.8s_43.4kHz_120.0MHz"
    assert (run / "config_used.yaml").exists()
    assert (run / "AA2_60.0min_20channels_6.8s_43.4kHz_120.0MHz.ms").is_dir()
    settings = (run / "AA2_60.0min_20channels_6.8s_43.4kHz_120.0MHz_oskar_settings.txt").read_text()
    assert "rigid-rotation_model.tm" in settings  # cable-error model chosen
    mask = fits.getdata(run / "ws_AA2_60.0min_20channels_6.8s_43.4kHz_120.0MHz_robm1_mask.fits")
    assert mask.sum() > 0
    ws2 = (run / "wsclean2.log").read_text()
    assert "-abs-threshold 0.5mJy" in ws2 and "-fits-mask" in ws2
    summary = (run / "simulation_summary.log").read_text()
    assert "*oskar run took" in summary and "*ws2 run took" in summary
