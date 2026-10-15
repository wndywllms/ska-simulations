"""Shared fixtures: a fake data directory, a fake ``oskar`` module and fake DP3/wsclean."""
import os
import stat
import sys
import textwrap
import types

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    data = tmp_path / "data"
    tel = tmp_path / "teldir"
    data.mkdir()
    (tel / "SKA-Low_AA2_68S_rigid-rotation_model_noerr.tm").mkdir(parents=True)
    (tel / "SKA-Low_AA2_68S_rigid-rotation_model.tm").mkdir(parents=True)
    rng = np.random.default_rng(1)
    n = 200
    ra, dec, flux = rng.uniform(-5, 5, n), rng.uniform(-5, 5, n), rng.uniform(0, 2e-3, n)
    ra[0], dec[0], flux[0] = 0.0, 0.0, 1.0   # one bright source at the field centre
    Table({"RA": ra, "DEC": dec, "I": flux, "Maj": np.zeros(n), "Min": np.zeros(n),
           "PA": np.zeros(n)}).write(data / "LoTSS_deep_merge_renorm.fits")
    Table({"RA": [0.0, 0.1], "DEC": [0.0, 0.1], "I": [1.0, 0.5], "Maj": [0.0, 0.0],
           "Min": [0.0, 0.0], "PA": [0.0, 0.0]}).write(data / "test_skymodel_simplesquare.fits")
    Table({"RA": [150.0], "DEC": [2.2], "Total_flux": [0.01], "DC_Maj": [1.0], "DC_Min": [1.0],
           "DC_PA": [0.0], "alpha": [0.7], "extra": [1]}).write(
        data / "SRCnet_Combined_input_catalogue_alpha.fits")
    for kind in ("freqs", "rms"):
        (data / f"oskar_noise_{kind}_6.8s_43.4kHz.txt").write_text("1\n")

    # fake external tools on PATH
    bindir = tmp_path / "bin"
    bindir.mkdir()
    wsclean = textwrap.dedent(f"""\
        #!{sys.executable}
        import sys
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        a = sys.argv
        name = a[a.index("-name") + 1]
        n = int(a[a.index("-size") + 1])
        w = WCS(naxis=2); w.wcs.ctype = ["RA---SIN", "DEC--SIN"]; w.wcs.crval = [0, -30]
        w.wcs.crpix = [n / 2, n / 2]; w.wcs.cdelt = [-1.5 / 3600, 1.5 / 3600]
        fits.PrimaryHDU(np.zeros((n, n), np.float32), w.to_header()).writeto(name + "-image.fits")
        print("fake wsclean", " ".join(a[1:]))
        """)
    dp3 = "#!/bin/sh\necho fake DP3 \"$@\"\n"
    for name, body in (("wsclean", wsclean), ("DP3", dp3)):
        p = bindir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    # fake oskar module, importable by the subprocess too
    mod = tmp_path / "fakemods" / "oskar"
    mod.mkdir(parents=True)
    (mod / "__init__.py").write_text(textwrap.dedent("""\
        import os
        class SettingsTree:
            def __init__(self, app): self.d = {}
            def from_dict(self, d): self.d = dict(d)
        class Sky:
            @staticmethod
            def from_array(a): return a
        class Interferometer:
            def __init__(self, settings): self.s = settings.d
            def set_sky_model(self, sky): self.sky = sky
            def run(self):
                os.makedirs(self.s["interferometer/ms_filename"])
                print("fake oskar ran", self.s["observation/start_time_utc"])
        """))
    monkeypatch.setenv("PYTHONPATH", f"{mod.parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.syspath_prepend(str(mod.parent))

    cfg = tmp_path / "site.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        paths:
          data_dir: {data}
          telescope_dir: {tel}
          work_dir: {tmp_path / 'runs'}
          wsclean_temp_dir: null
          msoverview_script: null
        sensitivity:
          enabled: false
        imaging:
          abs_threshold: 0.5mJy
          npix: 256
        """))
    (tmp_path / "runs").mkdir()
    return tmp_path, cfg
