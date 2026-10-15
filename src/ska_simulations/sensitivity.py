"""
Helpers around the SKAO sensitivity calculator
(https://sensitivity-calculator.skao.int).

``ska-sensitivity`` (CLI) reproduces the old ``get_sens.py``: it estimates the
per-visibility noise for a given integration time and channel width, and can
write OSKAR noise data files in the naming convention the simulator expects.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

DEFAULT_URL = "https://sensitivity-calculator.skao.int/api/v11/low/continuum/calculate"
SUBARRAYS = {"AA2": "LOW_AA2_all", "AAstarmod": "LOW_AAstar_all"}


def get_ska_sens_calc(ra: float, dec: float, f0: float, bw: float, inttime: float,
                      AA: str = "AA2", weight: str = "natural", url: str = DEFAULT_URL,
                      elevation_limit: float = 20.0, subarray: str | None = None,
                      timeout: float = 60) -> dict:
    """Continuum image sensitivity from the SKAO calculator.

    Parameters
    ----------
    ra, dec : pointing centre (deg)
    f0      : start frequency (MHz); the band centre f0 + bw/2 is used
    bw      : bandwidth (MHz)
    inttime : integration time (hours)

    Returns the calculator's ``weighted_continuum_sensitivity`` dict
    (``{'value': <Jy/beam>, 'unit': ...}``).
    """
    import requests

    if subarray is None:
        if AA not in SUBARRAYS:
            raise ValueError(f"No sensitivity-calculator subarray known for {AA}")
        subarray = SUBARRAYS[AA]
    params = {
        "integration_time_h": inttime,
        "pointing_centre": f"{ra} {dec}",
        "elevation_limit": elevation_limit,
        "freq_centre_mhz": f0 + bw / 2.0,
        "spectral_averaging_factor": 1,
        "bandwidth_mhz": bw,
        "n_subbands": 1,
        "subarray_configuration": subarray,
        "weighting_mode": weight,
        "robustness": None,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["transformed_result"]["weighted_continuum_sensitivity"]


def sensitivity_from_config(cfg: dict, ra: float, dec: float) -> dict:
    obs, s = cfg["observation"], cfg["sensitivity"]
    bw_mhz = obs["freq_step_khz"] * obs["nchan"] / 1e3
    return get_ska_sens_calc(
        ra, dec, obs["start_freq_mhz"], bw_mhz, obs["nhours"], AA=obs["array"],
        weight=s["weighting"], url=s["url"], elevation_limit=s["elevation_limit_deg"],
        subarray=s["subarray"].get(obs["array"]), timeout=s["timeout_s"],
    )


def visibility_noise(time_step_s: float, freq_step_khz: float, freqs_mhz=None, AA: str = "AA2",
                     nstations: int = 68, url: str = DEFAULT_URL, dec: float = -27.0):
    """Per-visibility rms noise (Jy) at ``freqs_mhz`` for one baseline, one polarisation.

    Queries the calculator for a 1 s / 0.1 MHz natural-weighted continuum
    sensitivity at each frequency, converts that to a station SEFD and then to
    the noise on a single visibility of ``time_step_s`` x ``freq_step_khz``.
    """
    import astropy.units as u
    import requests

    df = 0.1                # MHz used for the calculator query
    dt = 1.0 / 3600         # hours used for the calculator query
    if freqs_mhz is None:
        freqs_mhz = np.arange(100 + 0.5 * df, 150 + 0.5 * df, 10)
    freqs_mhz = np.atleast_1d(freqs_mhz).astype(float)
    sens = np.zeros_like(freqs_mhz)
    for i, freq in enumerate(freqs_mhz):
        resp = requests.get(url, params={
            "integration_time_h": dt,
            "subarray_configuration": SUBARRAYS.get(AA, AA),
            "pointing_centre": f"0 {dec}",
            "freq_centre_mhz": freq,
            "bandwidth_mhz": df,
            "spectral_averaging_factor": 1,
            "weighting_mode": "natural",
            "elevation_limit": 88,
        }, timeout=60)
        resp.raise_for_status()
        sens[i] = resp.json()["transformed_result"]["weighted_continuum_sensitivity"]["value"]
        print(f"Frequency {freq} sensitivity {sens[i] * 1e3} mJy/bm")
    ns = nstations
    sefd = sens * u.Jy * np.sqrt((ns * (ns - 1) * df * u.MHz * dt * 3600 * u.s).decompose())
    vis_noise = np.sqrt(sefd**2 / (2 * (freq_step_khz * u.kHz * time_step_s * u.s).decompose()))
    return freqs_mhz, vis_noise.to(u.Jy).value


def main(argv=None):
    p = argparse.ArgumentParser(description="Per-visibility noise from the SKAO sensitivity calculator")
    p.add_argument("--array", default="AA2", choices=sorted(SUBARRAYS))
    p.add_argument("--nstations", type=int, default=68)
    p.add_argument("--timestep", type=float, default=3.4, help="Integration time (s)")
    p.add_argument("--freqstep", type=float, default=21.7, help="Channel width (kHz)")
    p.add_argument("--freqs", type=float, nargs="+", default=None, help="Frequencies (MHz)")
    p.add_argument("--outdir", default=None,
                   help="If given, write oskar_noise_freqs_/oskar_noise_rms_ files here")
    a = p.parse_args(argv)
    freqs, noise = visibility_noise(a.timestep, a.freqstep, a.freqs, a.array, a.nstations)
    for f, n in zip(freqs, noise):
        print(f"{f:8.3f} MHz  {n:.4f} Jy per visibility")
    if a.outdir:
        tag = f"{a.timestep:g}s_{a.freqstep:g}kHz"
        ffile = os.path.join(a.outdir, f"oskar_noise_freqs_{tag}.txt")
        rfile = os.path.join(a.outdir, f"oskar_noise_rms_{tag}.txt")
        np.savetxt(ffile, freqs * 1e6)
        np.savetxt(rfile, noise)
        print(f"Wrote {ffile} and {rfile}")


if __name__ == "__main__":
    main()
