"""
Run an OSKAR interferometer simulation of SKA-Low.

Can be used as a library (:func:`simulate`, :func:`run_oskar`) or from the
command line::

    ska-oskar-sim --config my_run.yaml --skytabfits sky.fits
    ska-oskar-sim --array AA2 --nhours 1 --timestep 6.8 --freq 120 \\
                  --freqstep 43.4 --nchan 500 --ra 0 --dec -30 --skytabfits sky.fits

Explicit command-line options override values from ``--config`` (or the
packaged defaults if no config is given). The pipeline (``ska-runsim``) runs
this as a subprocess so that OSKAR's own output is captured in ``oskarsim.log``.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.table import Table
from astropy.time import Time

from .config import (data_path, deep_update, load_config, noise_file_paths, parse_overrides,
                     telescope_model_path)
from .utils import root_name_from_config

# SKA-Low core reference position
SKAO_LOW_CORE = EarthLocation.from_geodetic(116.7644482, -26.82472208, 365.0, ellipsoid="WGS84")


def transit_time(ra: u.Quantity, location: EarthLocation, date: Time, offset: float = 0) -> Time:
    """Time of transit of ``ra`` after ``date``, minus ``offset`` hours.

    Coarse 2-minute search over 24 h, then refined over a 4-minute window.
    Accurate to around a second.
    """
    ra_hours = ra.to(u.hourangle).value

    times_coarse = date + np.linspace(0, 24, 720) * u.hour
    lst_coarse = times_coarse.sidereal_time("apparent", longitude=location.lon).hour
    diff = np.abs(lst_coarse - ra_hours)
    diff = np.minimum(diff, 24 - diff)
    t_centre = times_coarse[np.argmin(diff)]

    times_fine = t_centre + np.linspace(-2, 2, 1000) * u.minute
    lst_fine = times_fine.sidereal_time("apparent", longitude=location.lon).hour
    diff_fine = np.abs(lst_fine - ra_hours)
    diff_fine = np.minimum(diff_fine, 24 - diff_fine)
    return times_fine[np.argmin(diff_fine)] - offset * u.hour


def load_sky(skytabfits: str):
    """Read a sky table written by :mod:`ska_simulations.skymodels` into an ``oskar.Sky``."""
    import oskar

    t = Table.read(skytabfits)
    cols = ["RA", "DEC", "I", "Q", "U", "V", "Freq", "Spidx", "RM", "Maj", "Min", "PA"]
    sky_data = np.array([t[c] for c in cols], dtype=float).T
    return oskar.Sky.from_array(sky_data)


def oskar_settings(cfg: dict, *, ra_deg: float, dec_deg: float, start_freq_hz: float,
                   nchan: int, outroot: str) -> dict:
    """Build the OSKAR ``oskar_sim_interferometer`` settings dict for one (sub)band."""
    obs, sim = cfg["observation"], cfg["simulator"]
    time_step_s = float(obs["time_step_s"])
    freq_step_hz = float(obs["freq_step_khz"]) * 1e3
    length_s = float(obs["nhours"]) * 3600.0
    ntime = int(np.floor(length_s / time_step_s))
    length_f = str(timedelta(seconds=ntime * time_step_s))

    telescope = telescope_model_path(cfg)
    noise_freq_file, noise_rms_file = noise_file_paths(cfg)
    for label, path in (("telescope model", telescope), ("noise frequency file", noise_freq_file),
                        ("noise rms file", noise_rms_file)):
        if label.startswith("noise") and not sim["noise"]["enable"]:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"OSKAR {label} is missing: {path}")

    time0 = transit_time(ra_deg * u.deg, SKAO_LOW_CORE, Time(obs["start_date"]),
                         float(obs["nhours"]) / 2.0 + float(obs["time_offset_hours"]))

    params = {
        "simulator/use_gpus": bool(sim["use_gpu"]),
        "simulator/max_sources_per_chunk": sim["max_sources_per_chunk"],
        "simulator/keep_log_file": sim["keep_log_file"],
        "simulator/write_status_to_log_file": True,
        "simulator/double_precision": sim["double_precision"],
        "observation/num_channels": int(nchan),
        "observation/start_frequency_hz": start_freq_hz,
        "observation/frequency_inc_hz": freq_step_hz,
        "observation/phase_centre_ra_deg": ra_deg,
        "observation/phase_centre_dec_deg": dec_deg,
        "observation/num_time_steps": ntime,
        "observation/start_time_utc": str(time0),
        "observation/length": length_f,
        "telescope/pol_mode": obs["pol_mode"],
        "telescope/input_directory": telescope,
        "telescope/normalise_beams_at_phase_centre": sim["normalise_beams_at_phase_centre"],
        "telescope/aperture_array/array_pattern/normalise": sim["normalise_array_pattern"],
        "interferometer/oskar_vis_filename": f"{outroot}.vis",
        "interferometer/ms_filename": f"{outroot}.ms",
        "interferometer/ms_dish_diameter": sim["ms_dish_diameter"],
        "interferometer/channel_bandwidth_hz": freq_step_hz,
        "interferometer/time_average_sec": time_step_s,
        "interferometer/max_time_samples_per_block": sim["max_time_samples_per_block"],
        "interferometer/max_channels_per_block": int(nchan),
        "interferometer/noise/enable": sim["noise"]["enable"],
    }
    if sim["noise"]["enable"]:
        params.update({
            "interferometer/noise/seed": sim["noise"]["seed"],
            "interferometer/noise/freq": "Data file",
            "interferometer/noise/freq/file": noise_freq_file,
            "interferometer/noise/rms": "Data file",
            "interferometer/noise/rms/file": noise_rms_file,
        })
    tec = data_path(cfg, sim["ionosphere"]["tec_screen"])
    if tec is not None:
        if not os.path.isfile(tec):
            raise FileNotFoundError(f"TEC screen {tec} does not exist!")
        params["telescope/ionosphere_screen_type"] = "External"
        params["telescope/external_tec_screen/input_fits_file"] = tec
        if sim["ionosphere"]["screen_height_km"] is not None:
            params["telescope/external_tec_screen/screen_height_km"] = sim["ionosphere"]["screen_height_km"]
    params.update(sim.get("extra_settings") or {})
    return params


def run_oskar(params: dict, skytabfits: str | None = None, settings_file: str | None = None) -> None:
    """Run ``oskar.Interferometer`` with the given settings dict and optional sky table."""
    import oskar

    if settings_file:
        with open(settings_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key}={value}\n")
    print(f"Using telescope model {params['telescope/input_directory']}")
    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(params)
    sim = oskar.Interferometer(settings=settings)
    if skytabfits:
        sim.set_sky_model(load_sky(skytabfits))
    print("simulating observation")
    sim.run()


def subband_freqs(cfg: dict) -> list[np.ndarray]:
    obs = cfg["observation"]
    freqs = obs["start_freq_mhz"] * 1e6 + obs["freq_step_khz"] * 1e3 * np.arange(int(obs["nchan"]))
    return np.array_split(freqs, int(obs["nsubbands"]))


def simulate(cfg: dict, ra_deg: float, dec_deg: float, skytabfits: str | None = None,
             dry_run: bool = False) -> list[str]:
    """Simulate all sub-bands; return the list of measurement sets written."""
    root = root_name_from_config(cfg)
    chunks = subband_freqs(cfg)
    print(f"Array is {cfg['observation']['array']}")
    written = []
    for i, freqs_i in enumerate(chunks):
        outroot = root if len(chunks) == 1 else f"{root}_SB{i:03d}"
        print(f"starting oskar simulation for subband {i} with {len(freqs_i)} channels -> {outroot}.ms")
        params = oskar_settings(cfg, ra_deg=ra_deg, dec_deg=dec_deg, start_freq_hz=float(freqs_i[0]),
                                nchan=len(freqs_i), outroot=outroot)
        if dry_run:
            for key, value in params.items():
                print(f"  {key}={value}")
        else:
            run_oskar(params, skytabfits, settings_file=f"{outroot}_oskar_settings.txt")
        written.append(f"{outroot}.ms")
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run OSKAR simulations of SKA-Low")
    p.add_argument("--config", help="YAML run configuration (defaults to the packaged defaults)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override any config value, e.g. --set simulator.noise.seed=1")
    p.add_argument("--nogpu", action="store_true", help="Turn off GPUs in the OSKAR simulation")
    p.add_argument("--array", default=None, help="SKA array release to simulate (e.g. AA2, AAstarmod)")
    p.add_argument("--TM_cableerrs", action="store_true", default=None,
                   help="Use the telescope model with cable errors")
    p.add_argument("--ra", type=float, default=None, help="Phase centre RA in deg")
    p.add_argument("--dec", type=float, default=None, help="Phase centre DEC in deg")
    p.add_argument("--nhours", type=float, default=None, help="Number of hours")
    p.add_argument("--offset_start_time", type=float, default=None,
                   help="Offset the start time of the observation by this value in hours")
    p.add_argument("--timestep", type=float, default=None, help="Time step in s")
    p.add_argument("--freq", type=float, default=None, help="Start frequency in MHz")
    p.add_argument("--freqstep", type=float, default=None, help="Frequency step in kHz")
    p.add_argument("--nchan", type=int, default=None, help="Number of channels")
    p.add_argument("--nsubbands", type=int, default=None, help="Number of subbands to split the simulation into")
    p.add_argument("--skytabfits", type=str, default=None, help="Sky catalogue in FITS format")
    p.add_argument("--tecscreenfits", type=str, default=None, help="TEC screen in FITS format")
    p.add_argument("--pol_mode", type=str, default=None, help="OSKAR telescope polarisation mode")
    p.add_argument("--dry-run", action="store_true", help="Print the OSKAR settings without running")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cli = {
        "observation": {k: v for k, v in {
            "array": args.array, "ra_deg": args.ra, "dec_deg": args.dec, "nhours": args.nhours,
            "time_offset_hours": args.offset_start_time, "time_step_s": args.timestep,
            "start_freq_mhz": args.freq, "freq_step_khz": args.freqstep, "nchan": args.nchan,
            "nsubbands": args.nsubbands, "pol_mode": args.pol_mode}.items() if v is not None},
        "simulator": {},
    }
    if args.nogpu:
        cli["simulator"]["use_gpu"] = False
    if args.TM_cableerrs:
        cli["simulator"]["cable_errors"] = True
    if args.tecscreenfits is not None:
        cli["simulator"]["ionosphere"] = {"tec_screen": None if args.tecscreenfits == "None"
                                          else os.path.abspath(args.tecscreenfits)}
    cfg = load_config(args.config, deep_update(cli, parse_overrides(args.set)))
    ra = cfg["observation"]["ra_deg"]
    dec = cfg["observation"]["dec_deg"]
    if ra is None or dec is None:
        from .skymodels import sky_centre
        ra, dec = sky_centre(cfg)
    sky = None if args.skytabfits in (None, "None") else args.skytabfits
    try:
        simulate(cfg, ra, dec, sky, dry_run=args.dry_run)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
