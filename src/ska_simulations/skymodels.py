"""
Build OSKAR-ready sky tables (FITS) from the various input catalogues.

Every builder writes a table with the columns
``RA DEC I Q U V Freq Spidx RM Maj Min PA`` (degrees, Jy, Hz, rad/m^2, arcsec, deg)
which is what :func:`ska_simulations.oskar_sim.load_sky` expects.
"""
from __future__ import annotations

import numpy as np
from astropy.table import Column, Table

from .config import data_path

OSKAR_COLUMNS = ["RA", "DEC", "I", "Q", "U", "V", "Freq", "Spidx", "RM", "Maj", "Min", "PA"]


def complete_sky_table(skytab: Table, ref_freq_hz: float = 120e6,
                       spectral_index: float = -0.7) -> Table:
    """Keep the OSKAR columns and add any that are missing (Q=U=V=RM=0, Freq, Spidx)."""
    n = len(skytab)
    keep = [c for c in OSKAR_COLUMNS if c in skytab.colnames]
    skytab.keep_columns(keep)
    for name in ("Q", "U", "V", "RM"):
        if name not in skytab.colnames:
            skytab.add_column(Column(name=name, data=np.zeros(n)))
    if "Freq" not in skytab.colnames:
        skytab.add_column(Column(name="Freq", data=ref_freq_hz * np.ones(n)))
    if "Spidx" not in skytab.colnames:
        skytab.add_column(Column(name="Spidx", data=spectral_index * np.ones(n)))
    for name in ("Maj", "Min", "PA"):
        if name not in skytab.colnames:
            skytab.add_column(Column(name=name, data=np.zeros(n)))
    return skytab


def apply_flux_limit(skytab: Table, fluxlim: float) -> Table:
    if fluxlim and fluxlim > 0:
        skytab = skytab[skytab["I"] >= fluxlim]
    print(f"{len(skytab)} sources in sky")
    return skytab


def get_cosmos_sky_table(skytabfits: str, catalogue: str, fluxlim: float = 0,
                         ref_freq_hz: float = 120e6) -> Table:
    """SRCnet combined COSMOS catalogue (absolute positions)."""
    skytab = Table.read(catalogue)
    print(f"{len(skytab)} sources in catalogue {catalogue}")
    skytab.rename_column("Total_flux", "I")
    skytab.rename_column("DC_Maj", "Maj")
    skytab.rename_column("DC_Min", "Min")
    skytab.rename_column("DC_PA", "PA")
    skytab.rename_column("alpha", "Spidx")
    skytab["Spidx"] = -1 * skytab["Spidx"]  # catalogue alpha is positive; OSKAR wants S ~ nu^Spidx
    skytab = complete_sky_table(skytab, ref_freq_hz=ref_freq_hz)
    skytab = apply_flux_limit(skytab, fluxlim)
    skytab.write(skytabfits, overwrite=True)
    return skytab


def get_lotss_deep_sky_table(skytabfits: str, catalogue: str, ra0: float = 0, dec0: float = -30,
                             fluxlim: float = 0, box_halfwidth_deg: float = 3.4,
                             ref_freq_hz: float = 120e6, spectral_index: float = -0.7) -> Table:
    """LoTSS-deep fields (combined, resampled and recentred on 0,0), shifted to (ra0, dec0)."""
    skytab = Table.read(catalogue)
    h = box_halfwidth_deg
    sel = (skytab["RA"] > -h) & (skytab["RA"] < h) & (skytab["DEC"] > -h) & (skytab["DEC"] < h)
    skytab = skytab[sel]
    skytab["RA"] = skytab["RA"] + ra0
    skytab["DEC"] = skytab["DEC"] + dec0
    skytab = apply_flux_limit(skytab, fluxlim)
    skytab = complete_sky_table(skytab, ref_freq_hz, spectral_index)
    skytab.write(skytabfits, overwrite=True)
    return skytab


def get_test_sky_table(skytabfits: str, catalogue: str, ra0: float = 0, dec0: float = -30,
                       fluxlim: float = 0, ref_freq_hz: float = 120e6,
                       spectral_index: float = -0.7) -> Table:
    """Simple test model (offsets from the centre), shifted to (ra0, dec0)."""
    skytab = Table.read(catalogue)
    skytab["RA"] = skytab["RA"] + ra0
    skytab["DEC"] = skytab["DEC"] + dec0
    skytab = apply_flux_limit(skytab, fluxlim)
    skytab = complete_sky_table(skytab, ref_freq_hz, spectral_index)
    skytab.write(skytabfits, overwrite=True)
    return skytab


def get_sky_table_cal(skytabfits: str, ra0: float = 0, dec0: float = -30, fluxcal: float = 100,
                      ref_freq_hz: float = 120e6, spectral_index: float = -0.7) -> Table:
    """Single point-source calibrator at (ra0, dec0)."""
    skytab = Table({"RA": [ra0], "DEC": [dec0], "I": [fluxcal]})
    skytab = complete_sky_table(skytab, ref_freq_hz, spectral_index)
    skytab.write(skytabfits, overwrite=True)
    return skytab


# ---------------------------------------------------------------------------
# Config-driven entry points
# ---------------------------------------------------------------------------

_SECTION = {"LoTSSDEEP": "lotss_deep", "COSMOS": "cosmos", "testsky": "testsky", "cal": "cal"}


def sky_centre(cfg: dict) -> tuple[float, float]:
    """Phase centre: observation.ra_deg/dec_deg if set, otherwise the sky model's centre."""
    obs = cfg["observation"]
    model = cfg["sky"]["model"]
    default = cfg["sky"][_SECTION[model]]["centre"] if model in _SECTION else (0.0, -30.0)
    ra = obs["ra_deg"] if obs["ra_deg"] is not None else default[0]
    dec = obs["dec_deg"] if obs["dec_deg"] is not None else default[1]
    return float(ra), float(dec)


def cosmos_catalogue(cfg: dict) -> str:
    c = cfg["sky"]["cosmos"]
    if c["variable"]:
        print("using variables in cosmos field")
        return data_path(cfg, c["catalogue_variable"])
    if c["ddcal"]:
        print("using ddcals in the cosmos field")
        return data_path(cfg, c["catalogue_ddcal"])
    return data_path(cfg, c["catalogue"])


def build_sky_table(cfg: dict, skytabfits: str) -> str | None:
    """Write the sky table selected by ``cfg['sky']['model']``; return its path (None for 'none')."""
    sky = cfg["sky"]
    model = sky["model"]
    ra0, dec0 = sky_centre(cfg)
    common = dict(ref_freq_hz=sky["ref_freq_hz"])
    if model == "COSMOS":
        if cfg["observation"]["ra_deg"] is not None or cfg["observation"]["dec_deg"] is not None:
            print("Note: COSMOS catalogue has absolute positions; it is not shifted to ra_deg/dec_deg")
        get_cosmos_sky_table(skytabfits, cosmos_catalogue(cfg), fluxlim=sky["flux_limit_jy"], **common)
    elif model == "LoTSSDEEP":
        s = sky["lotss_deep"]
        cat = data_path(cfg, s["catalogue_smallfield"] if s["smallfield"] else s["catalogue"])
        get_lotss_deep_sky_table(skytabfits, cat, ra0, dec0, fluxlim=sky["flux_limit_jy"],
                                 box_halfwidth_deg=s["box_halfwidth_deg"],
                                 spectral_index=sky["spectral_index"], **common)
    elif model == "testsky":
        get_test_sky_table(skytabfits, data_path(cfg, sky["testsky"]["catalogue"]), ra0, dec0,
                           fluxlim=sky["flux_limit_jy"], spectral_index=sky["spectral_index"], **common)
    elif model == "cal":
        get_sky_table_cal(skytabfits, ra0, dec0, fluxcal=sky["cal"]["flux_jy"],
                          spectral_index=sky["spectral_index"], **common)
    elif model == "none":
        return None
    else:
        raise ValueError(f"Unknown sky model {model!r}")
    return skytabfits
