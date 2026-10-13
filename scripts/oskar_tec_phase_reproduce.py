#!/usr/bin/env python3
"""
oskar_tec_phase_reproduce.py — Independently reproduce the per-station,
per-direction ionospheric phase (and, optionally, Faraday-rotation Jones
matrix) that OSKAR applied from an external TEC screen, using only the
measurement set OSKAR wrote and the TEC screen FITS file OSKAR read from.

Implements, outside of OSKAR, exactly the algorithm documented for OSKAR's
"External" ionosphere screen mode (RADEC/phase-tracking case), i.e. the two
kernels in OSKAR's define_evaluate_tec_screen.h:

Scalar phase (OSKAR_EVALUATE_TEC_SCREEN):

    world_x = (station_u + s_l * screen_height_m) / pixel_size_m
    world_y = (station_v + s_m * screen_height_m) / pixel_size_m
    pix_x   = screen_num_pixels_x // 2 + round(world_x)
    pix_y   = screen_num_pixels_y // 2 + round(world_y)
    tec     = screen[pix_y, pix_x]            # nearest-pixel, 0 outside
    phase   = tec * (-8.44797245e9) / freq_hz  # cold-plasma dispersion

where station_u/station_v are each station's position projected into the
plane perpendicular to the phase-centre direction (i.e. the same u,v an
interferometric baseline UVW would use, but per-station rather than
per-baseline), and s_l/s_m are the direction cosines of each source
relative to the phase centre.

Faraday rotation (OSKAR_EVALUATE_TEC_SCREEN_WITH_FARADAY_ROTATION, enabled
with --faraday): on top of the scalar phase above, OSKAR evaluates a 2x2
polarimetric Jones matrix per station/source/channel/time:

    b             = -(hor_e * field_e + hor_n * field_n + hor_u * field_u)
    b_tesla       = b * 1e-9                      # nT -> T (Wb/m^2)
    faraday_angle = 236.0 * b_tesla * tec * (1e9 / freq_hz)**2
                                                   # ITU-R P.531-6, eq. 2
    J = exp(i * phase) * [[cos(faraday_angle), -sin(faraday_angle)],
                          [ sin(faraday_angle),  cos(faraday_angle)]]

    (J = identity when tec == 0, matching OSKAR's kernel exactly)

where (hor_e, hor_n, hor_u) is the East/North/Up unit vector from the
station towards the source (the kernel's "hor_x/y/z", negated because the
line of sight runs from source to observer), and (field_e, field_n,
field_u) is that station's geomagnetic field in nT, in the same East/
North/Up frame — i.e. exactly the output of station_magnetic_field() in
igrf14.py (the IGRF-14 port produced earlier from OSKAR's
oskar_station_evaluate_magnetic_field.c / igrf14syn()). This script
imports that module directly rather than re-deriving the field; see the
"Requires" section below.

ASSUMPTIONS / things to verify against your own setup
-------------------------------------------------------
1. Station reference position: station_u/v need a common reference point
   for "the station position" before projection. This script uses the mean
   of all ANTENNA table positions (a reasonable stand-in for the array
   "centre" OSKAR's telescope model is built around). If your reproduction
   doesn't match OSKAR's own numbers, this is the first thing to check —
   an overall constant offset here shifts every pierce point by the same
   amount, which (because the pixel lookup is nonlinear/nearest-pixel) is
   NOT simply an overall phase offset that cancels out.
2. Screen geometry (pixel_size_m, screen_height_m) are read from common
   FITS header keywords if present, but ARatmospy/OSKAR header conventions
   vary — pass --pixel-size-m / --screen-height-km explicitly if the
   auto-detected values look wrong (the script prints what it found/used).
3. The small-angle approximation OSKAR itself uses (l*H instead of
   (l/n)*H) is reproduced as-is for fidelity to OSKAR's behaviour, not
   "corrected" — this script intentionally matches OSKAR, not the exact
   geometric pierce point.
4. Assumes a single spectral window and a single FIELD (phase centre) in
   the MS, consistent with how rapthor/OSKAR MSs in this project are set
   up. If your MS has more than one of either, check SPECTRAL_WINDOW /
   FIELD selection near the top of main().
5. The TEC screen FITS cube's frequency axis: if it has length 1, that one
   plane is used regardless of the MS's actual channel frequency (TEC
   itself is frequency-independent; the phase formula below applies the
   MS's true channel frequency separately). If it has more than one
   frequency plane, the nearest plane to each requested frequency is used.
6. (--faraday only) Horizontal frame convention: hor_e/hor_n/hor_u are
   assumed to follow the same East/North/Up axis convention OSKAR itself
   uses for "horizontal" (hor_x/hor_y/hor_z) direction cosines elsewhere
   in its source code, which is also the convention igrf14.py's
   station_magnetic_field() returns its field components in. This pairing
   is internally consistent but has not been cross-checked against an
   OSKAR-internal reference value — if your Faraday rotation angles come
   out with the wrong overall sign, swapping hor_e<->hor_n (equivalently,
   measuring azimuth from East rather than North) is the first thing to
   try.
7. (--faraday only) Azimuth/elevation of each source as seen from each
   station are computed with astropy's AltAz frame (no refraction, no
   diurnal aberration correction beyond what AltAz applies by default) —
   i.e. standard geometric topocentric coordinates, not whatever internal
   routine OSKAR itself uses to derive hor_x/y/z. For modest array extents
   this should be indistinguishable from OSKAR's own numbers, but it is a
   separate, independent computation, not a copy of OSKAR's code (unlike
   the phase/Faraday formulas above, which are transcribed verbatim).
8. (--faraday only) The IGRF epoch (decimal year) defaults to the first value of
   the MS's observation times; override with --year if you want a specific
   epoch (e.g. to match whatever `year` OSKAR was actually configured
   with for oskar_station_evaluate_magnetic_field()).
9. (--faraday only) No horizon/elevation masking is applied — OSKAR's own
   kernel computes the dot product regardless of source elevation; any
   below-horizon exclusion is presumably applied upstream of this kernel
   in OSKAR and is out of scope here.
10. (--h5parm only) Without --faraday, the scalar phase is written as a
    plain ["dir", "ant", "freq", "time"] phase soltab (write_h5parm()).
    With --faraday, the full 2x2 Jones matrix is written instead
    (write_h5parm_fulljones()), as an amplitude+phase soltab PAIR sharing
    a ["pol", "dir", "ant", "freq", "time"] axis layout — there is no
    single "complex" soltype in standard LoSoTo h5parm usage, and because
    the Faraday Jones matrix isn't just a per-element phase (its entries'
    amplitudes vary with the Faraday angle — see faraday_jones()), a
    phase-only soltab would silently lose that polarisation-leakage
    information. `pol` has 4 entries (default order XX, XY, YX, YY, i.e.
    the LOFAR/DP3-ecosystem convention — see JONES_ELEMENT_INDEX /
    DEFAULT_POL_LABELS); LoSoTo itself doesn't enforce an order (it reads
    the label strings, not a fixed position), but DP3/other consumers
    generally expect this one. faraday_angle stays npz-only either way —
    it's redundant once the Jones matrix itself is in the h5parm.
11. (--h5parm only) The antenna table's name field is fixed at 16 ASCII
    characters by LoSoTo's h5parm schema (and the source/direction name
    field at 128); longer antenna names are silently truncated by
    PyTables. OSKAR's own station names are normally well under this, but
    worth checking if yours aren't.
12. Source directions now come from the `point` regions of a wsclean
    facet-region file (the same one passed to wsclean via
    -facet-regions), not a plain "name ra dec" text file — see
    read_directions_from_facet_region(). Only lines whose shape keyword is
    literally "point(" are used (anchored at the start of the line, not a
    substring match — a facet-boundary `polygon(...)` line can carry its
    own "point=..." marker-style attribute in its comment and would
    otherwise be misread as a point region, silently grabbing a polygon
    vertex as the facet centre). The text= label is read whether it's
    written as text={...} or text="...". Coordinates are assumed to be
    plain decimal degrees, per claude/wsclean-facet-region-parsing-gotchas.md
    — this pipeline's facet-region files are never sexagesimal in the
    first place, because wsclean's own parser can't read sexagesimal
    coordinates either.
13. main() ends with `return locals()`, and the __main__ block does
    `globals().update(main())` instead of a plain `main()` call, to expose
    everything main() built (geom, screen, phases, jones, src_directions,
    ...) as top-level names for interactive debugging. Two things worth
    knowing about that pattern: (a) a name that was only ever a loop
    variable (t, ti, freq, tec, j, angle, ...) ends up holding just its
    *last* iteration's value, not a history across the loop; and (b) this
    only helps interactively — run with `python -i
    oskar_tec_phase_reproduce.py ...` (or %run in IPython/Jupyter) so the
    process doesn't exit before you get a prompt to inspect those globals
    from, and be aware that a stale global from an earlier --faraday run
    (e.g. `jones`) isn't cleared if a later run in the same session omits
    --faraday.

Requires: python-casacore, astropy, numpy, and igrf14.py (the IGRF-14 port
of OSKAR's oskar_station_evaluate_magnetic_field.c, delivered earlier in
this conversation) in the same directory as this script — only needed if
you pass --faraday. --h5parm additionally requires losoto.
    pip install python-casacore astropy numpy losoto

Usage:
    python oskar_tec_phase_reproduce.py obs.ms tec_screen.fits facets.reg \\
        -o phases.npz

    # also compute the Faraday-rotation Jones matrix per station/source/
    # channel/time (requires igrf14.py alongside this script):
    python oskar_tec_phase_reproduce.py obs.ms tec_screen.fits facets.reg \\
        -o phases.npz --faraday

    # also write the scalar phase solutions out as a LoSoTo-compatible h5parm:
    python oskar_tec_phase_reproduce.py obs.ms tec_screen.fits facets.reg \\
        -o phases.npz --h5parm phases.h5

    # --faraday + --h5parm together write the full Jones matrix to the
    # h5parm instead (amplitude000 + phase000, each with a 4-entry pol axis):
    python oskar_tec_phase_reproduce.py obs.ms tec_screen.fits facets.reg \\
        -o phases.npz --faraday --h5parm jones.h5

    facets.reg: a DS9-format facet region file (as passed to wsclean's
    -facet-regions), with one "point(ra_deg,dec_deg) # text={name}" region
    per facet centre/direction.
"""

from __future__ import annotations

import argparse
import sys
import re

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

try:
    from casacore.tables import table
except ImportError:
    sys.exit("This script requires python-casacore: pip install python-casacore")

try:
    import igrf14
except ImportError:
    igrf14 = None  # Only required if --faraday is used; checked in main().

try:
    import losoto.h5parm as lh5
except ImportError:
    lh5 = None  # Only required if --h5parm is used; checked in main().


TEC_TO_PHASE_CONST = -8.44797245e9  # cold-plasma dispersion constant (rad * Hz / TECU)
FARADAY_CONST = 2.36e2  # ITU-R P.531-6 eq. 2 constant (rad * GHz^2 / (T * TECU))


# --------------------------------------------------------------------------
# Measurement set geometry
# --------------------------------------------------------------------------

def read_ms_geometry(ms_path, avg_freq, avg_time):
    """Read antenna positions, phase centre, unique times, and channel
    frequencies from the MS. Returns a dict of numpy arrays."""
    with table(f"{ms_path}::ANTENNA", ack=False) as t:
        ant_pos = np.asarray(t.getcol("POSITION"))  # (n_ant, 3), ITRF metres
        ant_names = list(t.getcol("NAME"))

    with table(f"{ms_path}::OBSERVATION", ack=False) as t:
        array_pos = np.asarray(t.getcol("ARRAY_CENTER"))[0]  # (3,), m

    with table(f"{ms_path}::FIELD", ack=False) as t:
        phase_dir = np.asarray(t.getcol("PHASE_DIR"))  # (n_field, npoly, 2), radians
        if phase_dir.shape[0] != 1:
            print(f"  [warning] FIELD table has {phase_dir.shape[0]} rows; "
                  f"using row 0 as the phase centre", file=sys.stderr)
        ra0, dec0 = phase_dir[0, 0, :]

    with table(f"{ms_path}::SPECTRAL_WINDOW", ack=False) as t:
        n_spw = t.nrows()
        if n_spw != 1:
            sys.exit(f"MS has {n_spw} spectral windows; this script assumes exactly "
                     f"one (see the 'Set should have exactly one spectral window' "
                     f"lesson from earlier — split the MS to a single SPW first)")
        chan_freq = np.asarray(t.getcol("CHAN_FREQ"))[0]  # (n_chan,), Hz

    with table(ms_path, ack=False) as t:
        times = np.unique(np.asarray(t.getcol("TIME")))  # MJD UTC, seconds

    #ref_pos = ant_pos.mean(axis=0)
    
    ###### NB hardcoded!!!!! ######
    #print (ref_pos)
    #print ("ignoring and using hardcoded values")
    #ref_pos = np.array([-2565036.65272184, 5085748.26817034, -2861053.70315483])
    #-2565036.65274187,  5085748.26821007, -2861053.70295649
    #print (ref_pos)
    ref_loc = EarthLocation(x=array_pos[0] * u.m, y=array_pos[1] * u.m, z=array_pos[2] * u.m)
    
    ant_pos_rel = ant_pos - array_pos  # metres, relative to array "centre"

    print(f"{len(times)} time points")
    print(f"{len(chan_freq)} freq points")

    times = times[::avg_time]
    chan_freq = chan_freq[::avg_freq]

    print(f"after sampling")
    print(f"{len(times)} time points")
    print(f"{len(chan_freq)} freq points")
    
    return {
        "ant_pos": ant_pos,          # absolute ITRF (geocentric) metres
        "ant_pos_rel": ant_pos_rel,
        "ant_names": ant_names,
        "ra0": float(ra0),
        "dec0": float(dec0),
        "times": times,
        "chan_freq": chan_freq,
        "ref_lat": ref_loc.lat,
        "ref_lon": ref_loc.lon,
        "ref_height": ref_loc.height,
    }


def greenwich_hour_angle(times_mjd_seconds, ra0_rad):
    """Greenwich hour angle (radians) of the phase centre at each MS time.

    Uses Greenwich (not local) sidereal time because the antenna positions
    are in the ITRF frame, whose X axis is fixed to the Greenwich meridian.
    """
    t = Time(times_mjd_seconds / 86400.0, format="mjd", scale="utc")
    gst = t.sidereal_time("apparent", "greenwich").radian
    return gst - ra0_rad


def hour_angle(times_mjd_seconds, ra0_rad, longitude):
    """ hour angle (radians) of the phase centre at each MS time.

     local sidereal time 
    """
    t = Time(times_mjd_seconds / 86400.0, format="mjd", scale="utc")
    gst = t.sidereal_time("apparent", longitude).radian
    return gst - ra0_rad

def station_uv(ant_pos_rel, ha, dec0):
    """Per-station (u, v): projection of each station's position (relative
    to the array reference position) onto the plane perpendicular to the
    phase-centre direction, using the standard interferometric UVW
    rotation (Thompson, Moran & Swenson eq. 4.1), with Greenwich hour angle
    since positions are in the ITRF frame.

    ant_pos_rel: (n_ant, 3) metres
    ha: scalar, radians (Greenwich hour angle of the phase centre at this time)
    dec0: scalar, radians (phase centre declination)

    Returns (u, v), each shape (n_ant,), metres.
    """
    x, y, z = ant_pos_rel[:, 0], ant_pos_rel[:, 1], ant_pos_rel[:, 2]
    sin_h, cos_h = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec0), np.cos(dec0)

    u_ = sin_h * x + cos_h * y
    v_ = -sin_d * cos_h * x + sin_d * sin_h * y + cos_d * z
    #w_ = cos_d * cos_h * x - cos_d * sin_h * y + sin_d * z   # i think...
    return u_, v_


def radec_to_lm(ra, dec, ra0, dec0):
    """Direction cosines (l, m) of (ra, dec) relative to phase centre
    (ra0, dec0), all in radians (standard orthographic/SIN projection,
    e.g. Perley "Synthesis Imaging" eq. 3.1)."""
    dra = ra - ra0
    l = np.cos(dec) * np.sin(dra)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
    return l, m


# --------------------------------------------------------------------------
# TEC screen
# --------------------------------------------------------------------------

def read_tec_screen(fits_path, pixel_size_m=None, screen_height_km=None):
    """Read the TEC screen FITS cube and header geometry.

    Returns a dict with the data array (shape (n_freq, n_time, ny, nx)),
    pixel size (m), screen height (m), and the time/frequency axis
    reference values needed for nearest-slice selection.
    """
    with fits.open(fits_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header

    # Normalise to 4D (freq, time, y, x) regardless of how many axes
    # are actually present in this particular file.
    while data.ndim < 4:
        data = data[np.newaxis, ...]

    n_freq, n_time, ny, nx = data.shape

    def hget(*keys, default=None):
        for k in keys:
            if k in hdr:
                return hdr[k]
        return default

    if pixel_size_m is None:
        # CDELT1/2 are common FITS WCS keywords for pixel scale; ARatmospy
        # screens are typically written with pixel scale directly in metres.
        pixel_size_m = hget("PIXSIZE", "CDELT1", default=None)
        if pixel_size_m is not None:
            pixel_size_m = abs(float(pixel_size_m))
    if pixel_size_m is None:
        sys.exit("Could not determine pixel size from the FITS header; "
                  "pass --pixel-size-m explicitly")

    if screen_height_km is None:
        screen_height_km = hget("HEIGHT", "SCRHEIGHT", default=300.0)
    screen_height_m = float(screen_height_km) * 1000.0

    # Time axis reference (CRVAL3/CDELT3/CRPIX3), seconds, if present;
    # otherwise assume slice index == time index with unit spacing.
    t0 = hget("CRVAL3", default=0.0)
    dt = hget("CDELT3", default=1.0)
    crpix_t = hget("CRPIX3", default=1.0)

    print(f"  TEC screen: {nx}x{ny} pixels, {n_time} time slice(s), "
          f"{n_freq} freq plane(s)")
    print(f"  pixel_size_m={pixel_size_m}, screen_height_km={screen_height_km}")

    return {
        "data": data,
        "pixel_size_m": pixel_size_m,
        "screen_height_m": screen_height_m,
        "n_freq": n_freq,
        "n_time": n_time,
        "nx": nx,
        "ny": ny,
        "t0": float(t0),
        "dt": float(dt),
        "crpix_t": float(crpix_t),
    }


def time_slice_index(screen, t_mjd_seconds, t_ref_mjd_seconds):
    """Nearest time-slice index, using elapsed time since the start of the
    observation and the screen's time interval (matches OSKAR's
    nearest-slice-in-time behaviour rather than interpolating)."""
    if screen["n_time"] <= 1:
        return 0
    elapsed = t_mjd_seconds - t_ref_mjd_seconds
    idx = int(round(elapsed / screen["dt"]))
    return min(max(idx, 0), screen["n_time"] - 1)


def pierce_tec(screen, u, v, l, m, freq_idx, time_idx, give_pixels=False):
    """Vectorised nearest-pixel TEC lookup for arrays of station (u, v)
    and source (l, m). Returns delta-TEC, 0 where the pierce point falls
    outside the screen. u/v: shape (n_ant,); l/m: shape (n_src,).
    Returns array of shape (n_ant, n_src).
    """
    nx, ny = screen["nx"], screen["ny"]
    pix_size = screen["pixel_size_m"]
    h = screen["screen_height_m"]

    world_x = (u[:, None] + l[None, :] * h) / pix_size
    world_y = (v[:, None] + m[None, :] * h) / pix_size

    pix_x = nx // 2 + np.round(world_x).astype(int)
    pix_y = ny // 2 + np.round(world_y).astype(int)

    in_bounds = (pix_x >= 0) & (pix_x < nx) & (pix_y >= 0) & (pix_y < ny)

    tec = np.zeros_like(world_x)
    plane = screen["data"][freq_idx, time_idx]  # (ny, nx)
    tec[in_bounds] = plane[pix_y[in_bounds], pix_x[in_bounds]]
    if give_pixels:
        return tec, pix_x, pix_y
    else:
        return tec


def nearest_freq_index(screen, freq_hz, hdr_freq_axis=None):
    """Pick the TEC-screen frequency plane nearest to freq_hz. If the
    screen has only one frequency plane, always returns 0 (TEC itself is
    frequency-independent; see module docstring)."""
    if screen["n_freq"] <= 1:
        return 0
    if hdr_freq_axis is None:
        # No frequency WCS info available — fall back to plane 0 and warn.
        print("  [warning] screen has multiple frequency planes but no "
              "frequency axis info was found; using plane 0", file=sys.stderr)
        return 0
    return int(np.argmin(np.abs(hdr_freq_axis - freq_hz)))


# --------------------------------------------------------------------------
# Faraday rotation (--faraday)
# --------------------------------------------------------------------------

def station_geodetic(ant_pos):
    """Convert absolute ITRF (geocentric, metres) antenna positions to
    geodetic (lon_deg, lat_deg, height_m), for IGRF and AltAz use.

    ant_pos: (n_ant, 3) metres. Returns three (n_ant,) arrays.
    """
    loc = EarthLocation.from_geocentric(
        ant_pos[:, 0] * u.m, ant_pos[:, 1] * u.m, ant_pos[:, 2] * u.m)
    lon_deg = loc.lon.to(u.deg).value
    lat_deg = loc.lat.to(u.deg).value
    height_m = loc.height.to(u.m).value
    return lon_deg, lat_deg, height_m


def station_magnetic_fields(lat_deg, lon_deg, height_m, year):
    """Per-station geomagnetic field via igrf14.station_magnetic_field().

    Returns four (n_ant,) arrays: field_e, field_n, field_u, field_f (all
    nT), in the same East/North/Up frame used throughout this script.
    """
    n_ant = len(lat_deg)
    field_e = np.zeros(n_ant)
    field_n = np.zeros(n_ant)
    field_u = np.zeros(n_ant)
    field_f = np.zeros(n_ant)
    for i in range(n_ant):
        e, n, up, f = igrf14.station_magnetic_field(
            lat_deg[i], lon_deg[i], height_m[i], year)
        field_e[i], field_n[i], field_u[i], field_f[i] = e, n, up, f
    return field_e, field_n, field_u, field_f


def horizontal_unit_vectors(lon_deg, lat_deg, height_m, ras, decs, t_mjd_seconds):
    """East/North/Up unit vector from each station towards each source, at
    one MS time. Returns (hor_e, hor_n, hor_u), each shape (n_ant, n_src).

    Uses astropy's AltAz frame (geometric topocentric alt/az, no
    refraction) — see assumption 7 in the module docstring.
    """
    n_ant = len(lon_deg)
    loc = EarthLocation(
        lon=lon_deg[:, None] * u.deg,
        lat=lat_deg[:, None] * u.deg,
        height=height_m[:, None] * u.m)  # (n_ant, 1)
    sc = SkyCoord(ra=ras[None, :] * u.rad, dec=decs[None, :] * u.rad,
                  frame="icrs")  # (1, n_src), broadcasts to (n_ant, n_src)
    t = Time(t_mjd_seconds / 86400.0, format="mjd", scale="utc")
    altaz = sc.transform_to(AltAz(obstime=t, location=loc))

    alt = altaz.alt.radian
    az = altaz.az.radian  # measured from North, increasing through East
    hor_e = np.cos(alt) * np.sin(az)
    hor_n = np.cos(alt) * np.cos(az)
    hor_u = np.sin(alt)
    return hor_e, hor_n, hor_u


def faraday_jones(tec, freq_hz, b_los_tesla):
    """OSKAR_EVALUATE_TEC_SCREEN_WITH_FARADAY_ROTATION, transcribed.

    tec, b_los_tesla: broadcastable arrays (e.g. both (n_ant, n_src)).
    freq_hz: scalar.
    Returns a complex array of shape tec.shape + (2, 2):
        [..., 0, 0] = a,  [..., 0, 1] = b
        [..., 1, 0] = c,  [..., 1, 1] = d
    matching OSKAR's matx.a/b/c/d layout. Identity where tec == 0.
    """
    inv_frequency_hz = 1.0 / freq_hz
    inv_frequency_ghz = 1e9 * inv_frequency_hz
    f2 = inv_frequency_ghz * inv_frequency_ghz

    phase = tec * TEC_TO_PHASE_CONST * inv_frequency_hz
    faraday_angle = FARADAY_CONST * b_los_tesla * tec * f2

    phase_factor = np.exp(1j * phase)
    cos_angle = np.cos(faraday_angle)
    sin_angle = np.sin(faraday_angle)

    mask = tec != 0
    a = np.where(mask, cos_angle * phase_factor, 1.0 + 0j)
    b = np.where(mask, -sin_angle * phase_factor, 0j)
    c = np.where(mask, sin_angle * phase_factor, 0j)
    d = np.where(mask, cos_angle * phase_factor, 1.0 + 0j)

    jones = np.empty(tec.shape + (2, 2), dtype=complex)
    jones[..., 0, 0] = a
    jones[..., 0, 1] = b
    jones[..., 1, 0] = c
    jones[..., 1, 1] = d
    return jones, faraday_angle


# --------------------------------------------------------------------------
# LoSoTo h5parm output (--h5parm)
# --------------------------------------------------------------------------

def _populate_antenna_and_source_tables(solset, ant_names, ant_positions_itrf,
                                         direction_names, directions):
    """Shared by write_h5parm() and write_h5parm_fulljones(): fill in the
    antenna/source tables makeSolset() creates empty. Factored out so both
    writers populate them identically rather than keeping two copies of
    this in sync by hand."""
    ant_table = solset.obj._f_get_child("antenna")
    ant_table.append(list(zip(
        [name.encode("ascii") for name in ant_names],
        ant_positions_itrf.tolist(),
    )))

    src_table = solset.obj._f_get_child("source")
    src_table.append(list(zip(
        [name.encode("ascii") for name in direction_names],
        directions.tolist(),
    )))


def write_h5parm(
    output_path: str,
    phase: np.ndarray,
    times_mjd_s: np.ndarray,
    freqs_hz: np.ndarray,
    ant_names: list[str],
    ant_positions_itrf: np.ndarray,
    directions: np.ndarray,
    direction_names: list[str] | None = None,
    solset_name: str = "sol000",
    soltab_name: str = "phase000",
    history: str = "",
) -> None:
    """
    Write a LOFAR-style h5parm file using the losoto API.

    Parameters
    ----------
    output_path        : path for the output .h5 file
    phase              : phase solutions in radians,
                          #shape (n_dir, n_ant, n_freq, n_time)
                          shape (n_time, n_freq, n_ant, n_dir)
    times_mjd_s        : time centroids in MJD seconds, shape (n_time,)
    freqs_hz           : frequencies in Hz, shape (n_freq,)
    ant_names          : antenna/station names, length n_ant
    ant_positions_itrf : antenna ITRF XYZ positions in metres, shape (n_ant, 3)
                          pass np.zeros((n_ant, 3)) if unavailable
    directions         : source directions [[ra, dec], ...] in radians,
                          shape (n_dir, 2)
    direction_names    : names for each direction; defaults to Dir00, Dir01, ...
    solset_name        : name of the solution set (default 'sol000')
    soltab_name        : name of the solution table (default 'phase000')
    history            : optional history string
    """

    #n_dir, n_ant, n_freq, n_time = phase.shape
    n_time, n_freq, n_ant, n_dir  = phase.shape

    assert len(times_mjd_s)         == n_time,  "times length mismatch"
    assert len(freqs_hz)            == n_freq,  "freqs length mismatch"
    assert len(ant_names)           == n_ant,   "ant_names length mismatch"
    assert ant_positions_itrf.shape == (n_ant, 3), "ant_positions shape mismatch"
    assert directions.shape         == (n_dir, 2), "directions shape mismatch"

    if direction_names is None:
        direction_names = [f"Dir{i:02d}" for i in range(n_dir)]
    assert len(direction_names) == n_dir

    # ------------------------------------------------------------------
    # 1. Create h5parm — losoto writes the version attribute automatically
    # ------------------------------------------------------------------
    h5 = lh5.h5parm(output_path, readonly=False)

    # ------------------------------------------------------------------
    # 2. Create solset — losoto populates the antenna and source tables
    # ------------------------------------------------------------------
    solset = h5.makeSolset(solset_name)

    # ------------------------------------------------------------------
    # 3-4. Populate the ANTENNA and SOURCE tables (makeSolset creates them
    #    empty)
    # ------------------------------------------------------------------
    _populate_antenna_and_source_tables(
        solset, ant_names, ant_positions_itrf, direction_names, directions)

    # ------------------------------------------------------------------
    # 5. Create the soltab with phase solutions
    #    axesVals order must match axesNames and the phase array axes
    # ------------------------------------------------------------------
    weights = np.ones(phase.shape, dtype=np.float64)
    # Flag (set weight=0) any NaN phase values
    weights[~np.isfinite(phase)] = 0.0
    # Replace NaN in phase with 0 so HDF5 stores cleanly
    phase_out = np.where(np.isfinite(phase), phase, 0.0)

    solset.makeSoltab(
        soltype    = "phase",
        soltabName = soltab_name,
        #axesNames  = ["dir", "ant", "freq", "time"],
        axesNames  = ["time", "freq","ant","dir"],
        #axesVals   = [direction_names, ant_names, freqs_hz, times_mjd_s],
        axesVals   = [times_mjd_s, freqs_hz, ant_names, direction_names ],
        vals       = phase_out.astype(np.float64),
        weights    = weights.astype(np.float64),
    )

    # ------------------------------------------------------------------
    # 6. Add history if provided
    # ------------------------------------------------------------------
    if history:
        soltab = solset.getSoltabs()[0]
        soltab.addHistory(history)

    h5.close()

    print(
        f"Written {solset_name}/{soltab_name} "
        f"shape {phase.shape} (dir, ant, freq, time) "
        f"to {output_path!r}"
    )


# Which Jones-matrix element each LoSoTo "pol" label corresponds to, as
# element_index[label] = (row, col) into the 2x2 Jones matrix — i.e. row 0
# is the station's X output, row 1 its Y output, and likewise col 0/1 for
# the X/Y input, matching OSKAR's matx.a/b/c/d layout from faraday_jones():
# a=(0,0)=XX, b=(0,1)=XY, c=(1,0)=YX, d=(1,1)=YY.
JONES_ELEMENT_INDEX = {"XX": (0, 0), "XY": (0, 1), "YX": (1, 0), "YY": (1, 1)}

# LoSoTo doesn't actually enforce a pol order — operations like reset.py
# read whichever label string is stored at each index and key off that
# ('XX'/'YY' == diagonal, anything else == off-diagonal) rather than an
# assumed position. This order, XX/XY/YX/YY, is nonetheless the one used
# throughout the LOFAR/DP3 ecosystem's own tools (e.g. the killMS2H5parm.py
# converter shipped with this losoto install uses exactly this order), so
# it's the default here for interoperability with DP3 applycal and similar
# — pass a different `pol_labels` order if you have a specific reason to.
DEFAULT_POL_LABELS = ("XX", "XY", "YX", "YY")


def write_h5parm_fulljones(
    output_path: str,
    jones: np.ndarray,
    times_mjd_s: np.ndarray,
    freqs_hz: np.ndarray,
    ant_names: list[str],
    ant_positions_itrf: np.ndarray,
    directions: np.ndarray,
    direction_names: list[str] | None = None,
    solset_name: str = "sol000",
    amplitude_soltab_name: str = "amplitude000",
    phase_soltab_name: str = "phase000",
    pol_labels: tuple[str, str, str, str] = DEFAULT_POL_LABELS,
    history: str = "",
) -> None:
    """
    Write a full-Jones (2x2, per station/direction/channel/time) solution
    to a LOFAR-style h5parm, LoSoTo-style: as a pair of soltabs —
    amplitude and phase — sharing a ["pol", "dir", "ant", "freq", "time"]
    axis layout, where "pol" has the 4 entries in `pol_labels` (default
    XX, XY, YX, YY). This is the standard way a full 2x2 Jones solution is
    represented in an h5parm; there is no single "complex" soltype, and a
    phase-only soltab (as write_h5parm() above writes for the scalar,
    non-Faraday case) would silently drop the amplitude/polarisation-
    leakage information a true Jones matrix carries off the unit circle —
    i.e. unlike the scalar case, |J_XX|, |J_XY| etc. are not all 1 here
    (only JONES @ JONES^H is unitary; see faraday_jones()'s own det/
    unitarity checks), so amplitude has to be written too.

    Parameters
    ----------
    output_path        : path for the output .h5 file
    jones               : complex Jones matrices, shape
                          (n_dir, n_ant, n_freq, n_time, 2, 2) — exactly
                          what faraday_jones() produces and main() already
                          reorders the --faraday "jones" array into.
    times_mjd_s        : time centroids in MJD seconds, shape (n_time,)
    freqs_hz           : frequencies in Hz, shape (n_freq,)
    ant_names          : antenna/station names, length n_ant
    ant_positions_itrf : antenna ITRF XYZ positions in metres, shape (n_ant, 3)
    directions         : source directions [[ra, dec], ...] in radians,
                          shape (n_dir, 2)
    direction_names    : names for each direction; defaults to Dir00, Dir01, ...
    solset_name        : name of the solution set (default 'sol000')
    amplitude_soltab_name : name of the amplitude solution table (default 'amplitude000')
    phase_soltab_name  : name of the phase solution table (default 'phase000')
    pol_labels         : the 4 polarisation labels, in the order they'll be
                          stored — see JONES_ELEMENT_INDEX for which Jones
                          element each label maps to. Default: ('XX', 'XY',
                          'YX', 'YY').
    history            : optional history string
    """
    n_dir, n_ant, n_freq, n_time = jones.shape[:4]

    assert jones.shape[4:] == (2, 2), f"expected a 2x2 Jones matrix, got {jones.shape[4:]}"
    assert len(times_mjd_s)         == n_time,  "times length mismatch"
    assert len(freqs_hz)            == n_freq,  "freqs length mismatch"
    assert len(ant_names)           == n_ant,   "ant_names length mismatch"
    assert ant_positions_itrf.shape == (n_ant, 3), "ant_positions shape mismatch"
    assert directions.shape         == (n_dir, 2), "directions shape mismatch"
    assert len(pol_labels) == 4, "pol_labels must have exactly 4 entries"
    assert set(pol_labels) <= set(JONES_ELEMENT_INDEX), (
        f"pol_labels must be a permutation of {tuple(JONES_ELEMENT_INDEX)}")

    if direction_names is None:
        direction_names = [f"Dir{i:02d}" for i in range(n_dir)]
    assert len(direction_names) == n_dir

    # (n_pol, n_dir, n_ant, n_freq, n_time), in the requested pol_labels order.
    complex_vals = np.empty((4, n_dir, n_ant, n_freq, n_time), dtype=complex)
    for p, label in enumerate(pol_labels):
        row, col = JONES_ELEMENT_INDEX[label]
        complex_vals[p] = jones[..., row, col]

    amplitude = np.abs(complex_vals)
    phase = np.angle(complex_vals)
    # Flag (weight=0) and zero out any non-finite values, same convention
    # as write_h5parm()'s scalar case.
    finite = np.isfinite(amplitude) & np.isfinite(phase)
    weights = finite.astype(np.float64)
    amplitude = np.where(finite, amplitude, 0.0)
    phase = np.where(finite, phase, 0.0)

    h5 = lh5.h5parm(output_path, readonly=False)
    solset = h5.makeSolset(solset_name)
    _populate_antenna_and_source_tables(
        solset, ant_names, ant_positions_itrf, direction_names, directions)

    axes_names = ["pol", "dir", "ant", "freq", "time"]
    axes_vals = [list(pol_labels), direction_names, ant_names, freqs_hz, times_mjd_s]

    amp_soltab = solset.makeSoltab(
        soltype    = "amplitude",
        soltabName = amplitude_soltab_name,
        axesNames  = axes_names,
        axesVals   = axes_vals,
        vals       = amplitude.astype(np.float64),
        weights    = weights,
    )
    phase_soltab = solset.makeSoltab(
        soltype    = "phase",
        soltabName = phase_soltab_name,
        axesNames  = axes_names,
        axesVals   = axes_vals,
        vals       = phase.astype(np.float64),
        weights    = weights,
    )

    if history:
        amp_soltab.addHistory(history)
        phase_soltab.addHistory(history)

    h5.close()

    print(
        f"Written {solset_name}/{amplitude_soltab_name} and "
        f"{solset_name}/{phase_soltab_name}, shape {amplitude.shape} "
        f"(pol={list(pol_labels)}, dir, ant, freq, time) to {output_path!r}"
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def read_sources(path):
    """Plain-text "name ra_deg dec_deg" source list. No longer called from
    main() now that source directions come from a facet region file (see
    read_directions_from_facet_region below) — left here in case you want
    that input mode back for a quick test without a region file."""
    names, ras, decs = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, ra, dec = line.split()[:3]
            names.append(name)
            ras.append(float(ra))
            decs.append(float(dec))
    return names, np.radians(ras), np.radians(decs)


def read_directions_from_facet_region(
    regionfile
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Parse source (name, ra, dec) from the `point` regions of a DS9-format
    facet region file — the same file passed to wsclean via -facet-regions.

    Coordinates are assumed to be plain decimal degrees, never sexagesimal
    (see claude/wsclean-facet-region-parsing-gotchas.md: this pipeline's
    facet-region files are always generated in decimal degrees in the first
    place, because wsclean's own parser can't read sexagesimal coordinates).

    Only lines whose shape keyword is "point" are used; `polygon(...)`
    facet-boundary lines are skipped even if their own comment attributes
    happen to mention "point=" (a stray marker-style attribute some region
    writers add to every shape regardless of type) — matching "point"
    anywhere in the line, rather than anchoring to the start of the shape,
    would otherwise pick up a polygon vertex as if it were the facet centre.
    The `text=` label is read whether it's written as text={...} or
    text="..." (both are valid DS9 syntax and both occur across different
    facet-region generators); a point region with no text= label gets a
    generated DirNN name (with a warning) rather than crashing.
    """
    with open(regionfile) as f:
        lines = f.readlines()

    names: list[str] = []
    ras = []
    decs = []
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith("point("):
            continue

        coord_match = re.search(r'\((.+?)\)', stripped)
        if coord_match is None:
            print(f"  [warning] {regionfile}:{lineno}: 'point(' with no "
                  f"matching ')', skipping", file=sys.stderr)
            continue
        ra_str, dec_str = coord_match.group(1).split(",")[:2]
        ra = float(ra_str)
        dec = float(dec_str)

        text_match = re.search(r'text=(\{([^}]*)\}|"([^"]*)")', stripped)
        if text_match:
            name = text_match.group(2) if text_match.group(2) is not None else text_match.group(3)
        else:
            name = f"Dir{len(names):02d}"
            print(f"  [warning] {regionfile}:{lineno}: point region has no "
                  f"text= label, using generated name {name!r}", file=sys.stderr)

        names.append(name)
        print(name, ra, dec)
        ras.append(np.radians(ra))
        decs.append(np.radians(dec))

    ras = np.array(ras)
    decs = np.array(decs)

    return names, ras, decs


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce OSKAR's TEC-screen phase corruption outside of OSKAR.")
    parser.add_argument("ms", help="Measurement set written by OSKAR")
    parser.add_argument("fits_screen", help="TEC screen FITS file OSKAR read from")
    parser.add_argument("regfile", help="Source DS9 facet-region file (the same one "
                                         "passed to wsclean -facet-regions)")
    parser.add_argument("-o", "--output", default="oskar_tec_phases.npz",
                         help="Output .npz file (default: oskar_tec_phases.npz)")
    parser.add_argument("--pixel-size-m", type=float, default=None,
                         help="Override the screen pixel size (metres) if not "
                              "auto-detected correctly from the FITS header")
    parser.add_argument("--screen-height-km", type=float, default=None,
                         help="Override the screen height (km) if not auto-detected "
                              "correctly from the FITS header (OSKAR default: 300)")
    parser.add_argument("--channel", type=int, default=None,
                         help="Compute phases for a single MS channel index only "
                              "(default: all channels)")
    parser.add_argument("--faraday", action="store_true",
                         help="Also compute the Faraday-rotation Jones matrix per "
                              "station/source/channel/time, using IGRF-14 "
                              "(via igrf14.py, required alongside this script) "
                              "for each station's geomagnetic field. See "
                              "assumptions 6-9 in the module docstring.")
    parser.add_argument("--year", type=float, default=None,
                         help="Decimal year for the IGRF-14 geomagnetic field "
                              "(--faraday only). Default: first value of the MS's "
                              "observation times.")
    parser.add_argument("--avg_time", type=int, default=1,
                         help="Take every avg_time time samples")
    parser.add_argument("--avg_freq", type=int, default=1,
                         help="Take every avg_freq channels")
    parser.add_argument("--h5parm", default=None,
                         help="Also write solutions to this path as a LoSoTo-"
                              "compatible h5parm file (requires losoto). Without "
                              "--faraday: the scalar phase, as a single phase000 "
                              "soltab. With --faraday: the full 2x2 Jones matrix, "
                              "as a pol/dir/ant/freq/time amplitude+phase soltab "
                              "pair instead — see write_h5parm_fulljones().")
    parser.add_argument("--solset-name", default="sol000",
                         help="h5parm solution-set name (--h5parm only; default: sol000)")
    parser.add_argument("--soltab-name", default="phase000",
                         help="h5parm phase solution-table name (--h5parm only; "
                              "default: phase000)")
    parser.add_argument("--amplitude-soltab-name", default="amplitude000",
                         help="h5parm amplitude solution-table name (--h5parm "
                              "--faraday only; default: amplitude000)")
    args = parser.parse_args()

    if args.faraday and igrf14 is None:
        sys.exit("--faraday requires igrf14.py (the IGRF-14 port of OSKAR's "
                  "oskar_station_evaluate_magnetic_field.c) in the same "
                  "directory as this script.")

    if args.h5parm and lh5 is None:
        sys.exit("--h5parm requires losoto: pip install losoto")

    print(f"Reading MS geometry from {args.ms}")
    geom = read_ms_geometry(args.ms, args.avg_freq, args.avg_time)
    print(f"  phase centre: RA={np.degrees(geom['ra0']):.6f} deg, "
          f"Dec={np.degrees(geom['dec0']):.6f} deg")
    print(f"  {len(geom['ant_pos_rel'])} antennas, {len(geom['times'])} unique times, "
          f"{len(geom['chan_freq'])} channels")


    print(f"Reading TEC screen from {args.fits_screen}")
    screen = read_tec_screen(args.fits_screen, args.pixel_size_m, args.screen_height_km)

    print(f"Reading source list from {args.regfile}")
    src_names, ras, decs = read_directions_from_facet_region(args.regfile)
    # Kept as a SkyCoord (not reduced to a plain array) so it's directly
    # inspectable (e.g. src_directions.to_string('hmsdms')) when exposed as
    # a global for debugging below; `ras`/`decs` (radians) remain the arrays
    # actually used everywhere downstream.
    src_directions = SkyCoord(ra=ras * u.rad, dec=decs * u.rad, frame="icrs")
    nsrc = len(src_names)

    l, m = radec_to_lm(ras, decs, geom["ra0"], geom["dec0"])
    print(f"  {len(src_names)} source(s)")
    chan_freq = geom["chan_freq"]
    if args.channel is not None:
        chan_freq = chan_freq[[args.channel]]

    n_time = len(geom["times"])
    n_ant = len(geom["ant_pos_rel"])
    n_src = len(src_names)
    n_chan = len(chan_freq)

    phases = np.zeros((n_time, n_ant, n_src, n_chan))
    tec_values = np.zeros((n_time, n_ant, n_src))
    pix_x_values = np.zeros((n_time, n_ant, n_src))
    pix_y_values = np.zeros((n_time, n_ant, n_src))

    if args.faraday:
        lon_deg, lat_deg, height_m = station_geodetic(geom["ant_pos"])
        if args.year is not None:
            year = args.year
        else:
            year = float(Time(geom["times"][0] / 86400.0,
                               format="mjd", scale="utc").decimalyear)
        print(f"  --faraday: using IGRF-14 epoch {year:.4f}")
        field_e, field_n, field_u, field_f = station_magnetic_fields(
            lat_deg, lon_deg, height_m, year)
        print(f"  station |B| range: {field_f.min():.1f}-{field_f.max():.1f} nT")

        jones = np.empty((n_time, n_ant, n_src, n_chan, 2, 2), dtype=complex)
        faraday_angle = np.zeros((n_time, n_ant, n_src, n_chan))

    t_ref = geom["times"][0]
    for ti, t in enumerate(geom["times"]):
        ha = greenwich_hour_angle(np.array([t]), geom["ra0"])[0]
        #ha = hour_angle(np.array([t]), geom["ra0"], geom["ref_lon"])[0]
        u_, v_ = station_uv(geom["ant_pos_rel"], ha, geom["dec0"])
        time_idx = time_slice_index(screen, t, t_ref)

        if args.faraday:
            hor_e, hor_n, hor_u = horizontal_unit_vectors(
                lon_deg, lat_deg, height_m, ras, decs, t)  # (n_ant, n_src)
            # Source vector components are negated because the line of
            # sight runs from source to observer, not observer to source.
            b_los_tesla = -(hor_e * field_e[:, None]
                            + hor_n * field_n[:, None]
                            + hor_u * field_u[:, None]) * 1e-9

        for ci, freq in enumerate(chan_freq):
            freq_idx = nearest_freq_index(screen, freq)
            tec, pix_x, pix_y = pierce_tec(screen, u_, v_, l, m, freq_idx, time_idx, give_pixels=True)  # (n_ant, n_src)
            if ci == 0:
                tec_values[ti] = tec
                pix_x_values[ti] = pix_x
                pix_y_values[ti] = pix_y
            phases[ti, :, :, ci] = tec * TEC_TO_PHASE_CONST / freq

            if args.faraday:
                j, angle = faraday_jones(tec, freq, b_los_tesla)
                jones[ti, :, :, ci, :, :] = j
                faraday_angle[ti, :, :, ci] = angle


    import matplotlib.pyplot as plt
    plt.figure()
    #plt.xlim(0,2000)
    #plt.ylim(0,2000)
    for stat_i in range(n_ant):
        for dir_i in range(n_src):
            plt.scatter(pix_x_values[:,stat_i,dir_i], pix_y_values[:,stat_i,dir_i], c=range(n_time))
    plt.savefig("pierce_points_vs_time.png")

    # Computed above as (time, ant, src, freq); reorder to (src, ant, freq, time)
    # to match the axis order expected by the downstream HDF5 writer.
    #phases = np.transpose(phases, (2, 1, 3, 0))
    #tec_values = np.transpose(tec_values, (2, 1, 0))  # (src, ant, time)
    #want time freq ant dir
    phases = np.transpose(phases, (0, 3,1,2))
    #tec_values = np.transpose(tec_values, (2, 1, 0))  # (src, ant, time)

    save_kwargs = dict(
        phases=phases, tec=tec_values, names=src_names,
        times=geom["times"], chan_freq=chan_freq)

    if args.faraday:
        # (time, ant, src, chan, 2, 2) -> (src, ant, chan, time, 2, 2)
        jones = np.transpose(jones, (2, 1, 3, 0, 4, 5))
        faraday_angle = np.transpose(faraday_angle, (2, 1, 3, 0))
        save_kwargs["jones"] = jones
        save_kwargs["faraday_angle"] = faraday_angle

    np.savez(args.output, **save_kwargs)
    msg = (f"Wrote phases (shape {phases.shape}, axes = [source, antenna, channel, "
           f"time]) and intermediate TEC values to {args.output}")
    if args.faraday:
        msg += (f"\nAlso wrote Faraday-rotation Jones matrices (shape {jones.shape}, "
                f"axes = [source, antenna, channel, time, 2, 2]) and the underlying "
                f"faraday_angle (radians, same leading axes) to the same file.")
    print(msg)

    if args.h5parm:
        # `phases`/`jones` are already (n_dir, n_ant, n_chan, n_time[, 2, 2])
        # from the reorder above, matching what the h5parm writers expect.
        directions = np.column_stack([ras, decs])  # (n_src, 2) radians
        history = (f"Produced by oskar_tec_phase_reproduce.py from {args.ms} "
                   f"and {args.fits_screen}")
        if args.faraday:
            write_h5parm_fulljones(
                output_path=args.h5parm,
                jones=jones,
                times_mjd_s=geom["times"],
                freqs_hz=chan_freq,
                ant_names=geom["ant_names"],
                ant_positions_itrf=geom["ant_pos"],
                directions=directions,
                direction_names=src_names,
                solset_name=args.solset_name,
                amplitude_soltab_name=args.amplitude_soltab_name,
                phase_soltab_name=args.soltab_name,
                history=history,
            )
        else:
            write_h5parm(
                output_path=args.h5parm,
                phase=phases,
                times_mjd_s=geom["times"],
                freqs_hz=chan_freq,
                ant_names=geom["ant_names"],
                ant_positions_itrf=geom["ant_pos"],
                directions=directions,
                direction_names=src_names,
                solset_name=args.solset_name,
                soltab_name=args.soltab_name,
                history=history,
            )
    return locals()   # everything main() built, as a dict


if __name__ == "__main__":
    #main()
    globals().update(main())


'''
2561069.95910563, -5088350.0774816 ,  2859918.14759729
116.764448240 -26.824722085 345.639

from astropy.coordinates import EarthLocation
import astropy.units as u

# Define ITRF Cartesian coordinates (in meters or kilometers with units)
x, y, z = -2561069.95910563,  5088350.0774816 , -2859918.14759729 # example values in meters
loc = EarthLocation(x=x * u.m, y=y * u.m, z=z * u.m)

# Convert to geodetic coordinates (latitude, longitude, height)
lat = loc.lat
lon = loc.lon
height = loc.height

print(f"Latitude: {lat}")
print(f"Longitude: {lon}")
print(f"Height: {height}")

lon, lat, h = 116.764448240, -26.824722085, 345.639
from astropy.coordinates import EarthLocation
loc = EarthLocation(lon=lon, lat=lat, height=h)

'''
