"""
Paint elliptical source footprints from a sky catalogue onto a template FITS
image to make a WSClean ``-fits-mask``.

CLI: ``ska-make-mask image.fits catalogue.fits mask.fits``
"""
from __future__ import annotations

import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


def paint_sources_onto_image(
    image_path: str,
    catalogue_path: str,
    output_path: str,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    maj_col: str = "Maj",
    min_col: str = "Min",
    pa_col: str = "PA",
    shape_unit: u.Unit = u.arcsec,
    min_radius_pix: float = 2,
) -> None:
    """
    Paint elliptical source footprints (value=1) onto a template FITS image.

    Parameters
    ----------
    image_path   : path to template FITS image (provides WCS and shape)
    catalogue_path : path to FITS catalogue with source positions and shapes
    output_path  : path for the output FITS image
    ra_col       : catalogue column name for Right Ascension (degrees)
    dec_col      : catalogue column name for Declination (degrees)
    maj_col      : catalogue column name for semi-major axis
    min_col      : catalogue column name for semi-minor axis
    pa_col       : catalogue column name for position angle (degrees E of N)
    shape_unit   : unit of maj/min columns (default arcsec)
    min_radius_pix : minimum semi-axis painted, in pixels (default 2)
    """

    # --- 1. Load image and WCS ---
    with fits.open(image_path) as hdul:
        header = hdul[0].header
        data   = hdul[0].data.copy()
        data   = data*0

    # Handle 4D FITS cubes (e.g. Stokes/freq axes): work on the 2D sky plane
    # WCS needs the full header but we index into the array with [..., y, x]
    wcs_full = WCS(header)
    wcs      = wcs_full.celestial          # drops non-spatial axes

    # Image dimensions (last two axes are always sky for standard FITS)
    ny, nx = data.shape[-2], data.shape[-1]

    # Build output array — same shape as input, initialised to zero
    out = np.zeros_like(data, dtype=np.float32)

    # --- 2. Pixel scale (arcsec / pixel) from WCS ---
    # Take the mean of the two axes in case pixels are slightly non-square
    scales     = wcs.proj_plane_pixel_scales()   # returns Quantity array
    pix_scale  = np.mean([s.to(shape_unit).value for s in scales])

    # --- 3. Load catalogue ---
    with fits.open(catalogue_path) as hdul:
        cat = hdul[1].data                       # BinTableHDU is extension 1

    ra_deg  = cat[ra_col].astype(float)
    dec_deg = cat[dec_col].astype(float)
    maj     = cat[maj_col].astype(float)         # in shape_unit
    min_ax  = cat[min_col].astype(float)
    pa_deg  = cat[pa_col].astype(float)          # position angle E of N

    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    # --- 4. Convert source centres to pixel coordinates ---
    # wcs.world_to_pixel returns (x, y) for a SkyCoord
    cx_arr, cy_arr = wcs.world_to_pixel(coords)  # float arrays, 0-indexed

    # --- 5. Paint each source ---
    for cx, cy, a, b, pa in zip(cx_arr, cy_arr, maj, min_ax, pa_deg):

        # Skip sources whose centres fall outside the image
        if not (0 <= cx < nx and 0 <= cy < ny):
            continue

        # Convert ellipse axes from sky units to pixels
        a_pix = a / pix_scale          # semi-major axis in pixels
        b_pix = b / pix_scale          # semi-minor axis in pixels
        
        a_pix = max(a_pix, min_radius_pix)     # ensure at least min_radius_pix
        b_pix = max(b_pix, min_radius_pix)

        # Bounding box: pad by the semi-major axis to be safe
        pad    = int(np.ceil(a_pix)) + 1
        x0, x1 = max(0, int(cx) - pad), min(nx, int(cx) + pad + 1)
        y0, y1 = max(0, int(cy) - pad), min(ny, int(cy) + pad + 1)

        # Pixel coordinate grids within the bounding box
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        xx, yy = np.meshgrid(xs, ys)

        # Rotate pixel offsets by position angle
        # PA is measured E of N; in pixel space N is +y, E is -x (for
        # standard RA/Dec WCS where RA increases to the left)
        # there's an offset of 90 deg here
        dx = xx - cx
        dy = yy - cy
        pa_rad = np.deg2rad(pa + 90)

        # Rotate into ellipse frame
        dx_rot =  dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
        dy_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)

        # Ellipse equation: points with value <= 1 are inside
        ellipse = (dx_rot / a_pix) ** 2 + (dy_rot / b_pix) ** 2

        mask = ellipse <= 1.0

        # Paint into the correct slice (handles 2D, 3D, 4D FITS)
        out[..., y0:y1, x0:x1][..., mask] = 1.0

    # --- 6. Write output ---
    hdu = fits.PrimaryHDU(data=out, header=header)
    hdu.writeto(output_path, overwrite=True)
    print(f"Written {out.sum():.0f} pixels painted across {len(cat)} sources")
    print(f"Output: {output_path!r}")



def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Make a FITS CLEAN mask by painting catalogue sources onto a template image"
    )
    parser.add_argument("image_template", help="The image to use as a template")
    parser.add_argument("source_catalogue", help="The source catlogue")
    parser.add_argument("mask_file", help="The name output mask file")
    parser.add_argument("--min-radius-pix", type=float, default=2, help="Minimum ellipse semi-axis in pixels")

    args = parser.parse_args(argv)
    
    paint_sources_onto_image(
    image_path     = args.image_template,
    catalogue_path = args.source_catalogue,
    output_path    = args.mask_file,
    ra_col         = "RA",
    dec_col        = "DEC",
    maj_col        = "MAJ",       # semi-major axis
    min_col        = "MIN",       # semi-minor axis
    pa_col         = "PA",        # position angle, degrees E of N
    shape_unit     = u.arcsec,
    min_radius_pix = args.min_radius_pix,
    )

if __name__ == "__main__":
    main()
