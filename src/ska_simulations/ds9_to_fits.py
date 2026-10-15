#!/usr/bin/env python3
"""
ds9_to_fits.py — Convert a DS9 region file into a FITS table readable with
astropy.table.Table.read().

Supports the common DS9 sky-coordinate shapes: point, circle, ellipse,
box/rectangle, and polygon (e.g. the facet regions produced by a Voronoi
tessellation: one polygon + one point per facet, tagged with the same
`text=` label). Regions must be in a sky frame (fk5/icrs/galactic/...) —
region files given in image/physical pixel coordinates can't be converted
without a WCS and are skipped with a warning.

Output columns:
    id           running row index
    region_type  'point' | 'circle' | 'ellipse' | 'box' | 'polygon'
    ra, dec      degrees. For shapes with a defined centre, this is that
                 centre; for polygons, this is the mean (centroid) of the
                 vertices (a simple average — not corrected for RA wrap or
                 spherical distortion, fine for compact regions)
    radius       degrees, circles only (NaN otherwise)
    width,height degrees, ellipses/boxes only (NaN otherwise) — DS9's
                 ellipse() semi-axes are stored here as full axis lengths
                 (2x the semi-major/semi-minor radii), matching the
                 `regions` package convention
    angle        degrees, ellipses/boxes only (NaN otherwise)
    text         the region's `text=` label/tag, if any (empty string if none)
    vertices     polygons only: "ra1,dec1;ra2,dec2;..." string of vertex
                 coordinates in degrees (empty string otherwise)

Requires: astropy, regions (pip install astropy regions)

Usage:
    python ds9_to_fits.py input.reg -o output.fits
"""

import argparse
import sys

import numpy as np
from astropy.table import Table

try:
    import regions
except ImportError:
    sys.exit("This script requires the 'regions' package: pip install regions")


def region_to_row(idx, region):
    """Convert one parsed region object into a row dict, or None to skip it."""
    text = region.meta.get("text", "")
    row = {
        "id": idx,
        "region_type": "",
        "ra": np.nan,
        "dec": np.nan,
        "radius": np.nan,
        "width": np.nan,
        "height": np.nan,
        "angle": np.nan,
        "text": text,
        "vertices": "",
    }

    cls = type(region).__name__

    if cls == "PointSkyRegion":
        row["region_type"] = "point"
        row["ra"] = region.center.ra.deg
        row["dec"] = region.center.dec.deg

    elif cls == "CircleSkyRegion":
        row["region_type"] = "circle"
        row["ra"] = region.center.ra.deg
        row["dec"] = region.center.dec.deg
        row["radius"] = region.radius.to("deg").value

    elif cls == "EllipseSkyRegion":
        row["region_type"] = "ellipse"
        row["ra"] = region.center.ra.deg
        row["dec"] = region.center.dec.deg
        row["width"] = region.width.to("deg").value
        row["height"] = region.height.to("deg").value
        row["angle"] = region.angle.to("deg").value

    elif cls == "RectangleSkyRegion":
        row["region_type"] = "box"
        row["ra"] = region.center.ra.deg
        row["dec"] = region.center.dec.deg
        row["width"] = region.width.to("deg").value
        row["height"] = region.height.to("deg").value
        row["angle"] = region.angle.to("deg").value

    elif cls == "PolygonSkyRegion":
        row["region_type"] = "polygon"
        ra_v = region.vertices.ra.deg
        dec_v = region.vertices.dec.deg
        row["ra"] = float(np.mean(ra_v))
        row["dec"] = float(np.mean(dec_v))
        row["vertices"] = ";".join(f"{r:.8f},{d:.8f}" for r, d in zip(ra_v, dec_v))

    else:
        print(f"  [skip] row {idx}: unsupported or pixel-based region type {cls}", file=sys.stderr)
        return None

    return row


def convert(input_path, output_path):
    parsed = regions.Regions.read(input_path, format="ds9")

    rows = []
    for i, region in enumerate(parsed):
        row = region_to_row(i, region)
        if row is not None:
            rows.append(row)

    if not rows:
        sys.exit("No convertible (sky-coordinate) regions found in the input file.")

    table = Table(
        rows=rows,
        names=["id", "region_type", "ra", "dec", "radius", "width", "height", "angle", "text", "vertices"],
    )
    table.write(output_path, format="fits", overwrite=True)
    print(f"Wrote {len(table)} regions to {output_path}")
    print(table)


def main():
    parser = argparse.ArgumentParser(description="Convert a DS9 region file to a FITS table.")
    parser.add_argument("input", help="Input DS9 region file (.reg)")
    parser.add_argument("-o", "--output", default="regions.fits", help="Output FITS filename (default: regions.fits)")
    args = parser.parse_args()

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
