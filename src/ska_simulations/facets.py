"""
Voronoi-tessellate an image into facets around catalogue sources and write a
DS9 region file suitable for WSClean ``-facet-regions``.

CLI example (as used for the COSMOS DD-calibrator facets)::

    ska-make-facets ws_..._robm1_mask-MFS-image.fits AA2_..._skytab.fits \
        facets_ddcals46_sources.reg --label-col NAME --min-flux 0.25 --merge-radius 1
"""
from __future__ import annotations

import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import Voronoi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def merge_close_sources(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    flux: np.ndarray,
    labels: list[str],
    merge_radius: u.Quantity,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Merge sources within merge_radius of each other into a single
    flux-weighted position. Uses connected components so chains of
    close sources (A-B close, B-C close) are merged as one group,
    not just pairs.

    Parameters
    ----------
    ra_deg, dec_deg : source positions in degrees
    flux            : flux values used as weights for the merged centroid
    labels          : per-source labels (merged group keeps the label
                       of its brightest member)
    merge_radius    : angular radius within which sources are merged,
                       e.g. 30*u.arcsec

    Returns
    -------
    merged_ra, merged_dec, merged_labels
    """
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    n = len(coords)

    # Find all pairs within merge_radius (self-matches excluded)
    idx1, idx2, sep2d, _ = coords.search_around_sky(coords, merge_radius)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    # Build a sparse adjacency graph and find connected components —
    # this correctly groups chains/clusters, not just isolated pairs
    graph = coo_matrix(
        (np.ones(len(idx1)), (idx1, idx2)), shape=(n, n)
    )
    n_groups, group_id = connected_components(
        graph, directed=False
    )

    merged_ra, merged_dec, merged_labels = [], [], []

    for g in range(n_groups):
        members = np.where(group_id == g)[0]

        if len(members) == 1:
            i = members[0]
            merged_ra.append(ra_deg[i])
            merged_dec.append(dec_deg[i])
            merged_labels.append(labels[i])
            continue

        # Flux-weighted centroid for the merged group
        w = flux[members]
        w_sum = w.sum()
        if w_sum <= 0:
            w = np.ones_like(w)   # fall back to simple mean if flux invalid
            w_sum = w.sum()

        ra_c  = np.sum(ra_deg[members] * w) / w_sum
        dec_c = np.sum(dec_deg[members] * w) / w_sum

        merged_ra.append(ra_c)
        merged_dec.append(dec_c)

        # Label the merged group after its brightest member
        brightest = members[np.argmax(flux[members])]
        merged_labels.append(labels[brightest])

        print(
            f"Merged {len(members)} sources within "
            f"{merge_radius} into one facet centre "
            f"(brightest: {labels[brightest]})"
        )

    return np.array(merged_ra), np.array(merged_dec), merged_labels


def voronoi_tessellate_image(
    image_path: str,
    catalogue_path: str,
    output_path: str,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    filter_Icol: str = "I",
    filter_I: float = 0,
    label_col: str | None = None,
    merge_radius: u.Quantity | None = None,
) -> None:
    """
    Tessellate a FITS image using Voronoi decomposition around N points
    of interest, bounded by the image edges. Writes a DS9-compatible
    region file with polygon vertices in sky coordinates.

    Parameters
    ----------
    image_path     : FITS image with WCS header
    catalogue_path : FITS catalogue with source sky coordinates
    output_path    : output region file path (e.g. facets.reg)
    ra_col         : RA column name in catalogue (degrees)
    dec_col        : Dec column name in catalogue (degrees)
    filter_Icol    : column used to filter sources by flux
    filter_I       : minimum flux to keep a source
    label_col      : optional column for facet labels; defaults to facet_0, facet_1 ...
    merge_radius   : if given, sources within this angular separation are
                      merged into a single flux-weighted facet centre
                      before tessellation, e.g. 30*u.arcsec
    """

    # ------------------------------------------------------------------
    # 1. Load image WCS and build the image bounding polygon in pixel space
    # ------------------------------------------------------------------
    with fits.open(image_path) as hdul:
        header = hdul[0].header
        ny     = header["NAXIS2"]
        nx     = header["NAXIS1"]

    wcs = WCS(header).celestial

    corner_pix = np.array([
        [0,    0   ],
        [nx-1, 0   ],
        [nx-1, ny-1],
        [0,    ny-1],
    ])
    image_bbox_pix = Polygon(corner_pix)

    # ------------------------------------------------------------------
    # 2. Load catalogue, filter, and optionally merge close sources
    # ------------------------------------------------------------------
    cat = Table.read(catalogue_path)
    cat = cat[cat[filter_Icol] > filter_I]

    ra_deg  = np.array(cat[ra_col],       dtype=float)
    dec_deg = np.array(cat[dec_col],      dtype=float)
    flux    = np.array(cat[filter_Icol],  dtype=float)

    labels = (
        [str(v) for v in cat[label_col]]
        if label_col and label_col in cat.colnames
        else [f"facet_{i}" for i in range(len(ra_deg))]
    )

    if merge_radius is not None:
        ra_deg, dec_deg, labels = merge_close_sources(
            ra_deg, dec_deg, flux, labels, merge_radius
        )

    n_src = len(ra_deg)

    # world_to_pixel returns (x, y) — note x=col, y=row
    px, py = wcs.all_world2pix(ra_deg, dec_deg, 0)
    points_pix = np.column_stack([px, py])

    # ------------------------------------------------------------------
    # 3. Voronoi tessellation in pixel space
    # ------------------------------------------------------------------
    pad = max(nx, ny) * 3
    mirrors = np.array([
        [ pad,    0   ],
        [-pad,    0   ],
        [0,       pad ],
        [0,      -pad ],
        [ pad,    pad ],
        [-pad,    pad ],
        [ pad,   -pad ],
        [-pad,   -pad ],
    ])
    all_points = np.vstack([points_pix, mirrors])

    vor = Voronoi(all_points)

    # ------------------------------------------------------------------
    # 4. Clip each Voronoi cell to the image bounding box
    # ------------------------------------------------------------------
    facets_pix = []
    for i in range(n_src):
        region_idx = vor.point_region[i]
        region     = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            print(f"Warning: source {labels[i]} produced an unbounded cell, skipping")
            facets_pix.append(None)
            continue

        verts = vor.vertices[region]
        cell  = Polygon(verts)
        clipped = cell.intersection(image_bbox_pix)
        facets_pix.append(clipped)

    # ------------------------------------------------------------------
    # 5. Convert clipped pixel vertices back to sky coordinates and write
    # ------------------------------------------------------------------
    with open(output_path, "w") as f:
        f.write("# Region file format: DS9\n")
        f.write("fk5\n")

        for i, (geom, label) in enumerate(zip(facets_pix, labels)):
            if geom is None or geom.is_empty:
                continue

            polys = (
                list(geom.geoms)
                if isinstance(geom, MultiPolygon)
                else [geom]
            )

            for poly in polys:
                verts_pix = np.array(poly.exterior.coords)[:-1]

                ra_v, dec_v = wcs.all_pix2world(
                    verts_pix[:, 0], verts_pix[:, 1], 0
                )

                coord_pairs = ", ".join(
                    f"{r:.6f}, {d:.6f}" for r, d in zip(ra_v, dec_v)
                )
                f.write(f'polygon({coord_pairs}) # text="{label}"\n')
                f.write(f'point({ra_deg[i]:.6f}, {dec_deg[i]:.6f}) # text="{label}"\n')

    print(f"Written {n_src} facets to {output_path!r}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description="Make a Voronoi facet region file from a source catalogue")
    p.add_argument("image", help="FITS image with WCS header (sets the facet bounds)")
    p.add_argument("catalogue", help="FITS catalogue with source positions")
    p.add_argument("output", help="Output DS9 region file, e.g. facets.reg")
    p.add_argument("--ra-col", default="RA")
    p.add_argument("--dec-col", default="DEC")
    p.add_argument("--flux-col", default="I", help="Column used for the flux cut and merge weights")
    p.add_argument("--min-flux", type=float, default=0.25, help="Only use sources at least this bright")
    p.add_argument("--label-col", default=None, help="Column with facet names (default facet_0, facet_1...)")
    p.add_argument("--merge-radius", type=float, default=None,
                   help="Merge sources closer than this (arcsec) into one facet centre")
    a = p.parse_args(argv)
    voronoi_tessellate_image(
        image_path=a.image,
        catalogue_path=a.catalogue,
        output_path=a.output,
        ra_col=a.ra_col,
        dec_col=a.dec_col,
        filter_Icol=a.flux_col,
        filter_I=a.min_flux,
        label_col=a.label_col,
        merge_radius=None if a.merge_radius is None else a.merge_radius * u.arcsec,
    )


if __name__ == "__main__":
    main()
