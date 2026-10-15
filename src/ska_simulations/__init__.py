"""
ska_simulations - tools for running OSKAR simulations of SKA-Low and imaging the results.

Main entry points
-----------------
ska-runsim       full pipeline driven by a YAML config  (:mod:`ska_simulations.pipeline`)
ska-oskar-sim    OSKAR simulation only                  (:mod:`ska_simulations.oskar_sim`)

Library modules
---------------
config        load/merge/validate YAML configurations
utils         command running, logging and naming helpers
skymodels     build OSKAR sky tables from catalogues
sensitivity   SKAO sensitivity-calculator helpers
masks         paint sources into FITS CLEAN masks
facets        Voronoi facet region files for WSClean
ds9_to_fits   DS9 region file -> FITS table
igrf14        IGRF-14 magnetic field (port of OSKAR's C code)
tec_phase     reproduce OSKAR's TEC-screen phases / write h5parms
plot_dd_phase_sols  compare direction-dependent phase solutions
"""
__version__ = "0.2.0"
