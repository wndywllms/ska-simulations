# ska-simulations

Tools for running OSKAR simulations of SKA-Low and imaging the results.

The pipeline:

1. builds a sky model
2. simulates visibilities with OSKAR (optionally with an external TEC screen and cable errors)
3. optionally averages the data and applies the beam with DP3
4. makes a dirty image with WSClean
5. paints the sky-model sources into a CLEAN mask
6. deconvolves down to the noise level from the SKAO sensitivity calculator

Every tunable parameter is set in a YAML configuration file.

## Installation

```bash
git clone git@github.com:wndywllms/ska-simulations.git
cd ska-simulations
pip install -e .            # core: numpy, astropy, pyyaml, requests
pip install -e ".[all]"     # + scipy/shapely (facets), regions, python-casacore/losoto/matplotlib
```

The simulation step needs OSKAR's Python bindings (`import oskar`), which are not on PyPI. Install them from the OSKAR source tree. Imaging needs `DP3` and `wsclean` on your `PATH`. If these tools live in an Apptainer or Singularity image, see [Running inside a container](#running-inside-an-apptainer-or-singularity-container-without-installing). You don't need to install anything in the image.

## Running a simulation

```bash
# see every parameter with its documentation
ska-runsim --print-default-config > my_run.yaml

# run one of the examples (only values that differ from the defaults are needed)
ska-runsim configs/examples/aa2_cosmos_ionosphere.yaml

# check what would be run, without running anything
ska-runsim configs/examples/aa2_cosmos_ionosphere.yaml --dry-run

# override single values without editing the file
ska-runsim my_run.yaml --set observation.nchan=100 --set run.identifier=_quick

# re-run only the imaging in an existing run directory
ska-runsim my_run.yaml --restart --steps image_dirty mask image_clean

# print the fully merged configuration
ska-runsim my_run.yaml --print-config
```

Each run creates a directory `<work_dir>/<root><identifier>`, where `root` is e.g. `AA2_60.0min_500channels_6.8s_43.4kHz_120.0MHz`. The run directory contains:

- the measurement sets and images
- a log file for each step (`oskarsim.log`, `dppp_*.log`, `wsclean1.log`, `wsclean2.log`, …)
- `simulation_summary.log`, with the timing of each step and the sensitivity estimate
- `config_input.yaml`, a copy of your config file
- `config_used.yaml`, the fully resolved configuration, which is enough to reproduce the run

These config files replace the old practice of copying the scripts into the run directory.

### Site paths

The `paths:` section holds the locations of the data directory, the telescope models, the WSClean temp directory and helper scripts. The defaults point at the original locations under `/home/w.williams/projects/skasims`. To change them on another machine, put a `paths:` block in your run config. Relative catalogue, noise-file and TEC-screen names are resolved against `paths.data_dir`. Environment variables (`$HOME/...`) are expanded.

### Configuration sections

| Section | Contents |
|---|---|
| `paths`, `telescope_models` | site-specific locations; OSKAR telescope model per array (with and without cable errors) |
| `run` | run-directory identifier, restart/overwrite behaviour, dry run, list of steps |
| `observation` | array, length, time/frequency resolution, channels, sub-bands, start date, transit offset, phase centre |
| `simulator` | OSKAR settings: GPU, precision, cable errors, noise files and seed, TEC screen, extra raw OSKAR settings |
| `sky` | sky model (`LoTSSDEEP`, `COSMOS`, `testsky`, `cal`, `none`), flux limit, per-model catalogues and centres |
| `sensitivity` | SKAO sensitivity calculator query (sets the CLEAN threshold) |
| `postprocess` | DP3 averaging and beam application |
| `imaging` | WSClean image size, cell, weighting, channels, threshold, extra arguments, source finding |

### Example configurations (`configs/examples/`)

These were extracted from the commented-out parameter blocks in the old `runsimjob.py`.

| File | Description |
|---|---|
| `aa2_srcnet_v2_cosmos.yaml` | AA2 v2 SRCnet dataset: 4 h, COSMOS, 6912 ch, 6 sub-bands |
| `aastarmod_srcnet_v1_cosmos_full.yaml` | AA\* v1 SRCnet dataset: 30 min, 50 MHz start, 13824 ch, 12 sub-bands |
| `aa2_lotss_deep_4h_2bands.yaml` | AA2 4 h LoTSS-deep, S > 0.45 mJy, 2 sub-bands |
| `aa2_raw_10min_target.yaml` / `aa2_raw_10min_cal.yaml` | 10 min at raw resolution (0.85 s, 5.4 kHz), target and calibrator |
| `aa2_cosmos_ionosphere.yaml` | 4 h COSMOS with `scan_0.fits` TEC screen, ×4 averaging |
| `aa2_cosmos_ionosphere_variables.yaml` | as above, with the variable-source catalogue |
| `aa2_cosmos_ddcals.yaml` | DD-calibrator-only COSMOS model for deriving DD solutions |
| `aa2_testsky_tec_line2.yaml` | 1 h, 5-source test sky, `blank_line2_iono.fits` TEC screen |
| `aa2_sidereal_repeat_20270101.yaml` | 10 min repeat observation for the sidereal-averaging test |
| `aa2_ical_test_mid_band_obs1.yaml` | 7.5 min, cable errors and TEC screen, offset +1 h from transit |
| `aastarmod_4h_1000ch.yaml` | AA\* 4 h, 1000 ch. The note in the old script says this run took days and died at 99% |

## Running inside an Apptainer or Singularity container (without installing)

DP3, WSClean, EveryBeam and OSKAR usually come from an Apptainer or Singularity image. You don't need to install this package in the image. `bin/ska-apptainer` runs it straight from your git checkout by binding the repo into the container and putting `src/` on `PYTHONPATH`. Edits you make take effect on the next run. Every step runs inside the container, including the OSKAR step, which the pipeline starts as a separate process that inherits `PYTHONPATH`.

```bash
export SKA_SIF=/path/to/ska-sims.sif
export SKA_BINDS=/data-hot,/home/w.williams/projects/skasims   # anything not bound by default
export PATH=/path/to/ska-simulations/bin:$PATH

ska-apptainer configs/examples/aa2_cosmos_ionosphere.yaml          # = ska-runsim ...
ska-apptainer my_run.yaml --dry-run --set observation.nchan=100
ska-apptainer oskar_sim --config my_run.yaml --dry-run             # any module: masks, facets, tec_phase, ...
ska-apptainer --shell                                              # interactive shell, package importable
```

| Variable | Meaning |
|---|---|
| `SKA_SIF` | container image (required) |
| `SKA_CONTAINER_RUNTIME` | `apptainer` or `singularity`. By default the script uses `apptainer` if it's installed, otherwise `singularity` |
| `SKA_BINDS` | extra bind mounts, comma separated. Apptainer usually binds only `$HOME`, `/tmp` and the current directory, so add the data, telescope-model and WSClean temp directories if they live elsewhere |
| `SKA_PYTHON` | python inside the container (default `python3`) |
| `SKA_APPTAINER_OPTS` | options for `apptainer exec` / `singularity exec` (default `--nv`, which OSKAR needs for the GPU) |

The script starts the container in the directory you ran it from (`--pwd "$PWD"`, with `$PWD` bound). Without this, Apptainer falls back to `$HOME` whenever it can't bind the current directory, and relative paths such as the default `paths.work_dir: .` would put run directories in your home directory. If you call `apptainer exec` or `singularity exec` yourself, add `--pwd "$PWD" -B "$PWD"`. With Singularity, the commands below work the same with `singularity` in place of `apptainer`.

The image's Python needs `numpy`, `astropy`, `pyyaml` and `requests` (plus `oskar`). Check with:

```bash
apptainer exec $SKA_SIF python3 -c "import numpy, astropy, yaml, requests, oskar"
```

If something is missing, install it outside the image into a directory you own and add that directory to the path:

```bash
apptainer exec $SKA_SIF python3 -m pip install --target ~/.ska-deps pyyaml requests
export PYTHONPATH=~/.ska-deps        # ska-apptainer appends this after src/
```

For a batch job, call the same script from your job script, e.g. `ska-apptainer my_run.yaml --overwrite yes`.

## Command-line tools

| Command | Replaces | Purpose |
|---|---|---|
| `ska-runsim` | `scripts/runsimjob.py` | full pipeline from a YAML config |
| `ska-oskar-sim` | `scripts/oskar_sim.py` | OSKAR simulation only (same options as before, plus `--config`, `--set`, `--dry-run`) |
| `ska-sensitivity` | `scripts/get_sens.py` | per-visibility noise from the sensitivity calculator; `--outdir` writes OSKAR noise files |
| `ska-make-mask` | `scripts/make_mask_from_sources.py` | paint catalogue sources into a FITS CLEAN mask |
| `ska-make-facets` | `scripts/make_facets_regionfile.py` | Voronoi facet region file for WSClean |
| `ska-ds9-to-fits` | `scripts/ds9_to_fits.py` | DS9 region file to FITS table |
| `ska-tec-phase` | `scripts/oskar_tec_phase_reproduce.py` | reproduce OSKAR's TEC-screen phases and write h5parms |
| `ska-plot-dd-phases` | `scripts/plot_dd_phase_sols_compare.py` | compare DD phase solutions in two h5parms |

## Using the library

```python
from ska_simulations.config import load_config
from ska_simulations.pipeline import SimulationRun
from ska_simulations.skymodels import build_sky_table, get_lotss_deep_sky_table
from ska_simulations.sensitivity import get_ska_sens_calc
from ska_simulations.utils import run_log, run_root_name

cfg = load_config("configs/examples/aa2_cosmos_ionosphere.yaml",
                  overrides={"observation": {"nchan": 100}})
SimulationRun(cfg).run(steps=["sky", "oskar"])
```

## Package layout

```
src/ska_simulations/
    config.py           load / merge / validate YAML configs
    data/default_config.yaml   documented defaults for every parameter
    pipeline.py         the simulation + imaging pipeline (ska-runsim)
    oskar_sim.py        OSKAR wrapper (ska-oskar-sim)
    skymodels.py        sky-table builders
    sensitivity.py      SKAO sensitivity-calculator helpers
    utils.py            command running/logging, coloured output, run naming
    masks.py  facets.py  ds9_to_fits.py  tec_phase.py  igrf14.py  plot_dd_phase_sols.py
configs/examples/       example run configurations
tests/                  pytest suite (uses fake oskar / DP3 / wsclean)
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests run the whole pipeline end to end with stand-ins for OSKAR, DP3 and WSClean, so they need no real data or GPUs.
