"""
End-to-end SKA-Low simulation pipeline (replaces the old ``runsimjob.py``).

Steps (all configurable in the YAML ``run.steps`` list):

1. ``sky``          build the OSKAR sky table for the chosen sky model
2. ``sensitivity``  query the SKAO sensitivity calculator (sets the CLEAN threshold)
3. ``oskar``        run the OSKAR simulation (one MS per sub-band)
4. ``dp3``          optional averaging and applying the beam with DP3
5. ``msoverview``   summarise the output measurement sets
6. ``image_dirty``  WSClean dirty image
7. ``mask``         paint the sky-model sources into a CLEAN mask
8. ``image_clean``  masked, thresholded WSClean deconvolution
9. ``find_sources`` source finding on the final image (off by default)

Usage::

    ska-runsim configs/examples/aa2_cosmos_ionosphere.yaml
    ska-runsim my_run.yaml --set observation.nchan=100 --dry-run
    ska-runsim my_run.yaml --restart --steps image_dirty mask image_clean
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

from . import __version__
from .config import (STEPS, ConfigError, default_config_text, dump_config, load_config,
                     parse_overrides, resolve_path)
from .skymodels import build_sky_table, sky_centre
from .utils import (die, enter_launch_dir, log_time_to_file, ms_names, report, root_name_from_config, run,
                    separator, warn)

SUMMARY_LOG = "simulation_summary.log"
CONFIG_USED = "config_used.yaml"


class SimulationRun:
    """One configured simulation: derives all names from the config and runs the steps."""

    def __init__(self, cfg: dict, config_path: str | None = None):
        self.cfg = cfg
        self.config_path = os.path.abspath(config_path) if config_path else None
        self.dry_run = bool(cfg["run"]["dry_run"])

        # make all site paths absolute now, before we chdir into the run directory
        for key, val in cfg["paths"].items():
            cfg["paths"][key] = resolve_path(val, os.getcwd()) if val else val

        obs = cfg["observation"]
        self.root = root_name_from_config(cfg)
        report(f"Working directory {os.getcwd()}; paths.work_dir = {cfg['paths']['work_dir']}")
        self.workdir = os.path.normpath(os.path.join(cfg["paths"]["work_dir"],
                                                     f"{self.root}{cfg['run']['identifier']}"))
        self.ra, self.dec = sky_centre(cfg)
        self.skytab = f"{self.root}_skytab.fits" if cfg["sky"]["model"] != "none" else None

        self.ms = ms_names(self.root, int(obs["nsubbands"]))
        pp = cfg["postprocess"]
        if pp["apply_beam"]:
            suffix = "_applybeam"
        elif pp["average"]:
            suffix = "_avg"
        else:
            suffix = None
        self.ms_out = [m.replace(".ms", f"{suffix}.ms") for m in self.ms] if suffix else list(self.ms)

        im = cfg["imaging"]
        self.im_dirty = f"ws_{self.root}_{im['name_suffix']}"
        self.im_clean = f"{self.im_dirty}_mask"
        self.mask = f"{self.im_clean}.fits"
        self._sensitivity = None
        self._log = SUMMARY_LOG

    # ------------------------------------------------------------------ helpers
    def _run(self, cmd: str, log: str, fatal: bool = True) -> None:
        run(cmd, proceed=not fatal, dryrun=self.dry_run, log=log)

    def _timed(self, label: str, func, *a, **kw):
        if self.dry_run:
            return func(*a, **kw)
        start = log_time_to_file(f"{label} start", self._log)
        out = func(*a, **kw)
        log_time_to_file(f"*{label} run took", self._log, lastime=start)
        return out

    def _note(self, text: str) -> None:
        report(text)
        if not self.dry_run:
            with open(self._log, "a") as f:
                f.write(text + "\n")

    def _wsclean_common(self, name: str) -> str:
        im = self.cfg["imaging"]
        tmp = self.cfg["paths"]["wsclean_temp_dir"]
        temp = f"-temp-dir {tmp} " if tmp else ""
        return (f"wsclean {temp}-size {im['npix']} {im['npix']} -scale {im['cell']} "
                f"-name {name} -weight {im['weight']}")

    def threshold(self) -> str:
        im = self.cfg["imaging"]
        if im["abs_threshold"]:
            return str(im["abs_threshold"])
        if self._sensitivity is None:
            self.step_sensitivity()
        if self._sensitivity is None:
            if self.dry_run:
                return "<sensitivity>mJy"
            die("No CLEAN threshold: the sensitivity calculator failed and imaging.abs_threshold is not set")
        return f"{self._sensitivity['value'] * 1e3 * im['threshold_factor']:.3f}mJy"

    # -------------------------------------------------------------------- setup
    def setup(self) -> None:
        """Create (or reuse) the run directory, chdir into it and save the config used."""
        separator(f"Run directory {self.workdir}")
        restart = self.cfg["run"]["restart"]
        if self.dry_run:
            warn(f"Dry run: would {'reuse' if restart else 'create'} {self.workdir}")
            return
        if not restart and os.path.isdir(self.workdir):
            mode = self.cfg["run"]["overwrite"]
            if mode == "ask":
                print(f"working directory {self.workdir} already exists")
                while mode not in ("yes", "no"):
                    resp = input("(y) to Remove and continue or (n) to quit? [y/n]: ").strip().lower()
                    mode = {"y": "yes", "n": "no"}.get(resp, mode)
            if mode == "no":
                print("Exiting.")
                sys.exit(1)
            shutil.rmtree(self.workdir)
        if restart and not os.path.isdir(self.workdir):
            die(f"Cannot restart: {self.workdir} does not exist")
        os.makedirs(self.workdir, exist_ok=True)
        os.chdir(self.workdir)

        if self.config_path and os.path.isfile(self.config_path):
            shutil.copy(self.config_path, "config_input.yaml")
        resolved = dict(self.cfg)
        resolved["observation"] = dict(self.cfg["observation"], ra_deg=self.ra, dec_deg=self.dec)
        dump_config(resolved, CONFIG_USED,
                    header=f"Fully resolved configuration written by ska-simulations {__version__}\n"
                           f"on {datetime.now():%Y-%m-%d %H:%M:%S}")
        with open(self._log, "a" if restart else "w") as f:
            f.write(f"{'Restarting' if restart else 'Starting'} job (ska-simulations {__version__})\n")

    # -------------------------------------------------------------------- steps
    def step_sky(self) -> None:
        model = self.cfg["sky"]["model"]
        report(f"Sky model {model}: phase centre RA={self.ra} DEC={self.dec}")
        if self.dry_run:
            warn(f"Dry run: would write sky table {self.skytab}")
            return
        build_sky_table(self.cfg, self.skytab)

    def step_sensitivity(self) -> None:
        if not self.cfg["sensitivity"]["enabled"] or self.dry_run:
            if self.dry_run:
                warn("Dry run: skipping sensitivity calculator query")
            return
        from .sensitivity import sensitivity_from_config

        try:
            self._sensitivity = sensitivity_from_config(self.cfg, self.ra, self.dec)
        except Exception as e:  # network errors, API changes...
            warn(f"Sensitivity calculator query failed: {e}")
            return
        self._note(f"Sensitivity calculator reports {self._sensitivity} for these settings "
                   f"and {self.cfg['sensitivity']['weighting']} weighting")
        report(f"  = {self._sensitivity['value'] * 1e3:.2f} mJy/bm")

    def oskar_command(self) -> str:
        cmd = (f"{sys.executable} -m ska_simulations.oskar_sim --config {CONFIG_USED} "
               f"--ra {self.ra} --dec {self.dec}")
        if self.skytab:
            cmd += f" --skytabfits {self.skytab}"
        return cmd

    def step_oskar(self) -> None:
        self._timed("oskar", self._run, self.oskar_command(), "oskarsim.log")

    def dp3_commands(self) -> list[tuple[str, str]]:
        pp = self.cfg["postprocess"]
        steps = (["avg"] if pp["average"] else []) + (["applybeam"] if pp["apply_beam"] else [])
        if not steps:
            return []
        cmds = []
        for msi, mso in zip(self.ms, self.ms_out):
            cmd = f"DP3 msin={msi} steps=[{','.join(steps)}] msout={mso}"
            if pp["average"]:
                cmd += (f" avg.type=averager avg.timestep={pp['avg_time_step']}"
                        f" avg.freqstep={pp['avg_freq_step']}")
            cmds.append((cmd, f"dppp_{'_'.join(steps)}_{msi}.log"))
        return cmds

    def step_dp3(self) -> None:
        cmds = self.dp3_commands()
        if not cmds:
            warn("No DP3 steps requested (postprocess.apply_beam and .average both false)")
            return

        def _all():
            for cmd, log in cmds:
                self._run(cmd, log)
        self._timed("dppp", _all)

    def step_msoverview(self) -> None:
        script = self.cfg["paths"]["msoverview_script"]
        if not script:
            return
        for ms in self.ms_out:
            self._run(f"{sys.executable} {script} {ms}", f"msoverview_{ms}.log", fatal=False)

    def wsclean_dirty_command(self) -> str:
        im = self.cfg["imaging"]
        return f"{self._wsclean_common(self.im_dirty)} {im['dirty_args']} {' '.join(self.ms_out)}"

    def wsclean_clean_command(self) -> str:
        im = self.cfg["imaging"]
        mask = ""
        if self.skytab and (self.dry_run or os.path.isfile(self.mask)):
            mask = f"-fits-mask {self.mask} "
        elif self.skytab:
            warn(f"Mask {self.mask} not found; cleaning without a mask")
        return (f"{self._wsclean_common(self.im_clean)} -channels-out {im['channels_out']} "
                f"-deconvolution-channels {im['deconvolution_channels']} {mask}"
                f"-abs-threshold {self.threshold()} {' '.join(im['clean_args'].split())} "
                f"{' '.join(self.ms_out)}")

    def step_image_dirty(self) -> None:
        self._timed("ws1", self._run, self.wsclean_dirty_command(), "wsclean1.log")

    def step_mask(self) -> None:
        if self.cfg["imaging"]["dirty_only"]:
            return
        if not self.skytab:
            warn("No sky model, so no CLEAN mask")
            return
        cmd = (f"{sys.executable} -m ska_simulations.masks {self.im_dirty}-image.fits {self.skytab} "
               f"{self.mask} --min-radius-pix {self.cfg['imaging']['mask_min_radius_pix']}")
        self._run(cmd, "paint_mask.log")

    def step_image_clean(self) -> None:
        if self.cfg["imaging"]["dirty_only"]:
            return
        self._timed("ws2", self._run, self.wsclean_clean_command(), "wsclean2.log")

    def step_find_sources(self) -> None:
        if self.cfg["imaging"]["dirty_only"] or not self.cfg["imaging"]["find_sources"]:
            return
        script = self.cfg["paths"]["find_sources_script"]
        im = self.im_clean
        cmd = (f"{sys.executable} {script} --blankpb {im}-MFS-image-pb.fits "
               f"{im}-MFS-image-pb-pybdsf.fits {im}-MFS-image.fits")
        self._run(cmd, "pybdsf.log", fatal=False)

    # ---------------------------------------------------------------------- run
    def run(self, steps=None) -> None:
        steps = list(steps or self.cfg["run"]["steps"])
        self.setup()
        for step in STEPS:
            if step in steps:
                separator(step)
                getattr(self, f"step_{step}")()
        report(f"Finished: outputs in {self.workdir}")


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an SKA-Low OSKAR simulation + imaging pipeline "
                                            "from a YAML configuration file")
    p.add_argument("config", nargs="?", help="YAML run configuration")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value, e.g. --set observation.nchan=100 (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="Print what would be run without running it")
    p.add_argument("--restart", action="store_true", help="Reuse the existing run directory")
    p.add_argument("--overwrite", choices=["ask", "yes", "no"], default=None,
                   help="What to do if the run directory exists (default from config: ask)")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=None, help="Only run these steps")
    p.add_argument("--print-default-config", action="store_true",
                   help="Print the documented default configuration and exit")
    p.add_argument("--print-config", action="store_true",
                   help="Print the merged configuration and exit")
    return p


def main(argv=None) -> None:
    enter_launch_dir()
    args = build_parser().parse_args(argv)
    if args.print_default_config:
        print(default_config_text())
        return
    overrides = parse_overrides(args.set)
    run_over = overrides.setdefault("run", {})
    if args.dry_run:
        run_over["dry_run"] = True
    if args.restart:
        run_over["restart"] = True
    if args.overwrite:
        run_over["overwrite"] = args.overwrite
    if args.steps:
        run_over["steps"] = args.steps
    try:
        cfg = load_config(args.config, overrides)
    except (ConfigError, FileNotFoundError) as e:
        sys.exit(f"Configuration error: {e}")
    if args.print_config:
        import yaml
        print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
        return
    if args.config is None:
        warn("No config file given: running with the packaged defaults")
    SimulationRun(cfg, args.config).run()


if __name__ == "__main__":
    main()
