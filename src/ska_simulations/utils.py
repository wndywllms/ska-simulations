"""
General helpers: running external commands with logging, coloured console
messages, timing logs and the standard run naming convention.
"""
from __future__ import annotations

import select
import shutil
import subprocess
import sys
from datetime import datetime


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def separator(s: str) -> None:
    """Print a coloured ``==== s ====`` banner across the terminal."""
    print()
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    if len(s) % 2 == 1:
        s += " "
    sep = "=" * ((width - len(s) - 2) // 2)
    print(f"{bcolors.FAIL}{sep} {s} {sep}{bcolors.ENDC}")
    print()


def die(s: str) -> None:
    print(bcolors.FAIL + s + bcolors.ENDC)
    raise RuntimeError(s)


def report(s: str) -> None:
    print(bcolors.OKGREEN + s + bcolors.ENDC)


def warn(s: str) -> None:
    print(bcolors.OKBLUE + s + bcolors.ENDC)


def run_log(cmd: str, slogfile: str, quiet: bool = False) -> int:
    """Run a shell command, echoing its output and writing it, time-stamped, to a log file."""
    with open(slogfile, "w") as logfile:
        logfile.write("Running process with command: " + cmd + "\n")
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )
        while True:
            try:
                select.select([proc.stdout], [], [proc.stdout])
            except (select.error, ValueError):
                pass
            line = proc.stdout.readline()
            if line == "":
                break
            if not quiet:
                sys.stdout.write(line)
            ts = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
            logfile.write(ts + ": " + line)
            logfile.flush()
        retval = proc.wait()
        logfile.write("Process terminated with return value %i\n" % retval)
    return retval


def run(cmd: str, proceed: bool = False, dryrun: bool = False, log: str | None = None,
        quiet: bool = False) -> int | None:
    """Run a shell command (optionally logged); raise unless ``proceed`` if it fails."""
    report("Running: " + cmd)
    if dryrun:
        warn("Dry run, skipping this step")
        return None
    retval = subprocess.call(cmd, shell=True) if log is None else run_log(cmd, log, quiet)
    if not proceed and retval != 0:
        die(f"FAILED to run {cmd}: return value is {retval}")
    return retval


def log_time_to_file(s: str, filename: str, lastime: datetime | None = None) -> datetime:
    """Append ``s: <now>`` (or ``s: <elapsed since lastime>``) to a log file; return now."""
    timenow = datetime.now()
    writetime = timenow - lastime if lastime is not None else timenow
    with open(filename, "a") as logfile:
        logfile.write(f"{s}: {writetime} \n")
    return timenow


def run_root_name(array: str, nhours: float, nchan: int, time_step_s: float,
                  freq_step_khz: float, start_freq_mhz: float) -> str:
    """Standard name used for run directories, measurement sets and images.

    e.g. ``AA2_60.0min_500channels_6.8s_43.4kHz_120.0MHz``. Both the pipeline and
    the OSKAR wrapper use this function, so their file names always agree.
    """
    length_min = round(nhours * 60.0, 6)
    return (f"{array}_{length_min:.1f}min_{int(nchan)}channels_{time_step_s:g}s_"
            f"{freq_step_khz:g}kHz_{start_freq_mhz:.1f}MHz")


def root_name_from_config(cfg: dict) -> str:
    obs = cfg["observation"]
    return run_root_name(obs["array"], obs["nhours"], obs["nchan"], obs["time_step_s"],
                         obs["freq_step_khz"], obs["start_freq_mhz"])


def ms_names(root: str, nsubbands: int) -> list[str]:
    if nsubbands == 1:
        return [f"{root}.ms"]
    return [f"{root}_SB{i:03d}.ms" for i in range(nsubbands)]
