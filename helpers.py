"""Shared helpers for the SPHEREx moving-object pipeline."""
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Everything is resolved relative to the folder that contains this file, never to the
# current working directory or to a user-specific absolute path, so the repo can be cloned anywhere.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(REPO_DIR, "config.default.json")  # tracked template
# config.json is your local, auto-created copy of the template (it is git-ignored because the
# pipeline writes the target position, paths and progress flags into it).
# Override the location with the SPHEREX_CONFIG environment variable.
CONFIG_PATH = os.path.abspath(os.path.expanduser(
    os.environ.get("SPHEREX_CONFIG") or os.path.join(REPO_DIR, "config.json")))

FLAG_NAMES = ["configured", "downloaded", "detected", "gaia_matched",
              "static_filtered", "tracklets_linked", "known_objects_checked",
              "spectra_extracted", "classified"]

MJD_J2016 = 57388.0  # Gaia DR3 reference epoch (J2016.0)

MIN_PYTHON = (3, 10)
# import name -> pip name (keep in sync with requirements.txt)
REQUIRED_PACKAGES = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib",
    "astropy": "astropy", "astroquery": "astroquery", "photutils": "photutils", "pyvo": "pyvo",
}


# ---------------------------------------------------------------- environment
def check_dependencies():
    """Stop with one clear message if the Python version or any required package is missing."""
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(f"[!] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
                         f"(you have {sys.version_info.major}.{sys.version_info.minor}).")
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        raise SystemExit("[!] Missing Python packages: " + ", ".join(missing) + "\n"
                         "    Install everything with:\n"
                         f"        {os.path.basename(sys.executable) or 'python'} -m pip install -r requirements.txt")


# ---------------------------------------------------------------- config / flags
def load_config():
    """Load config.json; on first run create it from config.default.json."""
    if not os.path.exists(CONFIG_PATH):
        if not os.path.exists(DEFAULT_CONFIG_PATH):
            raise SystemExit(f"[!] Neither {CONFIG_PATH} nor {DEFAULT_CONFIG_PATH} exists.")
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        shutil.copyfile(DEFAULT_CONFIG_PATH, CONFIG_PATH)
        print(f"[i] Created {CONFIG_PATH} from config.default.json")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    """Write config.json atomically so an interrupted run cannot leave a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CONFIG_PATH), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        os.replace(tmp, CONFIG_PATH)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def set_flag(name, value=True):
    config = load_config()
    config["flags"][name] = value
    save_config(config)


def require_flag(config, flag, step_hint):
    if not config["flags"].get(flag, False):
        raise SystemExit(f"[!] '{flag}' is not done yet. Run {step_hint} first.")


def output_root(config):
    """Base output folder: $SPHEREX_OUTPUT_DIR if set, else info_dict.output_root (default 'output')."""
    return os.environ.get("SPHEREX_OUTPUT_DIR") or config["info_dict"].get("output_root") or "output"


def abs_path(path):
    """Absolute, OS-correct path. Relative paths are taken relative to the repo folder.
    Accepts '/' or '\\' separators and '~' so a config written on Windows also works on Linux/macOS."""
    p = os.path.expandvars(os.path.expanduser(str(path)))
    if os.sep == "/":
        p = p.replace("\\", "/")
    if not os.path.isabs(p):
        p = os.path.join(REPO_DIR, p)
    return os.path.normpath(p)


def build_paths(root):
    """All pipeline file/folder locations for one target, as portable '/'-separated strings."""
    root = Path(root)
    c, t = root / "catalogs", root / "tracklets"
    paths = {
        "root": root,
        "images_dir": root / "images",
        "catalog_dir": c,
        "tracklet_dir": t,
        "plot_dir": root / "plots",
        "observations_csv": c / "observations.csv",
        "detections_csv": c / "1_detections.csv",
        "gaia_catalog_csv": c / "gaia_catalog.csv",
        "gaia_matched_csv": c / "2_detections_gaia.csv",
        "filtered_csv": c / "3_detections_filtered.csv",
        "tracklets_csv": t / "tracklets.csv",
        "members_csv": t / "tracklet_members.csv",
        "checked_csv": t / "tracklets_known_check.csv",
        "candidates_csv": t / "candidates.csv",
        # steps 8-9
        "spectra_dir": root / "spectra",
        "spectrum_csv": t / "spectrum_points.csv",
        "classification_csv": t / "classification.csv",
        "timing_csv": root / "timing.csv",
        # steps 8-9, static unlabeled sources
        "static_spectra_dir": root / "static_spectra",
        "static_spectrum_csv": c / "static_spectrum_points.csv",
        "static_classification_csv": c / "static_classification.csv",
    }
    return {k: v.as_posix() for k, v in paths.items()}


def get_paths(config):
    p = config.get("paths") or {}
    if not p:
        raise SystemExit("[!] config.json has no 'paths'. Run step1_config.py first.")
    # configs written by an older version lack the step 8-9 paths: fill them in from the root
    for k, v in build_paths(p.get("root") or "output").items():
        if k not in p:
            p[k] = v
    return {k: abs_path(v) for k, v in p.items()}


def get_logger(name):
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


# ---------------------------------------------------------------- FITS
def read_image(path):
    """Return (image, variance_or_None, header, celestial WCS) of a SPHEREx cutout."""
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path, memmap=False) as hdul:
        names = [h.name.upper() for h in hdul]
        # Cutouts normally carry an IMAGE extension; fall back to the first 2-D HDU.
        idx = names.index("IMAGE") if "IMAGE" in names else next(
            i for i, h in enumerate(hdul) if h.data is not None and h.data.ndim == 2)
        data = np.array(hdul[idx].data, dtype=float)
        header = hdul[idx].header.copy()
        var = np.array(hdul[names.index("VARIANCE")].data, dtype=float) if "VARIANCE" in names else None
        wcs = WCS(header, fobj=hdul).celestial
    return data, var, header, wcs


def header_mjd(header):
    """Mid-exposure MJD, or None if the header has no usable time keyword."""
    from astropy.time import Time

    for key in ("MJD-AVG", "MJD-OBS"):
        if key in header:
            try:
                return float(header[key])
            except (TypeError, ValueError):
                pass
    for key in ("DATE-AVG", "DATE-OBS", "DATE"):
        if key in header:
            try:
                return float(Time(str(header[key])).mjd)
            except Exception:  # noqa: BLE001
                pass
    return None


# ---------------------------------------------------------------- geometry
def tangent_offsets(ra, dec, ra0, dec0):
    """Small-field offsets (east, north) in arcsec relative to (ra0, dec0)."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    dra = (ra - ra0 + 180.0) % 360.0 - 180.0
    return dra * np.cos(np.radians(dec0)) * 3600.0, (dec - dec0) * 3600.0


def offsets_to_radec(x, y, ra0, dec0):
    """Inverse of tangent_offsets."""
    return ra0 + np.asarray(x) / 3600.0 / np.cos(np.radians(dec0)), dec0 + np.asarray(y) / 3600.0


def field_radius_deg(config):
    """Cone radius covering a square cutout, with a small safety margin."""
    return 0.5 * config["download"]["cutout_size_deg"] * np.sqrt(2) * 1.1 + 30.0 / 3600.0