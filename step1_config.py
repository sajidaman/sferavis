"""Step 1 - set the target (RA, Dec, search radius), validate config.json and create the output folders.

Run it and type the target RA, Dec and search radius when asked; they are written into
config.json for you:
    python step1_config.py
or give them directly:
    python step1_config.py --ra 180.0 --dec 0.0 --radius 0.1
    python step1_config.py --ra 20h23m57.8s --dec +38d30m27s --radius 0.1
RA / Dec: decimal degrees or sexagesimal. Radius: decimal degrees; it is used directly as
the cutout size (download.cutout_size_deg = radius), e.g. 0.1 -> 0.1 deg square cutout.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import helpers

log = helpers.get_logger("step1")
AUTO_NAME = re.compile(r"^RA_[0-9.]+_DEC_[+-][0-9.]+$")


def parse_position(ra, dec):
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    def as_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    fr, fd = as_float(ra), as_float(dec)
    if fr is not None and fd is not None:  # plain numbers = decimal degrees
        c = SkyCoord(fr * u.deg, fd * u.deg)
    else:  # sexagesimal text: RA in hours, Dec in degrees, e.g. "20h23m57.8s" "+38d30m27s"
        c = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))
    return float(c.ra.deg), float(c.dec.deg)


def ask_position(cur_ra, cur_dec, cur_radius):
    """Ask for RA, Dec and search radius in the terminal (Enter keeps the value in config.json)."""
    print("\nTarget - RA / Dec in decimal degrees (e.g. 180.0  -5.25) or sexagesimal "
          "(e.g. 20h23m57.8s  +38d30m27s); search radius in degrees (e.g. 0.1).")
    print("Press Enter to keep the value currently in config.json.")
    while True:
        ra = input("  RA: ").strip() or cur_ra
        dec = input("  Dec: ").strip() or cur_dec
        radius = input("  Search radius (deg): ").strip() or cur_radius
        try:
            ra_deg, dec_deg = parse_position(ra, dec)
            radius = float(radius)
            if not (0.0 <= ra_deg < 360.0 and -90.0 <= dec_deg <= 90.0):
                print("  [!] RA must be 0-360 deg and Dec -90..+90 deg. Try again.")
            elif radius <= 0:
                print("  [!] The search radius must be > 0 deg. Try again.")
            else:
                return ra_deg, dec_deg, radius
        except Exception as e:  # noqa: BLE001
            print(f"  [!] Could not read that input ({e}). Try again.")


def main(ra=None, dec=None, radius=None, interactive=None):
    """ra/dec (and radius) given, e.g. from the command line -> used directly; otherwise ask in the
    terminal when running interactively; otherwise keep what is already in config.json."""
    if interactive is None:
        interactive = sys.stdin.isatty()
    config = helpers.load_config()
    info = config["info_dict"]
    cur_radius = info.get("radius_deg") or config["download"]["cutout_size_deg"]

    if ra is not None and dec is not None:
        ra, dec = parse_position(ra, dec)
        radius = float(radius) if radius is not None else float(cur_radius)
    elif interactive:
        ra, dec, radius = ask_position(info["ra"], info["dec"], cur_radius)
    else:
        ra, dec = parse_position(info["ra"], info["dec"])
        radius = float(radius) if radius is not None else float(cur_radius)
    if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
        raise SystemExit(f"[!] Invalid position RA={ra}, Dec={dec}")
    if radius <= 0:
        raise SystemExit(f"[!] Invalid search radius {radius} deg (must be > 0)")
    info["ra"], info["dec"] = round(ra, 6), round(dec, 6)
    info["radius_deg"] = round(radius, 6)
    config["download"]["cutout_size_deg"] = round(radius, 6)  # cutout size = the value you typed
    # The folder name always follows the target position unless you typed your own name.
    # (A name that looks auto-generated, e.g. left over from a previous target, is regenerated.)
    name = info.get("name") or ""
    if not name or AUTO_NAME.match(name):
        name = f"RA_{ra:.4f}_DEC_{dec:+.4f}"
    info["name"] = ""  # keep empty in config.json so the next position change renames the folder too

    # Portable, '/'-separated paths relative to the repo folder (or absolute if you set an
    # absolute output_root / $SPHEREX_OUTPUT_DIR). Nothing machine-specific is stored.
    root = Path(helpers.output_root(config), name).as_posix()
    paths = helpers.build_paths(root)
    for key in ("root", "images_dir", "catalog_dir", "tracklet_dir", "plot_dir"):
        os.makedirs(helpers.abs_path(paths[key]), exist_ok=True)
    config["paths"] = paths

    for flag in helpers.FLAG_NAMES:
        config["flags"][flag] = False
    config["flags"]["configured"] = True

    helpers.save_config(config)
    log.info("Target: %s  (RA=%.5f, Dec=%+.5f, radius=%.4f deg, cutout %.4f deg)",
             name, ra, dec, radius, radius)
    log.info("Output folder: %s", helpers.abs_path(root))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Set the target position and create the output folders.")
    ap.add_argument("--ra"), ap.add_argument("--dec")
    ap.add_argument("--radius", type=float, help="search radius in degrees")
    ap.add_argument("--no-prompt", action="store_true", help="do not ask; use config.json as is")
    a = ap.parse_args()
    main(a.ra, a.dec, a.radius, interactive=False if a.no_prompt else None)