"""Step 4 - cross-match detections with Gaia DR3 (proper-motion corrected per epoch) and SIMBAD.

Detections that coincide with a Gaia star or a SIMBAD object (stars, galaxies, ...) are known
stationary sources, not Solar System movers. SIMBAD does not list asteroids.
Use gaia.catalog_file (CSV with ra, dec [, pmra, pmdec in mas/yr]) to supply your own
catalogue instead of querying VizieR.
"""
import os

import numpy as np
import pandas as pd

import helpers

log = helpers.get_logger("step4")


def fetch_gaia(config, paths):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    gcfg = config["gaia"]
    if gcfg.get("catalog_file"):
        cat = pd.read_csv(helpers.abs_path(gcfg["catalog_file"]))
        cat = cat.rename(columns={c: c.lower() for c in cat.columns})
        for col in ("pmra", "pmdec"):
            if col not in cat:
                cat[col] = 0.0
        return cat[["ra", "dec", "pmra", "pmdec"]]

    info = config["info_dict"]
    center = SkyCoord(info["ra"] * u.deg, info["dec"] * u.deg)
    radius = helpers.field_radius_deg(config)

    def from_esa():
        from astroquery.gaia import Gaia
        q = ("SELECT ra, dec, pmra, pmdec FROM gaiadr3.gaia_source WHERE 1=CONTAINS("
             f"POINT('ICRS', ra, dec), CIRCLE('ICRS', {info['ra']}, {info['dec']}, {radius}))")
        return Gaia.launch_job(q).get_results().to_pandas()

    def from_vizier(server):
        def run():
            from astroquery.vizier import conf as vconf
            vconf.server = server
            viz = Vizier(columns=["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag"], row_limit=-1)
            viz.VIZIER_SERVER = server
            res = viz.query_region(center, radius=radius * u.deg, catalog="I/355/gaiadr3")
            if len(res) == 0:
                return pd.DataFrame(columns=["ra", "dec", "pmra", "pmdec"])
            return res[0].to_pandas().rename(columns={"RA_ICRS": "ra", "DE_ICRS": "dec",
                                                      "pmRA": "pmra", "pmDE": "pmdec"})
        return run

    sources = [("ESA Gaia archive", from_esa)] + [
        (f"VizieR {s}", from_vizier(s)) for s in
        ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu", "vizier.nao.ac.jp", "vizier.iucaa.in")]
    errors, got_empty = [], False
    for name, fn in sources:
        try:
            t = fn()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}")
            log.warning("%s failed (%s)", name, type(e).__name__)
            continue
        if len(t) == 0:  # a real Gaia cone of this size is rarely empty: try the next source
            got_empty = True
            log.warning("%s returned 0 stars - trying the next source", name)
            continue
        t[["pmra", "pmdec"]] = t[["pmra", "pmdec"]].fillna(0.0)
        log.info("Gaia DR3: %d stars from %s", len(t), name)
        return t[["ra", "dec", "pmra", "pmdec"]]

    if got_empty:
        log.warning("Every reachable source returned 0 Gaia stars for this field.")
        return pd.DataFrame(columns=["ra", "dec", "pmra", "pmdec"])
    raise SystemExit("[!] Could not download a Gaia catalogue (" + "; ".join(errors) + ").\n"
                     "    Try again later, give your own CSV (columns ra, dec[, pmra, pmdec]) in\n"
                     "    config.json -> gaia.catalog_file, or set gaia.enabled to false (step 5 still\n"
                     "    removes stationary stars because they repeat across epochs).")


def fetch_simbad(config):
    """All SIMBAD objects in the field -> DataFrame(main_id, ra, dec) in degrees."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.simbad import Simbad

    info = config["info_dict"]
    center = SkyCoord(info["ra"] * u.deg, info["dec"] * u.deg)
    radius = helpers.field_radius_deg(config)
    try:
        tab = Simbad.query_region(center, radius=radius * u.deg)
    except Exception:  # noqa: BLE001 - older/newer astroquery: fall back to a direct TAP query
        q = ("SELECT main_id, ra, dec FROM basic WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {info['ra']}, {info['dec']}, {radius}))")
        tab = Simbad.query_tap(q)
    if tab is None or len(tab) == 0:
        return pd.DataFrame(columns=["main_id", "ra", "dec"])
    df = tab.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    if df["ra"].dtype == object:  # old astroquery returns sexagesimal strings (hours / degrees)
        sc = SkyCoord(df["ra"].astype(str).values, df["dec"].astype(str).values, unit=(u.hourangle, u.deg))
        df["ra"], df["dec"] = sc.ra.deg, sc.dec.deg
    df["main_id"] = df["main_id"].astype(str).str.strip()
    return df[["main_id", "ra", "dec"]].dropna(subset=["ra", "dec"])


def main():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    config = helpers.load_config()
    helpers.require_flag(config, "detected", "step3_detect_sources.py")
    paths = helpers.get_paths(config)
    det = pd.read_csv(paths["detections_csv"])
    det["gaia_match"] = False
    det["gaia_sep_arcsec"] = np.nan

    if config["gaia"]["enabled"]:
        gaia = fetch_gaia(config, paths)
        gaia.to_csv(paths["gaia_catalog_csv"], index=False)
        log.info("Reference catalogue: %d stars", len(gaia))
        if len(gaia):
            rad = config["gaia"]["match_radius_arcsec"]
            for _, idx in det.groupby("epoch_idx").groups.items():
                sub = det.loc[idx]
                dt_yr = (sub["mjd"].iloc[0] - helpers.MJD_J2016) / 365.25
                cosd = np.cos(np.radians(gaia["dec"].values))
                gra = gaia["ra"].values + gaia["pmra"].values / 3.6e6 * dt_yr / np.maximum(cosd, 1e-6)
                gdec = gaia["dec"].values + gaia["pmdec"].values / 3.6e6 * dt_yr
                gc = SkyCoord(gra * u.deg, gdec * u.deg)
                dc = SkyCoord(sub["ra"].values * u.deg, sub["dec"].values * u.deg)
                _, sep, _ = dc.match_to_catalog_sky(gc)
                sep = sep.to(u.arcsec).value
                det.loc[idx, "gaia_sep_arcsec"] = sep
                det.loc[idx, "gaia_match"] = sep <= rad
    else:
        log.info("Gaia matching disabled in config.")

    det["simbad_match"] = False
    det["simbad_id"] = ""
    scfg = config.get("simbad", {"enabled": False, "match_radius_arcsec": 4.0})
    if scfg["enabled"]:
        try:
            sim = fetch_simbad(config)
            sim.to_csv(os.path.join(paths["catalog_dir"], "simbad_catalog.csv"), index=False)
            log.info("SIMBAD: %d objects in the field", len(sim))
            if len(sim):
                sc = SkyCoord(sim["ra"].values * u.deg, sim["dec"].values * u.deg)
                dc = SkyCoord(det["ra"].values * u.deg, det["dec"].values * u.deg)
                idx, sep, _ = dc.match_to_catalog_sky(sc)
                hit = sep.to(u.arcsec).value <= scfg["match_radius_arcsec"]
                det["simbad_match"] = hit
                det.loc[hit, "simbad_id"] = sim["main_id"].values[idx[hit]]
        except Exception as e:  # noqa: BLE001 - SIMBAD is optional; never kill the run
            log.warning("SIMBAD query failed (%s: %s) - continuing without it", type(e).__name__, e)
    else:
        log.info("SIMBAD matching disabled in config.")

    det.to_csv(paths["gaia_matched_csv"], index=False)
    log.info("%d / %d detections match Gaia stars, %d match SIMBAD objects", int(det["gaia_match"].sum()),
             len(det), int(det["simbad_match"].sum()))
    helpers.set_flag("gaia_matched")


if __name__ == "__main__":
    main()