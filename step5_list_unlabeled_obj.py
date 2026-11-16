"""Step 5 - quality cuts, astrometric error budget, and static / moving classification.

What it does
  1. Quality cuts (S/N, shape, edge distance).
  2. Per-epoch astrometric calibration against Gaia DR3 propagated to each epoch's MJD:
     median (east, north) shift and scatter of the matched reference stars. The shift is
     applied to every detection of that epoch (ra/dec = corrected, ra_raw/dec_raw = original).
  3. Per-detection 1-sigma position error (per axis, arcsec), combined in quadrature:
         err = sqrt(err_centroid^2 + err_calib^2 + err_gaia_ref^2)
       err_centroid ~ sigma_PSF / SNR        (from FWHM and S/N)
       err_calib    = per-epoch scatter of Gaia reference stars (minus their own centroid
                      error) plus the error of the shift estimate
       err_gaia_ref = Gaia catalogue error (tiny constant)
  4. Classification of every non-Gaia / non-SIMBAD detection by position +- error:
       two detections in different epochs are 'the same place' if their separation is
       <= n_sigma * sqrt(err_i^2 + err_j^2). Chains of such detections form a cluster.
       A cluster seen in >= static_min_epochs epochs spanning >= static_min_span_days is
       'static' (a fixed source Gaia does not list); everything else is 'moving'
       (= not shown to be stationary; step 6 decides whether it really moves).

Outputs (catalogs/ folder unless a path is set in config.json -> paths)
  4_unlabeled_objects.csv      every unlabeled detection (one row per object per epoch) with
                               ra, dec, pos_err_arcsec, category (static|moving), object_id
  4_unlabeled_static_objects.csv  one row per static object: weighted-mean position + error
  epoch_astrometry.csv         per-epoch Gaia calibration (n_ref, shift, rms, err_calib)
  3_detections_all_flagged.csv every detection with its category (known / known_star / bad_quality / static / moving)
  3_detections_filtered.csv    the 'moving' detections -> input of step 6
  4_static_near_misses.csv     unlabeled position clusters seen in >= 2 epochs that did NOT become static,
                               with n_epochs, span_days, mean_snr and the reason
  known_in_field.csv           KNOWN asteroids/comets (SkyBoT) that were inside the cutouts, where
                               each should have been in every exposure, and what step 5 did with the
                               detection found there (outcome = known / static / moving / bad_quality /
                               known_star / not_detected). Replaces check_known_in_field.py.

Optional config.json -> "filtering" keys (defaults in brackets)
  static_n_sigma [3.0]           match threshold in sigmas
  static_floor_arcsec [1.0]      smallest allowed match radius
  static_max_radius_arcsec [max(static_radius_arcsec, 10)]   largest allowed match radius
  static_min_span_days [1.0]     a static source must persist at least this long
  static_min_snr [min_snr]       S/N threshold used ONLY for the static search. Faint QSOs / galaxies are
                                 rarely above min_snr in a single exposure (each exposure samples a different
                                 wavelength), so set e.g. 3.0 to find them. Detections between static_min_snr
                                 and min_snr that do not join a static cluster are dropped (bad_quality);
                                 they never reach the mover search.
  static_min_epochs_faint [5]    a cluster whose MEDIAN S/N is below min_snr needs at least this many epochs
                                 (random low-S/N noise lines up in 3 epochs by chance; real sources persist).
  min_ref_stars [5]              Gaia stars needed to calibrate an epoch
  apply_epoch_shift [true]       apply the per-epoch Gaia shift to ra/dec
  centroid_error_scale [1.0]     multiplier on sigma_PSF/SNR
  astrometric_floor_arcsec [1.0] err_calib used when no epoch has enough reference stars
  gaia_ref_err_arcsec [0.002]    Gaia catalogue error added in quadrature
  known_in_field [true]          run the SkyBoT known-object check (needs internet, ~2 s per visit)
  known_in_field_tol_arcsec [8]  a detection this close to a predicted position counts as a hit
"""
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import helpers

log = helpers.get_logger("step5")

FWHM_TO_SIGMA = 2.3548      # FWHM = 2.3548 sigma for a Gaussian
RAYLEIGH_MEDIAN = 1.1774    # median of a 2-D radial offset = 1.1774 * sigma (per axis)


# ---------------------------------------------------------------- helpers
def out_path(paths, key, filename):
    """Use paths[key] from config.json if present, else put the file in the catalogs folder."""
    return paths.get(key) or os.path.join(paths["catalog_dir"], filename)


def pixel_scale_arcsec(det, paths, default=6.15):
    """Pixel scale from the first image's WCS (SPHEREx is about 6.15 arcsec/pixel)."""
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        *_, wcs = helpers.read_image(os.path.join(paths["images_dir"], det["file"].iloc[0]))
        return float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read the pixel scale (%s); using %.2f arcsec/pix", e, default)
        return default


def centroid_error(det, pix_arcsec, scale):
    """sigma_x ~ sigma_PSF / SNR  (arcsec per axis); floor of 0.02 pixel."""
    snr = det["snr"].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = scale * (det["fwhm"].values / FWHM_TO_SIGMA) / snr * pix_arcsec
    err = np.where(np.isfinite(err) & (snr > 0), err, np.nan)
    return np.maximum(err, 0.02 * pix_arcsec)


def gaia_at_epoch(gaia, mjd):
    """Gaia DR3 positions (J2016.0) propagated to the epoch with proper motion."""
    dt_yr = (mjd - helpers.MJD_J2016) / 365.25
    cosd = np.maximum(np.cos(np.radians(gaia["dec"].values)), 1e-6)
    ra = gaia["ra"].values + gaia["pmra"].values / 3.6e6 * dt_yr / cosd
    dec = gaia["dec"].values + gaia["pmdec"].values / 3.6e6 * dt_yr
    return ra, dec


EPOCH_COLS = ["epoch_idx", "n_ref", "shift_east_arcsec", "shift_north_arcsec",
              "ref_rms_arcsec", "err_calib_arcsec"]


def epoch_astrometry(det, good, gaia, ra0, dec0, match_radius, min_ref):
    """Per epoch: compare good detections with Gaia propagated to that epoch's MJD."""
    rows = []
    if gaia is None or len(gaia) == 0:
        return pd.DataFrame(rows, columns=EPOCH_COLS)
    for e, sub in det[good].groupby("epoch_idx"):
        gra, gdec = gaia_at_epoch(gaia, float(sub["mjd"].iloc[0]))
        gx, gy = helpers.tangent_offsets(gra, gdec, ra0, dec0)
        dx, dy = helpers.tangent_offsets(sub["ra"].values, sub["dec"].values, ra0, dec0)
        dist, j = cKDTree(np.column_stack([gx, gy])).query(np.column_stack([dx, dy]),
                                                           distance_upper_bound=match_radius)
        m = np.isfinite(dist)
        n = int(m.sum())
        row = {"epoch_idx": e, "n_ref": n, "shift_east_arcsec": np.nan, "shift_north_arcsec": np.nan,
               "ref_rms_arcsec": np.nan, "err_calib_arcsec": np.nan}
        if n >= min_ref:
            ox, oy = dx[m] - gx[j[m]], dy[m] - gy[j[m]]          # detected - Gaia
            sx, sy = np.median(ox), np.median(oy)
            sigma = float(np.median(np.hypot(ox - sx, oy - sy)) / RAYLEIGH_MEDIAN)   # per axis
            own = float(np.nanmedian(sub["err_centroid_arcsec"].values[m]))          # their centroid error
            row.update(shift_east_arcsec=sx, shift_north_arcsec=sy, ref_rms_arcsec=sigma,
                       err_calib_arcsec=float(np.sqrt(max(sigma ** 2 - own ** 2, 0.0) + sigma ** 2 / n)))
        rows.append(row)
    return pd.DataFrame(rows, columns=EPOCH_COLS)


def cluster_detections(x, y, sigma, epoch, n_sigma, floor, cap):
    """Link detections from DIFFERENT epochs whose separation <= n_sigma*sqrt(si^2+sj^2)
    (clipped to [floor, cap] arcsec). Returns a cluster label per detection."""
    n = len(x)
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(cap, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n)
    i, j = pairs[:, 0], pairs[:, 1]
    sep = np.hypot(x[i] - x[j], y[i] - y[j])
    thr = np.clip(n_sigma * np.hypot(sigma[i], sigma[j]), floor, cap)
    ok = (sep <= thr) & (epoch[i] != epoch[j])
    graph = coo_matrix((np.ones(int(ok.sum())), (i[ok], j[ok])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def summarize_static(d):
    """One row per static object: inverse-variance weighted position and its error."""
    rows = []
    for oid, g in d.groupby("object_id"):
        ra_ref, dec_ref = float(g["ra"].iloc[0]), float(g["dec"].iloc[0])
        x, y = helpers.tangent_offsets(g["ra"].values, g["dec"].values, ra_ref, dec_ref)
        s = g["pos_err_arcsec"].values
        w = 1.0 / s ** 2
        xm, ym = np.sum(w * x) / w.sum(), np.sum(w * y) / w.sum()
        r2 = (x - xm) ** 2 + (y - ym) ** 2
        n = len(g)
        rms = float(np.sqrt(np.mean(r2) / 2.0))                       # scatter per axis
        chi2 = float(np.sum(w * r2) / (2.0 * (n - 1))) if n > 1 else np.nan
        ra_m, dec_m = helpers.offsets_to_radec(xm, ym, ra_ref, dec_ref)
        rows.append({"object_id": oid, "ra": float(ra_m), "dec": float(dec_m),
                     "pos_err_arcsec": float(max(1.0 / np.sqrt(w.sum()), rms / np.sqrt(n))),
                     "scatter_arcsec": rms, "chi2_red": chi2, "n_epochs": n,
                     "mjd_first": g["mjd"].min(), "mjd_last": g["mjd"].max(),
                     "span_days": g["mjd"].max() - g["mjd"].min(), "mean_snr": g["snr"].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- known objects in the field
KIF_COLS = ["mjd_mid", "name", "V_mag", "rate_arcsec_hr", "motion_in_visit_arcsec", "n_exposures",
            "n_detected_near_prediction", "median_sep_arcsec", "outcome", "n_hit_moving", "n_hit_static",
            "n_hit_known", "n_hit_known_star", "n_hit_bad_quality", "matched_det_ids"]


def skybot_cone(ra, dec, mjd, radius_deg):
    import warnings
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astroquery.imcce import Skybot

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NoResultsWarning on an empty cone
            res = Skybot.cone_search(SkyCoord(ra * u.deg, dec * u.deg), radius_deg * u.deg,
                                     Time(mjd, format="mjd"))
    except RuntimeError:  # older astroquery raises this on an empty cone
        return pd.DataFrame()
    if res is None or len(res) == 0:
        return pd.DataFrame()
    return res.to_pandas()


def known_in_field(det, config, paths, tol_arcsec=8.0):
    """Which KNOWN Solar System bodies (SkyBoT) were inside the cutouts, and what happened to them?

    For every visit it predicts each body's position in every exposure and looks for the nearest
    detection (corrected positions, all categories). The category of that detection says where the
    pipeline lost it: 'moving' = survived to step 6, 'static' = removed as stationary,
    'known' / 'known_star' = swallowed by a Gaia/SIMBAD match, 'bad_quality' = failed the cuts.
    """
    from collections import Counter
    from step6_link_tracklets import split_visits

    info = config["info_dict"]
    half = config["download"]["cutout_size_deg"] * 3600.0 / 2.0
    rad = helpers.field_radius_deg(config)
    xa, ya = helpers.tangent_offsets(det["ra"].values, det["dec"].values, info["ra"], info["dec"])
    epochs = det["epoch_idx"].values
    by_epoch = {e: np.where(epochs == e)[0] for e in np.unique(epochs)}
    cats, ids = det["category"].values, det["det_id"].values
    ep = det.groupby("epoch_idx")["mjd"].first()
    groups = split_visits(ep.values, config["tracklets"]["visit_gap_days"])

    rows = []
    for g in groups:
        mjds = ep.values[g]
        mid = float(np.mean(mjds))
        try:
            sb = skybot_cone(info["ra"], info["dec"], mid, rad)
        except Exception as e:  # noqa: BLE001
            log.warning("SkyBoT failed for MJD %.3f (%s: %s)", mid, type(e).__name__, e)
            continue
        if sb.empty:
            continue
        span_h = (mjds.max() - mjds.min()) * 24.0
        for _, o in sb.iterrows():
            x0, y0 = helpers.tangent_offsets(o["RA"], o["DEC"], info["ra"], info["dec"])
            if abs(x0) > half or abs(y0) > half:
                continue  # inside the search cone but outside the square cutout
            vx, vy = float(o.get("RA_rate", np.nan)), float(o.get("DEC_rate", np.nan))  # arcsec/hour
            hits, seps = [], []
            for e_idx, e_mjd in zip(ep.index[g], mjds):
                k = by_epoch.get(e_idx, [])
                if not len(k):
                    continue
                dt_h = (e_mjd - mid) * 24.0
                px = x0 + (0.0 if np.isnan(vx) else vx) * dt_h
                py = y0 + (0.0 if np.isnan(vy) else vy) * dt_h
                d = np.hypot(xa[k] - px, ya[k] - py)
                j = int(np.argmin(d))
                if d[j] <= tol_arcsec:
                    hits.append(k[j])
                    seps.append(d[j])
            cnt = Counter(cats[hits]) if hits else Counter()
            rate = float(np.hypot(vx, vy)) if not (np.isnan(vx) or np.isnan(vy)) else np.nan
            rows.append({
                "mjd_mid": round(mid, 4), "name": o.get("Name"), "V_mag": o.get("V"),
                "rate_arcsec_hr": round(rate, 1), "motion_in_visit_arcsec": round(rate * span_h, 1),
                "n_exposures": len(g), "n_detected_near_prediction": len(hits),
                "median_sep_arcsec": round(float(np.median(seps)), 2) if seps else np.nan,
                "outcome": cnt.most_common(1)[0][0] if cnt else "not_detected",
                "n_hit_moving": cnt["moving"], "n_hit_static": cnt["static"], "n_hit_known": cnt["known"],
                "n_hit_known_star": cnt["known_star"], "n_hit_bad_quality": cnt["bad_quality"],
                "matched_det_ids": ";".join(str(int(i)) for i in ids[hits])})
    out = pd.DataFrame(rows, columns=KIF_COLS)
    out["V_mag"] = pd.to_numeric(out["V_mag"], errors="coerce")
    return out.sort_values(["V_mag", "mjd_mid"]).reset_index(drop=True)


# ---------------------------------------------------------------- main
def main():
    config = helpers.load_config()
    helpers.require_flag(config, "gaia_matched", "step4_survey_match.py")
    paths = helpers.get_paths(config)
    f = config["filtering"]
    info = config["info_dict"]
    ra0, dec0 = info["ra"], info["dec"]
    det = pd.read_csv(paths["gaia_matched_csv"])
    n0 = len(det)

    # ---- 1. quality cuts
    shape_ok = (det["fwhm"].between(f["min_fwhm_pix"], f["max_fwhm_pix"])
                & (det["elongation"] <= f["max_elongation"]) & (det["edge_dist"] >= f["edge_margin_pix"]))
    good = (det["snr"] >= f["min_snr"]) & shape_ok
    snr_static = min(float(f.get("static_min_snr", f["min_snr"])), float(f["min_snr"]))
    good_lo = (det["snr"] >= snr_static) & shape_ok  # detections allowed into the STATIC search
    log.info("quality cuts: %d -> %d (S/N >= %.1f and shape ok); %d more fail ONLY the S/N cut, %d of them "
             "have S/N >= %.1f (static_min_snr)", n0, int(good.sum()), float(f["min_snr"]),
             int((shape_ok & ~good).sum()), int((good_lo & ~good).sum()), snr_static)

    # ---- 2. centroid errors and per-epoch Gaia calibration
    pix = pixel_scale_arcsec(det, paths)
    det["err_centroid_arcsec"] = centroid_error(det, pix, float(f.get("centroid_error_scale", 1.0)))

    gaia = None
    if config["gaia"]["enabled"] and os.path.exists(paths["gaia_catalog_csv"]):
        gaia = pd.read_csv(paths["gaia_catalog_csv"])
        gaia[["pmra", "pmdec"]] = gaia[["pmra", "pmdec"]].fillna(0.0)
    astro = epoch_astrometry(det, good, gaia, ra0, dec0, float(config["gaia"]["match_radius_arcsec"]),
                             int(f.get("min_ref_stars", 5))).set_index("epoch_idx")
    astro = astro.apply(pd.to_numeric, errors="coerce").astype(float)  # empty tables default to object dtype
    ep = det.groupby("epoch_idx")["mjd"].first().to_frame().join(astro)
    floor_cal = float(f.get("astrometric_floor_arcsec", 1.0))
    valid = ep["err_calib_arcsec"].dropna()
    fallback = float(valid.median()) if len(valid) else floor_cal
    n_cal = len(valid)
    ep["err_calib_arcsec"] = ep["err_calib_arcsec"].fillna(fallback)
    ep["n_ref"] = ep["n_ref"].fillna(0).astype(int)
    ep["calibrated"] = ep.index.isin(valid.index)
    for c in ("shift_east_arcsec", "shift_north_arcsec"):
        ep[c] = ep[c].fillna(0.0)
    ep.reset_index().to_csv(out_path(paths, "epoch_astrometry_csv", "epoch_astrometry.csv"), index=False)
    log.info("Gaia calibration: %d / %d epochs had >= %d reference stars; median err_calib %.2f arcsec%s",
             n_cal, len(ep), int(f.get("min_ref_stars", 5)), fallback,
             "" if n_cal else " (no reference stars: using astrometric_floor_arcsec)")

    det["ra_raw"], det["dec_raw"] = det["ra"], det["dec"]
    x, y = helpers.tangent_offsets(det["ra"].values, det["dec"].values, ra0, dec0)
    if f.get("apply_epoch_shift", True):
        x = x - det["epoch_idx"].map(ep["shift_east_arcsec"]).values
        y = y - det["epoch_idx"].map(ep["shift_north_arcsec"]).values
        ra_c, dec_c = helpers.offsets_to_radec(x, y, ra0, dec0)
        det["ra"], det["dec"] = ra_c, dec_c
    det["err_calib_arcsec"] = det["epoch_idx"].map(ep["err_calib_arcsec"]).values
    gaia_ref = float(f.get("gaia_ref_err_arcsec", 0.002))
    det["pos_err_arcsec"] = np.sqrt(det["err_centroid_arcsec"] ** 2 + det["err_calib_arcsec"] ** 2
                                    + gaia_ref ** 2)

    # ---- 3. known sources and static / moving classification
    known = det["gaia_match"].astype(bool)
    if "simbad_match" in det:
        known |= det["simbad_match"].astype(bool)
    good_v, known_v, lo_v = good.values, known.values, good_lo.values

    n_sig = float(f.get("static_n_sigma", 3.0))
    floor = float(f.get("static_floor_arcsec", 1.0))
    cap = float(f.get("static_max_radius_arcsec", max(float(f["static_radius_arcsec"]), 10.0)))
    min_ep = int(f["static_min_epochs"])
    min_span = float(f.get("static_min_span_days", 1.0))
    min_ep_faint = max(min_ep, int(f.get("static_min_epochs_faint", 5)))

    gi = np.where(lo_v)[0]  # static search uses the (possibly lower) static_min_snr
    lab = cluster_detections(x[gi], y[gi], det["pos_err_arcsec"].values[gi],
                             det["epoch_idx"].values[gi], n_sig, floor, cap)
    cl = pd.DataFrame({"lab": lab, "epoch": det["epoch_idx"].values[gi], "mjd": det["mjd"].values[gi],
                       "known": known_v[gi], "snr": det["snr"].values[gi]})
    agg = cl.groupby("lab").agg(n_ep=("epoch", "nunique"), t0=("mjd", "min"), t1=("mjd", "max"),
                                has_known=("known", "any"), med_snr=("snr", "median"))
    agg["span"] = agg["t1"] - agg["t0"]
    agg["need_ep"] = np.where(agg["med_snr"] < float(f["min_snr"]), min_ep_faint, min_ep)
    agg["is_static"] = (agg["n_ep"] >= agg["need_ep"]) & (agg["span"] >= min_span)
    a = agg.loc[lab]

    n_match = np.full(len(det), np.nan)
    span_d = np.full(len(det), np.nan)
    has_known = np.zeros(len(det), bool)
    static_row = np.zeros(len(det), bool)
    n_match[gi], span_d[gi] = a["n_ep"].values, a["span"].values
    has_known[gi], static_row[gi] = a["has_known"].values, a["is_static"].values

    cat = np.full(len(det), "bad_quality", dtype=object)
    cat[good_v & known_v] = "known"
    cat[good_v & ~known_v & ~static_row] = "moving"
    cat[lo_v & ~known_v & static_row] = "static"
    # unlabeled detection sitting in a cluster that contains a Gaia/SIMBAD match = a catalogued star
    # whose match failed in this epoch (e.g. larger astrometric offset): drop it from the unlabeled set
    cat[lo_v & ~known_v & has_known] = "known_star"
    det["category"] = cat
    det["n_epochs_match"], det["span_days"], det["cluster_has_known"] = n_match, span_d, has_known

    oid = np.full(len(det), "", dtype=object)
    names = {l: f"S{k:04d}" for k, l in enumerate(np.unique(lab[a["is_static"].values & ~known_v[gi]]))}
    lab_full = np.full(len(det), -1)
    lab_full[gi] = lab
    is_st = cat == "static"
    oid[is_st] = [names[l] for l in lab_full[is_st]]
    det["object_id"] = oid

    log.info("categories: %s", ", ".join(f"{k}={v}" for k, v in det["category"].value_counts().items()))

    # ---- why are there so few static objects? clusters that came close
    cl["x"], cl["y"] = x[gi], y[gi]
    agg = agg.join(cl.groupby("lab").agg(x=("x", "mean"), y=("y", "mean"), mean_snr=("snr", "mean")))
    unl = agg[~agg["has_known"]]
    few_ep, short = unl["n_ep"] < unl["need_ep"], unl["span"] < min_span
    log.info("unlabeled position clusters (not Gaia/SIMBAD): %d | static: %d | seen in only 1 epoch: %d | "
             "failed only n_epochs: %d | failed only span (<%g d): %d | failed both: %d",
             len(unl), int(unl["is_static"].sum()), int((unl["n_ep"] == 1).sum()),
             int((few_ep & ~short & (unl["n_ep"] >= 2)).sum()), min_span, int((~few_ep & short).sum()),
             int((few_ep & short & (unl["n_ep"] >= 2)).sum()))
    near = unl[~unl["is_static"] & (unl["n_ep"] >= 2)].copy()
    fe, sh = few_ep[near.index], short[near.index]
    near["reason"] = np.where(fe & sh, "n_epochs and span",
                              np.where(fe, "n_epochs<" + near["need_ep"].astype(int).astype(str), f"span<{min_span:g} d"))
    near["ra"], near["dec"] = helpers.offsets_to_radec(near["x"].values, near["y"].values, ra0, dec0)
    near = near.rename(columns={"n_ep": "n_epochs", "span": "span_days"})
    near_csv = out_path(paths, "static_near_misses_csv", "4_static_near_misses.csv")
    near[["n_epochs", "span_days", "mean_snr", "ra", "dec", "reason"]].sort_values(
        ["n_epochs", "mean_snr"], ascending=False).to_csv(near_csv, index=False)
    log.info("static near-misses (>= 2 epochs): %d -> %s", len(near), near_csv)

    # ---- 4. outputs
    det.to_csv(out_path(paths, "all_flagged_csv", "3_detections_all_flagged.csv"),
                                                  index=False)
    cols = ["object_id", "category", "det_id", "epoch_idx", "file", "mjd", "ra", "dec", "pos_err_arcsec",
            "err_centroid_arcsec", "err_calib_arcsec", "ra_raw", "dec_raw", "snr", "flux", "fwhm",
            "elongation", "peak", "npix", "edge_dist", "x", "y", "n_epochs_match", "span_days"]
    unl = det[det["category"].isin(["static", "moving"])][[c for c in cols if c in det]]
    unl = unl.sort_values(["category", "object_id", "mjd"]).reset_index(drop=True)
    unl.to_csv(out_path(paths, "unlabeled_csv", "4_unlabeled_objects.csv"), index=False)
    stat = summarize_static(unl[unl["category"] == "static"]) if (unl["category"] == "static").any() \
        else pd.DataFrame(columns=["object_id"])
    stat.to_csv(out_path(paths, "unlabeled_static_csv", "4_unlabeled_static_objects.csv"), index=False)
    log.info("Unlabeled detections: %d (%d static in %d objects, %d moving) -> %s",
             len(unl), int((unl["category"] == "static").sum()), len(stat),
             int((unl["category"] == "moving").sum()), out_path(paths, "unlabeled_csv", "4_unlabeled_objects.csv"))

    # ---- 5. known Solar System bodies in the field (SkyBoT): was each one recovered, and where was it lost?
    if f.get("known_in_field", True):
        try:
            kif = known_in_field(det, config, paths, float(f.get("known_in_field_tol_arcsec", 8.0)))
            kif.to_csv(out_path(paths, "known_in_field_csv", "known_in_field.csv"), index=False)
            if kif.empty:
                log.info("known_in_field: no known Solar System bodies inside the cutouts in any visit")
            else:
                log.info("known_in_field: %d body-visits -> %s", len(kif),
                         out_path(paths, "known_in_field_csv", "known_in_field.csv"))
                log.info("outcomes: %s", ", ".join(f"{k}={v}" for k, v in kif["outcome"].value_counts().items()))
                bright = kif[kif["V_mag"] <= 19.0]
                if len(bright):
                    log.info("bodies brighter than V=19:\n%s", bright[
                        ["mjd_mid", "name", "V_mag", "n_exposures", "n_detected_near_prediction", "outcome"]
                    ].to_string(index=False))
        except Exception as e:  # noqa: BLE001 - network problems must not kill step 5
            log.warning("known_in_field check skipped (%s: %s)", type(e).__name__, e)

    out = det[det["category"] == "moving"].reset_index(drop=True)
    out.to_csv(paths["filtered_csv"], index=False)
    log.info("Gaia/SIMBAD known removed: %d (+%d catalogued stars missed by the match), static removed: %d",
             int((det["category"] == "known").sum()), int((det["category"] == "known_star").sum()),
             int((det["category"] == "static").sum()))
    log.info("Moving-source candidates remaining: %d (from %d)", len(out), n0)
    if len(out) < int(config["tracklets"]["min_detections"]):
        raise SystemExit("[!] Too few detections left to form any tracklet.")
    helpers.set_flag("static_filtered")


if __name__ == "__main__":
    main()