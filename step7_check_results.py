"""Step 7 - check tracklets against known Solar System bodies (SkyBoT) and make the report.

A tracklet is 'known' if one SkyBoT object lies within match_radius of the measured
position at >= min_match_fraction of the probed epochs (first, middle, last).
Everything else is written to candidates.csv with a diagnostic plot per candidate, and a ranked
triage table (n_det, then S/N) is logged and saved as candidates_triage.csv.
"""
import os

import numpy as np
import pandas as pd

import helpers

log = helpers.get_logger("step7")


def skybot_names(ra, dec, mjd, radius_arcsec):
    """Names of SkyBoT objects within radius of (ra, dec) at the given MJD."""
    import warnings
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astroquery.imcce import Skybot

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NoResultsWarning on an empty cone
            res = Skybot.cone_search(SkyCoord(ra * u.deg, dec * u.deg), radius_arcsec * u.arcsec,
                                     Time(mjd, format="mjd"))
    except RuntimeError:  # older astroquery raises this when the cone is empty
        return set()
    if res is None or len(res) == 0 or "Name" not in res.colnames:
        return set()
    return {str(n) for n in res["Name"]}


def check_known(tr, members, kcfg):
    m = members[members["tracklet_id"] == tr["tracklet_id"]].sort_values("mjd")
    probes = m.iloc[sorted({0, len(m) // 2, len(m) - 1})]
    counts, n_ok = {}, 0
    for _, p in probes.iterrows():
        try:
            names = skybot_names(p["ra"], p["dec"], p["mjd"], kcfg["match_radius_arcsec"])
            n_ok += 1
        except Exception as e:  # noqa: BLE001 - network problems must not kill the run
            log.warning("SkyBoT query failed (%s: %s)", type(e).__name__, e)
            continue
        for n in names:
            counts[n] = counts.get(n, 0) + 1
    if n_ok == 0:
        return "unchecked", ""
    if counts:
        best = max(counts, key=counts.get)
        if counts[best] / n_ok >= kcfg["min_match_fraction"]:
            return "known", best
    return "unknown", ""


def make_plot(tr, members, paths, rcfg):
    """One stamp and one dot for EVERY detection of the tracklet (not one per MJD, not a subset)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ImageNormalize, ZScaleInterval

    m = members[members["tracklet_id"] == tr["tracklet_id"]].sort_values("mjd").reset_index(drop=True)
    max_st = int(rcfg.get("max_stamps", 24))
    sel = np.unique(np.linspace(0, len(m) - 1, min(len(m), max_st)).round().astype(int))
    ncol = min(len(sel), 8)
    nrow = int(np.ceil(len(sel) / ncol))
    half = int(rcfg["cutout_pix"]) // 2
    fig = plt.figure(figsize=(2.4 * max(ncol, 3), 2.5 * nrow + 4.5))
    gs = fig.add_gridspec(nrow + 2, ncol, height_ratios=[1] * nrow + [1.6, 1.2])
    for n, k in enumerate(sel):
        r = m.iloc[k]
        ax = fig.add_subplot(gs[n // ncol, n % ncol])
        ax.set_xticks([]), ax.set_yticks([])
        data, _, _, _ = helpers.read_image(os.path.join(paths["images_dir"], r["file"]))
        xc, yc = int(round(r["x_pix"])), int(round(r["y_pix"]))
        y0, x0 = max(yc - half, 0), max(xc - half, 0)
        stamp = data[y0:yc + half + 1, x0:xc + half + 1]
        if np.isfinite(stamp).any():
            ax.imshow(stamp, origin="lower", cmap="gray",
                      norm=ImageNormalize(stamp, interval=ZScaleInterval()))
        ax.plot(r["x_pix"] - x0, r["y_pix"] - y0, "o", mfc="none", mec="r", ms=14)
        ax.set_title(f"#{k} MJD {r['mjd']:.4f}\nSNR {r['snr']:.1f}", fontsize=6)
    axs = fig.add_subplot(gs[nrow, :])
    ra0, dec0 = m["ra"].mean(), m["dec"].mean()
    ex, ny = helpers.tangent_offsets(m["ra"], m["dec"], ra0, dec0)
    hrs = (m["mjd"] - m["mjd"].min()) * 24
    if "ra_fit" in m:
        fx, fy = helpers.tangent_offsets(m["ra_fit"], m["dec_fit"], ra0, dec0)
        axs.plot(fx, fy, "-", color="grey", lw=0.8, label="linear fit")
    else:
        axs.plot(ex, ny, "-", color="grey", lw=0.8)
    sc = axs.scatter(ex, ny, c=hrs, cmap="viridis", s=28, zorder=3)  # every detection is its own dot
    for i in range(len(m)):
        axs.annotate(str(i), (ex[i], ny[i]), fontsize=5, xytext=(2, 2), textcoords="offset points")
    axs.invert_xaxis()
    axs.set_xlabel("East offset (arcsec)")
    axs.set_ylabel("North offset (arcsec)")
    axs.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=axs, label="hours since first detection")
    axr = fig.add_subplot(gs[nrow + 1, :])
    axr.plot(hrs, m["snr"], "o-", ms=3, lw=0.7)
    axr.set_xlabel("hours since first detection")
    axr.set_ylabel("detection S/N")
    fig.suptitle(f"Tracklet {int(tr['tracklet_id'])}: {tr['rate_arcsec_hr']:.1f} arcsec/hr, PA {tr['pa_deg']:.0f} deg, "
                 f"{int(tr['n_det'])} detections, rms {tr['rms_arcsec']:.2f} arcsec", fontsize=10)
    out = os.path.join(paths["plot_dir"], f"candidate_{int(tr['tracklet_id']):04d}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    config = helpers.load_config()
    helpers.require_flag(config, "tracklets_linked", "step6_link_tracklets.py")
    paths = helpers.get_paths(config)
    tr = pd.read_csv(paths["tracklets_csv"])
    mem = pd.read_csv(paths["members_csv"])

    if tr.empty:
        log.info("No tracklets were found - nothing to check.")
        tr.to_csv(paths["checked_csv"], index=False)
        tr.to_csv(paths["candidates_csv"], index=False)
        helpers.set_flag("known_objects_checked")
        return

    status, match = [], []
    for _, row in tr.iterrows():
        if config["known_objects"]["enabled"]:
            s, n = check_known(row, mem, config["known_objects"])
        else:
            s, n = "unchecked", ""
        status.append(s)
        match.append(n)
        log.info("tracklet %d: %s %s", row["tracklet_id"], s, n)
    tr["known_status"], tr["known_name"] = status, match
    tr.to_csv(paths["checked_csv"], index=False)

    cand = tr[tr["known_status"] != "known"].sort_values(["n_det", "mean_snr"], ascending=False)
    cand.to_csv(paths["candidates_csv"], index=False)
    log.info("%d tracklets: %d known, %d candidates", len(tr), int((tr["known_status"] == "known").sum()), len(cand))
    if len(cand):  # ranked triage table: most detections / highest S/N first
        cols = ["tracklet_id", "group", "n_det", "mjd_mid", "rate_arcsec_hr", "pa_deg",
                "total_motion_arcsec", "rms_arcsec", "mean_snr", "known_status"]
        log.info("Candidates ranked by n_det, then S/N:\n%s",
                 cand[cols].sort_values(["n_det", "mean_snr"], ascending=False).round(2).to_string(index=False))
        triage_csv = os.path.join(paths["tracklet_dir"], "candidates_triage.csv")
        cand[cols].sort_values(["n_det", "mean_snr"], ascending=False).round(2).to_csv(triage_csv, index=False)
        log.info("Triage table saved: %s", triage_csv)
    for _, row in cand.head(int(config["report"]["max_candidate_plots"])).iterrows():
        log.info("plot: %s", make_plot(row, mem, paths, config["report"]))
    helpers.set_flag("known_objects_checked")
    log.info("Done. Candidates: %s", paths["candidates_csv"])


if __name__ == "__main__":
    main()