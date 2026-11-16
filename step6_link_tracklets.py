"""Step 6 - link the remaining detections into linear-motion tracklets.

Seed pairs (two detections in different epochs of one 'visit') define a constant
velocity; every other epoch is checked for a detection near the predicted position.
Tracklets with >= min_detections epochs and a good linear fit are kept.

Every member detection is written as its own row of tracklet_members.csv with its own fitted
position, residual, time and local sky-plane rate, and tracklets.csv keeps the fitted
motion model (p0x, p0y, vx, vy, mjd_ref) so step 8 can place a forced aperture on the
object in ANY exposure of the visit, detected or not.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import helpers

log = helpers.get_logger("step6")


def split_visits(epoch_mjd, gap_days):
    order = np.argsort(epoch_mjd)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if epoch_mjd[b] - epoch_mjd[a] > gap_days:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    return groups


def fit_line(t, x, y):
    """Least-squares x(t), y(t) = p0 + v*t. Returns (p0x, p0y, vx, vy, rms)."""
    A = np.column_stack([np.ones_like(t), t])
    cx, *_ = np.linalg.lstsq(A, x, rcond=None)
    cy, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = np.hypot(x - A @ cx, y - A @ cy)
    return cx[0], cy[0], cx[1], cy[1], float(np.sqrt(np.mean(res ** 2)))


def link_group(ep_ids, det, tcfg):
    """det: DataFrame with columns det_id, epoch_idx, t_hr, x, y. Returns list of dicts."""
    tol = tcfg["link_tolerance_arcsec"]
    rmin, rmax = tcfg["min_rate_arcsec_per_hour"], tcfg["max_rate_arcsec_per_hour"]
    epochs = []
    for e in ep_ids:
        d = det[det["epoch_idx"] == e]
        if len(d):
            epochs.append({"t": float(d["t_hr"].iloc[0]), "ids": d["det_id"].values,
                           "xy": d[["x", "y"]].values, "tree": cKDTree(d[["x", "y"]].values)})
    epochs.sort(key=lambda e: e["t"])
    n_ep = len(epochs)
    if n_ep < tcfg["min_detections"]:
        return []

    found, n_seeds = {}, 0
    for a in range(n_ep):
        for b in range(a + 1, n_ep):
            dt = epochs[b]["t"] - epochs[a]["t"]
            if dt <= 0:
                continue
            nbrs = epochs[a]["tree"].query_ball_tree(epochs[b]["tree"], r=rmax * dt)
            ia = np.concatenate([[i] * len(n) for i, n in enumerate(nbrs)]).astype(int) if nbrs else np.array([], int)
            ib = np.concatenate([n for n in nbrs]).astype(int) if nbrs else np.array([], int)
            if not len(ia):
                continue
            pa, pb = epochs[a]["xy"][ia], epochs[b]["xy"][ib]
            v = (pb - pa) / dt
            rate = np.hypot(v[:, 0], v[:, 1])
            ok = (rate >= rmin) & (rate <= rmax)
            ia, ib, pa, v = ia[ok], ib[ok], pa[ok], v[ok]
            n_seeds += len(ia)
            if n_seeds > tcfg["max_seed_pairs"]:
                log.warning("Seed-pair limit reached; tighten the rate limits or filters.")
                return list(found.values())
            if not len(ia):
                continue
            # nearest detection to the predicted position in every epoch (vectorised over seeds)
            n_in = np.zeros(len(ia), int)
            pick = np.full((len(ia), n_ep), -1, int)
            for k, ep in enumerate(epochs):
                pred = pa + v * (ep["t"] - epochs[a]["t"])
                dist, idx = ep["tree"].query(pred)
                hit = dist <= tol
                pick[hit, k] = idx[hit]
                n_in += hit
            for s in np.where(n_in >= tcfg["min_detections"])[0]:
                key = frozenset(int(epochs[k]["ids"][pick[s, k]]) for k in range(n_ep) if pick[s, k] >= 0)
                if key not in found:
                    found[key] = {k: int(pick[s, k]) for k in range(n_ep) if pick[s, k] >= 0}

    tracklets = []
    for members in found.values():
        sel = dict(members)
        for _ in range(2):  # refit, then re-collect nearest detections around the fitted line
            t = np.array([epochs[k]["t"] for k in sel])
            xy = np.array([epochs[k]["xy"][i] for k, i in sel.items()])
            p0x, p0y, vx, vy, rms = fit_line(t, xy[:, 0], xy[:, 1])
            new = {}
            for k, ep in enumerate(epochs):
                dist, idx = ep["tree"].query([p0x + vx * ep["t"], p0y + vy * ep["t"]])
                if dist <= tol:
                    new[k] = int(idx)
            if len(new) < tcfg["min_detections"]:
                break
            sel = new
        else:
            t = np.array([epochs[k]["t"] for k in sel])
            xy = np.array([epochs[k]["xy"][i] for k, i in sel.items()])
            p0x, p0y, vx, vy, rms = fit_line(t, xy[:, 0], xy[:, 1])
            rate = float(np.hypot(vx, vy))
            motion = rate * float(t.max() - t.min())
            if (len(sel) >= tcfg["min_detections"] and rms <= tol and rmin <= rate <= rmax
                    and motion >= tcfg["min_total_motion_arcsec"]):
                tracklets.append({
                    "det_ids": [int(epochs[k]["ids"][i]) for k, i in sel.items()],
                    "n_det": len(sel), "rms": rms, "rate": rate, "motion": motion,
                    "p0x": p0x, "p0y": p0y, "vx": vx, "vy": vy})
    return tracklets


def main():
    config = helpers.load_config()
    helpers.require_flag(config, "static_filtered", "step5_list_unlabeled_obj.py")
    paths = helpers.get_paths(config)
    tcfg, info = config["tracklets"], config["info_dict"]
    det = pd.read_csv(paths["filtered_csv"])
    det["x_pix"], det["y_pix"] = det["x"], det["y"]
    det["x"], det["y"] = helpers.tangent_offsets(det["ra"], det["dec"], info["ra"], info["dec"])
    mjd0 = det["mjd"].min()
    det["t_hr"] = (det["mjd"] - mjd0) * 24.0

    ep_mjd = det.groupby("epoch_idx")["mjd"].first()
    groups = split_visits(ep_mjd.values, tcfg["visit_gap_days"])
    ep_index = ep_mjd.index.values
    log.info("%d epochs split into %d visit group(s)", len(ep_mjd), len(groups))

    all_tr = []
    for gi, g in enumerate(groups):
        ids = [ep_index[i] for i in g]
        trs = link_group(ids, det, tcfg)
        for tr in trs:
            tr["group"] = gi
        log.info("group %d: %d epochs -> %d raw tracklets", gi, len(ids), len(trs))
        all_tr += trs

    # greedy de-duplication: best (most epochs, lowest rms) first, reject mostly-shared tracks
    all_tr.sort(key=lambda t: (-t["n_det"], t["rms"]))
    used, kept = set(), []
    for tr in all_tr:
        if len(set(tr["det_ids"]) & used) <= 0.5 * tr["n_det"]:
            kept.append(tr)
            used |= set(tr["det_ids"])

    by_id = det.set_index("det_id")
    rows, mrows = [], []
    for k, tr in enumerate(kept):
        m = by_id.loc[tr["det_ids"]].sort_values("mjd")
        mid = len(m) // 2
        t_mid = float(m["t_hr"].iloc[mid])
        xm, ym = tr["p0x"] + tr["vx"] * t_mid, tr["p0y"] + tr["vy"] * t_mid
        ra_m, dec_m = helpers.offsets_to_radec(xm, ym, info["ra"], info["dec"])
        rows.append({
            "tracklet_id": k, "group": tr["group"], "n_det": tr["n_det"],
            "mjd_start": m["mjd"].min(), "mjd_end": m["mjd"].max(), "mjd_mid": m["mjd"].iloc[mid],
            "ra_mid": float(ra_m), "dec_mid": float(dec_m),
            "rate_arcsec_hr": tr["rate"], "pa_deg": float(np.degrees(np.arctan2(tr["vx"], tr["vy"])) % 360),
            "total_motion_arcsec": tr["motion"], "rms_arcsec": tr["rms"], "mean_snr": float(m["snr"].mean()),
            "mjd_ref": float(mjd0), "p0x_arcsec": tr["p0x"], "p0y_arcsec": tr["p0y"],
            "vx_arcsec_hr": tr["vx"], "vy_arcsec_hr": tr["vy"]})
        # one row per detection: fitted position, residual and local rate between its neighbours
        t_arr = m["t_hr"].values
        x_arr, y_arr = m["x"].values, m["y"].values
        xf, yf = tr["p0x"] + tr["vx"] * t_arr, tr["p0y"] + tr["vy"] * t_arr
        ra_f, dec_f = helpers.offsets_to_radec(xf, yf, info["ra"], info["dec"])
        for i, (did, r) in enumerate(m.iterrows()):
            lo, hi = max(i - 1, 0), min(i + 1, len(m) - 1)
            dt = t_arr[hi] - t_arr[lo]
            lvx = (x_arr[hi] - x_arr[lo]) / dt if dt > 0 else np.nan
            lvy = (y_arr[hi] - y_arr[lo]) / dt if dt > 0 else np.nan
            mrows.append({"tracklet_id": k, "det_id": did, "det_order": i, "file": r["file"],
                          "epoch_idx": r["epoch_idx"], "mjd": r["mjd"], "t_hr": float(t_arr[i]),
                          "x_pix": r["x_pix"], "y_pix": r["y_pix"],
                          "ra": r["ra"], "dec": r["dec"], "ra_fit": float(ra_f[i]), "dec_fit": float(dec_f[i]),
                          "resid_arcsec": float(np.hypot(x_arr[i] - xf[i], y_arr[i] - yf[i])),
                          "local_rate_arcsec_hr": float(np.hypot(lvx, lvy)),
                          "snr": r["snr"], "flux": r["flux"]})
    mem = pd.DataFrame(mrows)
    tr_cols = ["tracklet_id", "group", "n_det", "mjd_start", "mjd_end", "mjd_mid", "ra_mid", "dec_mid",
               "rate_arcsec_hr", "pa_deg", "total_motion_arcsec", "rms_arcsec", "mean_snr",
               "mjd_ref", "p0x_arcsec", "p0y_arcsec", "vx_arcsec_hr", "vy_arcsec_hr"]
    mem_cols = ["tracklet_id", "det_id", "det_order", "file", "epoch_idx", "mjd", "t_hr", "x_pix", "y_pix",
                "ra", "dec", "ra_fit", "dec_fit", "resid_arcsec", "local_rate_arcsec_hr", "snr", "flux"]
    pd.DataFrame(rows, columns=tr_cols).to_csv(paths["tracklets_csv"], index=False)
    (mem if len(mem) else pd.DataFrame(columns=mem_cols)).to_csv(paths["members_csv"], index=False)
    if not kept:
        log.info("No tracklets. Typical reasons: too few detections left after step 5, or the settings are "
                 "too strict (try tracklets.min_detections=3, min_total_motion_arcsec=2, visit_gap_days=0.05).")
    log.info("Tracklets kept: %d -> %s", len(kept), paths["tracklets_csv"])
    helpers.set_flag("tracklets_linked")


if __name__ == "__main__":
    main()