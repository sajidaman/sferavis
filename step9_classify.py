"""Step 9 - classify every spectrum from step 8: movers (tracklets) AND static unlabeled sources.

The classifier section (from "SPECTRAL INDEX CLASSIFIER" to "STEP 9") is the original
sferavis_spectral_index.py code with its logic unchanged. The binned spectrum (wavelength in micron,
flux in Jy) goes to classify_spectrum(); fewer than 5 spectral points cannot be classified
('insufficient_points').

MOVERS (tracklets)  -> tracklets/classification.csv
    One row per tracklet: predicted class, the score of every class, n_points, wavelength range,
    known_status from step 7, plus the feature strengths found.

STATIC objects (S0000, S0001, ... = detections not in Gaia / SIMBAD that repeat at the same position,
from step 5; spectra from step 8)  -> catalogs/static_classification.csv
    Fixed sources (QSOs, supernovae, galaxies, brown dwarfs) are classified with the same classifier but
    restricted to the classes that make sense for them (Asteroids and Comets are dropped). The wrapper adds
      * margin: best minus second-best score,
      * a flat-spectrum baseline: a featureless spectrum already scores ~0.5 per feature window, so a flat
        spectrum with the SAME wavelength sampling is scored and subtracted -> excess_* columns and
        predicted_class_flatcorr ('featureless' if no class beats the baseline),
      * final_class: the predicted class, or insufficient_points / low_snr / ambiguous when the spectrum
        cannot support a call.
    Optional config.json -> "static_classification" keys (defaults in brackets)
      classes          [Brown dwarfs, Quasars, Supernovas, Galaxies]
      min_points       [8]     fewer binned points -> 'insufficient_points'
      min_margin       [0.15]  best minus second-best score below this -> 'ambiguous'
      min_median_snr   [3.0]   median S/N of the binned points below this -> 'low_snr'
      min_excess       [0.1]   a class must beat the flat baseline by this much, else 'featureless'
      max_objects      [null]  (used by step 8) measure only the N brightest static objects
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import helpers

log = helpers.get_logger("step9")

# ======================================================================
# SPECTRAL INDEX CLASSIFIER (unchanged)
# ======================================================================


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _overlap_length(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def _safe_percentile(a: np.ndarray, q: float, default: float = 0.0) -> float:
    a = np.asarray(a)
    if a.size == 0:
        return default
    return float(np.percentile(a, q))

def _local_continuum_level(w: np.ndarray, f: np.ndarray, center: float, width: float = 0.03) -> float:
    
    """Estimate a local continuum level as the median flux in a window around center (excluding the center bin if possible).
    width is in microns."""
    
    mask = (w >= center - width) & (w <= center + width)
    if not np.any(mask):
        # fallback: global median
        return float(np.median(f))
    return float(np.median(f[mask]))

def _absorption_strength(w: np.ndarray, f: np.ndarray, wmin: float, wmax: float, cont_center: float) -> float:
    
    """For absorption features: strength = (cont - feature)/|cont|, clamped to [0, 1] via normalization.
    Works even if flux is not continuum-normalized, as long as cont is representative."""
    
    mask_feat = (w >= wmin) & (w <= wmax)
    if not np.any(mask_feat):
        return 0.0

    cont = _local_continuum_level(w, f, center=cont_center, width=(wmax - wmin) * 0.75 + 0.02)
    feat = float(np.median(f[mask_feat]))

    # If cont is ~0, avoid division issues
    if np.abs(cont) < 1e-12:
        return 0.0

    strength_raw = (cont - feat) / abs(cont)
    # map to [0,1] softly
    return _clip01(0.5 + strength_raw)  # heuristic mapping

def _emission_strength(w: np.ndarray, f: np.ndarray, wmin: float, wmax: float, cont_center: float) -> float:
    
    """For emission features: strength = (feature - cont)/|cont|, clamped to [0, 1]."""
    
    mask_feat = (w >= wmin) & (w <= wmax)
    if not np.any(mask_feat):
        return 0.0

    cont = _local_continuum_level(w, f, center=cont_center, width=(wmax - wmin) * 0.75 + 0.02)
    feat = float(np.median(f[mask_feat]))

    if np.abs(cont) < 1e-12:
        return 0.0

    strength_raw = (feat - cont) / abs(cont)
    return _clip01(0.5 + strength_raw)  # heuristic mapping


@dataclass(frozen=True)
class ClassSpec:
    name: str
    range_min: float
    range_max: float
    # (feature_name, feature_weight, feature_type, wmin, wmax, cont_center)
    # feature_type: 'abs' or 'em'
    features: Tuple[Tuple[str, float, str, float, float, float], ...]
    # Optional: bands that are low-transmission / noisy (used for supernova penalty)
    penalty_windows: Tuple[Tuple[float, float, float], ...]  # (wmin, wmax, penalty_weight)


def _make_class_specs() -> Dict[str, ClassSpec]:
    # Brown dwarfs (from documentation_2.docx)
    # - FeH at 1.19 μm
    # - CO at 2.295 μm
    # - CH4 initial signatures in K band at ~2.20 μm
    # - CH4 appearance in H band at 1.667 μm
    # I implement small windows around each center (configurable via feature windows).
    brown = ClassSpec(
        name="Brown dwarfs",
        range_min=1.13, range_max=2.33,
        features=(
            ("FeH_1p19", 1.0, "abs", 1.175, 1.205, 1.19),
            ("CO_2p295", 0.9, "abs", 2.270, 2.320, 2.295),
            ("CH4_K_2p20", 1.0, "abs", 2.175, 2.225, 2.20),
            ("CH4_H_1p667", 1.0, "abs", 1.63, 1.70, 1.667),
        ),
        penalty_windows=(),
    )

    # Quasars:
    # Documentation provides spectral ranges/resolutions, but no explicit line centers.
    # I use template-less heuristics: presence of broad IR coverage + generic "non-stellar"
    # via slope/curvature features would normally be used. Here we keep it range-based
    # plus a weak template-less score using "variance vs continuum".
    # To keep this code faithful to your doc, we implement quasars as:
    # - range gating (dominant)
    # - low penalty for "no strong narrow absorption at BD key bands"
    # I'll approximate that via penalizing absorption at BD centers.
    
    quasars = ClassSpec(
        name="Quasars",
        range_min=0.27, range_max=3.70,  # composite range in doc (final optical–IR composite)
        features=(
            # No explicit line/feature centers in doc; set empty.
        ),
        penalty_windows=(),
    )

    # Supernovas (explicit ion/element wavelength windows from doc)
    supernova = ClassSpec(
        name="Supernovas",
        range_min=0.8, range_max=2.5,
        features=(
            ("CaII_0p85_0p866", 0.8, "abs", 0.850, 0.866, 0.86),
            ("MgII_0p922_0p924", 0.8, "abs", 0.922, 0.924, 0.923),
            ("MgII_CoII_blend_1p091_1p095", 0.6, "abs", 1.091, 1.095, 1.093),
            ("SiII_MgII_blend_1p676_1p691", 0.6, "abs", 1.676, 1.691, 1.684),
            ("FeGroup_1p50_1p90", 0.6, "abs", 1.50, 1.90, 1.70),
        ),
        # Noise/low transmission windows (penalize features there)
        penalty_windows=(
            (1.31, 1.38, 0.6),
            (1.78, 1.88, 0.6),
            (0.93, 0.96, 0.4),
            (1.12, 1.16, 0.4),
            (2.40, 2.50, 0.4),
        )
    )

    # Asteroids: doc gives range only; no explicit bands.
    asteroids = ClassSpec(
        name="Asteroids",
        range_min=0.3, range_max=4.0,
        features=(
            # I'll treat absence of BD and SN features as "evidence" via penalties later.
        ),
        penalty_windows=(),
    )

    # Galaxies: explicit emission lines + range
    galaxies = ClassSpec(
        name="Galaxies",
        range_min=0.9, range_max=2.5,
        features=(
            ("H2_emission_2p122", 1.0, "em", 2.10, 2.14, 2.122),
            ("Brgamma_emission_2p166", 0.9, "em", 2.145, 2.185, 2.166),
        ),
        penalty_windows=(),
    )

    # Comets: emission bands at ~3.4, ~10, ~20 μm; range 3–160 μm
    comets = ClassSpec(
        name="Comets",
        range_min=3.0, range_max=160.0,
        features=(
            ("C-H_emission_3p4", 1.0, "em", 3.25, 3.55, 3.4),
            ("Si-O_broad_emission_10", 0.9, "em", 9.0, 11.5, 10.0),
            ("Si-O-Si_emission_20", 0.8, "em", 18.0, 22.5, 20.0),
        ),
        penalty_windows=(),
    )

    # Quasars range-only: store with empty features
    quasars = ClassSpec(
        name="Quasars",
        range_min=0.27, range_max=3.70,
        features=tuple(),
        penalty_windows=(),
    )

    return {
        "Brown dwarfs": brown,
        "Quasars": quasars,
        "Supernovas": supernova,
        "Asteroids": asteroids,
        "Galaxies": galaxies,
        "Comets": comets,
    }


CLASS_SPECS = _make_class_specs()


# ---------------------------
# Scoring logic
# ---------------------------

def _range_score(w: np.ndarray, c: ClassSpec) -> float:
    wmin_obs = float(np.min(w))
    wmax_obs = float(np.max(w))
    overlap = _overlap_length(wmin_obs, wmax_obs, c.range_min, c.range_max)
    denom = max(1e-12, float(wmax_obs - wmin_obs))
    # if spectrum is very narrow, overlap fraction can be small; keep it bounded
    return _clip01(overlap / denom)

def _apply_supernova_penalty(w: np.ndarray, f: np.ndarray, base: float, c: ClassSpec) -> float:
    if not c.penalty_windows:
        return base
    # penalize if observation includes penalty windows with strong "signal"
    # I'll compute average absolute deviation in those windows.
    penalty = 0.0
    for (pw0, pw1, pw) in c.penalty_windows:
        mask = (w >= pw0) & (w <= pw1)
        if np.any(mask):
            # estimate "badness" as strong variation/abs/emission
            # Here I use percentile of abs(flux) variation proxy.
            bad = float(np.std(f[mask])) / (np.std(f) + 1e-12)
            penalty += pw * bad
    return base * _clip01(1.0 - penalty)

def extract_features(wavelength_um: np.ndarray, flux: np.ndarray) -> Dict[str, float]:
    
    """Returns a dict of extracted feature strengths for all classes.
    For abs features: absorption_strength in [0,1]
    For em features: emission_strength in [0,1]"""
    
    w = np.asarray(wavelength_um, dtype=float)
    f = np.asarray(flux, dtype=float)

    feats = {}


    m = np.isfinite(w) & np.isfinite(f)
    w = w[m]
    f = f[m]

    for cls in CLASS_SPECS.values():
        for (fname, _, ftype, wmin, wmax, cont_center) in cls.features:
            if ftype == "abs":
                feats[fname] = _absorption_strength(w, f, wmin, wmax, cont_center)
            elif ftype == "em":
                feats[fname] = _emission_strength(w, f, wmin, wmax, cont_center)
            else:
                raise ValueError(f"Unknown feature type: {ftype}")
    return feats


def classify_spectrum(wavelength_um: np.ndarray, flux: np.ndarray) -> Dict:
    
    """Returns:
      - predicted_class
      - scores per class
      - extracted feature strengths (for debugging)"""
      
    w = np.asarray(wavelength_um, dtype=float)
    f = np.asarray(flux, dtype=float)

    # Basic sanitation
    m = np.isfinite(w) & np.isfinite(f)
    w = w[m]
    f = f[m]
    if w.size < 5:
        raise ValueError("Spectrum too small after removing NaNs.")

    feats = extract_features(w, f)

    scores = {}
    for cname, c in CLASS_SPECS.items():
        s = 0.0

        # Strong range prior
        rscore = _range_score(w, c)
        s += 1.5 * rscore  # range dominates

        # Add class-specific features
        if c.features:
            for (fname, wgt, ftype, wmin, wmax, cont_center) in c.features:
                s += wgt * feats.get(fname, 0.0)

        # Extra logic for classes that don't list explicit features in the doc
        # - Quasars: penalize if strong BD methane/FeH/CO absorption exists
        # - Asteroids: penalize if strong SN line windows exist, or BD methane exists
        if cname == "Quasars":
            bd_pen = (
                feats.get("FeH_1p19", 0.0) +
                feats.get("CO_2p295", 0.0) +
                feats.get("CH4_K_2p20", 0.0) +
                feats.get("CH4_H_1p667", 0.0)
            ) / 4.0
            s -= 1.0 * bd_pen

        if cname == "Asteroids":
            bd_pen = (
                feats.get("FeH_1p19", 0.0) +
                feats.get("CO_2p295", 0.0) +
                feats.get("CH4_K_2p20", 0.0) +
                feats.get("CH4_H_1p667", 0.0)
            ) / 4.0
            sn_pen = (
                feats.get("CaII_0p85_0p866", 0.0) +
                feats.get("MgII_0p922_0p924", 0.0) +
                feats.get("FeGroup_1p50_1p90", 0.0)
            ) / 3.0
            s -= 1.0 * (0.6 * bd_pen + 0.4 * sn_pen)

        if cname == "Supernovas":
            s = _apply_supernova_penalty(w, f, s, c)

        scores[cname] = float(s)

    predicted = max(scores, key=scores.get)

    return {
        "predicted_class": predicted,
        "scores": scores,
        "extracted_features": feats,
        "observed_range_um": (float(np.min(w)), float(np.max(w))),
    }


# ======================================================================
# STEP 9
# ======================================================================

STATIC_DEFAULTS = {
    "classes": ["Brown dwarfs", "Quasars", "Supernovas", "Galaxies"],
    "min_points": 8,
    "min_margin": 0.15,
    "min_median_snr": 3.0,
    "min_excess": 0.1,
}


def classify_tracklets(paths):
    """Movers: one row per tracklet -> tracklets/classification.csv (original behaviour)."""
    trk_path = paths["checked_csv"] if os.path.exists(paths["checked_csv"]) else paths["tracklets_csv"]
    trk = pd.read_csv(trk_path)
    rows = []
    for tr in trk.itertuples():
        f = os.path.join(paths["spectra_dir"], f"tracklet_{tr.tracklet_id:04d}_spectrum.csv")
        row = {"tracklet_id": tr.tracklet_id, "n_det": tr.n_det, "rate_arcsec_hr": tr.rate_arcsec_hr,
               "known_status": getattr(tr, "known_status", ""), "known_name": getattr(tr, "known_name", ""),
               "predicted_class": "no_spectrum", "n_points": 0}
        if os.path.exists(f):
            sp = pd.read_csv(f).dropna(subset=["wavelength_um", "flux_jy"]).sort_values("wavelength_um")
            row["n_points"] = len(sp)
            try:
                res = classify_spectrum(sp["wavelength_um"].values, sp["flux_jy"].values)
                row.update(predicted_class=res["predicted_class"],
                           wave_min_um=res["observed_range_um"][0], wave_max_um=res["observed_range_um"][1])
                for k, v in res["scores"].items():
                    row[f"score_{k}"] = v
                row["features"] = json.dumps({k: round(v, 3) for k, v in res["extracted_features"].items()})
            except ValueError as e:
                row["predicted_class"] = "insufficient_points"
                log.info("tracklet %d: %s", tr.tracklet_id, e)
        rows.append(row)
        log.info("tracklet %d: %s (%d spectral points)", tr.tracklet_id, row["predicted_class"], row["n_points"])

    out = pd.DataFrame(rows)
    out.to_csv(paths["classification_csv"], index=False)
    if len(out):
        log.info("mover classes: %s", ", ".join(f"{k}={v}" for k, v in out["predicted_class"].value_counts().items()))
    log.info("Movers -> %s", paths["classification_csv"])


def classify_static(w, f, scfg):
    """Step-9 classifier restricted to the static classes, plus margin and the flat-spectrum baseline.
    Returns None when the spectrum has fewer than 5 points (classify_spectrum cannot run)."""
    try:
        res = classify_spectrum(w, f)
    except ValueError:
        return None
    allowed = [c for c in scfg["classes"] if c in res["scores"]]
    scores = {k: res["scores"][k] for k in allowed}
    order = sorted(scores, key=scores.get, reverse=True)
    margin = scores[order[0]] - scores[order[1]] if len(order) > 1 else np.nan
    base = classify_spectrum(w, np.ones_like(w))["scores"]   # flat spectrum, same wavelength sampling
    excess = {k: scores[k] - base[k] for k in allowed}
    best_x = max(excess, key=excess.get)
    if excess[best_x] < scfg["min_excess"]:  # no class beats a flat spectrum: do not pick one by dict order
        best_x = "featureless"
    return {"best": order[0], "margin": margin, "scores": scores, "excess": excess,
            "best_flatcorr": best_x, "res": res}


def static_final_label(n_points, median_snr, cl, scfg):
    if cl is None or n_points < scfg["min_points"]:
        return "insufficient_points"
    if not np.isfinite(median_snr) or median_snr < scfg["min_median_snr"]:
        return "low_snr"
    if not np.isfinite(cl["margin"]) or cl["margin"] < scfg["min_margin"]:
        return "ambiguous"
    return cl["best"]


def classify_static_objects(config, paths):
    """Static unlabeled sources: one row per object -> catalogs/static_classification.csv."""
    scfg = {**STATIC_DEFAULTS, **config.get("static_classification", {})}
    cols = ["object_id", "ra", "dec", "pos_err_arcsec", "det_mean_snr", "n_det_epochs", "n_exposures",
            "n_points", "wave_min_um", "wave_max_um", "median_snr", "final_class", "predicted_class",
            "margin", "predicted_class_flatcorr"]
    stat_csv = paths.get("unlabeled_static_csv") or os.path.join(paths["catalog_dir"],
                                                                 "4_unlabeled_static_objects.csv")
    stat = pd.read_csv(stat_csv) if os.path.exists(stat_csv) else pd.DataFrame()
    if stat.empty or "ra" not in stat:
        log.info("No static unlabeled objects - nothing to classify.")
        pd.DataFrame(columns=cols).to_csv(paths["static_classification_csv"], index=False)
        return
    stat = stat.sort_values("mean_snr", ascending=False).reset_index(drop=True)
    n_exp = {}
    if os.path.exists(paths["static_spectrum_csv"]):
        n_exp = pd.read_csv(paths["static_spectrum_csv"]).groupby("object_id").size().to_dict()

    rows = []
    for s in stat.itertuples():
        row = {"object_id": s.object_id, "ra": s.ra, "dec": s.dec, "pos_err_arcsec": s.pos_err_arcsec,
               "det_mean_snr": s.mean_snr, "n_det_epochs": s.n_epochs, "n_exposures": n_exp.get(s.object_id, 0),
               "n_points": 0, "final_class": "no_spectrum"}
        f = os.path.join(paths["static_spectra_dir"], f"{s.object_id}_spectrum.csv")
        if os.path.exists(f):
            b = pd.read_csv(f).dropna(subset=["wavelength_um", "flux_jy"]).sort_values("wavelength_um")
            snr = b["flux_jy"].values / b["err_jy"].values if len(b) else np.array([])
            med_snr = float(np.nanmedian(snr)) if len(snr) else np.nan
            cl = classify_static(b["wavelength_um"].values, b["flux_jy"].values, scfg) if len(b) else None
            row.update(n_points=len(b), median_snr=med_snr,
                       final_class=static_final_label(len(b), med_snr, cl, scfg))
            if cl is not None:
                row.update(predicted_class=cl["best"], margin=cl["margin"],
                           predicted_class_flatcorr=cl["best_flatcorr"],
                           wave_min_um=cl["res"]["observed_range_um"][0],
                           wave_max_um=cl["res"]["observed_range_um"][1])
                row.update({f"score_{k}": v for k, v in cl["scores"].items()})
                row.update({f"excess_{k}": v for k, v in cl["excess"].items()})
                row["features"] = json.dumps({k: round(v, 3) for k, v in cl["res"]["extracted_features"].items()})
        rows.append(row)
        log.info("static %s: %s (%d bins)", s.object_id, row["final_class"], row["n_points"])

    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out] + [c for c in out if c not in cols]]
    out.to_csv(paths["static_classification_csv"], index=False)
    log.info("static classes: %s", ", ".join(f"{k}={v}" for k, v in out["final_class"].value_counts().items()))
    log.info("Static objects -> %s", paths["static_classification_csv"])


def main():
    t0 = time.time()
    config = helpers.load_config()
    helpers.require_flag(config, "spectra_extracted", "step8_extract_spectra.py")
    paths = helpers.get_paths(config)

    classify_tracklets(paths)
    classify_static_objects(config, paths)
    log.info("Step 9 done in %.1f s", time.time() - t0)
    helpers.set_flag("classified")


if __name__ == "__main__":
    main()
