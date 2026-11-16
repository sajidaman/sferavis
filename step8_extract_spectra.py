"""Step 8 - spectrum of every tracklet (forced aperture photometry on the moving object).

Method (follows the approach of spxquery and spexpi for SPHEREx cutouts)
  * Every SPHEREx exposure samples ONE wavelength per detector pixel (linear variable filter), so a
    moving object gives a different wavelength in each exposure: its flux-versus-time table is at the same
    time its low-resolution spectrum (flux-versus-wavelength).
  * Wavelength (and bandwidth) of each measurement: spectral WCS of the cutout (alternate WCS key 'W',
    WCS-WAVE lookup table) evaluated at the object's pixel; if unavailable, the nominal centre of the band.
  * Photometry: circular aperture on the object's position, local annulus background (sigma-clipped
    median), bad pixels from FLAGS excluded, optional ZODI subtraction, errors from VARIANCE plus the
    background-estimate error. MJy/sr -> Jy with the pixel solid angle from the cutout WCS.
  * WHICH exposures: every exposure that covers the object, not only the ones with a detection.
      - known body (SkyBoT match in step 7): its Miriade (IMCCE) ephemeris at the MJD of EVERY downloaded
        exposure, so the spectrum is built from all visits/days;
      - unknown tracklet: the whole visit (all exposures within visit_gap_days of each other), using the
        linear motion model. Without an orbit nothing reliable can be said between visits.
  * Position per exposure: the measured detection centroid for exposures where the tracklet has a
    detection (every detection is its own measurement); for the other exposures of the visit the position
    of the step-6 linear motion model is used (forced photometry), with the step-5 per-epoch Gaia shift undone.
  * Trailing: a mover smears over rate*exposure; the aperture radius is widened by half the trail.
  * Spectrum: points are binned per detector in wavelength bins of width lambda/R (R from the SPHEREx
    detector resolving powers), inverse-variance weighted, after a MAD outlier clip.

Speed: every FITS file is opened ONCE and all tracklet positions in it are measured together. The step logs
progress with an ETA and stops cleanly (keeping everything done so far) when photometry.max_runtime_s is reached.

STATIC unlabeled sources (QSO / supernova / galaxy / brown-dwarf candidates) are extracted too
  * Step 5 lists every detection that is not in Gaia / SIMBAD and repeats at the same sky position in
    several epochs (catalogs/4_unlabeled_static_objects.csv, ids S0000, S0001, ...).
  * A fixed source is measured with the same forced aperture photometry at its fixed position in EVERY
    downloaded exposure (the step-5 Gaia epoch shift is added back to get the raw frame position, no
    trail correction). Each exposure puts it on a different wavelength, so the exposures together give
    its spectrum; there is no lightcurve confusion as for a mover.

Outputs
  tracklets/spectrum_points.csv        one row per (tracklet, exposure) = the unbinned spectral points,
                                       with a 'good' column (only good rows are used for the spectrum)
  spectra/tracklet_XXXX_spectrum.csv   binned spectrum of each tracklet (input of step 9)
  spectra/tracklet_XXXX_spectrum.png   spectrum plot of EVERY tracklet that has a spectrum
  catalogs/static_spectrum_points.csv  same, one row per (static object, exposure)
  static_spectra/S####_spectrum.csv    binned spectrum of each static object (input of step 9)
  static_spectra/S####_spectrum.png    spectrum plot of EVERY static object that has a spectrum
  timing of each phase is logged

PNG policy: a PNG is saved for every spectrum. photometry.max_spectrum_plots (default null = all) can cap
the number per kind (tracklets / static objects) if a huge field makes that too many; the old
photometry.max_plots key is no longer used.
"""
import os
import re
import time

import numpy as np
import pandas as pd
from astropy import log as astropy_log

import helpers

log = helpers.get_logger("step8")
astropy_log.setLevel("WARNING")  # silence the per-image "SIP ... CTYPE" INFO notice (positions are unaffected)

# SPHEREx detector bands: nominal (lambda_min, lambda_max) in micron and resolving power
BANDS = {1: (0.75, 1.09, 39), 2: (1.10, 1.62, 41), 3: (1.63, 2.41, 41),
         4: (2.42, 3.82, 35), 5: (3.83, 4.41, 112), 6: (4.42, 5.00, 128)}

# SPHEREx FLAGS bits treated as fatal (same list as spexpi)
BAD_BITS = (0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 19, 22, 24, 26, 27, 28, 29)

DEFAULTS = {
    "aperture_radius_pix": 2.0,
    "annulus_inner_scale": 2.0,
    "annulus_outer_scale": 3.0,
    "sigma_clip": 3.0,
    "min_bg_pixels": 10,
    "max_bad_fraction": 0.3,
    "subtract_zodi": True,
    "trail_correction": True,
    "exposure_s": 100.0,
    "max_aperture_pix": 6.0,
    "aperture_correction": 1.0,      # used only when the cutout has no PSF extension
    "psf_aperture_correction": True,  # divide un-trailed aperture flux by the PSF encircled energy in the aperture
    "psf_photometry": False,          # STATIC sources: weighted PSF fit instead of aperture (opt-in; check psf_chi2r ~ 1)
    "psf_kernel_radius_native": 5,    # half-size (pixels) of the PSF stamp that is fitted
    "psf_min_valid_pixels": 6,
    "scope": "visit",            # unknown tracklets: "visit" = whole visit, "span" = first..last detection only
    "forced_pad_days": 0.0,      # extra time added around the span (scope "span" only)
    "use_ephemeris_for_known": True,   # known bodies: ephemeris position in every downloaded exposure
    "only_candidates": False,
    "max_tracklets": None,
    "max_runtime_s": 1800,
    "outlier_sigma": 3.0,
    "max_spectrum_plots": None,   # PNGs per kind (tracklets / static objects); None or 0 = one for every spectrum
    "ignore_flag_bits": [],   # FLAGS bits NOT treated as bad (e.g. if a bright mover is masked as a transient)
}


# ---------------------------------------------------------------- image access
def detector_of(name, fallback=None):
    m = re.search(r"D([1-6])", str(name))
    return int(m.group(1)) if m else fallback


def read_cube(path):
    """IMAGE, VARIANCE, FLAGS, ZODI, header, celestial WCS, spectral WCS (or None), pixel scale, pixel sr."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_area, proj_plane_pixel_scales

    with fits.open(path, memmap=False) as hdul:
        names = [h.name.upper() for h in hdul]
        idx = names.index("IMAGE") if "IMAGE" in names else next(
            i for i, h in enumerate(hdul) if h.data is not None and h.data.ndim == 2)
        img = np.array(hdul[idx].data, float)
        hdr = hdul[idx].header.copy()

        def ext(name, dtype):
            return np.array(hdul[names.index(name)].data, dtype) if name in names else None

        var, flags, zodi = ext("VARIANCE", float), ext("FLAGS", np.int64), ext("ZODI", float)
        wcs = WCS(hdr, fobj=hdul).celestial
        psf = np.array(hdul[names.index("PSF")].data, float) if "PSF" in names else None
        psf_over = int(hdul[names.index("PSF")].header.get("OVERSAMP", 10)) if psf is not None else 10
        try:
            swcs = WCS(hdr, fobj=hdul, key="W")
        except Exception:  # noqa: BLE001
            swcs = None
    scale = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
    sr = float(proj_plane_pixel_area(wcs)) * (np.pi / 180.0) ** 2
    return {"img": img, "var": var, "flags": flags, "zodi": zodi, "hdr": hdr, "wcs": wcs,
            "swcs": swcs, "scale": scale, "sr": sr, "psf": psf, "psf_over": psf_over}


def wavelength_at(swcs, x, y):
    """(wavelength, bandwidth) in micron from the spectral WCS at pixel (x, y); NaN if unavailable."""
    if swcs is None:
        return np.nan, np.nan
    try:
        res = swcs.pixel_to_world(x, y)
        wave, dwave = res[0], res[1]
        conv = lambda q: float(q.to("um").value) if hasattr(q, "unit") else float(q)  # noqa: E731
        return conv(wave), conv(dwave)
    except Exception:  # noqa: BLE001
        try:
            vals = swcs.all_pix2world(x, y, 0)
            return float(vals[0]), float(vals[1])
        except Exception:  # noqa: BLE001
            return np.nan, np.nan


# ---------------------------------------------------------------- PSF (SPHEREx PSF extension, as in spexpi)
def psf_native_kernel(c, x, y, radius_native=5):
    """Native-pixel PSF stamp for pixel (x, y): zone plane of the oversampled PSF cube, sub-pixel shifted to
    the source phase, re-binned to detector pixels, sum = 1. Raises ValueError if there is no usable PSF."""
    from scipy.ndimage import shift as ndi_shift
    cube = c.get("psf")
    if cube is None or cube.ndim != 3:
        raise ValueError("no PSF extension")
    over = int(c["psf_over"])
    ny, nx = c["img"].shape
    npl = cube.shape[0]
    ng = int(round(np.sqrt(npl))) if int(round(np.sqrt(npl))) ** 2 == npl else 11
    zx = min(ng - 1, max(0, int(np.floor(x / nx * ng))))
    zy = min(ng - 1, max(0, int(np.floor(y / ny * ng))))
    p = np.clip(np.nan_to_num(cube[min(npl - 1, zy * ng + zx)], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if p.sum() <= 0:
        raise ValueError("empty PSF plane")
    p = p / p.sum()
    p = np.clip(ndi_shift(p, ((y - round(y)) * over, (x - round(x)) * over), order=3, mode="constant"), 0, None)
    p = p / p.sum()
    # crop (2R+1)*over super-pixels centred on the PSF centre, then bin by `over` so the middle native pixel is
    # centred on the PSF. For an even oversampling factor the middle native pixel's centre lies half a super-pixel
    # before the central super-pixel, so shift by -0.5 super-pixel first.
    if over % 2 == 0:
        p = np.clip(ndi_shift(p, (-0.5, -0.5), order=1, mode="constant"), 0, None)
    n = (2 * int(radius_native) + 1) * over
    cy, cx = p.shape[0] // 2, p.shape[1] // 2
    pad = n // 2 + 1
    p = np.pad(p, pad)
    cut = p[cy + pad - n // 2:cy + pad - n // 2 + n, cx + pad - n // 2:cx + pad - n // 2 + n]
    m = n // over
    k = cut.reshape(m, over, m, over).sum(axis=(1, 3))
    if k.sum() <= 0:
        raise ValueError("PSF crop empty")
    return k / k.sum()


def encircled_energy(k, r, dx=0.0, dy=0.0):
    """Fraction of the PSF stamp k inside radius r pixels around its middle pixel + (dx, dy) (fractional edge)."""
    yy, xx = np.indices(k.shape)
    d = np.hypot(xx - (k.shape[1] - 1) / 2.0 - dx, yy - (k.shape[0] - 1) / 2.0 - dy)
    return float(np.sum(k * np.clip(r + 0.5 - d, 0.0, 1.0)))


def psf_photometry(c, x, y, cfg):
    """Weighted linear PSF fit  image = flux*PSF + constant  on a (2R+1)^2 stamp. Returns dict (Jy) or None."""
    try:
        k = psf_native_kernel(c, x, y, cfg["psf_kernel_radius_native"])
    except Exception:  # noqa: BLE001
        return None
    img, var, flags = c["img"], c["var"], c["flags"]
    if c["zodi"] is not None and cfg["subtract_zodi"]:
        img = img - c["zodi"]
    to_jy = c["sr"] * 1e6
    ph, pw = k.shape
    x0, y0 = int(round(x)), int(round(y))
    lw, lh = pw // 2, ph // 2
    X1, X2 = max(0, x0 - lw), min(img.shape[1], x0 + pw - lw)
    Y1, Y2 = max(0, y0 - lh), min(img.shape[0], y0 + ph - lh)
    cut = img[Y1:Y2, X1:X2] * to_jy
    kk = k[lh - (y0 - Y1):lh - (y0 - Y1) + cut.shape[0], lw - (x0 - X1):lw - (x0 - X1) + cut.shape[1]]
    if cut.shape != kk.shape or var is None:
        return None
    v = var[Y1:Y2, X1:X2] * to_jy ** 2
    ok = np.isfinite(cut) & np.isfinite(v) & (v > 0)
    if flags is not None:
        mask = 0
        for b in BAD_BITS:
            if b not in cfg["ignore_flag_bits"]:
                mask |= 1 << b
        ok &= (flags[Y1:Y2, X1:X2] & mask) == 0
    if ok.sum() < int(cfg["psf_min_valid_pixels"]):
        return None
    A = np.column_stack([kk[ok], np.ones(int(ok.sum()))])
    w = 1.0 / v[ok]
    Aw = A * np.sqrt(w)[:, None]
    try:
        cov = np.linalg.inv(Aw.T @ Aw)
    except np.linalg.LinAlgError:
        return None
    par = cov @ (Aw.T @ (cut[ok] * np.sqrt(w)))
    chi2 = float(np.sum((cut[ok] - A @ par) ** 2 * w))
    dof = int(ok.sum()) - 2
    flux, err = float(par[0]), float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.nan
    return {"flux_jy": flux, "err_jy": err, "snr": flux / err if err > 0 else np.nan,
            "psf_chi2r": chi2 / dof if dof > 0 else np.nan}


# ---------------------------------------------------------------- photometry
def aperture_photometry(c, x, y, r, cfg, use_ee=False):
    """Forced circular-aperture photometry at pixel (x, y). Returns a dict (flux/err in Jy)."""
    img, var, flags = c["img"], c["var"], c["flags"]
    if c["zodi"] is not None and cfg["subtract_zodi"]:
        img = img - c["zodi"]
    ny, nx = img.shape
    r_in, r_out = r * cfg["annulus_inner_scale"], r * cfg["annulus_outer_scale"]
    out = {"flux_jy": np.nan, "err_jy": np.nan, "snr": np.nan, "bg_mjysr": np.nan,
           "n_ap_pix": 0.0, "bad_frac": np.nan, "flag": "", "flag_bits": "", "ap_corr": np.nan,
           "method": "aperture", "psf_chi2r": np.nan}
    if not (-0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5):
        out["flag"] = "outside"
        return out
    half = int(np.ceil(r_out)) + 1
    x0, x1 = max(int(round(x)) - half, 0), min(int(round(x)) + half + 1, nx)
    y0, y1 = max(int(round(y)) - half, 0), min(int(round(y)) + half + 1, ny)
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    bad = ~np.isfinite(sub)
    if var is not None:
        v = var[y0:y1, x0:x1]
        bad |= ~np.isfinite(v) | (v <= 0)
    if flags is not None:
        mask = 0
        for b in BAD_BITS:
            if b not in cfg["ignore_flag_bits"]:
                mask |= 1 << b
        fsub = flags[y0:y1, x0:x1]
        bad |= (fsub & mask) != 0
        in_ap = int(np.bitwise_or.reduce(fsub[d <= r + 0.5])) if (d <= r + 0.5).any() else 0
        out["flag_bits"] = "|".join(str(b) for b in range(32) if in_ap >> b & 1)  # bits set inside the aperture
    w = np.clip(r + 0.5 - d, 0.0, 1.0)  # fractional pixel coverage of the aperture edge
    full = w.sum()
    w_good = np.where(bad, 0.0, w)
    out["bad_frac"] = float(1.0 - w_good.sum() / full) if full > 0 else 1.0
    ann = (d >= r_in) & (d <= r_out) & ~bad
    if ann.sum() < cfg["min_bg_pixels"]:
        out["flag"] = "no_background"
        return out
    vals = sub[ann]
    for _ in range(5):  # iterative sigma clip of the annulus
        med, sd = np.median(vals), np.std(vals)
        keep = np.abs(vals - med) <= cfg["sigma_clip"] * max(sd, 1e-30)
        if keep.all():
            break
        vals = vals[keep]
    bg, bg_sd, n_bg = float(np.median(vals)), float(np.std(vals)), len(vals)
    if out["bad_frac"] > cfg["max_bad_fraction"]:
        out["flag"] = "bad_pixels"
    # sum in MJy/sr; w_good is renormalised so masked pixels do not bias the flux low
    scale_up = full / w_good.sum() if w_good.sum() > 0 else np.nan
    clean = np.where(bad, 0.0, sub)  # 0 * NaN = NaN, so blank the masked pixels before the weighted sum
    s = float(np.sum(w_good * (clean - bg))) * scale_up
    if var is not None:
        s_var = float(np.sum(w_good ** 2 * np.where(bad, 0.0, var[y0:y1, x0:x1]))) * scale_up ** 2
    else:
        s_var = float(w_good.sum()) * bg_sd ** 2 * scale_up ** 2
    s_var += (w_good.sum() * scale_up) ** 2 * bg_sd ** 2 / max(n_bg, 1)
    corr = cfg["aperture_correction"]
    if use_ee and cfg.get("psf_aperture_correction", True) and c.get("psf") is not None:
        try:  # un-trailed aperture: correct for the PSF flux that falls outside the aperture
            corr = 1.0 / max(encircled_energy(psf_native_kernel(c, x, y, cfg["psf_kernel_radius_native"]), r,
                                              x - round(x), y - round(y)), 0.05)
        except Exception:  # noqa: BLE001
            pass
    out["ap_corr"] = float(corr)
    to_jy = c["sr"] * 1e6 * corr
    out.update(flux_jy=s * to_jy, err_jy=float(np.sqrt(s_var)) * to_jy, bg_mjysr=bg,
               n_ap_pix=float(w_good.sum()))
    if out["err_jy"] > 0:
        out["snr"] = out["flux_jy"] / out["err_jy"]
    return out


# ---------------------------------------------------------------- ephemeris of known bodies
def ephemeris_radec(name, mjds):
    """Geocentric RA/Dec (deg) of a known body at the given MJDs from IMCCE Miriade; None if unavailable.

    Miriade.get_ephemerides takes ONE start epoch (an astropy Time) plus a step, not a list of epochs
    (a list fails with "'list' object has no attribute 'jd'"), so every distinct exposure time is queried
    on its own; exposures sharing an MJD (simultaneous detectors) share one query."""
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        from astroquery.imcce import Miriade

        def to_deg(ra, dec):
            if getattr(ra, "unit", None) not in (None, u.dimensionless_unscaled):
                return float(u.Quantity(ra).to(u.deg).value), float(u.Quantity(dec).to(u.deg).value)
            if isinstance(ra, (str, bytes)) or getattr(ra, "dtype", None) in (object,) or getattr(getattr(ra, "dtype", None), "kind", "") in "US":
                sc = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))
                return float(sc.ra.deg), float(sc.dec.deg)
            return float(ra), float(dec)

        keys = np.round(np.asarray(mjds, float), 6)
        uniq = np.unique(keys)
        log.info("querying Miriade for %s at %d distinct exposure times", name, len(uniq))
        pos = {}
        for m in uniq:
            t = Miriade.get_ephemerides(targetname=name, objtype="asteroid",
                                        epoch=Time(float(m) + 2400000.5, format="jd"), location="500")
            if len(t) != 1:
                return None
            pos[m] = to_deg(t["RA"][0], t["DEC"][0])
        return np.array([pos[k][0] for k in keys]), np.array([pos[k][1] for k in keys])
    except Exception as e:  # noqa: BLE001 - network/name problems must not kill the step
        log.warning("ephemeris for %s unavailable (%s: %s)", name, type(e).__name__, e)
        return None


# ---------------------------------------------------------------- spectrum binning
def bin_spectrum(pts, outlier_sigma=3.0):
    """pts: DataFrame(wavelength_um, flux_jy, err_jy, detector). Bins of width lambda/R per detector."""
    rows = []
    for det, g in pts.groupby("detector"):
        if det not in BANDS:
            continue
        R = BANDS[det][2]
        w, f, e = g["wavelength_um"].values, g["flux_jy"].values, g["err_jy"].values
        step = np.log1p(1.0 / R)
        b = np.floor(np.log(w / w.min()) / step).astype(int)
        for k in np.unique(b):
            m = b == k
            ww, ff, ee = w[m], f[m], e[m]
            if len(ff) >= 4:  # MAD outlier clip
                med = np.median(ff)
                mad = 1.4826 * np.median(np.abs(ff - med))
                keep = np.abs(ff - med) <= outlier_sigma * max(mad, 1e-30)
                ww, ff, ee = ww[keep], ff[keep], ee[keep]
            ivar = 1.0 / ee ** 2
            fm = float(np.sum(ivar * ff) / ivar.sum())
            err = float(1.0 / np.sqrt(ivar.sum()))
            if len(ff) > 1:
                err = max(err, float(np.std(ff, ddof=1) / np.sqrt(len(ff))))
            rows.append({"wavelength_um": float(np.sum(ivar * ww) / ivar.sum()), "flux_jy": fm,
                         "err_jy": err, "detector": int(det), "n_points": int(len(ff))})
    return pd.DataFrame(rows, columns=["wavelength_um", "flux_jy", "err_jy", "detector", "n_points"]
                        ).sort_values("wavelength_um").reset_index(drop=True)


# ---------------------------------------------------------------- plots
def plot_spectrum(title, pts, binned, out_png):
    """spexpi-style diagnostic: filtered individual measurements (dots), lambda/R-binned points (open circles
    with error bars), a PCHIP curve through the bins, detector boundaries (dashed red) -> PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    finite = np.abs(np.concatenate([pts["flux_jy"].values, binned["flux_jy"].values]
                                   if len(binned) else pts["flux_jy"].values))
    scale, unit = (1.0, "Jy")
    ref = float(np.nanmedian(finite)) if finite.size else 0.0
    if ref < 1e-3:
        scale, unit = 1e6, "\u00b5Jy"
    elif ref < 1.0:
        scale, unit = 1e3, "mJy"
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for edge in [BANDS[d][0] for d in range(2, 7)] + [BANDS[6][1]]:
        ax.axvline(edge, color="red", ls=":", lw=0.8, alpha=0.6, zorder=0)
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.plot(pts["wavelength_um"], pts["flux_jy"] * scale, ".", color="k", ms=3.5, zorder=2,
            label="unbinned filtered measurements")
    if len(binned):
        bw, bf, be = (binned["wavelength_um"].values, binned["flux_jy"].values * scale,
                      binned["err_jy"].values * scale)
        if len(binned) >= 4:
            from scipy.interpolate import PchipInterpolator
            o = np.argsort(bw)
            bw_u, idx = np.unique(bw[o], return_index=True)
            if len(bw_u) >= 4:
                xs = np.linspace(bw_u[0], bw_u[-1], 600)
                ax.plot(xs, PchipInterpolator(bw_u, bf[o][idx])(xs), "-", color="0.45", lw=0.9, zorder=1,
                        label="PCHIP through binned spectrum")
        ax.errorbar(bw, bf, yerr=be, fmt="o", mfc="white", mec="k", ecolor="0.5", ms=4.5, lw=0.6, zorder=3,
                    label="binned spectrum points")
    ax.plot([], [], ":", color="red", label="detector boundaries")
    ax.set_xlim(0.7, 5.05)
    ax.set_xlabel("Wavelength (\u00b5m)")
    ax.set_ylabel(f"Flux ({unit})")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    ax.set_title(title, fontsize=10)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def noise_suspect(pts):
    """True when the exposures WITH a detection are significant but the forced exposures (no detection) are
    consistent with zero: the typical signature of a noise tracklet (detections sit on noise peaks)."""
    if "detected" not in pts or len(pts) < 4:
        return False
    d, f = pts[pts["detected"].astype(bool)], pts[~pts["detected"].astype(bool)]
    if len(d) < 2 or len(f) < 2:
        return False
    snr = lambda g: abs(np.sum(g["flux_jy"] / g["err_jy"] ** 2) / np.sqrt(np.sum(1.0 / g["err_jy"] ** 2)))  # noqa: E731
    return bool(snr(d) > 3.0 and snr(f) < 2.0)


def plot_tracklet(tid, trk, pts, binned, out_png):
    warn = "  [SUSPECT: forced points consistent with zero]" if noise_suspect(pts) else ""
    plot_spectrum(f"Tracklet {tid}: {trk['rate_arcsec_hr']:.1f} arcsec/hr, {int(trk['n_det'])} detections, "
                  f"{len(pts)} spectral points{warn}", pts, binned, out_png)


def _plot_ok(n_done, cfg):
    """True while fewer than photometry.max_spectrum_plots PNGs were written (None / 0 = no limit)."""
    lim = cfg.get("max_spectrum_plots")
    return not lim or n_done < int(lim)


# ---------------------------------------------------------------- exposure table, epoch shifts
def exposure_table(paths):
    """Downloaded exposures with epoch_idx (numbered as in step 3), MJD and band."""
    from astropy.io import fits

    obs = pd.read_csv(paths["observations_csv"])
    obs = obs[obs["status"].isin(["exists", "downloaded"])].reset_index(drop=True)
    obs["epoch_idx"] = obs.index  # fallback numbering (same as step 3)
    mjd_map, ep_map = {}, {}
    if os.path.exists(paths["gaia_matched_csv"]):
        d = pd.read_csv(paths["gaia_matched_csv"], usecols=["file", "mjd", "epoch_idx"]).drop_duplicates("file")
        mjd_map, ep_map = dict(zip(d["file"], d["mjd"])), dict(zip(d["file"], d["epoch_idx"]))
    for f in obs["file"]:
        if f not in mjd_map:  # exposure without any detection: read just the header
            try:
                mjd_map[f] = helpers.header_mjd(fits.getheader(os.path.join(paths["images_dir"], f), "IMAGE"))
            except Exception:  # noqa: BLE001
                mjd_map[f] = None
    obs["mjd"] = obs["file"].map(mjd_map)
    # epoch_idx of step 3 / step 5 (taken from the detections file when available, so the shifts line up)
    obs["epoch_idx"] = [int(ep_map.get(f, e)) for f, e in zip(obs["file"], obs["epoch_idx"])]
    return obs.dropna(subset=["mjd"]).reset_index(drop=True)


def load_shifts(config, paths):
    """Per-epoch (east, north) Gaia shift that step 5 SUBTRACTED from positions ({} if it was not applied)."""
    if not config.get("filtering", {}).get("apply_epoch_shift", True):
        return {}
    f = paths.get("epoch_astrometry_csv") or os.path.join(paths["catalog_dir"], "epoch_astrometry.csv")
    if not os.path.exists(f):
        return {}
    e = pd.read_csv(f)
    return {int(r.epoch_idx): (float(r.shift_east_arcsec), float(r.shift_north_arcsec))
            for r in e.itertuples() if np.isfinite(r.shift_east_arcsec) and np.isfinite(r.shift_north_arcsec)}


# ---------------------------------------------------------------- static sources
STATIC_COLS = ["object_id", "file", "epoch_idx", "mjd", "detector", "wavelength_um", "bandwidth_um",
               "x_pix", "y_pix", "aperture_pix", "flux_jy", "err_jy", "snr", "bg_mjysr", "n_ap_pix",
               "bad_frac", "flag", "flag_bits", "ap_corr", "method", "psf_chi2r", "ap_flux_jy", "ap_err_jy"]


def measure_static(stat, obs, shifts, paths, info, cfg, t_start):
    """Forced photometry of every static object in every exposure; each FITS file is opened once."""
    ex, ny = helpers.tangent_offsets(stat["ra"].values, stat["dec"].values, info["ra"], info["dec"])
    oids = stat["object_id"].values
    rows, t_ph, status = [], time.time(), "complete"
    for i, o in enumerate(obs.itertuples(), 1):
        if time.time() - t_start > float(cfg["max_runtime_s"]):
            status = "stopped at photometry.max_runtime_s"
            log.warning("Time budget reached after %d/%d exposures - keeping what is done", i - 1, len(obs))
            break
        try:
            c = read_cube(os.path.join(paths["images_dir"], o.file))
        except Exception as e:  # noqa: BLE001
            log.warning("%s skipped: %s: %s", o.file, type(e).__name__, e)
            continue
        sx, sy = shifts.get(int(o.epoch_idx), (0.0, 0.0))
        ra, dec = helpers.offsets_to_radec(ex + sx, ny + sy, info["ra"], info["dec"])  # raw frame position
        xs, ys = c["wcs"].world_to_pixel_values(ra, dec)
        xs, ys = np.atleast_1d(xs).astype(float), np.atleast_1d(ys).astype(float)
        nyi, nxi = c["img"].shape
        det_no = detector_of(getattr(o, "bandpass", ""), None)
        for k in np.where((xs >= -0.5) & (xs <= nxi - 0.5) & (ys >= -0.5) & (ys <= nyi - 0.5))[0]:
            x, y = float(xs[k]), float(ys[k])
            ph = aperture_photometry(c, x, y, cfg["aperture_radius_pix"], cfg, use_ee=True)
            if cfg.get("psf_photometry", True) and np.isfinite(ph["flux_jy"]) and ph["flag"] in ("", "bad_pixels"):
                pf = psf_photometry(c, x, y, cfg)  # static source: PSF fit is the best estimator
                if pf is not None and np.isfinite(pf["flux_jy"]) and pf["err_jy"] > 0:
                    ph["ap_flux_jy"], ph["ap_err_jy"] = ph["flux_jy"], ph["err_jy"]
                    ph.update(flux_jy=pf["flux_jy"], err_jy=pf["err_jy"], snr=pf["snr"],
                              psf_chi2r=pf["psf_chi2r"], method="psf")
            wl, dwl = wavelength_at(c["swcs"], x, y)
            if not np.isfinite(wl) and det_no in BANDS:  # fall back to the band centre
                wl = 0.5 * (BANDS[det_no][0] + BANDS[det_no][1])
                ph["flag"] = (ph["flag"] + "+band_centre_wavelength").strip("+")
            rows.append({"object_id": oids[k], "file": o.file, "epoch_idx": int(o.epoch_idx), "mjd": o.mjd,
                         "detector": det_no, "wavelength_um": wl, "bandwidth_um": dwl, "x_pix": x, "y_pix": y,
                         "aperture_pix": cfg["aperture_radius_pix"], **ph})
        if i % max(len(obs) // 10, 1) == 0 or i == len(obs):
            el = time.time() - t_ph
            log.info("static photometry %d/%d exposures, %d measurements, %.1f s elapsed, ETA %.1f s",
                     i, len(obs), len(rows), el, el / i * (len(obs) - i))
    lc = pd.DataFrame(rows, columns=STATIC_COLS)
    lc["flag"] = lc["flag"].fillna("").astype(str)
    lc["good"] = (np.isfinite(lc["flux_jy"]) & (lc["err_jy"] > 0) & np.isfinite(lc["wavelength_um"])
                  & ~lc["flag"].str.contains("outside|no_background|bad_pixels", regex=True))
    return lc.sort_values(["object_id", "mjd"]).reset_index(drop=True), status


def extract_static_spectra(config, paths, cfg, info, obs, shifts):
    """Spectra of the unlabeled static objects of step 5. Returns a short status string."""
    t_start = time.time()
    stat_csv = paths.get("unlabeled_static_csv") or os.path.join(paths["catalog_dir"],
                                                                 "4_unlabeled_static_objects.csv")
    stat = pd.read_csv(stat_csv) if os.path.exists(stat_csv) else pd.DataFrame()
    if stat.empty or "ra" not in stat:
        log.info("No static unlabeled objects (%s) - no static spectra to extract.", stat_csv)
        return "no static objects"
    stat = stat.sort_values("mean_snr", ascending=False)
    max_obj = config.get("static_classification", {}).get("max_objects")
    if max_obj:
        stat = stat.head(int(max_obj))
    stat = stat.reset_index(drop=True)
    log.info("Extracting %d static unlabeled object(s) from %d exposures", len(stat), len(obs))

    lc, status = measure_static(stat, obs, shifts, paths, info, cfg, t_start)
    lc.to_csv(paths["static_spectrum_csv"], index=False)
    log.info("static spectral points: %d total, %d good -> %s", len(lc), int(lc["good"].sum()),
             paths["static_spectrum_csv"])

    t_sp, n_plot, n_spec = time.time(), 0, 0
    for oid in stat["object_id"]:
        pts = lc[(lc["object_id"] == oid) & lc["good"]]
        if pts.empty:
            continue
        binned = bin_spectrum(pts, cfg["outlier_sigma"])
        binned.to_csv(os.path.join(paths["static_spectra_dir"], f"{oid}_spectrum.csv"), index=False)
        n_spec += 1
        if _plot_ok(n_plot, cfg):
            plot_spectrum(f"Static object {oid}: {len(pts)} exposures, {len(binned)} bins", pts, binned,
                          os.path.join(paths["static_spectra_dir"], f"{oid}_spectrum.png"))
            n_plot += 1
    log.info("[time] static binning + %d PNG plots (%d spectra): %.1f s -> %s", n_plot, n_spec,
             time.time() - t_sp, paths["static_spectra_dir"])
    return status


# ---------------------------------------------------------------- moving objects (tracklets)
def extract_tracklet_spectra(config, paths, cfg, info, obs, shifts):
    """Spectra of the step-6 tracklets (movers). Returns a short status string."""
    t_start = time.time()
    trk = pd.read_csv(paths["tracklets_csv"])
    mem = pd.read_csv(paths["members_csv"])
    if trk.empty:
        log.info("No tracklets - no mover spectra to extract.")
        return "no tracklets"
    if cfg["only_candidates"] and os.path.exists(paths["candidates_csv"]):
        keep = pd.read_csv(paths["candidates_csv"])["tracklet_id"]
        trk = trk[trk["tracklet_id"].isin(keep)]
    trk = trk.sort_values("n_det", ascending=False)
    if cfg["max_tracklets"]:
        trk = trk.head(int(cfg["max_tracklets"]))
    log.info("Extracting %d tracklet(s)", len(trk))

    # ---- job list: file -> measurements needed
    from step6_link_tracklets import split_visits
    obs = obs.reset_index(drop=True).copy()
    vid = np.zeros(len(obs), int)
    for gi, g in enumerate(split_visits(obs["mjd"].values, config["tracklets"]["visit_gap_days"])):
        vid[g] = gi
    obs["visit"] = vid
    known = {}
    if os.path.exists(paths["checked_csv"]):
        k = pd.read_csv(paths["checked_csv"])
        if "known_name" in k:
            known = {int(r.tracklet_id): str(r.known_name) for r in k.itertuples()
                     if getattr(r, "known_status", "") == "known" and isinstance(r.known_name, str) and r.known_name}
    half = config["download"]["cutout_size_deg"] * 3600.0 / 2.0
    jobs, fe = {}, dict(zip(obs["file"], obs["epoch_idx"]))
    pad = float(cfg["forced_pad_days"])
    n_src = {"detection": 0, "ephemeris": 0, "visit_model": 0}
    for tr in trk.itertuples():
        m = mem[mem["tracklet_id"] == tr.tracklet_id]
        det_files = set(m["file"])
        for r in m.itertuples():  # every detection = its own measurement at its centroid
            jobs.setdefault(r.file, []).append(dict(tid=tr.tracklet_id, mjd=r.mjd, detected=True, src="detection",
                                                    det_id=int(r.det_id), x=r.x_pix, y=r.y_pix,
                                                    rate=tr.rate_arcsec_hr))
            n_src["detection"] += 1
        eph = None
        name = known.get(int(tr.tracklet_id))
        if name and cfg["use_ephemeris_for_known"]:
            eph = ephemeris_radec(name, obs["mjd"].values)
            if eph is not None:
                ex_all, ny_all = helpers.tangent_offsets(eph[0], eph[1], info["ra"], info["dec"])
                dm = m.set_index("file")
                seps = []
                for o, ex_, ny_ in zip(obs.itertuples(), ex_all, ny_all):
                    if o.file in dm.index:  # validation: detection centroid vs ephemeris
                        dx, dy = helpers.tangent_offsets(dm.loc[o.file, "ra"] if np.isscalar(dm.loc[o.file, "ra"])
                                                         else dm.loc[o.file, "ra"].iloc[0],
                                                         dm.loc[o.file, "dec"] if np.isscalar(dm.loc[o.file, "dec"])
                                                         else dm.loc[o.file, "dec"].iloc[0], info["ra"], info["dec"])
                        seps.append(float(np.hypot(dx - ex_, dy - ny_)))
                        continue
                    if abs(ex_) > half * 1.05 or abs(ny_) > half * 1.05:
                        continue  # the body is outside this cutout
                    jobs.setdefault(o.file, []).append(dict(
                        tid=tr.tracklet_id, mjd=o.mjd, detected=False, src="ephemeris", det_id=-1, x=None, y=None,
                        fit=(float(ex_), float(ny_)), rate=tr.rate_arcsec_hr,
                        shift=shifts.get(int(fe[o.file]), (0.0, 0.0))))
                    n_src["ephemeris"] += 1
                if seps:
                    log.info("tracklet %d (%s): median |detection - ephemeris| = %.1f arcsec (%d detections)",
                             tr.tracklet_id, name, float(np.median(seps)), len(seps))
        if eph is None:  # unknown tracklet (or no ephemeris): linear motion model
            if cfg["scope"] == "visit":
                g = obs.loc[(obs["mjd"] - tr.mjd_mid).abs().idxmin(), "visit"]
                span = obs[obs["visit"] == g]
            else:
                span = obs[(obs["mjd"] >= tr.mjd_start - pad) & (obs["mjd"] <= tr.mjd_end + pad)]
            for o in span.itertuples():  # other exposures of the visit: forced position from the motion model
                if o.file in det_files:
                    continue
                th = (o.mjd - tr.mjd_ref) * 24.0
                jobs.setdefault(o.file, []).append(dict(
                    tid=tr.tracklet_id, mjd=o.mjd, detected=False, src="visit_model", det_id=-1, x=None, y=None,
                    fit=(tr.p0x_arcsec + tr.vx_arcsec_hr * th, tr.p0y_arcsec + tr.vy_arcsec_hr * th),
                    rate=tr.rate_arcsec_hr, shift=shifts.get(int(fe[o.file]), (0.0, 0.0))))
                n_src["visit_model"] += 1
    log.info("measurements planned: %s", n_src)
    n_jobs = sum(len(v) for v in jobs.values())
    log.info("%d exposures to open, %d measurements", len(jobs), n_jobs)

    # ---- photometry, one file open per exposure
    rows, t_ph, done = [], time.time(), 0
    status = "complete"
    for i, (fname, items) in enumerate(jobs.items(), 1):
        if time.time() - t_start > float(cfg["max_runtime_s"]):
            status = "stopped at photometry.max_runtime_s"
            log.warning("Time budget reached after %d/%d exposures - writing what is done", i - 1, len(jobs))
            break
        try:
            c = read_cube(os.path.join(paths["images_dir"], fname))
        except Exception as ex:  # noqa: BLE001
            log.warning("%s skipped: %s: %s", fname, type(ex).__name__, ex)
            continue
        det_no = detector_of(obs.loc[obs["file"] == fname, "bandpass"].iloc[0] if (obs["file"] == fname).any()
                             else "", None)
        exp_s = float(c["hdr"].get("EXPTIME", cfg["exposure_s"]) or cfg["exposure_s"])
        for it in items:
            if it["x"] is None:  # forced position: undo the step-5 epoch shift, then world -> pixel
                ex_, ny_ = it["fit"][0] + it["shift"][0], it["fit"][1] + it["shift"][1]
                ra, dec = helpers.offsets_to_radec(ex_, ny_, info["ra"], info["dec"])
                x, y = c["wcs"].world_to_pixel_values(ra, dec)
                x, y = float(x), float(y)
            else:
                x, y = float(it["x"]), float(it["y"])
            r = cfg["aperture_radius_pix"]
            if cfg["trail_correction"]:
                trail = it["rate"] * exp_s / 3600.0 / c["scale"]
                r = min(r + 0.5 * trail, cfg["max_aperture_pix"])
            ph = aperture_photometry(c, x, y, r, cfg, use_ee=(r <= cfg["aperture_radius_pix"] + 1e-9))  # EE only if un-trailed
            wl, dwl = wavelength_at(c["swcs"], x, y)
            if not np.isfinite(wl) and det_no in BANDS:  # fall back to the band centre
                wl = 0.5 * (BANDS[det_no][0] + BANDS[det_no][1])
                ph["flag"] = (ph["flag"] + "+band_centre_wavelength").strip("+")
            ra_o, dec_o = c["wcs"].pixel_to_world_values(x, y)
            rows.append({"tracklet_id": it["tid"], "file": fname, "epoch_idx": fe.get(fname, -1),
                         "mjd": it["mjd"], "detector": det_no, "wavelength_um": wl, "bandwidth_um": dwl,
                         "x_pix": x, "y_pix": y, "ra": float(ra_o), "dec": float(dec_o),
                         "detected": it["detected"], "source": it["src"], "det_id": it["det_id"], "aperture_pix": r, **ph})
            done += 1
        if i % max(len(jobs) // 10, 1) == 0 or i == len(jobs):
            el = time.time() - t_ph
            log.info("photometry %d/%d exposures, %d measurements, %.1f s elapsed, ETA %.1f s",
                     i, len(jobs), done, el, el / i * (len(jobs) - i))
    log.info("[time] photometry: %.1f s", time.time() - t_ph)

    cols = ["tracklet_id", "file", "epoch_idx", "mjd", "detector", "wavelength_um", "bandwidth_um", "x_pix",
            "y_pix", "ra", "dec", "detected", "source", "det_id", "aperture_pix", "flux_jy", "err_jy", "snr",
            "bg_mjysr", "n_ap_pix", "bad_frac", "flag", "flag_bits"]
    lc = pd.DataFrame(rows, columns=cols).sort_values(["tracklet_id", "mjd"]).reset_index(drop=True)
    good = (np.isfinite(lc["flux_jy"]) & (lc["err_jy"] > 0) & np.isfinite(lc["wavelength_um"])
            & ~lc["flag"].str.contains("outside|no_background|bad_pixels", regex=True))
    lc["good"] = good
    lc.to_csv(paths["spectrum_csv"], index=False)
    log.info("spectral points: %d total, %d good -> %s", len(lc), int(good.sum()), paths["spectrum_csv"])

    # ---- binned spectra and plots
    t_sp = time.time()
    n_plot = 0
    for tr in trk.itertuples():
        pts = lc[(lc["tracklet_id"] == tr.tracklet_id) & lc["good"]]
        if pts.empty:
            continue
        binned = bin_spectrum(pts, cfg["outlier_sigma"])
        binned.to_csv(os.path.join(paths["spectra_dir"], f"tracklet_{tr.tracklet_id:04d}_spectrum.csv"),
                      index=False)
        if _plot_ok(n_plot, cfg):
            plot_tracklet(tr.tracklet_id, tr._asdict(), pts, binned,
                          os.path.join(paths["spectra_dir"], f"tracklet_{tr.tracklet_id:04d}_spectrum.png"))
            n_plot += 1
    log.info("[time] binning + %d PNG plots: %.1f s -> %s", n_plot, time.time() - t_sp, paths["spectra_dir"])
    return status


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    config = helpers.load_config()
    helpers.require_flag(config, "tracklets_linked", "step6_link_tracklets.py")
    paths = helpers.get_paths(config)
    cfg = {**DEFAULTS, **config.get("photometry", {})}
    info = config["info_dict"]
    os.makedirs(paths["spectra_dir"], exist_ok=True)
    os.makedirs(paths["static_spectra_dir"], exist_ok=True)

    t0 = time.time()
    obs = exposure_table(paths)
    shifts = load_shifts(config, paths)
    log.info("[time] exposure table: %.1f s (%d exposures, %d with a Gaia epoch shift)",
             time.time() - t0, len(obs), len(shifts))

    st_trk = extract_tracklet_spectra(config, paths, cfg, info, obs, shifts)
    st_static = extract_static_spectra(config, paths, cfg, info, obs, shifts)
    log.info("Step 8 done in %.1f s (tracklets: %s; static objects: %s)", time.time() - t_start, st_trk, st_static)
    helpers.set_flag("spectra_extracted")


if __name__ == "__main__":
    main()