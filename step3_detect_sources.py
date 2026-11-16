"""Step 3 - detect sources in every epoch and convert pixel positions to RA/Dec.

Pure-Python (photutils): 2-D background -> Gaussian-smoothed threshold detection ->
segmentation catalog -> astrometry from each cutout's own WCS.
"""
import os

import numpy as np
import pandas as pd

import helpers

log = helpers.get_logger("step3")

COLUMNS = ["det_id", "epoch_idx", "file", "mjd", "x", "y", "ra", "dec",
           "flux", "snr", "fwhm", "elongation", "peak", "npix", "edge_dist"]


def _attr(cat, *names):
    """Read a SourceCatalog attribute under its new or old photutils name."""
    for n in names:
        if hasattr(cat, n):
            return np.asarray(getattr(cat, n), float)
    raise AttributeError(names)


def detect_image(path, cfg):
    from astropy.convolution import convolve
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import SourceCatalog, detect_sources
    from photutils.segmentation import make_2dgaussian_kernel

    data, var, header, wcs = helpers.read_image(path)
    mask = ~np.isfinite(data)
    if var is not None:
        mask |= ~np.isfinite(var) | (var <= 0)
    if mask.all():
        return None, data.shape, None, header

    n_valid = int((~mask).sum())
    if n_valid < 100:
        return None, data.shape, None, header
    try:
        box = int(min(cfg["box_size_pix"], min(data.shape) // 2))
        # exclude_percentile=100: accept boxes that contain ANY valid pixels (real cutouts
        # often have many NaN / non-finite pixels, which made every box get rejected).
        bkg = Background2D(np.where(mask, np.nan, data), (box, box), filter_size=(3, 3),
                           mask=mask, bkg_estimator=MedianBackground(), exclude_percentile=100.0)
        background, rms = bkg.background, bkg.background_rms
    except Exception as e:  # noqa: BLE001 - fall back to one global sigma-clipped level
        from astropy.stats import sigma_clipped_stats
        _, med, std = sigma_clipped_stats(data[~mask], sigma=3.0)
        log.debug("Background2D failed (%s); using global background", e)
        background, rms = np.full(data.shape, med), np.full(data.shape, std)
    if not np.isfinite(rms[~mask]).all() or np.nanmin(rms[~mask]) <= 0:
        return None, data.shape, None, header
    sub = np.where(mask, 0.0, data - background)

    kernel = make_2dgaussian_kernel(cfg["fwhm_pix"], size=5)
    conv = convolve(sub, kernel, normalize_kernel=True)
    segm = detect_sources(conv, cfg["detect_sigma"] * rms, int(cfg["min_pixels"]), mask=mask)
    if segm is None:
        return pd.DataFrame(columns=COLUMNS[3:]), data.shape, wcs, header

    try:  # split blended neighbours so a mover next to a star is still detected
        from photutils.segmentation import deblend_sources
        try:  # photutils >= 3.0 renamed nlevels -> n_levels and dropped progress_bar
            segm = deblend_sources(conv, segm, int(cfg["min_pixels"]), n_levels=32, contrast=0.005)
        except TypeError:
            segm = deblend_sources(conv, segm, int(cfg["min_pixels"]), nlevels=32,
                                   contrast=0.005, progress_bar=False)
    except Exception as e:  # noqa: BLE001
        log.debug("deblending skipped: %s", e)

    cat = SourceCatalog(sub, segm, convolved_data=conv, error=rms, mask=mask)
    x, y = _attr(cat, "x_centroid", "xcentroid"), _attr(cat, "y_centroid", "ycentroid")
    a, b = _attr(cat, "semimajor_axis", "semimajor_sigma"), _attr(cat, "semiminor_axis", "semiminor_sigma")
    flux, ferr = _attr(cat, "segment_flux"), _attr(cat, "segment_flux_err", "segment_fluxerr")
    ny, nx = data.shape
    edge = np.minimum.reduce([x, y, nx - 1 - x, ny - 1 - y])
    ra, dec = wcs.pixel_to_world_values(x, y)
    df = pd.DataFrame({
        "x": x, "y": y, "ra": ra, "dec": dec, "flux": flux,
        "snr": np.where(ferr > 0, flux / ferr, np.nan),
        "fwhm": 2.3548 * np.sqrt((a ** 2 + b ** 2) / 2.0),
        "elongation": a / np.maximum(b, 1e-6),
        "peak": _attr(cat, "max_value"),
        "npix": _attr(cat, "area"),
        "edge_dist": edge,
    })
    return df.dropna(subset=["x", "y", "ra", "dec"]), data.shape, wcs, header


def main():
    config = helpers.load_config()
    helpers.require_flag(config, "downloaded", "step2_download_obs.py")
    paths = helpers.get_paths(config)
    obs = pd.read_csv(paths["observations_csv"])
    obs = obs[obs["status"].isin(["exists", "downloaded"])].reset_index(drop=True)

    frames = []
    for i, row in obs.iterrows():
        fpath = os.path.join(paths["images_dir"], row["file"])
        try:
            df, _, _, header = detect_image(fpath, config["detection"])
        except Exception as e:  # noqa: BLE001
            log.warning("[%d/%d] %s skipped: %s: %s", i + 1, len(obs), row["file"], type(e).__name__, e)
            continue
        if df is None:
            log.warning("[%d/%d] %s has no valid pixels", i + 1, len(obs), row["file"])
            continue
        mjd = helpers.header_mjd(header)
        if mjd is None:
            mjd = float(row["mjd_start"])
        df.insert(0, "mjd", mjd)
        df.insert(0, "file", row["file"])
        df.insert(0, "epoch_idx", i)
        frames.append(df)
        log.info("[%d/%d] %s: %d sources (MJD %.5f)", i + 1, len(obs), row["file"], len(df), mjd)

    if not frames:
        raise SystemExit("[!] No detections in any image.")
    det = pd.concat(frames, ignore_index=True)
    det.insert(0, "det_id", np.arange(len(det)))
    det[COLUMNS].to_csv(paths["detections_csv"], index=False)
    log.info("Total detections: %d -> %s", len(det), paths["detections_csv"])
    helpers.set_flag("detected")


if __name__ == "__main__":
    main()