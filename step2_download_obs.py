#!/usr/bin/env python3
"""
Step 2 - SPHEREx exposure downloader.

Features
--------
1. Reuses cached cutout_urls.csv when available.
2. Queries IRSA TAP only when no URL cache exists.
3. Uses one worker to reduce server load.
4. Streams FITS downloads directly to disk.
5. Retries incomplete downloads, timeouts and HTTP 503.
6. Validates FITS files before accepting them.
7. Preserves previously completed FITS files.
8. Saves observations.csv after every completed exposure.
9. Reports coverage separately for each SPHEREx band.

Important:
IRSA's cutout endpoint currently ignores HTTP Range requests.
Interrupted downloads must therefore restart from byte zero.
"""

import concurrent.futures
import hashlib
import io
import os
import random
import re
import time
import urllib.error
import urllib.request

import pandas as pd
from astropy.io import fits

import helpers


log = helpers.get_logger("step2")

OK = ("exists", "downloaded")

PERMANENT_HTTP_ERRORS = (400, 401, 403, 404, 410)

CHUNK_SIZE = 1024 * 1024

IRSA_BASE_URL = os.environ.get("IRSA_BASE_URL", "https://irsa.ipac.caltech.edu/")


# ============================================================
# QUERY
# ============================================================

def build_query(
    ra,
    dec,
    size_deg,
    bandpasses,
    mjd_min,
    mjd_max
):

    where = [
        f"1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)"
    ]

    if bandpasses:

        names = ", ".join(
            f"'{b}'" for b in bandpasses
        )

        where.append(
            f"p.energy_bandpassname IN ({names})"
        )

    if mjd_min is not None:

        where.append(
            f"p.time_bounds_lower >= {float(mjd_min)}"
        )

    if mjd_max is not None:

        where.append(
            f"p.time_bounds_lower <= {float(mjd_max)}"
        )

    return f"""
SELECT
    '{IRSA_BASE_URL}' || a.uri ||
    '?center={ra},{dec}d&size={size_deg}' AS uri,
    p.time_bounds_lower,
    p.energy_bandpassname
FROM spherex.artifact a
JOIN spherex.plane p
    ON a.planeid = p.planeid
WHERE {' AND '.join(where)}
ORDER BY p.time_bounds_lower
"""


# ============================================================
# BAND IDENTIFICATION
# ============================================================

def infer_band(uri):

    """
    Extract D1-D6 from SPHEREx artifact filename.

    Example:
        ..._1D6_spx_l2b...
        -> D6
    """

    match = re.search(
        r"_\d?D([1-6])_",
        str(uri)
    )

    if match:
        return f"D{match.group(1)}"

    return "unknown"


# ============================================================
# FITS VALIDATION
# ============================================================

def validate_fits(path):

    try:

        with fits.open(
            path,
            memmap=False,
            lazy_load_hdus=False
        ) as hdul:

            hdul.verify("exception")

            if len(hdul) == 0:
                return False

        return True

    except Exception:

        return False


# ============================================================
# URL CACHE
# ============================================================

def load_or_query_urls(
    config,
    paths
):

    from astroquery.ipac.irsa import Irsa

    info = config["info_dict"]
    dl = config["download"]

    catalog_dir = os.path.dirname(
        paths["observations_csv"]
    )

    os.makedirs(
        catalog_dir,
        exist_ok=True
    )

    urls_csv = os.path.join(
        catalog_dir,
        "cutout_urls.csv"
    )

    # --------------------------------------------------------
    # Reuse existing URL cache
    # --------------------------------------------------------

    if os.path.exists(urls_csv):

        log.info(
            "Found existing URL cache: %s",
            urls_csv
        )

        cached = pd.read_csv(urls_csv)

        if (
            not cached.empty
            and {"file", "uri"}.issubset(cached.columns)
        ):

            cached = cached[
                ["file", "uri"]
            ].copy()

            cached["mjd_start"] = float("nan")

            cached["bandpass"] = cached["uri"].apply(
                infer_band
            )

            log.info(
                "Reusing %d cached exposure URLs. "
                "Skipping IRSA TAP query.",
                len(cached)
            )

            return cached

        log.warning(
            "URL cache is invalid. Re-querying IRSA."
        )

    # --------------------------------------------------------
    # Query IRSA if no usable cache exists
    # --------------------------------------------------------

    query = build_query(
        info["ra"],
        info["dec"],
        dl["cutout_size_deg"],
        dl["bandpasses"],
        dl["mjd_min"],
        dl["mjd_max"]
    )

    log.info(
        "Querying IRSA TAP for SPHEREx exposures..."
    )

    start = time.time()

    results = (
        Irsa.query_tap(query)
        .to_table()
        .to_pandas()
    )

    log.info(
        "Found %d exposures in %.1f seconds",
        len(results),
        time.time() - start
    )

    if results.empty:

        raise SystemExit(
            "No SPHEREx exposures found."
        )

    results = results.rename(
        columns={
            "time_bounds_lower": "mjd_start",
            "energy_bandpassname": "bandpass"
        }
    )

    results["file"] = [
        (
            f"cutout_{i:04d}_"
            f"{hashlib.md5(uri.encode()).hexdigest()[:8]}.fits"
        )
        for i, uri in enumerate(results["uri"])
    ]

    results["bandpass"] = (
        results["bandpass"]
        .fillna("unknown")
        .astype(str)
    )

    # Save URL cache immediately.
    results[
        ["file", "uri"]
    ].to_csv(
        urls_csv,
        index=False
    )

    log.info(
        "Saved URL cache: %s",
        urls_csv
    )

    return results[
        ["file", "uri", "mjd_start", "bandpass"]
    ].copy()


# ============================================================
# SINGLE DOWNLOAD
# ============================================================

def download_one(
    uri,
    out_file,
    timeout,
    max_retries=5
):

    filename = os.path.basename(out_file)

    os.makedirs(
        os.path.dirname(out_file),
        exist_ok=True
    )

    tmp = out_file + ".part"

    # --------------------------------------------------------
    # Reuse valid existing FITS
    # --------------------------------------------------------

    if os.path.exists(out_file):

        if validate_fits(out_file):
            return "exists"

        log.warning(
            "%s: existing FITS is invalid. Removing.",
            filename
        )

        os.remove(out_file)

    # Remove stale partial file.
    if os.path.exists(tmp):

        os.remove(tmp)

    # --------------------------------------------------------
    # Download with retries
    # --------------------------------------------------------

    for attempt in range(max_retries + 1):

        try:

            request = urllib.request.Request(
                uri,
                headers={
                    "User-Agent":
                        "Mozilla/5.0 SPHEREx-Research-Pipeline/2.0",
                    "Accept": "*/*",
                    "Connection": "close"
                }
            )

            bytes_written = 0

            with urllib.request.urlopen(
                request,
                timeout=timeout
            ) as response:

                with open(tmp, "wb") as f:

                    while True:

                        chunk = response.read(
                            CHUNK_SIZE
                        )

                        if not chunk:
                            break

                        f.write(chunk)

                        bytes_written += len(chunk)

                    f.flush()
                    os.fsync(f.fileno())

            if bytes_written == 0:

                raise IOError(
                    "IRSA returned an empty response"
                )

            # ------------------------------------------------
            # Validate completed FITS
            # ------------------------------------------------

            if not validate_fits(tmp):

                raise IOError(
                    "Downloaded file failed FITS validation"
                )

            # ------------------------------------------------
            # Atomic final write
            # ------------------------------------------------

            os.replace(
                tmp,
                out_file
            )

            return "downloaded"

        except Exception as e:

            if os.path.exists(tmp):

                try:
                    os.remove(tmp)

                except OSError:
                    pass

            if (
                isinstance(e, urllib.error.HTTPError)
                and e.code in PERMANENT_HTTP_ERRORS
            ):

                return (
                    f"failed: HTTP {e.code} "
                    "(permanent error)"
                )

            if attempt >= max_retries:

                return (
                    f"failed after {max_retries + 1} attempts: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )

            # ------------------------------------------------
            # Exponential backoff
            # ------------------------------------------------

            delay = min(
                15 * (2 ** attempt),
                180
            ) + random.uniform(0, 5)

            if isinstance(e, urllib.error.HTTPError):

                retry_after = e.headers.get(
                    "Retry-After"
                )

                if retry_after:

                    try:

                        delay = max(
                            delay,
                            float(retry_after)
                        )

                    except ValueError:
                        pass

            log.warning(
                "%s | attempt %d/%d failed: %s | "
                "retrying in %.1f seconds",
                filename,
                attempt + 1,
                max_retries + 1,
                str(e)[:150],
                delay
            )

            time.sleep(delay)

    return "failed: retry loop terminated unexpectedly"


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_observations(results, observations_csv):

    os.makedirs(
        os.path.dirname(observations_csv),
        exist_ok=True
    )

    results.to_csv(
        observations_csv,
        index=False
    )


# ============================================================
# BAND REPORT
# ============================================================

def band_report(results):

    log.info("")
    log.info("=" * 65)
    log.info("SPHEREx BAND COVERAGE")
    log.info("=" * 65)

    for band, group in results.groupby(
        "bandpass_key"
    ):

        n_ok = int(
            group["status"].isin(OK).sum()
        )

        total = len(group)

        fraction = (
            n_ok / total
            if total else 0
        )

        log.info(
            "%-12s %4d / %4d  (%5.1f%%)",
            band,
            n_ok,
            total,
            fraction * 100
        )

    log.info("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    config = helpers.load_config()

    helpers.require_flag(
        config,
        "configured",
        "step1_config.py"
    )

    paths = helpers.get_paths(config)

    dl = config["download"]

    images_dir = paths["images_dir"]
    observations_csv = paths["observations_csv"]

    os.makedirs(
        images_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Conservative download settings
    # --------------------------------------------------------

    workers = 1

    timeout = int(
        dl.get("timeout_s", 180)
    )

    max_retries = max(
        0,
        int(dl.get("max_retries", 2))
    )

    log.info("=" * 65)
    log.info("SPHEREx DOWNLOAD PIPELINE")
    log.info("=" * 65)

    log.info(
        "Workers: %d | Timeout: %ds | Retries: %d",
        workers,
        timeout,
        max_retries
    )

    # --------------------------------------------------------
    # Load cached URLs or query IRSA
    # --------------------------------------------------------

    results = load_or_query_urls(
        config,
        paths
    )

    if results.empty:

        raise SystemExit(
            "No exposure URLs available."
        )

    # Optional maximum image limit.
    if dl.get("max_images"):

        results = results.iloc[
            :int(dl["max_images"])
        ].copy()

    results = results.reset_index(
        drop=True
    )

    results["bandpass_key"] = (
        results["bandpass"]
        .fillna("unknown")
        .astype(str)
    )

    results["status"] = "pending"

    # --------------------------------------------------------
    # Check existing files before downloading
    # --------------------------------------------------------

    log.info("")
    log.info("CHECKING EXISTING FITS FILES")

    pending = []

    for row in results.itertuples():

        out_file = os.path.join(
            images_dir,
            row.file
        )

        if os.path.exists(out_file):

            if validate_fits(out_file):

                results.loc[
                    results["file"] == row.file,
                    "status"
                ] = "exists"

                continue

            log.warning(
                "Removing invalid existing file: %s",
                row.file
            )

            os.remove(out_file)

        pending.append(
            (row.file, row.uri)
        )

    # Save initial state before downloads.
    save_observations(
        results,
        observations_csv
    )

    n_existing = int(
        results["status"].isin(OK).sum()
    )

    log.info(
        "Already available: %d / %d",
        n_existing,
        len(results)
    )

    log.info(
        "Remaining downloads: %d",
        len(pending)
    )

    # --------------------------------------------------------
    # Download one exposure at a time
    # --------------------------------------------------------

    log.info("")
    log.info("STARTING DOWNLOADS")
    log.info("=" * 65)

    total = len(pending)

    for i, (filename, uri) in enumerate(
        pending,
        1
    ):

        log.info(
            "[%d/%d] Downloading %s",
            i,
            total,
            filename
        )

        out_file = os.path.join(
            images_dir,
            filename
        )

        status = download_one(
            uri,
            out_file,
            timeout,
            max_retries
        )

        results.loc[
            results["file"] == filename,
            "status"
        ] = status

        # Save after EVERY exposure.
        save_observations(
            results,
            observations_csv
        )

        log.info(
            "[%d/%d] %s: %s",
            i,
            total,
            filename,
            status
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    band_report(results)

    n_ok = int(
        results["status"].isin(OK).sum()
    )

    n_failed = len(results) - n_ok

    log.info("")
    log.info("=" * 65)
    log.info("DOWNLOAD SUMMARY")
    log.info("=" * 65)

    log.info(
        "Total exposures: %d",
        len(results)
    )

    log.info(
        "Usable FITS files: %d",
        n_ok
    )

    log.info(
        "Missing/failed: %d",
        n_failed
    )

    log.info(
        "Coverage: %.2f%%",
        100 * n_ok / len(results)
    )

    log.info(
        "Observations CSV: %s",
        observations_csv
    )

    log.info(
        "Images directory: %s",
        images_dir
    )

    log.info("=" * 65)

    # --------------------------------------------------------
    # Pipeline flag
    # --------------------------------------------------------

    if n_ok >= 3:

        helpers.set_flag(
            "downloaded",
            True
        )

        log.info(
            "Download stage completed with usable FITS files."
        )

    else:

        log.warning(
            "Fewer than 3 usable FITS files. "
            "Download stage is incomplete."
        )


if __name__ == "__main__":

    main()