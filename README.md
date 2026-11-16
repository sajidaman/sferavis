# Sferavis: SPHEREx Moving-Object Detection and Classification Pipeline

Since 2025, NASA’s SPHEREx mission has been mapping the entire sky every six months in 102 bands of near-infrared light, returning images of more than a billion objects. Comets, asteroids, stars, brown dwarfs, and even new planets reveal themselves by shifting position between images, but no one person can sift through all that data alone. Your challenge is to create a public-facing web tool to display images of the sky from SPHEREx mission, making it easy for anyone to quickly see how they change over time.

---

## Table of Contents

1. [Physical Background](#physical-background)
2. [Running the Pipeline](#running-the-pipeline)
3. [Step 1: step1_config.py](#step-1-step1_configpy)
4. [Step 2: step2_download_obs.py](#step-2-step2_download_obspy)
5. [Step 3: step3_detect_sources.py](#step-3-step3_detect_sourcespy)
6. [Step 4: step4_survey_match.py](#step-4-step4_survey_matchpy)
7. [Step 5: step5_list_unlabeled_obj.py](#step-5-step5_list_unlabeled_objpy)
8. [Step 6: step6_link_tracklets.py](#step-6-step6_link_trackletspy)
9. [Step 7: step7_check_results.py](#step-7-step7_check_resultspy)
10. [Step 8: step8_extract_spectra.py](#step-8-step8_extract_spectrapy)
11. [Step 9: step9_classify.py](#step-9-step9_classifypy)

---

## Physical Background

### Why Look for Moving Objects in SPHEREx?

SPHEREx is an all-sky near-infrared spectral survey (0.75 to 5 µm). Its images are not designed for Solar System science, but the same patch of sky is imaged repeatedly, in several exposures minutes apart (a *visit*), and again at other times. Anything in the Solar System moves against the fixed stars and galaxies during that time, so it shows up as a source that **changes position from exposure to exposure** while everything else stays put.

| Object type | Behaviour in repeated exposures |
|-------------|--------------------------------|
| Stars, galaxies | Fixed position (stars drift by milli-arcsec per year, see Gaia proper motion) |
| Main-belt asteroids | Move by tens of arcsec per hour |
| Near-Earth objects | Can move by hundreds to thousands of arcsec per hour |
| Image artifacts, noise peaks | Appear at one epoch only, or at random positions |

### What We Measure

| Observable | Physical Information | How We Get It |
|------------|---------------------|---------------|
| Source position (RA, Dec) per exposure | Where the object was at that MJD | Detection + WCS |
| Position change between exposures | Sky-plane motion rate and direction | Linear fit of position vs time |
| Rate of motion | Rough orbital class (main belt vs NEO) | Tracklet fit |
| Match to SkyBoT | Whether the object is already catalogued | Cone search at the tracklet position |
| Flux at a known wavelength | Low-resolution 0.75 to 5 µm spectrum | Forced aperture photometry in every exposure (step 8) |
| Spectral features | Rough object class (brown dwarf, quasar, supernova, galaxy, ...) | Spectral-index classifier (step 9) |

### The Goal of This Pipeline

Given a target position, collect every SPHEREx exposure that fully covers a small square around it, detect all sources, **remove everything that is stationary** (Gaia stars, repeated fixed sources), link the remaining detections into **straight-line tracklets**, and report the tracklets that are **not known** Solar System bodies as candidates. Finally, measure a **spectrum** for every tracklet and every unlabeled static source, and give each a rough **class**.

### Pipeline Overview

```
step1_config.py              target -> square, config.json, output folders
step2_download_obs.py        IRSA TAP query (4 corners) + cutout download + verification
step3_detect_sources.py      background, smoothing, segmentation, deblending, RA/Dec
step4_survey_match.py        Gaia DR3 (proper-motion corrected) and optional SIMBAD match
step5_list_unlabeled_obj.py  quality cuts, per-epoch Gaia astrometric calibration,
                             static vs moving classification
step6_link_tracklets.py      seed pairs -> constant-velocity tracklets
step7_check_results.py       SkyBoT known-object check, candidate ranking, plots
step8_extract_spectra.py     forced aperture photometry -> spectrum of every tracklet
                             and every unlabeled static source
step9_classify.py            spectral-index classifier -> class of every spectrum
```

---

## Running the Pipeline

**Files:** `run_pipeline.py`, `helpers.py`

```bash
python run_pipeline.py                                  # asks for RA, Dec, radius
python run_pipeline.py --ra 180 --dec 0 --radius 0.1    # no questions
python run_pipeline.py --ra 20h23m57.8s --dec +38d30m27s --radius 0.1
python run_pipeline.py --no-prompt                      # reuse config.json as is
```

`run_pipeline.py` calls `helpers.check_dependencies()`, runs `step1_config.main(...)`, then imports and runs steps 2 to 9 in order. Each step can also be run alone (`python step3_detect_sources.py`); each one refuses to start if the previous step's flag is not set.

### Requirements

| Package | Minimum version | Used for |
|---------|-----------------|----------|
| Python | 3.10 | whole pipeline |
| numpy | 1.24 | arrays |
| scipy | 1.10 | KD-trees, sparse graphs, PSF shifting and PCHIP curves (step 8) |
| pandas | 2.0 | catalogs / CSV |
| astropy | 6.0 | FITS, WCS, coordinates, time |
| photutils | 1.13 | background, detection, deblending (step 3) |
| astroquery | 0.4.7 | IRSA, Gaia, VizieR, SIMBAD, SkyBoT, Miriade (step 8) |
| pyvo | 1.5 | TAP client used by astroquery |
| matplotlib | 3.7 | candidate plots (step 7), spectrum plots (step 8) |

### Environment Variables

| Variable | Meaning |
|----------|---------|
| `SPHEREX_CONFIG` | Alternative path to `config.json` |
| `SPHEREX_OUTPUT_DIR` | Overrides `info_dict.output_root` (default `output`) |
| `IRSA_BASE_URL` | IRSA base URL (default `https://irsa.ipac.caltech.edu/`) |

All paths are resolved relative to the folder containing `helpers.py`, never the current directory. On the first run `config.json` is created from the tracked template `config.default.json`.

---

## Step 1: step1_config.py

**File:** `step1_config.py`
**Purpose:** Set the target, validate `config.json`, create the output folders, reset progress flags

### Configuration Parameters

#### `ra`, `dec` (float or sexagesimal text)
- **What it is:** The target position, which is the **center of the square** that will be analysed
- **Accepted forms:** decimal degrees (`180.0`, `-5.25`) or sexagesimal (`20h23m57.8s`, `+38d30m27s`). Plain numbers are read as degrees; text is read as RA in hours and Dec in degrees
- **Validation:** 0 ≤ RA < 360, -90 ≤ Dec ≤ +90
- **Stored as:** rounded to 6 decimals

#### `radius` → `radius_deg` and `download.cutout_size_deg` (float, degrees)
- **What it is:** The value typed as "search radius". It is used **directly as the full side length of the square cutout**, not as half of it
- **Example:** `0.1` gives a 0.1° × 0.1° square (360″ per side, about 58 SPHEREx pixels)
- **Validation:** must be > 0
- **Where it's used:** Step 2 (corner calculation, cutout request), Step 4 (Gaia cone radius), Step 5 (known-object field edge), Step 8 (ephemeris field edge)

#### `name` (string)
- **What it is:** Output folder name
- **Behaviour:** Left empty in `config.json`; the folder is auto-named `RA_{ra:.4f}_DEC_{dec:+.4f}` (for example `RA_180.0000_DEC_+0.0000`). A name that looks auto-generated is regenerated when the position changes

#### `output_root` (path)
- **Default:** `output`

### Folders Created

```
output/RA_180.0000_DEC_+0.0000/
├── images/          # FITS cutouts (step 2)
├── catalogs/        # detection and matching tables (steps 2-5, 8-9)
├── tracklets/       # tracklets, candidates, triage, spectral points, classification (steps 6-9)
├── plots/           # candidate diagnostic plots (step 7)
├── spectra/         # binned spectrum CSV + PNG per tracklet (step 8)
└── static_spectra/  # binned spectrum CSV + PNG per static object (step 8)
```

### Output: `config.json`

```json
{
    "info_dict": {
        "name": "",
        "ra": 180.0,
        "dec": 0.0,
        "output_root": "output",
        "radius_deg": 0.1
    },
    "download": { "cutout_size_deg": 0.1, "...": "..." },
    "paths": { "root": "output/RA_180.0000_DEC_+0.0000", "...": "..." },
    "flags": {
        "configured": true,
        "downloaded": false,
        "...": false
    }
}
```

### Processing Flags

All flags are reset to `false` every time step 1 runs, except `configured`, which is set to `true`.

| Flag | Set True When | Meaning |
|------|--------------|---------|
| `configured` | Step 1 complete | Target and folders defined |
| `downloaded` | Step 2 complete (at least 3 usable FITS files) | Cutouts on disk and verified |
| `detected` | Step 3 complete | Per-exposure source catalogs built |
| `gaia_matched` | Step 4 complete | Detections flagged against Gaia / SIMBAD |
| `static_filtered` | Step 5 complete | Static sources removed, moving detections isolated |
| `tracklets_linked` | Step 6 complete | Linear-motion tracklets built |
| `known_objects_checked` | Step 7 complete | Tracklets compared with SkyBoT |
| `spectra_extracted` | Step 8 complete | Spectra measured for tracklets and static objects |
| `classified` | Step 9 complete | Every spectrum has a predicted class |

---

## Step 2: step2_download_obs.py

**File:** `step2_download_obs.py`
**Purpose:** Find every SPHEREx exposure that fully contains the target square, download only that square, delete everything else

### Square Geometry

The target is the center of a square of side `s = cutout_size_deg`. With `h = s/2`:

```math
\alpha_{corner} = \alpha_0 \pm \frac{h}{\cos\delta_0}, \qquad \delta_{corner} = \delta_0 \pm h
```

| Corner | RA | Dec |
|--------|----|-----|
| 1 | α₀ - h/cos δ₀ | δ₀ - h |
| 2 | α₀ + h/cos δ₀ | δ₀ - h |
| 3 | α₀ + h/cos δ₀ | δ₀ + h |
| 4 | α₀ - h/cos δ₀ | δ₀ + h |

RA is wrapped modulo 360°. If |δ₀| + h ≥ 90° the step stops (the square would reach a pole).

**Example (RA 33, Dec 33, 1° square):** corners at RA 32.40382 and 33.59618, Dec +32.5 and +33.5.

### The Coverage Rule

An exposure is used only if its footprint polygon contains **all four corners**. This guarantees the cutout is a complete square with no missing edge.

IRSA's ADQL parser rejected one query with four `CONTAINS` clauses, so the step sends **four single-point queries** (the form IRSA accepts) and keeps only the exposures returned by all four:

```math
\mathcal{E}_{kept} = \mathcal{E}_{c_1} \cap \mathcal{E}_{c_2} \cap \mathcal{E}_{c_3} \cap \mathcal{E}_{c_4}
```

### IRSA TAP Query (one per corner)

```sql
SELECT '{IRSA_BASE_URL}' || a.uri || '?center={ra},{dec}d&size={size_deg}' AS uri,
       p.time_bounds_lower, p.energy_bandpassname
FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid
WHERE 1 = CONTAINS(POINT('ICRS', {corner_ra}, {corner_dec}), p.poly)
  AND p.energy_bandpassname IN ('SPHEREx-D1', ..., 'SPHEREx-D6')
  [AND p.time_bounds_lower >= {mjd_min}] [AND p.time_bounds_lower <= {mjd_max}]
ORDER BY p.time_bounds_lower
```

The cutout URL carries the **target center and the square size**, so IRSA returns only the square, not the full detector image.

### Download Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `download.bandpasses` | `SPHEREx-D1` to `SPHEREx-D6` | The six detector bands (roughly 0.75 to 5 µm in total) |
| `download.cutout_size_deg` | `0.1` (from radius) | Full side of the square |
| `download.mjd_min`, `mjd_max` | `null` | Optional time window on exposure start |
| `download.max_images` | `null` | Optional cap on how many of the selected exposures are downloaded |
| `download.timeout_s` | `180` | Seconds before a request times out |
| `download.max_retries` | `2` | Retries per file (3 attempts in total) |
| Workers | `1` (fixed in code) | One file at a time, to be gentle on IRSA |
| Chunk size | 1 MiB | Streamed to disk |
| Permanent HTTP errors | 400, 401, 403, 404, 410 | Not retried |

**Retry back-off** (also honours `Retry-After`):

```math
t_{wait} = \min\left(15 \cdot 2^{attempt},\ 180\right) + U(0, 5)\ \mathrm{s}
```

IRSA's cutout endpoint ignores HTTP Range requests, so an interrupted download restarts from byte zero.

### File Naming

```
cutout_{index:04d}_{md5(uri)[:8]}.fits
```

### URL Cache

`catalogs/cutout_urls.csv` stores `file, uri, mjd_start, bandpass`. `catalogs/cutout_urls.meta.json` stores what the selection depended on:

```json
{"ra": 180.0, "dec": 0.0, "size_deg": 0.1, "bandpasses": [...],
 "mjd_min": null, "mjd_max": null, "selection": "all_four_corners"}
```

The cache is reused **only if this metadata matches the current run**. A cache without metadata (from an older overlap-based version) or for a different target or size triggers a new IRSA query.

### Residual Deletion and Verification

1. **Before downloading:** every `cutout_*.fits` and `cutout_*.fits.part` in `images/` that is not in the current selection is deleted
2. **Existing files** are opened and validated (`hdul.verify("exception")`); invalid ones are deleted and re-downloaded
3. **After downloading:** each file is opened, the four corners are converted to pixel coordinates with the file's own WCS, and the file is kept only if all four land inside the image **on finite pixels**. Others are deleted and removed from `observations.csv`
4. `observations.csv` is saved after every exposure, so an interrupted run resumes without losing progress

### Output Files

| File | Contents |
|------|----------|
| `images/cutout_*.fits` | Verified square cutouts |
| `catalogs/observations.csv` | `file, uri, mjd_start, bandpass, bandpass_key, status` |
| `catalogs/cutout_urls.csv` | URL cache |
| `catalogs/cutout_urls.meta.json` | Cache validity key |

The step prints a per-band coverage table and sets `downloaded` only if at least **3** usable files exist.

---

## Step 3: step3_detect_sources.py

**File:** `step3_detect_sources.py`
**Purpose:** Find sources in every exposure and convert pixel centroids to RA/Dec

### Reading an Image

`helpers.read_image` opens the `IMAGE` extension (or the first 2-D HDU), the `VARIANCE` extension if present, and the celestial WCS of the cutout.

**Pixel mask:** a pixel is masked if the data is not finite, or the variance is not finite or ≤ 0. An image with no valid pixels, or fewer than **100** valid pixels, is skipped.

### Detection Parameters

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| `detect_sigma` | `3.0` | Detection threshold in units of the local background RMS |
| `min_pixels` | `3` | Minimum connected pixels above threshold |
| `fwhm_pix` | `2.5` | FWHM of the Gaussian smoothing kernel in pixels |
| `box_size_pix` | `16` | Background mesh size (reduced to half the smaller image side if the cutout is small) |

### Algorithm

**1. Background and noise.** `photutils.Background2D` with a 3×3 median filter, `MedianBackground`, and `exclude_percentile=100` (accept boxes containing any valid pixel, because real cutouts contain many NaNs). If it fails, a single global 3σ-clipped median and standard deviation are used.

**2. Smoothing.** The background-subtracted image is convolved with a normalised 5×5 Gaussian kernel of FWHM 2.5 pixels (a matched filter for a point source at SPHEREx's pixel scale).

**3. Segmentation.** Pixels above `detect_sigma × background_rms` that form groups of at least `min_pixels` pixels:

```math
I_{smooth}(x,y) > 3.0 \times \sigma_{bkg}(x,y)
```

**4. Deblending.** Touching sources are split with `deblend_sources` (32 levels, contrast 0.005), so a mover next to a star is still detected.

**5. Measurements** (`photutils.SourceCatalog`):

| Column | Definition |
|--------|-----------|
| `x`, `y` | Centroid in pixels (0-indexed) |
| `ra`, `dec` | `wcs.pixel_to_world_values(x, y)` from the cutout's own WCS |
| `flux` | Segment flux (sum over the segment, background-subtracted) |
| `snr` | `flux / flux_err` |
| `fwhm` | Gaussian-equivalent FWHM, see below |
| `elongation` | `a / b` (semi-major over semi-minor sigma) |
| `peak` | Maximum pixel value in the segment |
| `npix` | Segment area in pixels |
| `edge_dist` | Distance to the nearest image edge in pixels |

```math
\mathrm{FWHM} = 2.3548 \sqrt{\frac{a^2 + b^2}{2}}, \qquad \mathrm{edge\_dist} = \min(x,\ y,\ n_x - 1 - x,\ n_y - 1 - y)
```

**Timing:** `mjd` is read from `MJD-AVG`, else `MJD-OBS`, else parsed from `DATE-AVG`, `DATE-OBS` or `DATE`; if none exist, `mjd_start` from `observations.csv` is used. Each exposure gets an `epoch_idx`.

### Output Files

| File | Contents |
|------|----------|
| `catalogs/1_detections.csv` | `det_id, epoch_idx, file, mjd, x, y, ra, dec, flux, snr, fwhm, elongation, peak, npix, edge_dist` |

---

## Step 4: step4_survey_match.py

**File:** `step4_survey_match.py`
**Purpose:** Flag detections that coincide with catalogued stationary sources

### Reference Catalog (Gaia DR3)

The Gaia cone covers the whole square with a safety margin:

```math
r_{cone} = 0.5 \times s \times \sqrt{2} \times 1.1 + \frac{30''}{3600}\ \mathrm{deg}
```

For `s = 0.1°` this is about 0.086°. Sources are tried in order: ESA Gaia archive (`gaiadr3.gaia_source`), then VizieR mirrors (`I/355/gaiadr3`) at `vizier.cds.unistra.fr`, `vizier.cfa.harvard.edu`, `vizier.nao.ac.jp`, `vizier.iucaa.in`. A source that returns 0 stars is skipped in favour of the next. Missing proper motions are set to 0. You can supply your own CSV (`ra, dec[, pmra, pmdec]`) through `gaia.catalog_file`.

### Proper-Motion Propagation

Gaia DR3 positions are at epoch J2016.0 (MJD 57388.0). For every exposure at time `MJD`:

```math
\Delta t = \frac{MJD - 57388.0}{365.25}\ \mathrm{yr}
```

```math
\alpha' = \alpha + \frac{\mu_{\alpha*}}{3.6\times10^{6}} \cdot \frac{\Delta t}{\cos\delta}, \qquad \delta' = \delta + \frac{\mu_\delta}{3.6\times10^{6}} \cdot \Delta t
```

(`μ` in mas/yr; the factor 3.6×10⁶ converts mas to degrees.) The `cos δ` is clipped to ≥ 10⁻⁶.

### Matching

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `gaia.enabled` | `true` | Turn Gaia matching on or off |
| `gaia.match_radius_arcsec` | `4.0` | Nearest-neighbour match radius, about 0.65 SPHEREx pixel |
| `gaia.catalog_file` | `null` | Own catalog instead of querying |
| `simbad.enabled` | `false` | SIMBAD matching is off by default |
| `simbad.match_radius_arcsec` | `4.0` | SIMBAD match radius (SIMBAD does not list asteroids) |

Each epoch is matched separately with `SkyCoord.match_to_catalog_sky` against Gaia propagated to that epoch.

### Output Files

| File | Contents |
|------|----------|
| `catalogs/gaia_catalog.csv` | `ra, dec, pmra, pmdec` of the reference stars |
| `catalogs/simbad_catalog.csv` | SIMBAD objects in the field (only if enabled) |
| `catalogs/2_detections_gaia.csv` | Step 3 table plus `gaia_match, gaia_sep_arcsec, simbad_match, simbad_id` |

---

## Step 5: step5_list_unlabeled_obj.py

**File:** `step5_list_unlabeled_obj.py`
**Purpose:** Keep only good detections, calibrate positions against Gaia, and separate stationary sources from possible movers

### 5.1 Quality Cuts

A detection is *good* if all of the following hold:

| Cut | Value | Removes |
|-----|-------|---------|
| `min_snr` | `5.0` | Faint noise peaks |
| `min_fwhm_pix` | `0.8` | Hot pixels and cosmic-ray-like spikes |
| `max_fwhm_pix` | `8.0` | Extended or blended structures |
| `max_elongation` | `3.0` | Streaks and trails |
| `edge_margin_pix` | `3` | Sources too close to the cutout edge |

Detections that fail are labelled `bad_quality`.

### 5.2 Position Error Budget

Each detection gets a one-sigma position error per axis, in arcsec:

```math
\sigma_{pos} = \sqrt{\sigma_{centroid}^2 + \sigma_{calib}^2 + \sigma_{Gaia}^2}
```

**Centroid error** (scale 1.0, floor 0.02 pixel):

```math
\sigma_{centroid} = \max\left(\frac{\mathrm{FWHM}/2.3548}{\mathrm{S/N}} \times p,\ 0.02\,p\right), \qquad p = \text{pixel scale in arcsec}
```

`p` is read from the first image's WCS (SPHEREx is about 6.15″/pixel; this is the fallback if it cannot be read).

**Per-epoch Gaia calibration.** For each epoch, good detections are matched (within `gaia.match_radius_arcsec`) to Gaia propagated to that epoch. If at least `min_ref_stars = 5` pairs exist:

```math
(\Delta_E, \Delta_N) = \mathrm{median}\left(\text{detected} - \text{Gaia}\right)
```

```math
\sigma_{ref} = \frac{\mathrm{median}\ \sqrt{(o_x - \Delta_E)^2 + (o_y - \Delta_N)^2}}{1.1774}
```

1.1774 is the median of a 2-D Rayleigh distribution in units of σ, so `σ_ref` is the per-axis scatter.

```math
\sigma_{calib} = \sqrt{\max\left(\sigma_{ref}^2 - \sigma_{own}^2,\ 0\right) + \frac{\sigma_{ref}^2}{n}}
```

where `σ_own` is the median centroid error of those reference stars (subtracted so it is not counted twice) and `n` is the number of reference stars.

- Epochs with fewer than 5 reference stars use the **median** `σ_calib` of the calibrated epochs; if no epoch is calibrated, `astrometric_floor_arcsec = 1.0″` is used
- The shift `(Δ_E, Δ_N)` is **subtracted from every detection** of that epoch (`apply_epoch_shift = true`). The original positions are kept as `ra_raw`, `dec_raw`
- `σ_Gaia = gaia_ref_err_arcsec = 0.002″`

### 5.3 Static vs Moving Classification

Detections in different epochs are "the same place" if

```math
d_{ij} \le \mathrm{clip}\left(n_\sigma \sqrt{\sigma_i^2 + \sigma_j^2},\ r_{floor},\ r_{cap}\right)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `static_n_sigma` | `3.0` | Match threshold in sigmas |
| `static_floor_arcsec` | `1.0` | Smallest allowed match radius |
| `static_max_radius_arcsec` | `max(static_radius_arcsec, 10) = 10` | Largest allowed match radius (`static_radius_arcsec = 3.0`) |
| `static_min_epochs` | `3` | Epochs a source must appear in |
| `static_min_span_days` | `1.0` | Time the source must persist |

Linked detections form chains found with connected components on a sparse graph (pairs found with a KD-tree within the cap radius). A chain is **static** if it appears in at least 3 different epochs **and** spans at least 1 day.

**Categories assigned to every detection:**

| Category | Meaning |
|----------|---------|
| `bad_quality` | Failed the quality cuts |
| `known` | Good detection matched to Gaia or SIMBAD |
| `known_star` | Unmatched detection that sits in a chain containing a Gaia/SIMBAD match (a catalogued star whose match failed in this epoch) |
| `static` | Unlabeled but stationary over at least 3 epochs and 1 day; given an ID `S0000`, `S0001`, ... |
| `moving` | Everything else: not shown to be stationary; step 6 decides whether it moves coherently |

**Static object summary:** inverse-variance weighted position `Σ(x/σ²)/Σ(1/σ²)`, error `max(1/√Σ(1/σ²), rms/√n)`, per-axis scatter `rms`, and reduced χ².

### 5.4 Known Solar System Bodies in the Field

When `known_in_field = true`, for every visit step 5 queries SkyBoT at the mean visit MJD, keeps bodies inside the square, predicts each body's position in every exposure from its sky rate, and looks for the nearest detection within `known_in_field_tol_arcsec = 8.0`. The category of that detection shows where the pipeline lost the body: `moving` (survived), `static`, `known`, `known_star`, `bad_quality` or `not_detected`. Bodies brighter than V = 19 are listed in the log. Network failure skips this check without stopping the run.

### Output Files

| File | Contents |
|------|----------|
| `catalogs/3_detections_all_flagged.csv` | Every detection with its category and corrected position |
| `catalogs/4_unlabeled_objects.csv` | All `static` and `moving` detections with `pos_err_arcsec` |
| `catalogs/4_unlabeled_static_objects.csv` | One row per static object (input of steps 8 and 9) |
| `catalogs/epoch_astrometry.csv` | `n_ref`, shift, `ref_rms_arcsec`, `err_calib_arcsec` per epoch (the shift is used again in step 8) |
| `catalogs/known_in_field.csv` | SkyBoT body-visits and what happened to each |
| `catalogs/3_detections_filtered.csv` | The `moving` detections, input to step 6 |

The step stops with an error if fewer than `tracklets.min_detections` moving detections remain.

---

## Step 6: step6_link_tracklets.py

**File:** `step6_link_tracklets.py`
**Purpose:** Link the `moving` detections into straight-line, constant-velocity tracklets

### Coordinates

Positions are converted to small-field tangent offsets in arcsec relative to the target, and time to hours since the first detection:

```math
x_E = \left[(\alpha - \alpha_0 + 180) \bmod 360 - 180\right] \cos\delta_0 \times 3600, \qquad y_N = (\delta - \delta_0) \times 3600
```

### Visits

Epochs are sorted by MJD and a new *visit* starts whenever the gap to the previous epoch exceeds `visit_gap_days = 0.0025` d (216 s). Linking is done **inside each visit separately**. The function `split_visits` that does this is reused by step 8.

### Parameters

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| `visit_gap_days` | `0.0025` | Gap (216 s) that separates visits |
| `min_detections` | `3` | Minimum epochs in a tracklet |
| `link_tolerance_arcsec` | `5.0` | Max distance between a detection and the predicted position |
| `min_rate_arcsec_per_hour` | `1.0` | Slowest motion accepted |
| `max_rate_arcsec_per_hour` | `3000.0` | Fastest motion accepted |
| `min_total_motion_arcsec` | `2.0` | Minimum distance travelled over the tracklet |
| `max_seed_pairs` | `5000000` | Safety limit on candidate pairs |

### Algorithm

1. **Seed pairs.** For every pair of epochs `(a, b)` with `Δt > 0`, pair detections within `r_max × Δt` of each other (KD-tree range query). The pair defines a velocity:

```math
\vec{v} = \frac{\vec{p}_b - \vec{p}_a}{\Delta t}, \qquad r_{min} \le |\vec{v}| \le r_{max}
```

2. **Prediction.** For every epoch, the position predicted by the seed is compared with the nearest detection; it counts if within `link_tolerance_arcsec`:

```math
\vec{p}_{pred}(t) = \vec{p}_a + \vec{v}\,(t - t_a), \qquad |\vec{p}_{pred} - \vec{p}_{det}| \le 5''
```

3. **Selection.** Seeds with at least `min_detections` hits are kept.
4. **Refinement.** Twice: fit a line by least squares (`x(t) = p₀ + v·t` and likewise for `y`), then re-collect the nearest detection around the fitted line in every epoch.
5. **Acceptance.** The tracklet is kept if the final fit has: at least 3 members, `rms ≤ 5″`, rate within 1 to 3000″/hr, and total motion `rate × (t_max - t_min) ≥ 2″`:

```math
\mathrm{rms} = \sqrt{\frac{1}{N}\sum_i \left[(x_i - x_{fit})^2 + (y_i - y_{fit})^2\right]}
```

6. **De-duplication.** Tracklets are sorted by most detections, then lowest rms; a tracklet is dropped if more than 50% of its detections already belong to a kept tracklet.

### Derived Quantities

| Quantity | Definition |
|----------|-----------|
| `rate_arcsec_hr` | `√(vₓ² + v_y²)` of the fit |
| `pa_deg` | Position angle, `atan2(vₓ, v_y) mod 360`, measured from North through East |
| `total_motion_arcsec` | `rate × (t_max - t_min)` |
| `mean_snr` | Mean S/N of the member detections |
| `ra_mid`, `dec_mid` | Fitted position at the middle member's time |
| `mjd_ref`, `p0x_arcsec`, `p0y_arcsec`, `vx_arcsec_hr`, `vy_arcsec_hr` | The fitted linear motion model: offset (East, North, arcsec from the target) at reference time `mjd_ref`, and velocity per hour. Step 8 uses it to place the aperture in exposures without a detection |

### Output Files

| File | Contents |
|------|----------|
| `tracklets/tracklets.csv` | `tracklet_id, group, n_det, mjd_start, mjd_end, mjd_mid, ra_mid, dec_mid, rate_arcsec_hr, pa_deg, total_motion_arcsec, rms_arcsec, mean_snr, mjd_ref, p0x_arcsec, p0y_arcsec, vx_arcsec_hr, vy_arcsec_hr` |
| `tracklets/tracklet_members.csv` | `tracklet_id, det_id, file, epoch_idx, mjd, x_pix, y_pix, ra, dec, snr, flux` |

---

## Step 7: step7_check_results.py

**File:** `step7_check_results.py`
**Purpose:** Separate known Solar System bodies from new candidates and produce a ranked report

### SkyBoT Check

For each tracklet, three **probe epochs** are used: the first, the middle and the last member detection. At each, SkyBoT (IMCCE) is queried with a cone around the measured position at that MJD.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `known_objects.enabled` | `true` | Turn the check on or off |
| `match_radius_arcsec` | `20.0` | SkyBoT cone radius |
| `min_match_fraction` | `0.5` | Fraction of probes in which the same body must appear |

```math
\frac{N_{probes\ with\ the\ same\ body}}{N_{probes\ answered}} \ge 0.5 \;\Rightarrow\; \text{known}
```

| Status | Meaning |
|--------|---------|
| `known` | One SkyBoT body matches in at least 50% of the probes |
| `unknown` | Queries worked but no body matched consistently |
| `unchecked` | No query could be completed (network), or the check is disabled |

A failed query is logged and does not stop the run.

### Candidates

Every tracklet whose status is **not** `known` (that is, `unknown` or `unchecked`) is a candidate. Candidates are ranked by number of detections, then by mean S/N, both descending.

### Diagnostic Plot (one per candidate)

| Setting | Value |
|---------|-------|
| `report.max_candidate_plots` | `20` |
| `report.cutout_pix` | `21` (stamp size in pixels) |
| Stamps shown | Up to 8, evenly spaced along the tracklet |
| Image stretch | ZScale |

The top row shows the image stamps with the measured position circled; the bottom panel shows the East/North offsets in arcsec, coloured by hours since the first detection.

### Output Files

| File | Contents |
|------|----------|
| `tracklets/tracklets_known_check.csv` | All tracklets with `known_status`, `known_name` (input of steps 8 and 9) |
| `tracklets/candidates.csv` | Non-known tracklets, ranked |
| `tracklets/candidates_triage.csv` | Ranked table: `tracklet_id, group, n_det, mjd_mid, rate_arcsec_hr, pa_deg, total_motion_arcsec, rms_arcsec, mean_snr, known_status` |
| `plots/candidate_{id:04d}.png` | Diagnostic plot per candidate |

---

## Step 8: step8_extract_spectra.py

**File:** `step8_extract_spectra.py`
**Purpose:** Measure the flux of every tracklet and every unlabeled static source in every exposure that covers it, and turn the flux-versus-wavelength table into a binned low-resolution spectrum

### Why Photometry Gives a Spectrum

SPHEREx uses linear variable filters: **each detector pixel samples one wavelength**, and that wavelength depends on where on the detector the source falls. A moving object lands on a different part of the detector in every exposure, so its flux-versus-time table is at the same time its flux-versus-wavelength table. A fixed source seen in many exposures is likewise sampled at many different wavelengths. The method follows the approach of `spxquery` and `spexpi` for SPHEREx cutouts.

### Detector Bands

| Detector | Wavelength range (µm) | Resolving power R |
|----------|-----------------------|-------------------|
| D1 | 0.75 to 1.09 | 39 |
| D2 | 1.10 to 1.62 | 41 |
| D3 | 1.63 to 2.41 | 41 |
| D4 | 2.42 to 3.82 | 35 |
| D5 | 3.83 to 4.41 | 112 |
| D6 | 4.42 to 5.00 | 128 |

### Reading an Exposure

`read_cube` opens each FITS file **once** and returns the `IMAGE`, `VARIANCE`, `FLAGS`, `ZODI` and `PSF` extensions (when present), the celestial WCS, the spectral WCS, the pixel scale and the pixel solid angle:

- **Wavelength and bandwidth** are read from the spectral WCS (alternate WCS key `W`, a WCS-WAVE lookup table) evaluated at the object's pixel. If it is unavailable, the **nominal centre of the detector band** is used and the flag `band_centre_wavelength` is added to the measurement
- **Pixel solid angle** (`sr`) comes from the cutout WCS and converts MJy/sr to Jy
- **Bad pixels:** the `FLAGS` bits 0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 19, 22, 24, 26, 27, 28, 29 are fatal (same list as `spexpi`). Bits listed in `photometry.ignore_flag_bits` are not treated as bad (useful if a bright mover is masked as a transient)

### Which Exposures Are Measured

| Object | Exposures measured | Position used |
|--------|--------------------|---------------|
| Tracklet, **known** body (SkyBoT match in step 7) | **Every** downloaded exposure in which the body is inside the cutout (all visits and days) | Miriade (IMCCE) ephemeris at the MJD of each exposure; detections keep their own measured centroid |
| Tracklet, **unknown** | The whole **visit** around the tracklet (`scope = "visit"`), or only first to last detection (`scope = "span"`) | Measured centroid where the tracklet has a detection; otherwise the step-6 linear motion model |
| **Static** unlabeled source (`S####`) | **Every** downloaded exposure that contains the position | Fixed position |

Without an orbit nothing reliable can be said about an unknown mover between visits, which is why its spectrum is limited to one visit.

Each measurement is tagged with a `source`: `detection`, `ephemeris` or `visit_model`.

**Positions in the forced exposures** are in the step-5 *Gaia-corrected* frame, so the step-5 per-epoch shift `(Δ_E, Δ_N)` is **added back** before converting to pixels with the file's own WCS:

```math
(E, N)_{raw} = (E, N)_{model} + (\Delta_E, \Delta_N)_{epoch}
```

**Ephemeris:** `Miriade.get_ephemerides` accepts one epoch at a time, so each distinct exposure time is queried separately (exposures sharing an MJD share one query). The median distance between the detection centroids and the ephemeris is logged as a sanity check. If the ephemeris cannot be obtained (network, name), the step falls back to the linear motion model.

### Aperture Photometry

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `aperture_radius_pix` | `2.0` | Circular aperture radius |
| `annulus_inner_scale` | `2.0` | Annulus inner radius = 2 × aperture radius |
| `annulus_outer_scale` | `3.0` | Annulus outer radius = 3 × aperture radius |
| `sigma_clip` | `3.0` | Clipping of the annulus pixels (up to 5 iterations) |
| `min_bg_pixels` | `10` | Fewer good annulus pixels gives flag `no_background` |
| `max_bad_fraction` | `0.3` | More masked aperture area gives flag `bad_pixels` |
| `subtract_zodi` | `true` | Subtract the `ZODI` extension first |
| `aperture_correction` | `1.0` | Used only when the cutout has no `PSF` extension |
| `psf_aperture_correction` | `true` | Divide un-trailed aperture flux by the PSF encircled energy inside the aperture |

1. **Pixel weights.** Fractional coverage of the aperture edge: `w = clip(r + 0.5 - d, 0, 1)`, where `d` is the distance of the pixel centre from the object. Masked pixels get weight 0 and the remaining weights are rescaled so masked pixels do not bias the flux low
2. **Background.** Sigma-clipped **median** of the annulus pixels (`bg`), with scatter `bg_sd` and `n_bg` pixels
3. **Flux.** In MJy/sr: `S = Σ w (I - bg)`; converted with the pixel solid angle:

```math
F_{Jy} = S \times \Omega_{pix}\,[\mathrm{sr}] \times 10^{6} \times C_{ap}
```

4. **Error.** From the `VARIANCE` extension plus the uncertainty of the background estimate (if there is no variance, the annulus scatter is used):

```math
\sigma_S^2 = \sum w^2 \sigma_{var}^2 + \left(\sum w\right)^2 \frac{\sigma_{bg}^2}{n_{bg}}
```

5. **Aperture correction** `C_ap` is `1 / EE(r)`, where `EE` is the encircled energy of the PSF stamp (SPHEREx `PSF` extension, zone plane of the oversampled cube, sub-pixel shifted to the source phase and re-binned to detector pixels; minimum `EE` of 0.05). It is applied only for **un-trailed** apertures (see below). If the PSF is unavailable, `aperture_correction` is used

### Trailing

A mover smears over `rate × exposure time` during an exposure. For tracklets the aperture is widened by half the trail, up to a cap:

```math
\mathrm{trail} = \frac{\mathrm{rate}\,[''/\mathrm{hr}] \times t_{exp}\,[\mathrm{s}]}{3600 \times p\,['']}, \qquad r = \min\left(r_0 + \tfrac{1}{2}\,\mathrm{trail},\ r_{max}\right)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `trail_correction` | `true` | Widen the aperture for movers |
| `exposure_s` | `100.0` | Fallback exposure time (header `EXPTIME` is used when present) |
| `max_aperture_pix` | `6.0` | Largest allowed aperture radius |

Static sources are never widened.

### Optional PSF Photometry (static sources)

With `psf_photometry = true` a **weighted linear PSF fit** `image = flux × PSF + constant` is done on a `(2R+1)²` stamp (`R = psf_kernel_radius_native = 5`, at least `psf_min_valid_pixels = 6` good pixels). The result replaces the aperture flux (`method = "psf"`), the aperture values are kept in `ap_flux_jy`, `ap_err_jy`, and `psf_chi2r` is recorded (it should be close to 1). The option is **off by default** and applies to static sources only.

### Measurement Quality

A spectral point is `good` if its flux and error are finite, `err_jy > 0`, the wavelength is finite, and its flag does not contain `outside`, `no_background` or `bad_pixels`. Only `good` rows go into the spectrum.

| Flag | Meaning |
|------|---------|
| `outside` | Position lies outside the image |
| `no_background` | Fewer than `min_bg_pixels` usable annulus pixels |
| `bad_pixels` | More than `max_bad_fraction` of the aperture is masked |
| `band_centre_wavelength` | Spectral WCS missing, nominal band centre used |

`flag_bits` lists the `FLAGS` bits that are set anywhere inside the aperture.

### Binning

Good points of each tracklet or static object are binned **per detector** in logarithmic wavelength bins of width λ/R:

```math
\Delta\ln\lambda = \ln\left(1 + \frac{1}{R}\right), \qquad k = \left\lfloor \frac{\ln(\lambda/\lambda_{min})}{\Delta\ln\lambda} \right\rfloor
```

Inside a bin with at least 4 points, a **MAD outlier clip** (`outlier_sigma = 3.0`, `σ_MAD = 1.4826 × MAD`) is applied. Then, with `w = 1/σ²`:

```math
F = \frac{\sum w_i F_i}{\sum w_i}, \qquad \lambda = \frac{\sum w_i \lambda_i}{\sum w_i}, \qquad \sigma_F = \max\left(\frac{1}{\sqrt{\sum w_i}},\ \frac{\mathrm{std}(F_i)}{\sqrt{n}}\right)
```

The second term in `σ_F` is used only when the bin holds more than one point, so scatter larger than the formal errors is not hidden.

### Noise-Tracklet Check

`noise_suspect` flags a tracklet when the exposures **with** a detection are significant (combined |S/N| > 3) but the **forced** exposures (no detection) are consistent with zero (combined |S/N| < 2). That is the signature of a tracklet whose detections sit on noise peaks. The plot title then carries `[SUSPECT: forced points consistent with zero]`. It needs at least 4 points, with at least 2 detected and 2 forced.

### Runtime Control

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `only_candidates` | `false` | Use only the tracklets in `candidates.csv` |
| `max_tracklets` | `null` | Measure only the N tracklets with the most detections |
| `max_runtime_s` | `1800` | Time budget per kind (tracklets, static objects). On expiry the step stops cleanly and keeps everything measured so far |
| `max_spectrum_plots` | `null` | Cap on PNGs per kind (`null` or `0` = one for every spectrum) |
| `static_classification.max_objects` | `null` | Measure only the N brightest static objects |

Each FITS file is opened **once**, and all positions inside it are measured together. Progress and an ETA are logged about every 10% of the exposures, and the time of each phase is logged.

### Plots

One PNG per spectrum (style of `spexpi`): the unbinned measurements (dots), the λ/R-binned points (open circles with error bars), a PCHIP curve through the bins (needs at least 4 distinct bins), and the detector boundaries (dotted red). The flux axis is scaled to Jy, mJy or µJy from the median flux.

### Output Files

| File | Contents |
|------|----------|
| `tracklets/spectrum_points.csv` | One row per (tracklet, exposure): `tracklet_id, file, epoch_idx, mjd, detector, wavelength_um, bandwidth_um, x_pix, y_pix, ra, dec, detected, source, det_id, aperture_pix, flux_jy, err_jy, snr, bg_mjysr, n_ap_pix, bad_frac, flag, flag_bits, good` |
| `spectra/tracklet_XXXX_spectrum.csv` | Binned spectrum: `wavelength_um, flux_jy, err_jy, detector, n_points` (input of step 9) |
| `spectra/tracklet_XXXX_spectrum.png` | Spectrum plot |
| `catalogs/static_spectrum_points.csv` | One row per (static object, exposure): same columns with `object_id`, plus `ap_corr, method, psf_chi2r, ap_flux_jy, ap_err_jy, good` |
| `static_spectra/S####_spectrum.csv` | Binned spectrum of each static object (input of step 9) |
| `static_spectra/S####_spectrum.png` | Spectrum plot |

The step sets `spectra_extracted` when it finishes. Missing tracklets or static objects are not an error: the corresponding part is skipped and logged.

---

## Step 9: step9_classify.py

**File:** `step9_classify.py`
**Purpose:** Give every binned spectrum from step 8 a rough class, for movers (tracklets) and for static unlabeled sources

### The Spectral-Index Classifier

The classifier (`classify_spectrum`) takes the binned spectrum (wavelength in µm, flux in Jy). It needs **at least 5** finite points, otherwise it raises an error and the object is labelled `insufficient_points`.

Each class has a score:

```math
\mathrm{score}_c = 1.5 \times R_c + \sum_{f \in c} w_f\, s_f \; - \; \mathrm{penalties}
```

- **Range score** `R_c`: the fraction of the observed wavelength range that overlaps the class's valid range (clipped to 0 to 1). It carries a weight of 1.5, so the range dominates
- **Feature strength** `s_f` (0 to 1) of every spectral feature `f` with weight `w_f`: the median flux inside the feature window is compared with a local continuum (median flux in a window around the feature centre)

```math
s_{abs} = \mathrm{clip}\left(0.5 + \frac{F_{cont} - F_{feat}}{|F_{cont}|},\ 0,\ 1\right), \qquad s_{em} = \mathrm{clip}\left(0.5 + \frac{F_{feat} - F_{cont}}{|F_{cont}|},\ 0,\ 1\right)
```

A window with no data, or with a continuum of about zero, gives 0. A featureless spectrum gives **about 0.5** per feature window, not 0.

The predicted class is the one with the highest score.

### Classes

| Class | Valid range (µm) | Features (type, weight, window µm) |
|-------|------------------|------------------------------------|
| Brown dwarfs | 1.13 to 2.33 | FeH 1.19 (abs, 1.0, 1.175-1.205); CO 2.295 (abs, 0.9, 2.270-2.320); CH₄ K-band 2.20 (abs, 1.0, 2.175-2.225); CH₄ H-band 1.667 (abs, 1.0, 1.63-1.70) |
| Quasars | 0.27 to 3.70 | none (range only) |
| Supernovas | 0.8 to 2.5 | Ca II 0.85-0.866 (abs, 0.8); Mg II 0.922-0.924 (abs, 0.8); Mg II/Co II 1.091-1.095 (abs, 0.6); Si II/Mg II 1.676-1.691 (abs, 0.6); Fe group 1.50-1.90 (abs, 0.6) |
| Asteroids | 0.3 to 4.0 | none (range only) |
| Galaxies | 0.9 to 2.5 | H₂ 2.122 (em, 1.0, 2.10-2.14); Br γ 2.166 (em, 0.9, 2.145-2.185) |
| Comets | 3.0 to 160 | C-H 3.4 (em, 1.0, 3.25-3.55); Si-O 10 (em, 0.9, 9.0-11.5); Si-O-Si 20 (em, 0.8, 18.0-22.5) |

### Extra Penalties

| Class | Penalty subtracted from the score |
|-------|-----------------------------------|
| Quasars | `1.0 ×` mean of the four brown-dwarf absorption strengths (FeH, CO, CH₄ K, CH₄ H) |
| Asteroids | `1.0 × (0.6 × mean brown-dwarf strengths + 0.4 × mean of Ca II, Mg II and Fe-group strengths)` |
| Supernovas | Score multiplied by `clip(1 - Σ p_w × std(F in window) / std(F), 0, 1)` over the noisy / low-transmission windows 1.31-1.38 (0.6), 1.78-1.88 (0.6), 0.93-0.96 (0.4), 1.12-1.16 (0.4), 2.40-2.50 (0.4) |

### Movers (tracklets)

`classify_tracklets` reads `tracklets_known_check.csv` (or `tracklets.csv` if step 7 did not produce it) and, for each tracklet, the binned spectrum from step 8. All six classes are scored. The row holds the predicted class, the score of every class (`score_*`), `n_points`, the wavelength range, the step-7 `known_status` and `known_name`, and the feature strengths as JSON.

| `predicted_class` value | Meaning |
|-------------------------|---------|
| Brown dwarfs / Quasars / Supernovas / Asteroids / Galaxies / Comets | Class with the highest score |
| `insufficient_points` | Fewer than 5 spectral points |
| `no_spectrum` | Step 8 produced no spectrum for this tracklet |

A known SkyBoT body (`known_status = known`) is a useful cross-check of the classifier, because its class is already known.

### Static Objects

Fixed unlabeled sources (QSO, supernova, galaxy and brown-dwarf candidates) go through the same classifier, restricted to the classes that make sense for them (**Asteroids and Comets are dropped**). The wrapper adds:

1. **Margin:** best minus second-best score
2. **Flat-spectrum baseline.** A featureless spectrum already scores about 0.5 per feature window. The same wavelength sampling is therefore scored with a **flat spectrum** and subtracted:

```math
\mathrm{excess}_c = \mathrm{score}_c - \mathrm{score}_c^{flat}
```

If no class beats the baseline by `min_excess`, `predicted_class_flatcorr` is `featureless` (instead of picking a class by dictionary order)

3. **Final class**, decided in this order:

| Order | Condition | `final_class` |
|-------|-----------|---------------|
| 1 | No classification, or fewer than `min_points` binned points | `insufficient_points` |
| 2 | Median S/N of the binned points is not finite or below `min_median_snr` | `low_snr` |
| 3 | Margin is not finite or below `min_margin` | `ambiguous` |
| 4 | Otherwise | The predicted class |

Objects with no spectrum from step 8 get `no_spectrum`.

### Configuration

Optional `config.json` → `static_classification` keys (defaults shown):

| Key | Default | Meaning |
|-----|---------|---------|
| `classes` | `["Brown dwarfs", "Quasars", "Supernovas", "Galaxies"]` | Classes allowed for static objects |
| `min_points` | `8` | Fewer binned points gives `insufficient_points` |
| `min_margin` | `0.15` | Best minus second-best score below this gives `ambiguous` |
| `min_median_snr` | `3.0` | Median S/N of the binned points below this gives `low_snr` |
| `min_excess` | `0.1` | A class must beat the flat baseline by this much, else `featureless` |
| `max_objects` | `null` | Used by step 8: measure only the N brightest static objects |

### Output Files

| File | Contents |
|------|----------|
| `tracklets/classification.csv` | One row per tracklet: `tracklet_id, n_det, rate_arcsec_hr, known_status, known_name, predicted_class, n_points, wave_min_um, wave_max_um, score_<class>..., features` |
| `catalogs/static_classification.csv` | One row per static object: `object_id, ra, dec, pos_err_arcsec, det_mean_snr, n_det_epochs, n_exposures, n_points, wave_min_um, wave_max_um, median_snr, final_class, predicted_class, margin, predicted_class_flatcorr, score_<class>..., excess_<class>..., features` |

The step sets `classified` when it finishes.

---

## References

1. **SPHEREx:** Crill et al. (2020), SPIE 11443, 114430I
2. **Gaia DR3:** Gaia Collaboration, Vallenari et al. (2023), A&A 674, A1
3. **photutils:** Bradley et al., `photutils` (Zenodo)
4. **astropy:** Astropy Collaboration (2013, 2018, 2022)
5. **SkyBoT:** Berthier et al. (2006), ASP Conf. Ser. 351, 367
6. **Miriade:** Berthier et al. (2008), IMCCE ephemeris service
7. **IRSA SPHEREx archive and TAP:** https://irsa.ipac.caltech.edu/
8. **spxquery, spexpi:** SPHEREx cutout photometry and spectrum tools, whose approach step 8 follows

---

*Pipeline documentation for the SPHEREx moving-object detection pipeline*
*Last updated: 2026-11-15*
