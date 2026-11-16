"""Run the whole pipeline (steps 1-9) and report the execution time of every step.

    python run_pipeline.py                                    # asks for RA, Dec and search radius (deg)
    python run_pipeline.py --ra 180 --dec 0 --radius 0.1      # no questions
    python run_pipeline.py --no-prompt                        # reuse the values already in config.json
    python run_pipeline.py --start-at step6_link_tracklets    # re-run only from one step on (skips step 1,
                                                              # keeps the downloaded images and flags)

Works from any folder (all paths are relative to this repo), e.g.  python /path/to/run_pipeline.py
The timing table is printed at the end and saved to <output folder>/timing.csv.
"""
import argparse
import importlib
import time

import helpers

STEPS = ("step2_download_obs", "step3_detect_sources", "step4_survey_match",
         "step5_list_unlabeled_obj", "step6_link_tracklets", "step7_check_results",
         "step8_extract_spectra", "step9_classify")


def main():
    ap = argparse.ArgumentParser(description="SPHEREx moving-object pipeline")
    ap.add_argument("--ra"), ap.add_argument("--dec")
    ap.add_argument("--radius", type=float, help="search radius in degrees")
    ap.add_argument("--no-prompt", action="store_true", help="do not ask for RA/Dec/radius")
    ap.add_argument("--start-at", choices=STEPS, help="skip step 1 and every step before this one")
    args = ap.parse_args()

    helpers.check_dependencies()  # clear message instead of an ImportError traceback
    t_all = time.time()
    timing = []

    todo = STEPS
    if args.start_at:
        todo = STEPS[STEPS.index(args.start_at):]
    else:
        print("\n=== step1_config ===")
        t = time.time()
        importlib.import_module("step1_config").main(args.ra, args.dec, args.radius,
                                                     interactive=False if args.no_prompt else None)
        timing.append(("step1_config", time.time() - t))
    for name in todo:
        print(f"\n=== {name} ===")
        t = time.time()
        importlib.import_module(name).main()
        timing.append((name, time.time() - t))
        print(f"[time] {name}: {timing[-1][1]:.1f} s")

    total = time.time() - t_all
    print("\n=== execution time ===")
    for n, s in timing:
        print(f"  {n:28s} {s:9.1f} s  {100 * s / total:5.1f} %")
    print(f"  {'TOTAL':28s} {total:9.1f} s  ({total / 60:.1f} min)")
    try:
        import pandas as pd
        paths = helpers.get_paths(helpers.load_config())
        pd.DataFrame(timing + [("TOTAL", total)], columns=["step", "seconds"]).to_csv(paths["timing_csv"], index=False)
    except Exception as e:  # noqa: BLE001 - timing is informational only
        print(f"(timing table not saved: {e})")


if __name__ == "__main__":
    main()