#!/usr/bin/env python3
"""
preprocess.py
-------------
Preprocessing pipeline for MEG/EEG data.

Steps
-----
1. Load raw .fif file
2. Apply bandpass filter (from FILTER_CONFIGS in core.py)
3. Fit Picard ICA on the broadband signal
4. Auto-detect and mark EOG / ECG artefact components
5. Apply ICA to all filter bands
6. Interpolate bad channels
7. Save preprocessed raw to derivatives/prep/

All settings (filter bands, tasks, subject list) are read from core.py.
The script is safe to re-run: existing outputs are skipped unless --overwrite.

Usage
-----
    # Single subject, all tasks
    python preprocess.py --subjects P01

    # Skip ICA (first-pass inspection)
    python preprocess.py --subjects P01 --no-ica

    # All subjects
    python preprocess.py

    # Overwrite existing
    python preprocess.py --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np

from core import (
    TASKS,
    FILTER_CONFIGS,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)

DEFAULT_ROOT = Path("./bids_project")

# ---------------------------------------------------------------------------
# ICA settings
# ---------------------------------------------------------------------------

ICA_METHOD       = "picard"
ICA_N_COMPONENTS = 0.99        # retain 99% of variance
ICA_MAX_ITER     = 500
ICA_RANDOM_STATE = 42

# Amplitude rejection before ICA fit (peak-to-peak, in SI units)
# Adjust for your data — these defaults suit Neuromag-122 gradiometers
REJECT_FOR_ICA = {
    "grad": 4000e-13,   # fT/cm  → works for gradiometer systems
    # "mag": 4e-12,     # T      → uncomment for magnetometers
    # "eeg": 150e-6,    # V      → uncomment for EEG
}


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Load raw
# ---------------------------------------------------------------------------


def load_raw(paths: Paths, label: str, task: str, logger) -> mne.io.Raw | None:
    raw_file = paths.raw_meg(label, task)
    if not raw_file.exists():
        logger.warning("[sub-%s / %s]  Raw file not found: %s", label, task, raw_file)
        return None

    logger.info("[sub-%s / %s]  Loading raw: %s", label, task, raw_file.name)
    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)

    # Apply bad channels from bads JSON if it exists
    bads_file = paths.bads_file(label, task)
    if bads_file.exists():
        bads = json.loads(bads_file.read_text()).get("bad_channels", [])
        if bads:
            raw.info["bads"] = bads
            logger.info("[sub-%s / %s]  Marking %d bad channels", label, task, len(bads))

    return raw


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def filter_raw(
    raw: mne.io.Raw, l_freq: float, h_freq: float, logger, tag: str
) -> mne.io.Raw:
    logger.info("[%s]  Bandpass filter: %.1f – %.1f Hz", tag, l_freq, h_freq)
    return raw.copy().filter(
        l_freq=l_freq, h_freq=h_freq,
        method="fir", fir_design="firwin",
        verbose=False,
    )


# ---------------------------------------------------------------------------
# ICA
# ---------------------------------------------------------------------------


def fit_ica(
    raw: mne.io.Raw, logger, tag: str
) -> mne.preprocessing.ICA:
    logger.info(
        "[%s]  Fitting ICA (method=%s, n_components=%s)",
        tag, ICA_METHOD, ICA_N_COMPONENTS,
    )
    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        max_iter=ICA_MAX_ITER,
        random_state=ICA_RANDOM_STATE,
    )
    # Filter to 1 Hz before ICA fit (standard practice)
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(raw_for_ica, reject=REJECT_FOR_ICA, verbose=False)
    logger.info("[%s]  ICA: %d components fitted", tag, ica.n_components_)
    return ica


def detect_artefacts(
    ica: mne.preprocessing.ICA, raw: mne.io.Raw, logger, tag: str
) -> mne.preprocessing.ICA:
    """Auto-detect EOG and ECG artefact components."""
    exclude = set()

    # EOG
    try:
        eog_idx, eog_scores = ica.find_bads_eog(raw, verbose=False)
        if eog_idx:
            exclude.update(eog_idx)
            logger.info("[%s]  EOG components: %s", tag, eog_idx)
    except Exception as e:
        logger.warning("[%s]  EOG detection failed: %s", tag, e)

    # ECG
    try:
        ecg_idx, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
        if ecg_idx:
            exclude.update(ecg_idx)
            logger.info("[%s]  ECG components: %s", tag, ecg_idx)
    except Exception as e:
        logger.warning("[%s]  ECG detection failed: %s", tag, e)

    ica.exclude = sorted(exclude)
    logger.info(
        "[%s]  Excluding %d ICA components: %s", tag, len(ica.exclude), ica.exclude
    )
    return ica


# ---------------------------------------------------------------------------
# Main per-subject pipeline
# ---------------------------------------------------------------------------


def preprocess_one(
    paths: Paths,
    label: str,
    task: str,
    run_ica: bool,
    overwrite: bool,
    logger,
) -> int:
    """Preprocess one subject/task.  Returns number of bands written."""
    tag = f"sub-{label} / {task}"

    raw = load_raw(paths, label, task, logger)
    if raw is None:
        return 0

    # Fit ICA once on the broadband signal (reused for all filter bands)
    ica = None
    if run_ica:
        ica_file = paths.ica_file(label, task)
        if _exists(ica_file) and not overwrite:
            logger.info("[%s]  Loading existing ICA: %s", tag, ica_file.name)
            ica = mne.preprocessing.read_ica(ica_file, verbose=False)
        else:
            ica = fit_ica(raw, logger, tag)
            ica = detect_artefacts(ica, raw, logger, tag)
            ica_file.parent.mkdir(parents=True, exist_ok=True)
            ica.save(ica_file, overwrite=True, verbose=False)
            logger.info("[%s]  ICA saved: %s", tag, ica_file.name)

    # Apply filter + ICA for each configured band
    n_written = 0
    for band_name, filt_cfg in FILTER_CONFIGS.items():
        desc = f"{band_name}-preproc" if band_name != "preproc" else "preproc"
        out_file = paths.prep_raw(label, task, desc=desc)

        if _exists(out_file) and not overwrite:
            logger.info("[%s]  SKIP %s (exists)", tag, out_file.name)
            continue

        raw_band = filter_raw(
            raw, filt_cfg["l_freq"], filt_cfg["h_freq"], logger, tag
        )

        if ica is not None:
            logger.info("[%s]  Applying ICA to %s band", tag, band_name)
            ica.apply(raw_band, verbose=False)

        # Interpolate bad channels
        if raw_band.info["bads"]:
            logger.info("[%s]  Interpolating bad channels", tag)
            raw_band.interpolate_bads(verbose=False)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        raw_band.save(out_file, overwrite=True, verbose=False)
        logger.info("[%s]  Saved: %s", tag, out_file.name)
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing (filter + ICA) for MEG/EEG data."
    )
    parser.add_argument("--root",     type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks",    nargs="+", default=None, choices=TASKS)
    parser.add_argument(
        "--no-ica", action="store_true",
        help="Skip ICA (useful for first-pass inspection)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "preprocess")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks    = args.tasks    if args.tasks    else TASKS

    logger.info("Subjects : %s", subjects)
    logger.info("Tasks    : %s", tasks)
    logger.info("ICA      : %s", not args.no_ica)
    logger.info("Overwrite: %s", args.overwrite)

    n_ok = n_fail = 0
    for label in subjects:
        for task in tasks:
            try:
                n = preprocess_one(
                    paths, label, task,
                    run_ica=not args.no_ica,
                    overwrite=args.overwrite,
                    logger=logger,
                )
                n_ok += n
            except Exception as e:
                logger.error("[sub-%s / %s]  FAILED: %s", label, task, e, exc_info=True)
                n_fail += 1

    logger.info("Done.  Written: %d  |  Failed: %d", n_ok, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
