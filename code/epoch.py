#!/usr/bin/env python3
"""
epoch.py
--------
Trigger decoding and epoching for MEG/EEG data.

Steps
-----
1. Load preprocessed raw (from preprocess.py)
2. Find events from the STI channel
3. Apply epoch configuration from core.py (tmin, tmax, event_id, baseline)
4. Amplitude-based rejection
5. Save epochs to derivatives/epochs/

Epoch configurations (timing, trigger mapping) are defined in EPOCH_CONFIGS
in core.py — no changes needed here between projects.

Usage
-----
    # Single subject, default epoch config
    python epoch.py --subjects P01

    # Show trigger codes found in raw data (useful for first-pass)
    python epoch.py --subjects P01 --show-triggers

    # Specific epoch config
    python epoch.py --subjects P01 --epoch-config sweep

    # All subjects
    python epoch.py

    # Overwrite existing
    python epoch.py --overwrite
"""

import argparse
import sys
from pathlib import Path

import mne
import numpy as np

from core import (
    TASKS,
    EPOCH_CONFIGS,
    DEFAULT_EPOCH_CONFIG,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)

DEFAULT_ROOT = Path("./bids_project")

# ---------------------------------------------------------------------------
# Rejection threshold (peak-to-peak, SI units)
# Adjust in core.py via EPOCH_CONFIGS[...]["reject_grad"] if needed,
# or change the defaults here for your recording system.
# ---------------------------------------------------------------------------

DEFAULT_REJECT = {
    "grad": 6000e-13,   # fT/cm — Neuromag-122 gradiometers
    # "mag": 4e-12,     # T     — uncomment for magnetometers
    # "eeg": 150e-6,    # V     — uncomment for EEG
}


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Show trigger codes (diagnostic)
# ---------------------------------------------------------------------------


def show_triggers(paths: Paths, label: str, task: str, logger) -> None:
    """Print all trigger codes found in the raw STI channel."""
    raw_file = paths.raw_meg(label, task)
    if not raw_file.exists():
        logger.warning("[sub-%s / %s]  Raw file not found", label, task)
        return

    raw    = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
    events = mne.find_events(raw, verbose=False)
    unique = np.unique(events[:, 2])
    logger.info("[sub-%s / %s]  Found %d events, trigger codes: %s",
                label, task, len(events), unique.tolist())


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------


def epoch_one(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    overwrite: bool,
    logger,
) -> tuple[bool, str | None]:
    """Epoch one subject/task combination.

    Returns
    -------
    (success, qc_warning_or_None)
    """
    cfg  = EPOCH_CONFIGS[epoch_config]
    desc = f"{cfg['desc']}-preproc"
    tag  = f"sub-{label} / {task} / {epoch_config}"

    out_file = paths.epochs(label, task, desc=desc)
    if _exists(out_file) and not overwrite:
        logger.info("[%s]  SKIP (exists)", tag)
        return True, None

    # Load preprocessed raw
    prep_file = paths.prep_raw(label, task, desc="preproc")
    if not _exists(prep_file):
        logger.warning("[%s]  Preprocessed file not found: %s", tag, prep_file.name)
        return False, None

    logger.info("[%s]  Loading: %s", tag, prep_file.name)
    raw = mne.io.read_raw_fif(prep_file, preload=True, verbose=False)

    # Find events
    try:
        events = mne.find_events(raw, verbose=False)
    except Exception as e:
        logger.warning("[%s]  find_events failed: %s", tag, e)
        return False, None

    if len(events) == 0:
        logger.warning("[%s]  No events found", tag)
        return False, None

    # Event ID mapping — use config or accept all trigger codes
    event_id = cfg.get("event_id", None)
    if event_id is None:
        unique = np.unique(events[:, 2])
        event_id = {str(v): v for v in unique}
        logger.info("[%s]  No event_id in config — using all codes: %s",
                    tag, list(event_id.keys()))

    # Reject threshold — use per-config override or default
    reject = cfg.get("reject_grad", None)
    if reject is not None:
        reject_dict = {"grad": reject}
    else:
        reject_dict = DEFAULT_REJECT

    # Epoch
    logger.info(
        "[%s]  Epoching: tmin=%.2f s, tmax=%.2f s, event_id=%s",
        tag, cfg["tmin"], cfg["tmax"], list(event_id.keys()),
    )
    try:
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=cfg["tmin"],
            tmax=cfg["tmax"],
            baseline=cfg.get("baseline", (cfg["tmin"], 0)),
            reject=reject_dict,
            preload=True,
            verbose=False,
        )
    except Exception as e:
        logger.warning("[%s]  Epoching failed: %s", tag, e)
        return False, None

    n_total   = len(epochs) + epochs.drop_log.count(("IGNORED",))
    n_kept    = len(epochs)
    n_dropped = n_total - n_kept
    logger.info(
        "[%s]  Epochs: %d kept / %d dropped (%.0f%%)",
        tag, n_kept, n_dropped,
        100 * n_dropped / n_total if n_total else 0,
    )

    # QC warning if fewer epochs than expected
    expected = cfg.get("n_expected", None)
    qc_warn  = None
    if expected and n_kept < 0.8 * expected:
        qc_warn = (
            f"sub-{label} / {task} / {epoch_config}: "
            f"only {n_kept}/{expected} epochs kept ({100*n_kept/expected:.0f}%)"
        )
        logger.warning("[%s]  QC: %s", tag, qc_warn)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_file, overwrite=True, verbose=False)
    logger.info("[%s]  Saved: %s", tag, out_file.name)
    return True, qc_warn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Trigger decoding and epoching for MEG/EEG data."
    )
    parser.add_argument("--root",         type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects",     nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks",        nargs="+", default=None, choices=TASKS)
    parser.add_argument(
        "--epoch-config", default=DEFAULT_EPOCH_CONFIG,
        choices=list(EPOCH_CONFIGS.keys()),
    )
    parser.add_argument(
        "--show-triggers", action="store_true",
        help="Print trigger codes found in raw data and exit",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "epoch")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks    = args.tasks    if args.tasks    else TASKS

    if args.show_triggers:
        for label in subjects:
            for task in tasks:
                show_triggers(paths, label, task, logger)
        return

    logger.info("Subjects     : %s", subjects)
    logger.info("Tasks        : %s", tasks)
    logger.info("Epoch config : %s", args.epoch_config)
    logger.info("Overwrite    : %s", args.overwrite)

    n_ok = n_skip = n_fail = 0
    qc_warnings = []

    for label in subjects:
        for task in tasks:
            try:
                ok, warn = epoch_one(
                    paths, label, task,
                    epoch_config=args.epoch_config,
                    overwrite=args.overwrite,
                    logger=logger,
                )
                if ok:
                    n_ok += 1
                else:
                    n_skip += 1
                if warn:
                    qc_warnings.append(warn)
            except Exception as e:
                logger.error("[sub-%s / %s]  FAILED: %s", label, task, e, exc_info=True)
                n_fail += 1

    logger.info("Done.  OK: %d  |  Skipped: %d  |  Failed: %d", n_ok, n_skip, n_fail)
    if qc_warnings:
        logger.warning("QC warnings:")
        for w in qc_warnings:
            logger.warning("  %s", w)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
