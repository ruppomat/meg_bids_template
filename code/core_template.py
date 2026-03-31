#!/usr/bin/env python3
"""
core.py — Project configuration and shared utilities
=====================================================
Template for MEG/EEG student projects using a BIDS-compliant pipeline.

QUICKSTART
----------
1. Copy this file into your project's code/ directory.
2. Edit the sections marked  ← CONFIGURE THIS.
3. Leave everything below "Shared utilities" untouched.

Minimal changes for a new project
----------------------------------
- PROJECT_NAME       : used in log filenames
- TASKS              : list of BIDS task labels (often just one)
- EPOCH_CONFIGS      : epoch timing, description string, and optional markers
- ATLAS_CONFIGS      : which ROIs to extract (copy/paste from existing project)
- FILTER_CONFIGS     : filter bands you actually need

Everything else (Paths, load_subjects, sub_id, setup_logging) is project-
independent and can be reused verbatim.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# =============================================================================
# ← CONFIGURE THIS:  Project identity
# =============================================================================

PROJECT_NAME = "myproject"  # used in log filenames — no spaces or slashes

# =============================================================================
# ← CONFIGURE THIS:  Tasks
#
# List all BIDS task labels used in this project.
# Each label must match the task-<label> part of your raw .fif filenames.
#
# Examples:
#   Single-task:  TASKS = ["laser"]
#   Multi-task:   TASKS = ["passive", "active"]
#   RHT study:    TASKS = ["rhtpassive", "rhthigh", "rhtlow"]
# =============================================================================

TASKS: list[str] = [
    "mytask",  # ← replace with your BIDS task label(s)
]

# =============================================================================
# ← CONFIGURE THIS:  Epoch configurations
#
# One entry per epoch type you want to analyse.
# Keys are used as --epoch-config values on the command line.
#
# Required fields
# ---------------
# tmin        : float   Pre-trigger baseline start (s).  Typically -0.2 to -0.1.
# tmax        : float   Epoch end (s).
# desc        : str     Short label used in output filenames (no spaces).
#
# Optional fields
# ---------------
# event_id    : dict    Trigger value → condition name mapping.
#                       If omitted, all events found by epoch.py are used.
# baseline    : tuple   (tmin, tmax) for baseline correction.  None = no correction.
# melody_onset: float   For RHT-style paradigms — marks a secondary event for plots.
# reject_grad : float   Gradient rejection threshold in fT/cm (overrides global).
#
# Examples
# --------
#   Single trigger, short sweep (laser / SEP):
#     "sweep": {"tmin": -0.1, "tmax": 0.8, "desc": "sweep",
#               "event_id": {"stimulus": 1}, "baseline": (-0.1, 0)}
#
#   Auditory oddball:
#     "standard": {"tmin": -0.1, "tmax": 0.6, "desc": "standard",
#                  "event_id": {"standard": 1, "deviant": 2}}
#
#   Long paradigm with multiple windows:
#     "full":  {"tmin": -0.2, "tmax": 6.0, "desc": "full"},
#     "early": {"tmin": -0.2, "tmax": 1.0, "desc": "early"},
# =============================================================================

EPOCH_CONFIGS: dict[str, dict] = {
    "sweep": {  # ← rename and adjust
        "tmin": -0.5,  # ← baseline start (s)
        "tmax": 1.5,  # ← epoch end (s)
        "desc": "sweep",  # ← used in output filenames
        "event_id": {"stimulus": 1},  # ← {name: trigger_value}; remove if auto
        "baseline": (-0.2, 0),  # ← or None for no correction
    },
    # Add more windows here if needed, e.g.:
    # "late": {"tmin": -0.1, "tmax": 0.8, "desc": "late", ...},
}

DEFAULT_EPOCH_CONFIG: str = "sweep"  # ← must be a key in EPOCH_CONFIGS

# =============================================================================
# ← CONFIGURE THIS:  Filter / preprocessing bands
#
# Each entry produces one set of filtered + ICA-cleaned output files.
# "preproc" is the standard broadband used for epoching and source analysis.
# Add "broad" only if you also need a high-frequency band (e.g. for gamma).
# =============================================================================

FILTER_CONFIGS: dict[str, dict] = {
    "preproc": {"l_freq": 1.0, "h_freq": 100.0},
    # "broad":  {"l_freq": 1.0, "h_freq": 150.0},  # uncomment if needed
}

# =============================================================================
# ← CONFIGURE THIS:  Atlas / ROI definitions
#
# Each atlas entry defines:
#   parc     : str   FreeSurfer parcellation name.
#                    "aparc_sub"  — 448 labels, finer sulcal resolution
#                    "HCPMMP1"    — 360 labels, functional (Glasser 2016)
#                    "aparc"      — 68 labels, Desikan-Killiany (coarse)
#   rois     : dict  {roi_name: [label_names]}
#              Label names follow MNE convention after read_labels_from_annot:
#                    HCPMMP1  → "L_A1_ROI", "R_A1_ROI"  (no hemi suffix needed)
#                    aparc_sub → "lh.superiortemporal"   (no hemi suffix needed)
#              Bilateral extraction is automatic: L_ ↔ R_ and lh. ↔ rh.
#
# Minimal example — primary auditory cortex + one frontal region:
#
#   "hcpmmp1": {
#       "parc": "HCPMMP1",
#       "rois": {
#           "A1":   ["L_A1_ROI"],
#           "IFG":  ["L_44_ROI", "L_45_ROI"],
#       }
#   }
#
# Leave as an empty dict {} to skip ROI extraction entirely.
# =============================================================================

ATLAS_CONFIGS: dict[str, dict] = {
    "aparcsub": {
        "parc": "aparc_sub",
        "rois": {
            # ← replace with the regions relevant to your study
            "auditory": [
                "lh.transversetemporal",
                "lh.superiortemporal",
            ],
            "frontal": [
                "lh.parsopercularis",
                "lh.parstriangularis",
            ],
        },
    },
    "hcpmmp1": {
        "parc": "HCPMMP1",
        "rois": {
            "A1": ["L_A1_ROI"],
            "Belt": ["L_LBelt_ROI", "L_MBelt_ROI", "L_PBelt_ROI"],
            "IFG": ["L_44_ROI", "L_45_ROI"],
        },
    },
}

DEFAULT_ATLAS: str = "hcpmmp1"  # ← must be a key in ATLAS_CONFIGS

# Optional: flat list of all ROI names for convenience imports
ROI_NAMES: list[str] = sorted(
    {roi for cfg in ATLAS_CONFIGS.values() for roi in cfg["rois"]}
)


# =============================================================================
# Shared utilities — no changes needed below this line
# =============================================================================


def sub_id(label: str) -> str:
    """Return BIDS-prefixed subject ID from a bare label.

    Examples
    --------
    >>> sub_id("2827")
    'sub-2827'
    >>> sub_id("sub-2827")   # idempotent
    'sub-2827'
    """
    return label if label.startswith("sub-") else f"sub-{label}"


def sub_label(bids_id: str) -> str:
    """Strip the 'sub-' prefix from a BIDS subject ID.

    Examples
    --------
    >>> sub_label("sub-2827")
    '2827'
    >>> sub_label("2827")    # idempotent
    '2827'
    """
    return bids_id.removeprefix("sub-")


class Paths:
    """Centralised path construction for a BIDS project.

    All paths are derived from *root* (the BIDS project root, i.e. the
    directory that contains ``rawdata/`` and ``derivatives/``).

    Usage
    -----
    >>> paths = Paths(Path("/data/myproject"))
    >>> paths.raw_meg("2827", "laser")
    PosixPath('/data/myproject/rawdata/sub-2827/meg/sub-2827_task-laser_meg.fif')
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.raw = self.root / "rawdata"
        self.deriv = self.root / "derivatives"

    # ── Raw data ──────────────────────────────────────────────────────────

    def raw_meg(self, label: str, task: str) -> Path:
        return self.raw / sub_id(label) / "meg" / f"{sub_id(label)}_task-{task}_meg.fif"

    def events_tsv(self, label: str, task: str) -> Path:
        return (
            self.raw / sub_id(label) / "meg" / f"{sub_id(label)}_task-{task}_events.tsv"
        )

    # ── Derivatives: preprocessing ────────────────────────────────────────

    def prep_dir(self, label: str, task: str) -> Path:
        return self.deriv / "prep" / sub_id(label) / "meg"

    def prep_raw(self, label: str, task: str, desc: str = "preproc") -> Path:
        return (
            self.prep_dir(label, task)
            / f"{sub_id(label)}_task-{task}_desc-{desc}_meg.fif"
        )

    def ica_file(self, label: str, task: str) -> Path:
        return self.prep_dir(label, task) / f"{sub_id(label)}_task-{task}_ica.fif"

    def bads_file(self, label: str, task: str) -> Path:
        return self.deriv / "bads" / f"{sub_id(label)}_task-{task}_bads.json"

    # ── Derivatives: epochs ───────────────────────────────────────────────

    def epochs_dir(self, label: str, task: str) -> Path:
        return self.deriv / "epochs" / sub_id(label) / "meg"

    def epochs(self, label: str, task: str, desc: str = "preproc") -> Path:
        return (
            self.epochs_dir(label, task)
            / f"{sub_id(label)}_task-{task}_desc-{desc}_epo.fif"
        )

    # ── Derivatives: source analysis ─────────────────────────────────────

    def source_dir(self, label: str) -> Path:
        return self.deriv / "source" / sub_id(label)

    def bem_dir(self, label: str) -> Path:
        return self.source_dir(label) / "bem"

    def bem_sol(self, label: str) -> Path:
        return self.bem_dir(label) / f"{sub_id(label)}-5120-bem-sol.fif"

    def src(self, label: str) -> Path:
        return self.source_dir(label) / f"{sub_id(label)}-src.fif"

    def trans(self, label: str, task: str) -> Path:
        return self.deriv / "trans" / f"{sub_id(label)}_task-{task}_trans.fif"

    def stc_dir(self, label: str, task: str) -> Path:
        return self.source_dir(label) / f"task-{task}" / "meg" / "stc"

    def fwd(self, label: str, task: str) -> Path:
        return (
            self.source_dir(label)
            / f"task-{task}"
            / "meg"
            / f"{sub_id(label)}_task-{task}_fwd.fif"
        )

    def noise_cov(self, label: str, task: str) -> Path:
        return (
            self.source_dir(label)
            / f"task-{task}"
            / "meg"
            / f"{sub_id(label)}_task-{task}_cov.fif"
        )

    def inv(self, label: str, task: str) -> Path:
        return (
            self.source_dir(label)
            / f"task-{task}"
            / "meg"
            / f"{sub_id(label)}_task-{task}_inv.fif"
        )

    # ── Derivatives: ROI time courses ────────────────────────────────────

    def roi_dir(self, label: str, task: str) -> Path:
        return self.source_dir(label) / f"task-{task}" / "meg" / "roi"

    def roi_ave(
        self, label: str, task: str, roi: str, atlas: str, epoch_config: str
    ) -> Path:
        return (
            self.roi_dir(label, task)
            / f"{sub_id(label)}_task-{task}_{roi}_{atlas}_{epoch_config}_ave.h5"
        )

    def roi_epo(self, label: str, task: str, atlas: str, epoch_config: str) -> Path:
        return (
            self.roi_dir(label, task)
            / f"{sub_id(label)}_task-{task}_{atlas}_{epoch_config}_epo.h5"
        )

    # ── Derivatives: connectivity ─────────────────────────────────────────

    def connectivity_dir(self, label: str, task: str) -> Path:
        return self.deriv / "connectivity" / sub_id(label) / f"task-{task}"

    # ── Logging ──────────────────────────────────────────────────────────

    def log_dir(self) -> Path:
        return self.deriv / "logs"

    def freesurfer_dir(self) -> Path:
        return self.deriv / "freesurfer"


def load_subjects(paths: Paths) -> list[str]:
    """Return bare subject labels from participants.tsv.

    The file is expected at ``<root>/rawdata/participants.tsv`` with a
    ``participant_id`` column containing BIDS-prefixed IDs (``sub-XXXX``).
    Returns bare labels (without ``sub-`` prefix).

    Raises
    ------
    FileNotFoundError  if participants.tsv does not exist.
    ValueError         if the participant_id column is missing.
    """
    tsv = paths.raw / "participants.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {tsv}")

    df = pd.read_csv(tsv, sep="\t", dtype=str)
    if "participant_id" not in df.columns:
        raise ValueError(f"No 'participant_id' column in {tsv}")

    return [sub_label(pid) for pid in df["participant_id"].dropna()]


def setup_logging(paths: Paths, script_name: str) -> logging.Logger:
    """Configure and return a logger that writes to both stdout and a log file.

    Log files are written to ``derivatives/logs/<script_name>_<timestamp>.log``.

    Parameters
    ----------
    paths       : Paths
    script_name : str   Used as the log filename stem (e.g. "preprocess").

    Returns
    -------
    logging.Logger
    """
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logger = logging.getLogger(script_name)
    logger.info("Log file: %s", log_file)
    return logger
