#!/usr/bin/env python3
"""
core_laser.py — Example configuration for a laser-evoked potential study
=========================================================================

Copy this file to core.py for a laser pain/somatosensory MEG project.

Paradigm
--------
Single laser stimulus type, short epoch (~800 ms), somatosensory cortex ROIs.
One trigger code (e.g. 1 = laser pulse onset).

Rename to core.py before running any pipeline script.
"""

from __future__ import annotations
from pathlib import Path
import logging, sys, pandas as pd
from datetime import datetime

# =============================================================================
# Project settings  ← edit these four blocks
# =============================================================================

PROJECT_NAME = "laser_pain"

TASKS = ["laser"]

EPOCH_CONFIGS = {
    "sweep": {
        "tmin":       -0.1,    # 100 ms baseline
        "tmax":        0.8,    # 800 ms epoch
        "desc":       "sweep",
        "event_id":   {"laser": 1},   # trigger code 1 = laser pulse
        "baseline":   (-0.1, 0),
        "n_expected":  80,     # expected number of sweeps per subject (QC)
    },
    # Uncomment for a longer window including late components:
    # "long": {
    #     "tmin":  -0.1, "tmax": 1.5,
    #     "desc":  "long",
    #     "event_id": {"laser": 1},
    #     "baseline": (-0.1, 0),
    # },
}

DEFAULT_EPOCH_CONFIG = "sweep"

FILTER_CONFIGS = {
    "preproc": {"l_freq": 1.0, "h_freq": 40.0},
    # Uncomment for high-gamma analysis:
    # "broad": {"l_freq": 1.0, "h_freq": 150.0},
}

ATLAS_CONFIGS = {
    "hcpmmp1": {
        "parc": "HCPMMP1",
        "rois": {
            # Primary somatosensory cortex (areas 1, 2, 3a, 3b)
            "S1": [
                "L_1_ROI",
                "L_2_ROI",
                "L_3a_ROI",
                "L_3b_ROI",
            ],
            # Secondary somatosensory / parietal operculum
            "SII": [
                "L_OP1_ROI",
                "L_OP4_ROI",
                "L_43_ROI",
            ],
            # Insular cortex
            "Insula": [
                "L_Ig_ROI",
                "L_PoI1_ROI",
                "L_PoI2_ROI",
            ],
            # Anterior cingulate (pain affect)
            "ACC": [
                "L_24_ROI",
                "L_p24pr_ROI",
            ],
            # Primary auditory (control ROI — should show no response)
            "A1": ["L_A1_ROI"],
        },
    },
    "aparcsub": {
        "parc": "aparc_sub",
        "rois": {
            "S1":     ["lh.postcentral"],
            "SII":    ["lh.superiortemporal"],   # approximation
            "Insula": ["lh.insula"],
            "ACC":    ["lh.caudalanteriorcingulate",
                       "lh.rostralanteriorcingulate"],
        },
    },
}

DEFAULT_ATLAS = "hcpmmp1"

ROI_NAMES = sorted(
    {roi for cfg in ATLAS_CONFIGS.values() for roi in cfg["rois"]}
)


# =============================================================================
# Shared utilities — copy unchanged into every project's core.py
# =============================================================================


def sub_id(label: str) -> str:
    return label if label.startswith("sub-") else f"sub-{label}"


def sub_label(bids_id: str) -> str:
    return bids_id.removeprefix("sub-")


class Paths:
    def __init__(self, root: Path) -> None:
        self.root  = Path(root)
        self.raw   = self.root / "rawdata"
        self.deriv = self.root / "derivatives"

    def raw_meg(self, label: str, task: str) -> Path:
        return self.raw / sub_id(label) / "meg" / f"{sub_id(label)}_task-{task}_meg.fif"

    def events_tsv(self, label: str, task: str) -> Path:
        return self.raw / sub_id(label) / "meg" / f"{sub_id(label)}_task-{task}_events.tsv"

    def prep_dir(self, label: str, task: str) -> Path:
        return self.deriv / "prep" / sub_id(label) / "meg"

    def prep_raw(self, label: str, task: str, desc: str = "preproc") -> Path:
        return self.prep_dir(label, task) / f"{sub_id(label)}_task-{task}_desc-{desc}_meg.fif"

    def ica_file(self, label: str, task: str) -> Path:
        return self.prep_dir(label, task) / f"{sub_id(label)}_task-{task}_ica.fif"

    def bads_file(self, label: str, task: str) -> Path:
        return self.deriv / "bads" / f"{sub_id(label)}_task-{task}_bads.json"

    def epochs_dir(self, label: str, task: str) -> Path:
        return self.deriv / "epochs" / sub_id(label) / "meg"

    def epochs(self, label: str, task: str, desc: str = "preproc") -> Path:
        return self.epochs_dir(label, task) / f"{sub_id(label)}_task-{task}_desc-{desc}_epo.fif"

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
        return self.source_dir(label) / f"task-{task}" / "meg" / f"{sub_id(label)}_task-{task}_fwd.fif"

    def noise_cov(self, label: str, task: str) -> Path:
        return self.source_dir(label) / f"task-{task}" / "meg" / f"{sub_id(label)}_task-{task}_cov.fif"

    def inv(self, label: str, task: str) -> Path:
        return self.source_dir(label) / f"task-{task}" / "meg" / f"{sub_id(label)}_task-{task}_inv.fif"

    def roi_dir(self, label: str, task: str) -> Path:
        return self.source_dir(label) / f"task-{task}" / "meg" / "roi"

    def roi_ave(self, label: str, task: str, roi: str, atlas: str, epoch_config: str) -> Path:
        return self.roi_dir(label, task) / f"{sub_id(label)}_task-{task}_{roi}_{atlas}_{epoch_config}_ave.h5"

    def roi_epo(self, label: str, task: str, atlas: str, epoch_config: str) -> Path:
        return self.roi_dir(label, task) / f"{sub_id(label)}_task-{task}_{atlas}_{epoch_config}_epo.h5"

    def connectivity_dir(self, label: str, task: str) -> Path:
        return self.deriv / "connectivity" / sub_id(label) / f"task-{task}"

    def log_dir(self) -> Path:
        return self.deriv / "logs"

    def freesurfer_dir(self) -> Path:
        return self.deriv / "freesurfer"


def load_subjects(paths: Paths) -> list[str]:
    tsv = paths.raw / "participants.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {tsv}")
    df = pd.read_csv(tsv, sep="\t", dtype=str)
    if "participant_id" not in df.columns:
        raise ValueError(f"No participant_id column in {tsv}")
    return [sub_label(pid) for pid in df["participant_id"].dropna()]


def setup_logging(paths: Paths, script_name: str) -> logging.Logger:
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"{script_name}_{timestamp}.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(script_name)
    logger.info("Log file: %s", log_file)
    return logger
