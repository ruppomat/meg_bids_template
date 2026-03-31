#!/usr/bin/env python3
"""
source.py
---------
Source analysis pipeline for MEG/EEG data.

Steps
-----
1. Load BEM solution, source space, and coregistration transform
2. Compute forward solution (MEG only)
3. Estimate noise covariance from pre-stimulus baseline (-0.2 to 0.0 s)
4. Compute dSPM inverse operator
5. Apply inverse to evoked response → evoked STC
6. Apply inverse to single-trial epochs → single-trial STCs
7. Extract ROI time courses for each atlas configuration:
   - Averaged (from evoked STC)
   - Single-trial (from single-trial STCs, for connectivity)
8. Save all outputs to derivatives/source/

Atlases and epoch configurations are driven by ATLAS_CONFIGS and
EPOCH_CONFIGS in core.py — pass --atlases and --epoch-config on the
command line to select which combination to run.

Usage:
    # Single subject, default atlas and epoch config
    python source.py --subjects P01

    # Specific atlas
    python source.py --subjects P01 --atlases hcpmmp1 aparcsub

    # All subjects, no single-trial STCs (faster — skip for basic analyses)
    python source.py --no-single-trial

    # Overwrite existing outputs
    python source.py --overwrite
"""

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
import h5py

from core import (
    TASKS,
    EPOCH_CONFIGS,
    ATLAS_CONFIGS,
    DEFAULT_ATLAS,
    DEFAULT_EPOCH_CONFIG,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)

DEFAULT_ROOT = Path("./bids_project")


def _exists(path: Path) -> bool:
    """Return True only if path exists AND has non-zero size.

    Empty placeholder files created by create_bids_structure.py have 0 bytes
    and should be treated as missing so they are recomputed rather than
    loaded, which would cause a corrupt-file error.
    """
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Source analysis parameters
# ---------------------------------------------------------------------------

SNR = 3.0  # assumed signal-to-noise ratio
LAMBDA2 = 1.0 / SNR**2  # regularisation parameter
METHOD = "dSPM"
PICK_ORI = "normal"  # project onto cortical surface normal

# Forward model
MINDIST = 5.0  # mm — exclude sources closer to inner skull
MEG_ONLY = True  # set False to include EEG channels

# Noise covariance — estimated from pre-stimulus baseline
NOISE_COV_TMIN = -0.2  # s  (must match epoch tmin)
NOISE_COV_TMAX = 0.0  # s  (trigger onset)

# STC smoothing (number of steps for surface smoothing)
SMOOTH_STEPS = 5


# ---------------------------------------------------------------------------
# Step 1: Forward solution
# ---------------------------------------------------------------------------


def compute_forward(
    paths: Paths, label: str, task: str, overwrite: bool, logger
) -> mne.Forward | None:
    """Compute or load the forward solution for one subject/task.

    Parameters
    ----------
    paths    : Paths
    label    : str    Bare subject label.
    task     : str    BIDS task label.
    overwrite: bool
    logger   :

    Returns
    -------
    mne.Forward or None if a required file is missing.
    """
    fwd_file = paths.fwd(label, task)

    if _exists(fwd_file) and not overwrite:
        logger.info("[sub-%s / %s]  Loading forward: %s", label, task, fwd_file.name)
        return mne.read_forward_solution(fwd_file, verbose=False)

    # Check required files
    required = {
        "BEM solution": paths.bem_sol(label),
        "source space": paths.src(label),
        "trans": paths.trans(label, task),
        "raw MEG": paths.raw_meg(label, task),
    }
    missing = [name for name, p in required.items() if not p.exists()]
    if missing:
        logger.warning(
            "[sub-%s / %s]  Missing files for forward model: %s", label, task, missing
        )
        return None

    logger.info("[sub-%s / %s]  Computing forward solution", label, task)

    src_file = paths.src(label)
    if not _exists(src_file):
        logger.warning(
            "[sub-%s / %s]  Source space file missing or empty: %s",
            label,
            task,
            src_file,
        )
        return None
    src = mne.read_source_spaces(src_file, verbose=False)
    info = mne.io.read_info(paths.raw_meg(label, task), verbose=False)

    fwd = mne.make_forward_solution(
        info=info,
        trans=str(paths.trans(label, task)),
        src=src,
        bem=str(paths.bem_sol(label)),
        meg=MEG_ONLY,
        eeg=False,
        mindist=MINDIST,
        verbose=False,
    )

    fwd_file.parent.mkdir(parents=True, exist_ok=True)
    mne.write_forward_solution(fwd_file, fwd, overwrite=True, verbose=False)
    logger.info("[sub-%s / %s]  Forward saved: %s", label, task, fwd_file.name)
    return fwd


# ---------------------------------------------------------------------------
# Step 2: Noise covariance
# ---------------------------------------------------------------------------


def compute_noise_cov(
    paths: Paths, label: str, task: str, epochs: mne.Epochs, overwrite: bool, logger
) -> mne.Covariance | None:
    """Estimate or load noise covariance from pre-stimulus baseline.

    Parameters
    ----------
    paths  : Paths
    label  : str
    task   : str
    epochs : mne.Epochs  Must cover NOISE_COV_TMIN to NOISE_COV_TMAX.
    overwrite: bool
    logger :

    Returns
    -------
    mne.Covariance or None on failure.
    """
    cov_file = paths.noise_cov(label, task)

    if _exists(cov_file) and not overwrite:
        logger.info("[sub-%s / %s]  Loading noise cov: %s", label, task, cov_file.name)
        return mne.read_cov(cov_file, verbose=False)

    logger.info(
        "[sub-%s / %s]  Estimating noise cov from baseline (%.2f to %.2f s)",
        label,
        task,
        NOISE_COV_TMIN,
        NOISE_COV_TMAX,
    )

    noise_cov = mne.compute_covariance(
        epochs,
        tmin=NOISE_COV_TMIN,
        tmax=NOISE_COV_TMAX,
        method="empirical",
        verbose=False,
    )

    cov_file.parent.mkdir(parents=True, exist_ok=True)
    noise_cov.save(cov_file, overwrite=True, verbose=False)
    logger.info("[sub-%s / %s]  Noise cov saved: %s", label, task, cov_file.name)
    return noise_cov


# ---------------------------------------------------------------------------
# Step 3: Inverse operator
# ---------------------------------------------------------------------------


def compute_inverse(
    paths: Paths,
    label: str,
    task: str,
    fwd: mne.Forward,
    noise_cov: mne.Covariance,
    epochs: mne.Epochs,
    overwrite: bool,
    logger,
) -> mne.minimum_norm.InverseOperator | None:
    """Compute or load the dSPM inverse operator.

    Parameters
    ----------
    paths     : Paths
    label     : str
    task      : str
    fwd       : mne.Forward
    noise_cov : mne.Covariance
    epochs    : mne.Epochs   Used to supply channel info.
    overwrite : bool
    logger    :

    Returns
    -------
    InverseOperator or None on failure.
    """
    inv_file = paths.inv(label, task)

    if _exists(inv_file) and not overwrite:
        logger.info("[sub-%s / %s]  Loading inverse: %s", label, task, inv_file.name)
        return mne.minimum_norm.read_inverse_operator(inv_file, verbose=False)

    logger.info(
        "[sub-%s / %s]  Computing inverse operator (method=%s, SNR=%.1f)",
        label,
        task,
        METHOD,
        SNR,
    )

    inv = mne.minimum_norm.make_inverse_operator(
        epochs.info,
        fwd,
        noise_cov,
        loose=0.2,
        depth=0.8,
        verbose=False,
    )

    inv_file.parent.mkdir(parents=True, exist_ok=True)
    mne.minimum_norm.write_inverse_operator(
        inv_file, inv, overwrite=True, verbose=False
    )
    logger.info("[sub-%s / %s]  Inverse saved: %s", label, task, inv_file.name)
    return inv


# ---------------------------------------------------------------------------
# Step 4: Evoked STC
# ---------------------------------------------------------------------------


def compute_evoked_stc(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    inv: mne.minimum_norm.InverseOperator,
    evoked: mne.Evoked,
    overwrite: bool,
    logger,
) -> mne.SourceEstimate | None:
    """Apply dSPM inverse to the evoked response.

    Parameters
    ----------
    paths        : Paths
    label        : str
    task         : str
    epoch_config : str    Key from EPOCH_CONFIGS.
    inv          : InverseOperator
    evoked       : mne.Evoked
    overwrite    : bool
    logger       :

    Returns
    -------
    mne.SourceEstimate or None.
    """
    stc_dir = paths.stc_dir(label, task)
    stc_stem = stc_dir / f"{sub_id(label)}_task-{task}_desc-{epoch_config}_ave"

    # STC files are paired: *-lh.stc and *-rh.stc
    if _exists(Path(str(stc_stem) + "-lh.stc")) and not overwrite:
        logger.info("[sub-%s / %s]  Loading evoked STC", label, task)
        return mne.read_source_estimate(str(stc_stem))

    logger.info("[sub-%s / %s]  Applying inverse to evoked (%s)", label, task, METHOD)

    stc = mne.minimum_norm.apply_inverse(
        evoked,
        inv,
        lambda2=LAMBDA2,
        method=METHOD,
        pick_ori=PICK_ORI,
        verbose=False,
    )

    # Do not override stc.subject — it is set by MNE from the source space
    # and must stay consistent with it. Label subjects are normalised below.
    stc_dir.mkdir(parents=True, exist_ok=True)
    stc.save(str(stc_stem), overwrite=True, verbose=False)
    logger.info("[sub-%s / %s]  Evoked STC saved: %s", label, task, stc_stem.name)
    return stc


# ---------------------------------------------------------------------------
# Step 5: ROI extraction
# ---------------------------------------------------------------------------


def _get_labels(
    atlas_cfg: dict, subjects_dir: Path, label: str, logger, tag: str
) -> list:
    """Load parcellation labels for both hemispheres.

    Parameters
    ----------
    atlas_cfg    : dict   Entry from ATLAS_CONFIGS.
    subjects_dir : Path   FreeSurfer SUBJECTS_DIR.
    label        : str    Bare subject label.
    logger       :
    tag          : str    Log prefix.

    Returns
    -------
    list of mne.Label
    """
    parc = atlas_cfg["parc"]
    # FreeSurfer directories use the BIDS-prefixed name (sub-4163).
    # Labels will carry label.subject = "sub-4163".
    labels = mne.read_labels_from_annot(
        sub_id(label),
        parc=parc,
        subjects_dir=str(subjects_dir),
        verbose=False,
    )
    logger.info("[%s]  Loaded %d labels from %s", tag, len(labels), parc)
    return labels


def extract_roi_time_courses(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    atlas_key: str,
    src: mne.SourceSpaces,
    stc_evoked: mne.SourceEstimate,
    epochs: mne.Epochs,
    inv: mne.minimum_norm.InverseOperator,
    single_trial: bool,
    overwrite: bool,
    logger,
) -> None:
    """Extract averaged and single-trial ROI time courses.

    For each ROI defined in ATLAS_CONFIGS[atlas_key]:
    - Computes the mean time course across parcels (label_sign_flip weighted)
    - Saves averaged ROI time course to HDF5
    - Optionally saves single-trial ROI time courses to HDF5

    Parameters
    ----------
    paths        : Paths
    label        : str    Bare subject label.
    task         : str    BIDS task label.
    epoch_config : str    Key from EPOCH_CONFIGS.
    atlas_key    : str    Key from ATLAS_CONFIGS.
    src          : mne.SourceSpaces
    stc_evoked   : mne.SourceEstimate  Evoked STC.
    epochs       : mne.Epochs          For single-trial extraction.
    inv          : InverseOperator
    single_trial : bool   Also extract single-trial time courses.
    overwrite    : bool
    logger       :
    """
    atlas_cfg = ATLAS_CONFIGS[atlas_key]
    tag = f"sub-{label} / {task} / {epoch_config} / {atlas_key}"
    subjects_dir = paths.freesurfer_dir()

    all_labels = _get_labels(atlas_cfg, subjects_dir, label, logger, tag)

    # Build lookup: roi_name → list of matching mne.Label objects (both hemi)
    #
    # MNE appends hemisphere suffix to label names from read_labels_from_annot:
    #   HCPMMP1:  "L_44_ROI"  →  stored as "L_44_ROI-lh"
    #   aparc_sub: "lh.parsopercularis"  →  stored as "parsopercularis-lh"
    #
    # Strategy: match by checking if any loaded label name starts with the
    # configured label name (without hemi suffix), and also try the right-
    # hemisphere equivalent for bilateral extraction.

    def _label_base(name: str) -> str:
        """Strip MNE hemisphere suffix from a label name."""
        return name.removesuffix("-lh").removesuffix("-rh")

    roi_label_map: dict[str, list] = {}
    for roi_name, label_names in atlas_cfg["rois"].items():
        matched = []
        for ln in label_names:
            # Strip any existing hemi suffix from the configured name
            ln_base = _label_base(ln)
            # Also build right-hemisphere variant for bilateral extraction
            rh_base = ln_base.replace("L_", "R_").replace("lh.", "rh.")

            for lbl in all_labels:
                lbl_base = _label_base(lbl.name)
                if lbl_base == ln_base or lbl_base == rh_base:
                    matched.append(lbl)

        # Deduplicate while preserving order
        seen, unique = set(), []
        for lbl in matched:
            if lbl.name not in seen:
                seen.add(lbl.name)
                unique.append(lbl)

        if not unique:
            logger.warning(
                "[%s]  ROI '%s': no matching labels found (configured: %s)",
                tag,
                roi_name,
                label_names,
            )
        else:
            hemi_counts = {
                "lh": sum(1 for l in unique if l.name.endswith("-lh")),
                "rh": sum(1 for l in unique if l.name.endswith("-rh")),
            }
            logger.info(
                "[%s]  ROI '%s': %d labels %s", tag, roi_name, len(unique), hemi_counts
            )
            roi_label_map[roi_name] = unique

    if not roi_label_map:
        logger.warning("[%s]  No ROIs could be matched — skipping extraction", tag)
        return

    # ------------------------------------------------------------------
    # Averaged ROI time courses (from evoked STC)
    # ------------------------------------------------------------------
    # Normalise label.subject to match stc.subject (which MNE derives from
    # the source space and may be a bare label like "2827" rather than
    # the BIDS-prefixed "sub-2827"). MNE enforces exact equality.
    stc_subject = stc_evoked.subject
    for roi_labels_list in roi_label_map.values():
        for lbl in roi_labels_list:
            lbl.subject = stc_subject

    for roi_name, roi_labels in roi_label_map.items():
        out_ave = paths.roi_ave(label, task, roi_name, atlas_key, epoch_config)

        if _exists(out_ave) and not overwrite:
            logger.info("[%s]  SKIP averaged ROI %s", tag, roi_name)
            continue

        try:
            tc = mne.extract_label_time_course(
                stc_evoked,
                roi_labels,
                src,
                mode="mean_flip",
                verbose=False,
            )  # shape (n_labels, n_times)

            # Average across labels within the ROI
            tc_mean = tc.mean(axis=0)  # shape (n_times,)

            out_ave.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_ave, "w") as f:
                f.create_dataset("time_course", data=tc_mean)
                f.create_dataset("times", data=stc_evoked.times)
                f.attrs["roi"] = roi_name
                f.attrs["atlas"] = atlas_key
                f.attrs["task"] = task
                f.attrs["subject"] = sub_id(label)
                f.attrs["epoch_config"] = epoch_config
                f.attrs["method"] = METHOD
                f.attrs["n_labels"] = len(roi_labels)
                f.attrs["label_names"] = [l.name for l in roi_labels]

            logger.info("[%s]  Saved averaged ROI %s: %s", tag, roi_name, out_ave.name)

        except Exception as e:
            logger.warning("[%s]  Failed averaged ROI %s: %s", tag, roi_name, e)

    # ------------------------------------------------------------------
    # Single-trial ROI time courses
    # ------------------------------------------------------------------
    if not single_trial:
        return

    out_epo = paths.roi_epo(label, task, atlas_key, epoch_config)
    if _exists(out_epo) and not overwrite:
        logger.info("[%s]  SKIP single-trial ROI", tag)
        return

    logger.info(
        "[%s]  Extracting single-trial ROI time courses (%d epochs)", tag, len(epochs)
    )

    try:
        stcs_single = mne.minimum_norm.apply_inverse_epochs(
            epochs,
            inv,
            lambda2=LAMBDA2,
            method=METHOD,
            pick_ori=PICK_ORI,
            verbose=False,
        )

        # shape: (n_rois, n_epochs, n_times)
        roi_names = list(roi_label_map.keys())
        n_epochs = len(stcs_single)
        n_times = stcs_single[0].times.shape[0]
        data = np.zeros((len(roi_names), n_epochs, n_times), dtype=np.float32)

        for ei, stc in enumerate(stcs_single):
            for ri, (roi_name, roi_labels) in enumerate(roi_label_map.items()):
                tc = mne.extract_label_time_course(
                    stc,
                    roi_labels,
                    src,
                    mode="mean_flip",
                    verbose=False,
                )
                data[ri, ei, :] = tc.mean(axis=0)

        out_epo.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_epo, "w") as f:
            f.create_dataset(
                "time_courses", data=data, compression="gzip", compression_opts=4
            )
            f.create_dataset("times", data=stcs_single[0].times)
            f.create_dataset("roi_names", data=np.array(roi_names, dtype="S"))
            f.attrs["atlas"] = atlas_key
            f.attrs["task"] = task
            f.attrs["subject"] = sub_id(label)
            f.attrs["epoch_config"] = epoch_config
            f.attrs["method"] = METHOD
            f.attrs["n_epochs"] = n_epochs
            f.attrs["n_rois"] = len(roi_names)

        logger.info("[%s]  Saved single-trial ROI time courses: %s", tag, out_epo.name)

    except Exception as e:
        logger.warning("[%s]  Failed single-trial ROI extraction: %s", tag, e)


# ---------------------------------------------------------------------------
# Main per-subject/task pipeline
# ---------------------------------------------------------------------------


def source_one(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    atlases: list[str],
    single_trial: bool,
    overwrite: bool,
    logger,
) -> bool:
    """Run the full source analysis pipeline for one subject/task.

    Parameters
    ----------
    paths        : Paths
    label        : str          Bare subject label.
    task         : str          BIDS task label.
    epoch_config : str          Key from EPOCH_CONFIGS.
    atlases      : list[str]    Keys from ATLAS_CONFIGS to extract ROIs for.
    single_trial : bool         Extract single-trial ROI time courses.
    overwrite    : bool
    logger       :

    Returns
    -------
    bool  True on success.
    """
    tag = f"sub-{label} / {task} / {epoch_config}"
    cfg = EPOCH_CONFIGS[epoch_config]
    band = "preproc"  # source analysis always uses the standard band
    desc = f"{cfg['desc']}-{band}"

    # Load epochs
    epo_file = paths.epochs(label, task, desc=desc)
    if not epo_file.exists():
        logger.warning("[%s]  Epochs file not found: %s", tag, epo_file.name)
        return False

    logger.info("[%s]  Loading epochs: %s", tag, epo_file.name)
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    evoked = epochs.average()

    # Forward solution
    fwd = compute_forward(paths, label, task, overwrite, logger)
    if fwd is None:
        return False

    # Noise covariance
    noise_cov = compute_noise_cov(paths, label, task, epochs, overwrite, logger)
    if noise_cov is None:
        return False

    # Inverse operator
    inv = compute_inverse(paths, label, task, fwd, noise_cov, epochs, overwrite, logger)
    if inv is None:
        return False

    # Source space (needed for ROI extraction)
    src_file = paths.src(label)
    if not _exists(src_file):
        logger.warning("[%s]  Source space file missing or empty: %s", tag, src_file)
        return False
    src = mne.read_source_spaces(src_file, verbose=False)

    # Evoked STC
    stc_evoked = compute_evoked_stc(
        paths, label, task, epoch_config, inv, evoked, overwrite, logger
    )
    if stc_evoked is None:
        return False

    # ROI extraction for each requested atlas
    for atlas_key in atlases:
        extract_roi_time_courses(
            paths,
            label,
            task,
            epoch_config,
            atlas_key,
            src,
            stc_evoked,
            epochs,
            inv,
            single_trial=single_trial,
            overwrite=overwrite,
            logger=logger,
        )

    logger.info("[%s]  Done", tag)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Source analysis (dSPM) for the RHT MEG study."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"BIDS project root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Bare subject labels (default: all in participants.tsv)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=TASKS,
        metavar="TASK",
        help=f"Tasks to process (default: all — {TASKS})",
    )
    parser.add_argument(
        "--epoch-config",
        default=DEFAULT_EPOCH_CONFIG,
        choices=list(EPOCH_CONFIGS.keys()),
        help=f"Epoch configuration to use (default: {DEFAULT_EPOCH_CONFIG})",
    )
    parser.add_argument(
        "--atlases",
        nargs="+",
        default=[DEFAULT_ATLAS],
        choices=list(ATLAS_CONFIGS.keys()),
        help=f"Atlas configurations for ROI extraction (default: {DEFAULT_ATLAS})",
    )
    parser.add_argument(
        "--no-single-trial",
        action="store_true",
        help="Skip single-trial ROI extraction (faster, no connectivity outputs)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    args = parser.parse_args()

    paths = Paths(args.root)
    logger = setup_logging(paths, "source")

    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks = args.tasks if args.tasks else TASKS

    logger.info("Subjects     : %s", subjects)
    logger.info("Tasks        : %s", tasks)
    logger.info("Epoch config : %s", args.epoch_config)
    logger.info("Atlases      : %s", args.atlases)
    logger.info("Single-trial : %s", not args.no_single_trial)
    logger.info("Method       : %s  (SNR=%.1f, lambda2=%.4f)", METHOD, SNR, LAMBDA2)
    logger.info("Overwrite    : %s", args.overwrite)

    n_ok, n_skip, n_fail = 0, 0, 0

    for label in subjects:
        for task in tasks:
            try:
                ok = source_one(
                    paths,
                    label,
                    task,
                    epoch_config=args.epoch_config,
                    atlases=args.atlases,
                    single_trial=not args.no_single_trial,
                    overwrite=args.overwrite,
                    logger=logger,
                )
                if ok:
                    n_ok += 1
                else:
                    n_skip += 1
            except Exception as e:
                logger.error("[sub-%s / %s]  FAILED: %s", label, task, e, exc_info=True)
                n_fail += 1

    logger.info("─────────────────────────────────────────────")
    logger.info(
        "Done.  Success: %d  |  Skipped: %d  |  Failed: %d", n_ok, n_skip, n_fail
    )

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
