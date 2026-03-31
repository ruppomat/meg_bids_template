# Getting Started — MEG/EEG Analysis Pipeline

**For students at the MEG Core Facility, University of Heidelberg**

This guide walks you through setting up and running the pipeline on your own
computer — step by step, no prior experience with BIDS or MNE required.

---

## What you need

- A Mac (Apple Silicon M1/M2/M3 recommended) or Linux machine
- ~10 GB free disk space for software
- Your raw MEG/EEG data files (`.fif` format) from the lab

---

## Step 1 — Install Miniconda

If you don't have conda yet:

```bash
# Download and install Miniconda (Apple Silicon)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
# Follow the prompts, then restart your terminal
```

> **Linux:** replace `MacOSX-arm64` with `Linux-x86_64`

---

## Step 2 — Create the MNE environment

```bash
conda create -n mne python=3.11
conda activate mne
pip install mne h5py pandas numpy scipy joblib pyvistaqt

# Optional but recommended — GPU acceleration for permutation tests (Apple Silicon only)
pip install mlx
```

You only need to do this once. For all future work, just activate the environment:

```bash
conda activate mne
```

---

## Step 3 — Clone the pipeline

```bash
git clone https://github.com/ruppomat/meg_bids_template.git
cd meg_bids_template
```

---

## Step 4 — Configure your project

Copy the template and edit the four settings at the top:

```bash
cp code/core_template.py code/core.py
open code/core.py          # or use any text editor
```

The four things to set:

```python
PROJECT_NAME = "laser_pain"     # ← your project name (no spaces)

TASKS = ["laser"]               # ← your BIDS task label(s)

EPOCH_CONFIGS = {
    "sweep": {
        "tmin":     -0.1,       # ← baseline start in seconds
        "tmax":      0.8,       # ← epoch end in seconds
        "desc":     "sweep",    # ← short label for filenames
        "event_id": {"pain": 1},# ← {name: trigger_code}
        "baseline": (-0.1, 0),  # ← baseline window
    }
}

ATLAS_CONFIGS = {               # ← brain regions to extract
    "hcpmmp1": {
        "parc": "HCPMMP1",
        "rois": {
            "S1":  ["L_1_ROI", "L_2_ROI", "L_3a_ROI", "L_3b_ROI"],
            "SII": ["L_OP1_ROI", "L_OP4_ROI"],
        }
    }
}
```

**Not sure which trigger code to use?** Ask your supervisor, or check the events
file that came with your data.

**Not sure which ROI labels exist?** Run this in Python:

```python
import mne
labels = mne.read_labels_from_annot(
    "fsaverage", parc="HCPMMP1",
    subjects_dir="/path/to/derivatives/freesurfer"
)
for l in labels:
    print(l.name)
```

---

## Step 5 — Create the project folder structure

```bash
python code/create_bids_structure.py --root /path/to/my_project
```

This creates:

```
my_project/
├── rawdata/           ← put your .fif files here
└── derivatives/       ← all outputs go here automatically
```

---

## Step 6 — Copy your data

Place your raw MEG files in the correct location:

```
rawdata/
└── sub-YOURNAME/
    └── meg/
        └── sub-YOURNAME_task-laser_meg.fif
```

The subject label (`YOURNAME`) can be any short identifier (e.g. `P01`, `AB`,
`2827`) — no spaces or special characters.

> If you have multiple subjects, create one folder per subject:
> `sub-P01/meg/`, `sub-P02/meg/`, etc.

---

## Step 7 — Add participants

Edit `rawdata/participants.tsv` and add a line for each subject:

```
participant_id  age  sex  handedness
sub-P01         24   F    R
sub-P02         31   M    R
```

---

## Step 8 — Check your trigger codes

Before running the full pipeline, verify that the trigger codes in your data
match what you set in `core.py`:

```bash
python code/epoch.py \
    --root /path/to/my_project \
    --subjects P01 \
    --show-triggers
```

This prints all trigger codes found in the data. Adjust `event_id` in
`core.py` if needed.

---

## Step 9 — Run the pipeline

### One subject at a time (recommended for first test)

```bash
# Step 1: preprocess (filter + ICA — skip ICA on first pass for speed)
python code/preprocess.py --root /path/to/my_project --subjects P01 --no-ica

# Step 2: epoch
python code/epoch.py --root /path/to/my_project --subjects P01

# Step 3: source analysis (dSPM inverse)
python code/source.py --root /path/to/my_project --subjects P01 --no-single-trial
```

### All subjects at once

```bash
python code/batch.py --root /path/to/my_project
```

### Group analysis (after all subjects are processed)

```bash
python code/contrast.py --root /path/to/my_project
```

---

## Step 10 — Look at your results

### Interactive brain viewer

```bash
python code/visualize.py --root /path/to/my_project
```

This opens an interactive 3D brain showing the grand-average source
reconstruction. Use the time slider to browse through the epoch.

### ROI time courses

The group-level ROI time courses are saved as HDF5 files in:

```
derivatives/contrasts/group/roi_stats/
```

You can read them in Python:

```python
import h5py, numpy as np, matplotlib.pyplot as plt

with h5py.File("derivatives/contrasts/group/roi_stats/hcpmmp1/sweep/"
               "groupstats_task-laser_S1_hcpmmp1_sweep.h5") as f:
    times  = f["times"][:]
    median = f["median"][:]
    mad    = f["mad"][:]

plt.plot(times, median, label="S1 (median)")
plt.fill_between(times, median - mad, median + mad, alpha=0.3, label="±MAD")
plt.axvline(0, color="gray", linestyle="--", label="Stimulus")
plt.xlabel("Time (s)")
plt.ylabel("dSPM amplitude")
plt.legend()
plt.show()
```

### Statistical tests

```bash
# Test for a significant response in a specific time window
python code/permtest.py \
    --root /path/to/my_project \
    --contrasts laser-baseline \
    --tmin 0.05 --tmax 0.5 \
    --n-permutations 5000
```

---

## Common problems

| Problem                               | Solution                                                              |
| ------------------------------------- | --------------------------------------------------------------------- |
| `ModuleNotFoundError: mne`            | Run `conda activate mne` first                                        |
| `FileNotFoundError: participants.tsv` | Check that `--root` points to the right folder                        |
| `STC not found`                       | Source analysis hasn't run yet — run `source.py` first                |
| `No trigger codes found`              | Check raw data with `--show-triggers`, adjust `event_id` in `core.py` |
| `FreeSurfer reconstruction missing`   | Ask your supervisor — FreeSurfer needs to be run separately           |
| Window closes immediately             | Install `pyvistaqt`: `pip install pyvistaqt`                          |

---

## Where to find help

- **MNE documentation:** [mne.tools/stable](https://mne.tools/stable/)
- **BIDS specification:** [bids.neuroimaging.io](https://bids.neuroimaging.io/)
- **Your supervisor** at the MEG Core Facility

---

## File structure overview

After running the full pipeline, your project looks like this:

```
my_project/
├── rawdata/
│   ├── participants.tsv           ← subject list
│   ├── task-laser_events.json     ← trigger documentation
│   └── sub-P01/meg/*.fif          ← raw data (original, never modified)
│
└── derivatives/
    ├── prep/                      ← filtered + cleaned data
    ├── epochs/                    ← epoched data
    ├── source/sub-P01/            ← forward model, inverse, STCs, ROIs
    ├── contrasts/group/           ← grand averages + ROI statistics
    │   ├── roi_stats/             ← median ± MAD per ROI (read these!)
    │   └── *.stc                  ← brain maps
    ├── stats/permtest/            ← cluster test results
    └── logs/                      ← pipeline logs (check here if errors)
```

Everything in `rawdata/` is **never modified** by the pipeline.
All outputs go into `derivatives/`.
