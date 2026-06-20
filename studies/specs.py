from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScanPattern:
    role: str
    pattern: str
    input_kind: str
    n_frames: int


@dataclass(frozen=True)
class StudySpec:
    name: str
    source_root: Path
    data_dir: Path
    target_cols: tuple[str, ...]
    scans: tuple[ScanPattern, ...]
    default_ckpt: Path | None = None


TARGETS = ("wm_0bk", "wm_2bk", "wm_diff", "rel")

HCP_SCANS = (
    ScanPattern("rest", "rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii", "grayord", 1200),
    ScanPattern("rest", "rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii", "grayord", 1200),
    ScanPattern("rest", "rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii", "grayord", 1200),
    ScanPattern("rest", "rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii", "grayord", 1200),
    ScanPattern("task", "tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii", "grayord", 405),
    ScanPattern("task", "tfMRI_WM_RL_Atlas_MSMAll.dtseries.nii", "grayord", 405),
    ScanPattern("task", "tfMRI_RELATIONAL_LR_Atlas_MSMAll.dtseries.nii", "grayord", 232),
    ScanPattern("task", "tfMRI_RELATIONAL_RL_Atlas_MSMAll.dtseries.nii", "grayord", 232),
)

CHCP_SCANS = (
    ScanPattern("rest", "rfMRI_REST1_AP_Atlas_hp2000_clean.dtseries.nii", "grayord", 1200),
    ScanPattern("rest", "rfMRI_REST1_PA_Atlas_hp2000_clean.dtseries.nii", "grayord", 1200),
    ScanPattern("rest", "rfMRI_REST2_AP_Atlas_hp2000_clean.dtseries.nii", "grayord", 1200),
    ScanPattern("rest", "rfMRI_REST2_PA_Atlas_hp2000_clean.dtseries.nii", "grayord", 1200),
    ScanPattern("task", "tfMRI_WM_*Atlas*.dtseries.nii", "grayord", 405),
    ScanPattern("task", "tfMRI_RELATIONAL_*Atlas*.dtseries.nii", "grayord", 232),
)

ADNI_SCANS = (
    ScanPattern("rest", "*rest*.nii.gz", "vol", 300),
    ScanPattern("rest", "*rsfMRI*.nii.gz", "vol", 300),
    ScanPattern("rest", "*rfMRI*.nii.gz", "vol", 300),
)

ADNI_CLASS_TARGETS = ("AD", "CN", "MCI")

STUDIES = {
    "hcp": StudySpec("hcp", Path("/path/to/HCP_1200"), Path("data/hcp_grayord"), TARGETS, HCP_SCANS),
    "chcp": StudySpec("chcp", Path("/path/to/CHCP_DB"), Path("data/chcp_grayord"), TARGETS, CHCP_SCANS, Path("runs/hcp_r2t/last.pt")),
    "adni": StudySpec("adni", Path("/path/to/ADNI"), Path("data/adni_raw4d"), ("wm_0bk", "wm_2bk", "rel"), ADNI_SCANS, Path("runs/hcp_raw4d_r2t/last.pt")),
    "adni_classification": StudySpec("adni_classification", Path("/path/to/ADNI"), Path("data/adni_classification_raw4d"), ADNI_CLASS_TARGETS, ADNI_SCANS, Path("runs/hcp_raw4d_r2t/last.pt")),
}

REPRESENTATIONS = {
    "raw4d": {"data_dir": "data/hcp_raw4d", "input_kind": "vol", "encoder": "swift", "sequence_length": 300, "signature_dim": 1024},
    "grayord": {"data_dir": "data/hcp_grayord", "input_kind": "grayord", "encoder": "timesformer", "sequence_length": 300, "signature_dim": 1024},
    "ca718": {"data_dir": "data/hcp_ca718", "input_kind": "roi", "encoder": "timesformer", "sequence_length": 300, "signature_dim": 1024},
    "mmp379": {"data_dir": "data/hcp_mmp379", "input_kind": "roi", "encoder": "timesformer", "sequence_length": 300, "signature_dim": 1024},
}

TRAINING_MODES = {
    "rs_only": {"training_mode": "rest_only", "eval_role": "rest", "disable_contrastive": True},
    "t_only": {"training_mode": "task_only", "eval_role": "task", "disable_contrastive": True},
    "synthetic_t": {"training_mode": "synthetic_task", "eval_role": "rest", "lambda_synthetic": 1.0},
    "r2t": {"training_mode": "r2t", "eval_role": "rest", "lambda_contrast": 0.5, "supervised_view": "average"},
}

SCAN_LENGTHS = (100, 150, 200, 300, 400, 600, 800, 1200)
SIGNATURE_DIMS = (256, 512, 1024, 2048)
SEEDS = (42, 123, 999)
