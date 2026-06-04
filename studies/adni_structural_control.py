from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np


DEFAULT_PREDICTIONS = Path("adni_r2tnet_predictions.csv")
DEFAULT_COVARIATES = Path("adni_covariates.csv")
DEFAULT_STRUCTURAL = Path("adni_structural_features.csv")
DEFAULT_OUT_DIR = Path("runs/adni_structural_control")
DIAGNOSIS_ORDER = ["CN", "MCI", "AD"]
SCORE_COLUMNS = ["pred_wm0", "pred_wm2", "pred_rel", "pc1"]


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("This analysis requires pandas. Install pandas to run studies.adni_structural_control.") from exc
    return pd


def _require_statsmodels():
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("ANCOVA analyses require statsmodels. Install statsmodels to run analyses 1 and 2.") from exc
    return sm, smf


def _clean_column(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _load_csv(path):
    pd = _require_pandas()
    df = pd.read_csv(path)
    df = df.rename(columns={col: _clean_column(col) for col in df.columns})
    if "subject" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"subject": "subject_id"})
    if "scanner" in df.columns and "site" not in df.columns:
        df = df.rename(columns={"scanner": "site"})
    if "icv" in df.columns and "tiv" not in df.columns:
        df = df.rename(columns={"icv": "tiv"})
    if "csf_volume" in df.columns and "ventricle_volume" not in df.columns:
        df = df.rename(columns={"csf_volume": "ventricle_volume"})
    if "mean_cortical_thickness" not in df.columns and "cortical_thickness" in df.columns:
        df = df.rename(columns={"cortical_thickness": "mean_cortical_thickness"})
    if "subject_id" not in df.columns:
        raise ValueError(f"{path} must contain subject_id")
    df["subject_id"] = df["subject_id"].astype(str)
    return df


def _check_columns(df, columns, label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _merge_tables(predictions, covariates, structural, diagnosis_order):
    pd = _require_pandas()
    pred = _load_csv(predictions)
    cov = _load_csv(covariates)
    struct = _load_csv(structural)

    _check_columns(pred, ["subject_id", "diagnosis", "pred_wm0", "pred_wm2", "pred_rel"], predictions)
    _check_columns(cov, ["subject_id", "age", "sex", "site", "mean_fd"], covariates)
    _check_columns(
        struct,
        ["subject_id", "tiv", "total_gray_volume", "left_hippocampus", "right_hippocampus", "ventricle_volume"],
        structural,
    )

    df = pred.merge(cov, on="subject_id", how="inner").merge(struct, on="subject_id", how="inner")
    df["diagnosis"] = df["diagnosis"].astype(str).str.upper().str.strip()
    unknown = sorted(set(df["diagnosis"].dropna()) - set(diagnosis_order))
    if unknown:
        raise ValueError(f"Unexpected diagnosis labels {unknown}; expected {diagnosis_order}")
    df["diagnosis"] = pd.Categorical(df["diagnosis"], categories=diagnosis_order, ordered=True)

    df["hippocampus_volume"] = df["left_hippocampus"] + df["right_hippocampus"]
    df["gray_norm"] = df["total_gray_volume"] / df["tiv"]
    df["hippo_norm"] = df["hippocampus_volume"] / df["tiv"]
    df["ventricle_norm"] = df["ventricle_volume"] / df["tiv"]
    return df


def _score_columns(df):
    return [col for col in SCORE_COLUMNS if col in df.columns]


def _base_covariates():
    return ["age", "C(sex)", "C(site)", "mean_fd", "gray_norm", "hippo_norm", "ventricle_norm"]


def _analysis_df(df, columns):
    rows = df[["subject_id", *columns]].replace([np.inf, -np.inf], np.nan).dropna()
    return rows.copy()


def _diag_term(anova_index):
    for key in anova_index:
        if str(key).startswith("C(diagnosis"):
            return key
    raise KeyError("diagnosis term not found in ANOVA table")


def _diagnosis_formula(score):
    return f"{score} ~ C(diagnosis, Treatment(reference='CN')) + " + " + ".join(_base_covariates())


def _diagnosis_param(model, group):
    if group == "CN":
        return None
    matches = [name for name in model.params.index if name.startswith("C(diagnosis") and f"[T.{group}]" in name]
    if len(matches) != 1:
        raise KeyError(f"Could not find diagnosis coefficient for {group}; matches={matches}")
    return matches[0]


def _contrast_vector(model, left, right):
    vec = np.zeros(len(model.params))
    names = list(model.params.index)
    for group, sign in ((left, 1.0), (right, -1.0)):
        param = _diagnosis_param(model, group)
        if param is not None:
            vec[names.index(param)] += sign
    return vec


def run_ancova(df):
    pd = _require_pandas()
    sm, smf = _require_statsmodels()
    effect_rows = []
    pairwise_rows = []
    covariates = ["diagnosis", "age", "sex", "site", "mean_fd", "gray_norm", "hippo_norm", "ventricle_norm"]

    for score in _score_columns(df):
        data = _analysis_df(df, [score, *covariates])
        if data["diagnosis"].nunique() < 2:
            continue
        model = smf.ols(_diagnosis_formula(score), data=data).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        term = _diag_term(anova.index)
        ss_diag = float(anova.loc[term, "sum_sq"])
        ss_resid = float(anova.loc["Residual", "sum_sq"])
        effect_rows.append(
            {
                "score": score,
                "n": int(len(data)),
                "f_stat": float(anova.loc[term, "F"]),
                "p_value": float(anova.loc[term, "PR(>F)"]),
                "partial_eta2": ss_diag / (ss_diag + ss_resid) if ss_diag + ss_resid > 0 else np.nan,
            }
        )

        for left, right in (("CN", "MCI"), ("MCI", "AD"), ("CN", "AD")):
            try:
                test = model.t_test(_contrast_vector(model, left, right))
            except KeyError:
                continue
            pairwise_rows.append(
                {
                    "score": score,
                    "contrast": f"{left}_minus_{right}",
                    "estimate": float(test.effect[0]),
                    "t_stat": float(test.tvalue[0][0]),
                    "p_value": float(test.pvalue),
                }
            )
    return pd.DataFrame(effect_rows), pd.DataFrame(pairwise_rows)


def _assoc_formula(target, predictor):
    return f"{target} ~ {predictor} + " + " + ".join(_base_covariates())


def _reduced_assoc_formula(target):
    return f"{target} ~ " + " + ".join(_base_covariates())


def run_digit_span(df):
    pd = _require_pandas()
    _, smf = _require_statsmodels()
    if "digit_span_backward" not in df.columns:
        return pd.DataFrame()

    rows = []
    covariates = ["digit_span_backward", "age", "sex", "site", "mean_fd", "gray_norm", "hippo_norm", "ventricle_norm"]
    for predictor in _score_columns(df):
        data = _analysis_df(df, [predictor, *covariates])
        if len(data) < 10:
            continue
        full = smf.ols(_assoc_formula("digit_span_backward", predictor), data=data).fit()
        reduced = smf.ols(_reduced_assoc_formula("digit_span_backward"), data=data).fit()
        sse_full = float(np.sum(full.resid**2))
        sse_reduced = float(np.sum(reduced.resid**2))
        partial_r2 = (sse_reduced - sse_full) / sse_reduced if sse_reduced > 0 else np.nan
        rows.append(
            {
                "predictor": predictor,
                "n": int(len(data)),
                "beta": float(full.params[predictor]),
                "t_stat": float(full.tvalues[predictor]),
                "p_value": float(full.pvalues[predictor]),
                "partial_r2": partial_r2,
                "partial_r": math.copysign(math.sqrt(max(partial_r2, 0.0)), float(full.params[predictor])),
            }
        )
    return pd.DataFrame(rows)


def _optional_structural_features(df):
    base = ["gray_norm", "hippo_norm", "ventricle_norm"]
    optional = []
    explicit = [
        "mean_cortical_thickness",
        "entorhinal_thickness",
        "temporal_lobe_gray_matter_volume",
        "whole_brain_volume",
    ]
    for col in explicit:
        if col in df.columns:
            if col.endswith("volume") and f"{col}_norm" not in df.columns:
                df[f"{col}_norm"] = df[col] / df["tiv"]
                optional.append(f"{col}_norm")
            else:
                optional.append(col)
    return base + [col for col in optional if col not in base]


def _signature_columns(df):
    def key(col):
        return int(col.rsplit("_", 1)[1])

    return sorted([col for col in df.columns if re.fullmatch(r"signature_\d+", col)], key=key)


def _classification_data(df, features):
    y_col = "diagnosis"
    data = df[["subject_id", y_col, *features]].replace([np.inf, -np.inf], np.nan).dropna()
    return data["subject_id"].astype(str).to_numpy(), data[y_col].astype(str).to_numpy(), data[features].to_numpy(dtype=float)


def _aligned_proba(model, x, classes):
    proba = model.predict_proba(x)
    out = np.zeros((x.shape[0], len(classes)), dtype=float)
    for j, label in enumerate(model.named_steps["clf"].classes_):
        out[:, classes.index(label)] = proba[:, j]
    return out


def _safe_auc(y, proba, classes):
    from sklearn.metrics import roc_auc_score

    if len(set(y)) < len(classes):
        return np.nan
    try:
        return float(roc_auc_score(y, proba, labels=classes, multi_class="ovr", average="macro"))
    except ValueError:
        return np.nan


def _metrics(y, pred, proba, classes):
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_ovr_auc": _safe_auc(y, proba, classes),
    }


def _fit_predict_cv(subject_ids, y, x, classes, n_splits, seed, signature_pcs=None):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    counts = {label: int(np.sum(y == label)) for label in classes}
    usable_splits = min(n_splits, min(counts.values()))
    if usable_splits < 2:
        raise ValueError(f"Need at least two subjects per diagnosis class for CV; class counts={counts}")

    pred = np.empty_like(y, dtype=object)
    proba = np.zeros((len(y), len(classes)), dtype=float)
    fold_id = np.zeros(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits=usable_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        steps = [("scaler", StandardScaler())]
        if signature_pcs is not None:
            n_pc = min(signature_pcs, x.shape[1], len(train_idx))
            steps.append(("pca", PCA(n_components=n_pc, random_state=seed)))
        steps.append(
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                ),
            )
        )
        model = Pipeline(steps)
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
        proba[test_idx] = _aligned_proba(model, x[test_idx], classes)
        fold_id[test_idx] = fold

    return {
        "subject_id": subject_ids,
        "y_true": y,
        "y_pred": pred,
        "proba": proba,
        "fold": fold_id,
        "metrics": _metrics(y, pred, proba, classes),
    }


def _confusion_rows(name, result, classes):
    from sklearn.metrics import confusion_matrix

    mat = confusion_matrix(result["y_true"], result["y_pred"], labels=classes)
    rows = []
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            rows.append({"model": name, "true_label": true_label, "predicted_label": pred_label, "n": int(mat[i, j])})
    return rows


def _oof_rows(name, result, classes):
    rows = []
    for i, sid in enumerate(result["subject_id"]):
        row = {
            "model": name,
            "subject_id": sid,
            "fold": int(result["fold"][i]),
            "diagnosis": result["y_true"][i],
            "prediction": result["y_pred"][i],
        }
        for j, label in enumerate(classes):
            row[f"p_{label}"] = float(result["proba"][i, j])
        rows.append(row)
    return rows


def _bootstrap_diffs(results, comparisons, classes, n_bootstrap, seed):
    pd = _require_pandas()
    rng = np.random.default_rng(seed)
    rows = []
    for left, right in comparisons:
        if left not in results or right not in results:
            continue
        left_res = results[left]
        right_res = results[right]
        common = sorted(set(left_res["subject_id"]) & set(right_res["subject_id"]))
        if not common:
            continue
        left_idx = {sid: i for i, sid in enumerate(left_res["subject_id"])}
        right_idx = {sid: i for i, sid in enumerate(right_res["subject_id"])}
        li = np.array([left_idx[sid] for sid in common])
        ri = np.array([right_idx[sid] for sid in common])
        y = left_res["y_true"][li]
        if not np.array_equal(y, right_res["y_true"][ri]):
            raise ValueError(f"Mismatched labels in bootstrap comparison {left} vs {right}")

        observed_left = _metrics(y, left_res["y_pred"][li], left_res["proba"][li], classes)
        observed_right = _metrics(y, right_res["y_pred"][ri], right_res["proba"][ri], classes)
        draws = {metric: [] for metric in observed_left}
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(common), size=len(common))
            yy = y[idx]
            for metric in draws:
                if metric == "macro_ovr_auc" and len(set(yy)) < len(classes):
                    continue
                l_val = _metrics(yy, left_res["y_pred"][li][idx], left_res["proba"][li][idx], classes)[metric]
                r_val = _metrics(yy, right_res["y_pred"][ri][idx], right_res["proba"][ri][idx], classes)[metric]
                if not np.isnan(l_val) and not np.isnan(r_val):
                    draws[metric].append(l_val - r_val)

        for metric, values in draws.items():
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "comparison": f"{left}_minus_{right}",
                    "metric": metric,
                    "n": int(len(common)),
                    "observed_diff": observed_left[metric] - observed_right[metric],
                    "bootstrap_mean_diff": float(np.mean(arr)) if arr.size else np.nan,
                    "ci_low": float(np.quantile(arr, 0.025)) if arr.size else np.nan,
                    "ci_high": float(np.quantile(arr, 0.975)) if arr.size else np.nan,
                    "bootstrap_samples": int(arr.size),
                }
            )
    return pd.DataFrame(rows)


def run_classification(df, n_splits, seed, signature_pcs, n_bootstrap):
    pd = _require_pandas()
    classes = DIAGNOSIS_ORDER
    structural_features = _optional_structural_features(df)
    r2t_features = [col for col in ["pred_wm0", "pred_wm2", "pred_rel", "pc1"] if col in df.columns]
    signature_features = _signature_columns(df)
    specs = {
        "structural_only": structural_features,
        "r2t_score_only": r2t_features,
        "structural_plus_r2t": structural_features + r2t_features,
    }
    if signature_features and signature_pcs > 0:
        specs[f"r2t_signature_{signature_pcs}pc"] = signature_features

    results = {}
    metric_rows = []
    confusion_rows = []
    oof_rows = []
    for name, features in specs.items():
        if not features:
            continue
        subject_ids, y, x = _classification_data(df, features)
        if len(subject_ids) == 0:
            continue
        pcs = signature_pcs if name.startswith("r2t_signature") else None
        result = _fit_predict_cv(subject_ids, y, x, classes, n_splits, seed, signature_pcs=pcs)
        results[name] = result
        metric_rows.append({"model": name, "n": int(len(y)), "n_features": int(len(features)), **result["metrics"]})
        confusion_rows.extend(_confusion_rows(name, result, classes))
        oof_rows.extend(_oof_rows(name, result, classes))

    boot = _bootstrap_diffs(
        results,
        [("structural_plus_r2t", "structural_only"), ("r2t_score_only", "structural_only")],
        classes,
        n_bootstrap,
        seed,
    )
    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows), pd.DataFrame(oof_rows), boot


def parse_control_predictions(raw):
    out = []
    for item in raw:
        if "=" not in item:
            raise ValueError("--control-predictions entries must be name=path")
        name, path = item.split("=", 1)
        out.append((_clean_column(name), Path(path)))
    return out


def run_temporal_prediction_controls(args, base_df):
    pd = _require_pandas()
    effect_rows = []
    digit_rows = []
    class_rows = []
    controls = [("original", args.predictions)] + parse_control_predictions(args.control_predictions)
    for name, path in controls:
        df = _merge_tables(path, args.covariates, args.structural, DIAGNOSIS_ORDER)
        effects, _ = run_ancova(df)
        digit = run_digit_span(df)
        metrics, _, _, _ = run_classification(df, args.n_splits, args.seed, 0, 0)
        if not effects.empty:
            effects.insert(0, "condition", name)
            effect_rows.append(effects)
        if not digit.empty:
            digit.insert(0, "condition", name)
            digit_rows.append(digit)
        if not metrics.empty:
            metrics = metrics[metrics["model"] == "r2t_score_only"].copy()
            metrics.insert(0, "condition", name)
            class_rows.append(metrics)
    return (
        pd.concat(effect_rows, ignore_index=True) if effect_rows else pd.DataFrame(),
        pd.concat(digit_rows, ignore_index=True) if digit_rows else pd.DataFrame(),
        pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame(),
    )


def make_temporal_control_inputs(args):
    import torch
    from tqdm import tqdm

    if args.rsfmri_dir is None:
        raise ValueError("--rsfmri-dir is required with --write-temporal-controls")
    source = args.rsfmri_dir
    mean_dir = args.temporal_control_dir / "temporal_mean"
    shuffle_dir = args.temporal_control_dir / "time_shuffled"
    mean_dir.mkdir(parents=True, exist_ok=True)
    shuffle_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    files = sorted(source.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {source}")

    for path in tqdm(files, desc="temporal controls"):
        x = torch.load(path, map_location="cpu")
        mean_x = x.mean(dim=-1, keepdim=True).expand_as(x).clone()
        perm = torch.tensor(rng.permutation(x.shape[-1]), dtype=torch.long)
        shuffle_x = x.index_select(-1, perm)
        torch.save(mean_x, mean_dir / path.name)
        torch.save(shuffle_x, shuffle_dir / path.name)
    print(f"wrote temporal-mean inputs to {mean_dir}")
    print(f"wrote time-shuffled inputs to {shuffle_dir}")


def _rankdata(values):
    pd = _require_pandas()
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _spearman(x, y):
    xr = _rankdata(x)
    yr = _rankdata(y)
    if xr.size < 2 or float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _top_mask(values, frac):
    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    frac = min(max(frac, 0.0), 1.0)
    threshold = np.quantile(values, 1.0 - frac)
    return values >= threshold


def run_mni_overlap(args):
    if args.relevance_map is None or args.atrophy_map is None:
        return None
    import nibabel as nib

    rel = np.asarray(nib.load(args.relevance_map).get_fdata(), dtype=float)
    atrophy = np.asarray(nib.load(args.atrophy_map).get_fdata(), dtype=float)
    if rel.shape != atrophy.shape:
        raise ValueError(f"relevance and atrophy maps have different shapes: {rel.shape} vs {atrophy.shape}")

    finite = np.isfinite(rel) & np.isfinite(atrophy)
    if args.gray_mask is not None:
        gray = np.asarray(nib.load(args.gray_mask).get_fdata()) > 0
        finite &= gray

    rel_values = np.abs(rel[finite])
    atrophy_values = np.abs(atrophy[finite]) if args.abs_atrophy else atrophy[finite]
    rel_top = _top_mask(rel_values, args.top_fraction)
    atrophy_top = _top_mask(np.abs(atrophy[finite]), args.top_fraction)
    dice = 2.0 * np.sum(rel_top & atrophy_top) / (np.sum(rel_top) + np.sum(atrophy_top)) if np.sum(rel_top) + np.sum(atrophy_top) else np.nan

    rows = [
        {
            "analysis": "gray_mask_overlap" if args.gray_mask else "whole_map_overlap",
            "n_voxels": int(rel_values.size),
            "spearman_abs_relevance_vs_atrophy": _spearman(rel_values, atrophy_values),
            "top_fraction": float(args.top_fraction),
            "dice_top_relevance_top_atrophy": float(dice),
        }
    ]

    for label, path in (("gray_matter", args.gray_mask), ("white_matter", args.white_mask), ("csf_or_ventricles", args.csf_mask)):
        if path is None:
            continue
        mask = np.asarray(nib.load(path).get_fdata()) > 0
        if mask.shape != rel.shape:
            raise ValueError(f"{label} mask has shape {mask.shape}; expected {rel.shape}")
        denom = float(np.sum(np.abs(rel[np.isfinite(rel)])))
        numer = float(np.sum(np.abs(rel[mask & np.isfinite(rel)])))
        rows.append(
            {
                "analysis": f"relevance_fraction_{label}",
                "n_voxels": int(np.sum(mask & np.isfinite(rel))),
                "spearman_abs_relevance_vs_atrophy": np.nan,
                "top_fraction": np.nan,
                "dice_top_relevance_top_atrophy": np.nan,
                "relevance_fraction": numer / denom if denom > 0 else np.nan,
            }
        )
    return _require_pandas().DataFrame(rows)


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ADNI structural-atrophy control analyses for R2T-Net",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--covariates", type=Path, default=DEFAULT_COVARIATES)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signature-pcs", type=int, default=25)
    parser.add_argument(
        "--control-predictions",
        action="append",
        default=[],
        help="Optional temporal-control predictions as name=csv, e.g. temporal_mean=adni_temporal_mean_predictions.csv",
    )
    parser.add_argument("--write-temporal-controls", action="store_true")
    parser.add_argument("--rsfmri-dir", type=Path, default=Path("adni_rsfmri_windows"))
    parser.add_argument("--temporal-control-dir", type=Path, default=Path("adni_temporal_controls"))
    parser.add_argument("--relevance-map", type=Path)
    parser.add_argument("--atrophy-map", type=Path)
    parser.add_argument("--gray-mask", type=Path)
    parser.add_argument("--white-mask", type=Path)
    parser.add_argument("--csf-mask", type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--abs-atrophy", action="store_true", help="Use absolute atrophy t-values in the Spearman overlap.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.write_temporal_controls:
        make_temporal_control_inputs(args)

    df = _merge_tables(args.predictions, args.covariates, args.structural, DIAGNOSIS_ORDER)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(df, args.out_dir / "merged_adni_structural_control.csv")

    effects, pairwise = run_ancova(df)
    write_csv(effects, args.out_dir / "ancova_diagnosis_effects.csv")
    write_csv(pairwise, args.out_dir / "ancova_pairwise_group_differences.csv")

    digit = run_digit_span(df)
    write_csv(digit, args.out_dir / "digit_span_adjusted_associations.csv")

    metrics, confusion, oof, boot = run_classification(df, args.n_splits, args.seed, args.signature_pcs, args.bootstrap)
    write_csv(metrics, args.out_dir / "classification_metrics.csv")
    write_csv(confusion, args.out_dir / "classification_confusion_matrices.csv")
    write_csv(oof, args.out_dir / "classification_oof_predictions.csv")
    write_csv(boot, args.out_dir / "classification_bootstrap_differences.csv")

    if args.control_predictions:
        control_effects, control_digit, control_class = run_temporal_prediction_controls(args, df)
        write_csv(control_effects, args.out_dir / "temporal_control_diagnosis_effects.csv")
        write_csv(control_digit, args.out_dir / "temporal_control_digit_span_associations.csv")
        write_csv(control_class, args.out_dir / "temporal_control_classification_metrics.csv")

    overlap = run_mni_overlap(args)
    if overlap is not None:
        write_csv(overlap, args.out_dir / "mni_relevance_atrophy_overlap.csv")


if __name__ == "__main__":
    main()
