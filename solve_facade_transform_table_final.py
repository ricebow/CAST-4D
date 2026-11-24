#!/usr/bin/env python3
"""Estimate a 3D similarity transform from control points and evaluate check points.

Model:
    P_M = scale * R * P_C + t

Control points are used for fitting. Check points are excluded from fitting and
used only for independent residual evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_COLS = ["x_c", "y_c", "z_c"]
TARGET_COLS = ["x_m", "y_m", "z_m"]
REQUIRED_COLS = ["id", "role"] + SOURCE_COLS + TARGET_COLS


def normalize_colname(col):
    text = str(col).strip().lower()
    for old, new in [("（", "("), ("）", ")"), ("-", "_"), (" ", "_"),
                     ("/", "_"), (".", "_")]:
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text


def standardize_columns(df):
    aliases = {
        "id": {"id", "point", "point_id", "name", "点号", "点名", "编号"},
        "role": {"role", "type", "类别", "类型", "点类型", "控制_检查", "控制点_检查点"},
        "x_c": {"x_c", "xc", "x_source", "source_x", "x_众源", "众源_x", "x众源", "x_crowd", "crowd_x"},
        "y_c": {"y_c", "yc", "y_source", "source_y", "y_众源", "众源_y", "y众源", "y_crowd", "crowd_y"},
        "z_c": {"z_c", "zc", "z_source", "source_z", "z_众源", "众源_z", "z众源", "z_crowd", "crowd_z"},
        "x_m": {"x_m", "xm", "x_target", "target_x", "x_ref", "ref_x", "x_参考", "参考_x", "x_倾斜", "倾斜_x", "x_high", "high_x"},
        "y_m": {"y_m", "ym", "y_target", "target_y", "y_ref", "ref_y", "y_参考", "参考_y", "y_倾斜", "倾斜_x", "y_high", "high_y"},
        "z_m": {"z_m", "zm", "z_target", "target_z", "z_ref", "ref_z", "z_参考", "参考_z", "z_倾斜", "倾斜_z", "z_high", "high_z"},
        "weight": {"weight", "w", "权重"},
    }

    rename = {}
    used = set()
    for old in df.columns:
        normalized = normalize_colname(old)
        matched = next(
            (standard for standard, names in aliases.items() if normalized in names),
            None,
        )
        if matched and matched not in used:
            rename[old] = matched
            used.add(matched)
    return df.rename(columns=rename)


def read_input_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        error = None
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
                break
            except Exception as exc:
                error = exc
        else:
            raise RuntimeError(f"Could not read CSV: {error}")
    else:
        raise ValueError("Supported input formats: .xlsx, .xls, .csv")

    df = standardize_columns(df.dropna(how="all").dropna(axis=1, how="all"))
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    if "weight" not in df.columns:
        df["weight"] = 1.0
    return df


def normalize_role(value):
    if pd.isna(value):
        return "control"

    text = str(value).strip().lower()
    if text in {"control", "train", "fit", "gcp", "控制点", "拟合点", "控制"}:
        return "control"
    if text in {"check", "test", "validation", "validate", "检查点", "检核点", "验证点", "检查"}:
        return "check"
    return text


def numeric_array(df, columns):
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any(axis=None):
        bad = df.loc[values.isna().any(axis=1), "id"].astype(str).tolist()
        raise ValueError(f"Invalid coordinates for points: {bad}")

    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"NaN or Inf found in columns: {columns}")
    return array


def weights_array(df):
    weights = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).to_numpy(float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be non-negative and have a positive sum.")
    return weights


def geometry_check(points, name, line_eps=1e-4, plane_eps=1e-4):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError(f"{name}: at least three points are required.")

    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[0] <= 0:
        raise ValueError(f"{name}: degenerate point configuration.")

    line_ratio = singular[1] / singular[0]
    plane_ratio = singular[2] / singular[0]

    if line_ratio < line_eps:
        raise ValueError(f"{name}: control points are nearly collinear.")

    return {
        "name": name,
        "count": int(len(points)),
        "singular_value_1": float(singular[0]),
        "singular_value_2": float(singular[1]),
        "singular_value_3": float(singular[2]),
        "line_ratio_sv2_sv1": float(line_ratio),
        "plane_ratio_sv3_sv1": float(plane_ratio),
        "near_planar": bool(plane_ratio < plane_eps),
    }


def solve_weighted_umeyama(source, target, weights=None):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be N x 3 arrays.")

    n = len(source)
    if n < 3:
        raise ValueError("At least three point pairs are required.")

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(weights) != n or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid weights.")

    w = weights / weights.sum()
    mu_x = np.sum(source * w[:, None], axis=0)
    mu_y = np.sum(target * w[:, None], axis=0)

    x = source - mu_x
    y = target - mu_y

    covariance = (y * w[:, None]).T @ x
    u, singular, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1

    rotation = u @ correction @ vt
    variance = np.sum(w * np.sum(x ** 2, axis=1))
    if variance <= 0:
        raise ValueError("Source point variance is zero.")

    scale = np.trace(np.diag(singular) @ correction) / variance
    translation = mu_y - scale * (rotation @ mu_x)

    return float(scale), rotation, translation


def transform_points(source, scale, rotation, translation):
    source = np.asarray(source, dtype=float)
    return scale * (source @ rotation.T) + np.asarray(translation, dtype=float)


def non_collinear(points, indices, eps=1e-10):
    sample = points[indices]
    return np.linalg.norm(np.cross(sample[1] - sample[0], sample[2] - sample[0])) > eps


def ransac_similarity(source, target, weights=None, threshold=0.05, max_iter=2000, random_seed=42):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(source)

    if n < 4:
        scale, rotation, translation = solve_weighted_umeyama(source, target, weights)
        residuals = np.linalg.norm(
            transform_points(source, scale, rotation, translation) - target,
            axis=1,
        )
        return scale, rotation, translation, np.arange(n), residuals

    rng = np.random.default_rng(random_seed)
    best_inliers = None
    best_score = (-1, np.inf)

    for _ in range(max_iter):
        indices = rng.choice(n, size=3, replace=False)
        if not non_collinear(source, indices) or not non_collinear(target, indices):
            continue

        try:
            scale, rotation, translation = solve_weighted_umeyama(
                source[indices], target[indices]
            )
        except Exception:
            continue

        residuals = np.linalg.norm(
            transform_points(source, scale, rotation, translation) - target,
            axis=1,
        )
        inliers = np.where(residuals <= threshold)[0]
        if len(inliers) < 3:
            continue

        rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        score = (len(inliers), -rmse)
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        raise RuntimeError("RANSAC did not find a valid inlier set.")

    fit_weights = weights[best_inliers] if weights is not None else None
    scale, rotation, translation = solve_weighted_umeyama(
        source[best_inliers], target[best_inliers], fit_weights
    )
    residuals = np.linalg.norm(
        transform_points(source, scale, rotation, translation) - target,
        axis=1,
    )
    return scale, rotation, translation, best_inliers, residuals


def residual_stats(residuals):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) == 0:
        return {
            "count": 0,
            "rmse_m": None,
            "mean_m": None,
            "median_m": None,
            "std_m": None,
            "max_m": None,
            "p95_m": None,
        }

    return {
        "count": int(len(residuals)),
        "rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
        "mean_m": float(np.mean(residuals)),
        "median_m": float(np.median(residuals)),
        "std_m": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "max_m": float(np.max(residuals)),
        "p95_m": float(np.percentile(residuals, 95)),
    }


def transform_matrix(scale, rotation, translation):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def main():
    parser = argparse.ArgumentParser(description="3D similarity transform for facade control/check points.")
    parser.add_argument("--input", required=True, help="Input Excel or CSV table.")
    parser.add_argument("--outdir", default="facade_transform_result")
    parser.add_argument("--ransac_threshold", type=float, default=None,
                        help="Optional RANSAC inlier threshold in metres.")
    parser.add_argument("--ransac_iter", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input_table(args.input)
    df["role"] = df["role"].apply(normalize_role)
    df["id"] = df["id"].astype(str)

    source = numeric_array(df, SOURCE_COLS)
    target = numeric_array(df, TARGET_COLS)
    weights = weights_array(df)

    control_mask = df["role"].eq("control").to_numpy()
    check_mask = df["role"].eq("check").to_numpy()

    if control_mask.sum() < 3:
        raise ValueError("At least three control points are required.")

    source_control = source[control_mask]
    target_control = target[control_mask]
    weight_control = weights[control_mask]

    source_geometry = geometry_check(source_control, "source_control_points")
    target_geometry = geometry_check(target_control, "target_control_points")

    if args.ransac_threshold is None:
        scale, rotation, translation = solve_weighted_umeyama(
            source_control, target_control, weight_control
        )
        fit_mask = control_mask.copy()
        inlier_global = np.where(control_mask)[0]
    else:
        scale, rotation, translation, inlier_local, _ = ransac_similarity(
            source_control,
            target_control,
            weight_control,
            threshold=args.ransac_threshold,
            max_iter=args.ransac_iter,
            random_seed=args.random_seed,
        )
        control_indices = np.where(control_mask)[0]
        inlier_global = control_indices[inlier_local]
        fit_mask = np.zeros(len(df), dtype=bool)
        fit_mask[inlier_global] = True

    predicted = transform_points(source, scale, rotation, translation)
    residual_vector = predicted - target
    residual_norm = np.linalg.norm(residual_vector, axis=1)

    result = df.copy()
    result[["x_m_pred", "y_m_pred", "z_m_pred"]] = predicted
    result[["vx_m", "vy_m", "vz_m"]] = residual_vector
    result["residual_3d_m"] = residual_norm
    result["used_for_fit"] = fit_mask

    matrix = transform_matrix(scale, rotation, translation)
    summary = {
        "model": "P_M = scale * R * P_C + t",
        "target_unit": "m",
        "scale": scale,
        "rotation_matrix": rotation.tolist(),
        "translation_vector": translation.tolist(),
        "transform_matrix_4x4": matrix.tolist(),
        "point_count_total": int(len(df)),
        "point_count_control": int(control_mask.sum()),
        "point_count_check": int(check_mask.sum()),
        "point_count_used_for_fit": int(fit_mask.sum()),
        "geometry_source_control": source_geometry,
        "geometry_target_control": target_geometry,
        "residual_stats_fit_points": residual_stats(residual_norm[fit_mask]),
        "residual_stats_control_points": residual_stats(residual_norm[control_mask]),
        "residual_stats_check_points": residual_stats(residual_norm[check_mask]),
        "ransac": {
            "enabled": args.ransac_threshold is not None,
            "threshold_m": args.ransac_threshold,
            "iterations": args.ransac_iter if args.ransac_threshold is not None else None,
            "inlier_ids": result.iloc[inlier_global]["id"].tolist(),
        },
    }

    with open(output_dir / "transform_params.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(rotation).to_csv(output_dir / "rotation_matrix.csv", index=False, header=False)
    pd.DataFrame(matrix).to_csv(output_dir / "transform_matrix_4x4.csv", index=False, header=False)
    result.to_csv(output_dir / "transformed_points.csv", index=False, encoding="utf-8-sig")

    groups = {
        "fit_points": residual_norm[fit_mask],
        "control_points": residual_norm[control_mask],
        "check_points": residual_norm[check_mask],
    }
    rows = [{"group": name, **residual_stats(values)} for name, values in groups.items()]
    pd.DataFrame(rows).to_csv(output_dir / "residual_summary.csv", index=False)

    pd.DataFrame([source_geometry, target_geometry]).to_csv(
        output_dir / "geometry_report.csv", index=False
    )

    print(json.dumps(summary["residual_stats_check_points"], indent=2))
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
