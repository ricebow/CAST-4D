"""Crack skeleton and junction analysis with fixed parameters.

The same preprocessing, pruning, and junction criteria are used for all images.
No year-dependent compensation or temporal smoothing is applied.
"""

from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, convolve, label
from skimage.morphology import remove_small_objects, skeletonize


INPUT_FILES = [
    "2018crack2.png",
    "2018crack3.png",
    "2021crack2.png",
    "2021crack3.png",
    "2025crack2.png",
    "2025crack3.png",
]

MIN_OBJECT_SIZE = 200
MORPH_KERNEL_SIZE = (5, 5)
PRUNE_ITERATIONS = 10
JUNCTION_MERGE_DISTANCE = 15.0


def read_grayscale(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]],
        dtype=float,
    )
    if corners.mean() > 127:
        image = 255 - image

    binary = (image > 127).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask_bool = remove_small_objects(closed > 127, min_size=MIN_OBJECT_SIZE)
    return mask_bool, mask_bool.astype(np.uint8) * 255


def prune_skeleton(skeleton: np.ndarray) -> np.ndarray:
    skel = skeleton.astype(np.uint8).copy()
    endpoint_kernel = np.array(
        [[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8
    )

    for _ in range(PRUNE_ITERATIONS):
        neighborhood = convolve(skel, endpoint_kernel, mode="constant", cval=0)
        endpoints = (skel == 1) & (neighborhood == 11)
        if not np.any(endpoints):
            break
        skel[endpoints] = 0

    return skel.astype(bool)


def merge_junctions(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    for coord in coords:
        point = np.asarray(coord, dtype=float)
        if not merged:
            merged.append(coord)
            continue

        distances = [
            np.linalg.norm(point - np.asarray(existing, dtype=float))
            for existing in merged
        ]
        if all(distance > JUNCTION_MERGE_DISTANCE for distance in distances):
            merged.append(coord)

    return merged


def quantify(mask_bool: np.ndarray) -> dict:
    original = skeletonize(mask_bool)
    pruned = prune_skeleton(original)

    neighbor_kernel = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
    )
    neighbor_count = convolve(
        pruned.astype(np.uint8), neighbor_kernel, mode="constant", cval=0
    )

    junction_pixels = pruned & (neighbor_count >= 3)
    labeled, count = label(
        junction_pixels,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    coords = []
    for index in range(1, count + 1):
        coord = center_of_mass(junction_pixels, labeled, index)
        if np.isfinite(coord[0]) and np.isfinite(coord[1]):
            coords.append(coord)

    junctions = merge_junctions(coords)
    pruned_pixels = int(pruned.sum())

    return {
        "Original_Skeleton_Pixels": int(original.sum()),
        "Pruned_Skeleton_Pixels": pruned_pixels,
        "Skeleton_Length_per_1000px": pruned_pixels / 1000.0,
        "Junctions": len(junctions),
        "Skeleton": pruned,
    }


def make_heatmap(mask: np.ndarray, common_max: float) -> tuple[np.ndarray, float]:
    field = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    raw_max = float(field.max())

    if common_max <= 0:
        scaled = np.zeros_like(mask)
    else:
        scaled = np.clip(field / common_max * 255.0, 0, 255).astype(np.uint8)

    scaled = cv2.GaussianBlur(scaled, (3, 3), 0)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    heatmap[mask == 0] = 0
    return heatmap, raw_max


def parse_year_and_sample(path: Path) -> tuple[int, str]:
    year_match = re.search(r"20\d{2}", path.name)
    if not year_match:
        raise ValueError(f"Year not found in filename: {path.name}")

    sample_match = re.search(r"crack(\d+)", path.name, re.IGNORECASE)
    sample = f"crack{sample_match.group(1)}" if sample_match else path.stem
    return int(year_match.group()), sample


def create_yearly_summary(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby("Year", as_index=False)
        .agg(
            Skeleton_Length_Mean=("Skeleton_Length_per_1000px", "mean"),
            Skeleton_Length_Std=("Skeleton_Length_per_1000px", "std"),
            Junctions_Mean=("Junctions", "mean"),
            Junctions_Std=("Junctions", "std"),
            Crack_Area_Mean=("Crack_Area_px", "mean"),
            Max_Distance_Mean=("Max_Distance_px", "mean"),
        )
        .fillna(0)
    )


def plot_trend(summary: pd.DataFrame, output_dir: Path) -> None:
    df = summary.sort_values("Year")

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax2 = ax1.twinx()

    ax1.plot(
        df["Year"],
        df["Skeleton_Length_Mean"],
        marker="s",
        linestyle="--",
        label="Mean normalized skeleton length",
    )
    ax2.plot(
        df["Year"],
        df["Junctions_Mean"],
        marker="^",
        linestyle=":",
        label="Mean junction count",
    )

    ax1.set_xlabel("Observation year")
    ax1.set_ylabel("Skeleton length per 1000 pixels")
    ax2.set_ylabel("Junction count")
    ax1.set_xticks(df["Year"])
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    path = output_dir / "crack_skeleton_trend.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(input_dir: Path) -> None:
    input_dir = Path(input_dir)
    output_dir = input_dir / "CLD_Visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [input_dir / name for name in INPUT_FILES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))

    preprocessed = []
    for path in paths:
        image = read_grayscale(path)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        mask_bool, mask = preprocess(image)
        preprocessed.append((path, mask_bool, mask))

    fields = [cv2.distanceTransform(mask, cv2.DIST_L2, 5) for _, _, mask in preprocessed]
    common_max = max(float(field.max()) for field in fields)

    rows = []
    for path, mask_bool, mask in preprocessed:
        year, sample = parse_year_and_sample(path)
        quant = quantify(mask_bool)
        heatmap, max_distance = make_heatmap(mask, common_max)
        write_png(output_dir / f"Heatmap_{path.stem}.png", heatmap)

        rows.append(
            {
                "File": path.name,
                "Year": year,
                "Sample": sample,
                "Crack_Area_px": int(np.count_nonzero(mask)),
                "Original_Skeleton_Pixels": quant["Original_Skeleton_Pixels"],
                "Pruned_Skeleton_Pixels": quant["Pruned_Skeleton_Pixels"],
                "Skeleton_Length_per_1000px": quant["Skeleton_Length_per_1000px"],
                "Junctions": quant["Junctions"],
                "Max_Distance_px": max_distance,
            }
        )

    detail = pd.DataFrame(rows).sort_values(["Year", "Sample"])
    summary = create_yearly_summary(detail)

    detail.to_csv(
        output_dir / "crack_skeleton_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "crack_skeleton_yearly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_trend(summary, output_dir)

    print(detail.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


def main() -> None:
    input_dir = Path(__file__).resolve().parent / "序列图片"
    run_analysis(input_dir)


if __name__ == "__main__":
    main()
