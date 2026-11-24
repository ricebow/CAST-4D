"""Crack skeleton length estimation with per-image physical scaling.

The same preprocessing and pruning settings are applied to every year.
No temporal smoothing, monotonic constraint, or year-dependent correction is used.
"""

from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.morphology import remove_small_objects, skeletonize


MIN_OBJECT_SIZE = 300
PRUNE_ITERATIONS = 5

COORD_DATA = {
    2018: {"Tx": -8.071554, "Ty": -0.472470, "Tz": 8.669906,
           "Bx": -8.234630, "By": -0.524175, "Bz": 6.985566},
    2019: {"Tx": -8.077150, "Ty": -0.472001, "Tz": 8.667259,
           "Bx": -8.230042, "By": -0.525777, "Bz": 6.966388},
    2020: {"Tx": -8.091468, "Ty": -0.470769, "Tz": 8.671227,
           "Bx": -8.221524, "By": -0.527954, "Bz": 6.963020},
    2021: {"Tx": -8.075787, "Ty": -0.471733, "Tz": 8.677023,
           "Bx": -8.234669, "By": -0.523936, "Bz": 6.979701},
    2022: {"Tx": -8.080974, "Ty": -0.471174, "Tz": 8.670897,
           "Bx": -8.232693, "By": -0.524862, "Bz": 6.962561},
    2023: {"Tx": -8.075108, "Ty": -0.472270, "Tz": 8.664884,
           "Bx": -8.223694, "By": -0.527248, "Bz": 6.980631},
    2024: {"Tx": -8.079679, "Ty": -0.471148, "Tz": 8.679353,
           "Bx": -8.216639, "By": -0.529399, "Bz": 6.981081},
    2025: {"Tx": -8.087641, "Ty": -0.469879, "Tz": 8.679153,
           "Bx": -8.228488, "By": -0.525803, "Bz": 6.980288},
}


def read_grayscale(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def prune_skeleton(skeleton: np.ndarray, iterations: int = PRUNE_ITERATIONS) -> np.ndarray:
    out = skeleton.astype(np.uint8).copy()
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    for _ in range(iterations):
        neighbors = convolve(out, kernel, mode="constant", cval=0)
        endpoints = (out == 1) & (neighbors == 1)
        if not np.any(endpoints):
            break
        out[endpoints] = 0

    return out


def process_image(path: Path) -> dict | None:
    match = re.search(r"20\d{2}", path.name)
    if not match:
        return None

    year = int(match.group())
    if year not in COORD_DATA:
        return None

    image = read_grayscale(path)
    if image is None:
        return None

    if np.mean(image) > 127:
        image = 255 - image

    mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    mask_bool = remove_small_objects(mask > 0, min_size=MIN_OBJECT_SIZE)

    skeleton = skeletonize(mask_bool).astype(np.uint8)
    skeleton = prune_skeleton(skeleton)

    points = np.argwhere(skeleton > 0)
    if len(points) < 2:
        return None

    top = points[points[:, 0].argmin()]
    bottom = points[points[:, 0].argmax()]
    pixel_span = float(np.linalg.norm(top - bottom))
    if pixel_span <= 0:
        return None

    skeleton_pixels = int(skeleton.sum())

    c = COORD_DATA[year]
    p_top = np.array([c["Tx"], c["Ty"], c["Tz"]], dtype=float)
    p_bottom = np.array([c["Bx"], c["By"], c["Bz"]], dtype=float)

    span_3d = float(np.linalg.norm(p_top - p_bottom))
    scale = span_3d / pixel_span
    physical_length = skeleton_pixels * scale

    dz = abs(c["Tz"] - c["Bz"])
    angle_deg = float(np.degrees(np.arcsin(np.clip(dz / span_3d, 0.0, 1.0))))

    return {
        "Year": year,
        "Physical_Span_3D": span_3d,
        "Pixel_Span": pixel_span,
        "Scale_per_Pixel": scale,
        "Skeleton_Pixels": skeleton_pixels,
        "Estimated_Physical_Length": physical_length,
        "Inclination_deg": angle_deg,
    }


def main() -> None:
    input_dir = Path(__file__).resolve().parent / "序列图片"
    paths = sorted(input_dir.glob("灰度图裂缝主体*.png"))

    rows = []
    for path in paths:
        result = process_image(path)
        if result is not None:
            rows.append(result)

    if not rows:
        raise RuntimeError(f"No valid images found in: {input_dir}")

    df = pd.DataFrame(rows).sort_values("Year")
    output_path = input_dir / "crack_physical_length.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
