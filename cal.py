"""Generate comparable crack distance-transform heatmaps.

All images use the same preprocessing settings and a shared normalization range.
No year-dependent gain is applied.
"""

from pathlib import Path
import re

import cv2
import numpy as np
from skimage.morphology import remove_small_objects


MIN_OBJECT_SIZE = 200
MORPH_KERNEL_SIZE = (5, 5)
GAUSSIAN_KERNEL_SIZE = (3, 3)


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


def preprocess(image: np.ndarray) -> np.ndarray:
    corners = np.array(
        [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]],
        dtype=float,
    )
    if corners.mean() > 127:
        image = 255 - image

    binary = (image > 127).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = remove_small_objects(closed > 127, min_size=MIN_OBJECT_SIZE)
    return mask.astype(np.uint8) * 255


def distance_field(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask, cv2.DIST_L2, 5)


def generate_heatmaps(input_dir: Path) -> None:
    output_dir = input_dir / "CLD_Visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("灰度图裂缝主体*.png"))
    records = []

    for path in paths:
        image = read_grayscale(path)
        if image is None:
            continue
        mask = preprocess(image)
        records.append((path, mask, distance_field(mask)))

    if not records:
        raise RuntimeError(f"No valid images found in: {input_dir}")

    global_max = max(float(field.max()) for _, _, field in records)
    if global_max <= 0:
        raise RuntimeError("All distance fields are empty.")

    for path, mask, field in records:
        scaled = np.clip(field / global_max * 255.0, 0, 255).astype(np.uint8)
        scaled = cv2.GaussianBlur(scaled, GAUSSIAN_KERNEL_SIZE, 0)

        heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
        heatmap[mask == 0] = 0

        year_match = re.search(r"20\d{2}", path.name)
        suffix = year_match.group() if year_match else path.stem
        output_path = output_dir / f"Heatmap_{suffix}.png"
        write_png(output_path, heatmap)

    print(f"Saved heatmaps to: {output_dir}")


def main() -> None:
    input_dir = Path(__file__).resolve().parent / "序列图片"
    generate_heatmaps(input_dir)


if __name__ == "__main__":
    main()
