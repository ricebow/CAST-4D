"""Visualize detected crack growth between two registered years.

The overlay uses direct binary-mask differences. It does not create a dilated
"damage influence zone" or apply year-dependent enhancement.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from skimage.morphology import remove_small_objects, skeletonize


MIN_OBJECT_SIZE = 150


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


def binary_mask(path: Path) -> np.ndarray:
    image = read_grayscale(path)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    corners = np.array(
        [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]],
        dtype=float,
    )
    if corners.mean() > 127:
        binary = 255 - binary

    mask = remove_small_objects(binary > 127, min_size=MIN_OBJECT_SIZE)
    return mask.astype(np.uint8) * 255


def find_year_image(input_dir: Path, year: int) -> Path:
    matches = sorted(input_dir.glob(f"*{year}*.png"))
    matches = [path for path in matches if "灰度图裂缝主体" in path.name]
    if not matches:
        raise FileNotFoundError(f"No crack image found for {year} in {input_dir}")
    return matches[0]


def generate_growth_overlay(input_dir: Path, start_year: int = 2018, end_year: int = 2025) -> None:
    output_dir = input_dir / "Area_Growth_Analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_path = find_year_image(input_dir, start_year)
    end_path = find_year_image(input_dir, end_year)

    start = binary_mask(start_path) > 0
    end = binary_mask(end_path) > 0

    if start.shape != end.shape:
        raise ValueError("Input masks must have identical dimensions and be spatially registered.")

    new_pixels = end & ~start
    current_skeleton = skeletonize(end)

    canvas = np.full((*start.shape, 3), 255, dtype=np.uint8)
    canvas[start] = [120, 120, 120]
    canvas[new_pixels] = [0, 0, 220]
    canvas[current_skeleton] = [40, 40, 40]

    raw_path = output_dir / f"Growth_Overlay_{start_year}_{end_year}.png"
    write_png(raw_path, canvas)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    handles = [
        Patch(facecolor="#787878", label=f"Detected crack footprint ({start_year})"),
        Patch(facecolor="#DC0000", label=f"Newly detected pixels ({end_year} vs. {start_year})"),
        Line2D([0], [0], color="#282828", lw=2, label=f"Skeleton ({end_year})"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False)

    figure_path = output_dir / f"Growth_Annotated_{start_year}_{end_year}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved: {raw_path}")
    print(f"Saved: {figure_path}")


def main() -> None:
    input_dir = Path(__file__).resolve().parent / "序列图片"
    generate_growth_overlay(input_dir)


if __name__ == "__main__":
    main()
