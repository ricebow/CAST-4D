"""Plot crack metrics reported in Table 11 without temporal correction."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TABLE11_DATA = {
    "Year": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "CLD": [7.154, 7.242, 7.701, 7.824, 8.045, 8.505, 8.923, 10.915],
    "Junctions": [70, 69, 76, 79, 74, 78, 86, 106],
}


def plot_table11_crack_trend(output_dir: Path | None = None) -> None:
    df = pd.DataFrame(TABLE11_DATA)
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        df["Year"],
        df["CLD"],
        marker="s",
        linestyle="--",
        label=r"Crack Linear Density ($CLD$)",
    )
    line2 = ax2.plot(
        df["Year"],
        df["Junctions"],
        marker="^",
        linestyle=":",
        label=r"Topological Junction Count ($N_j$)",
    )

    ax1.set_xlabel("Observation year")
    ax1.set_ylabel(r"Crack Linear Density, $CLD$ (m$^{-1}$)")
    ax2.set_ylabel(r"Topological Junction Count ($N_j$)")
    ax1.set_xticks(df["Year"])
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Axis limits are derived from the data instead of being selected to alter
    # the apparent relationship between the two series.
    ax1.set_ylim(0, max(df["CLD"]) * 1.2)
    ax2.set_ylim(0, max(df["Junctions"]) * 1.2)

    for x, y in zip(df["Year"], df["CLD"]):
        ax1.annotate(f"{y:.3f}", (x, y), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=9)

    for x, y in zip(df["Year"], df["Junctions"]):
        ax2.annotate(f"{int(y)}", (x, y), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=9)

    handles = line1 + line2
    labels = [item.get_label() for item in handles]
    ax1.legend(handles, labels, loc="upper left")

    fig.tight_layout()
    png_path = output_dir / "Fig10_Table11.png"
    pdf_path = output_dir / "Fig10_Table11.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    plot_table11_crack_trend()
