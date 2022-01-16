from colorsys import hsv_to_rgb
from math import floor, sqrt
from pathlib import Path
from typing import Union

import seaborn as sns

from utils import Point

sns.set_style("darkgrid")


def plot_out_2d(
    out_file: Union[Path, str],
    output_file: Union[Path, str],
):
    point_ids = []
    x = []
    y = []
    cluster_ids = []
    with Path(out_file).open("r") as f:
        _ = f.readline()
        for line in f:
            if line.strip():
                line = line.strip()
                spt = line.split(",")
                point_ids.append(spt[0])
                x.append(float(spt[1]))
                y.append(float(spt[2]))
                cluster_ids.append(max(0, int(spt[-1])))  # plot noise points as 0s

    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    plot = sns.scatterplot(
        x=x,
        y=y,
        hue=cluster_ids,
        s=50,
        palette=_generate_sample_palette(cluster_ids),
    )

    if len(point_ids) < 50:
        for x_i, y_i, point_id in zip(x, y, point_ids):
            plot.text(
                x_i,
                y_i + 0.1,
                point_id,
                horizontalalignment="center",
                size="medium",
                color="black",
                weight="semibold",
            )

    plot.get_figure().savefig(str(output_file))


def plot_points_2d(
    points: list[Point],
    output_file: Union[Path, str],
):
    cluster_ids = [
        point.cluster_id if point.cluster_id != -1 else 0 for point in points
    ]
    x = [p.vals[0] for p in points]
    y = [p.vals[1] for p in points]

    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    plot = sns.scatterplot(
        x=x,
        y=y,
        hue=cluster_ids,
        s=50,
        palette=_generate_sample_palette(cluster_ids),
    )

    if len(points) < 50:
        for point in points:
            plot.text(
                point.vals[0],
                point.vals[1] + 0.1,
                point.id,
                horizontalalignment="center",
                size="medium",
                color="black",
                weight="semibold",
            )

    plot.get_figure().savefig(str(output_file))


def _generate_sample_palette(
    cluster_ids: list[int],
) -> dict[int, tuple[float, float, float]]:
    unique_ids = set(cluster_ids)
    n = len(unique_ids)

    phi = (1 + sqrt(5)) / 2
    hues = {i + 1: i * phi - floor(i * phi) for i in range(n)}
    palette = {i: hsv_to_rgb(hue, 1, 1) for i, hue in hues.items()}

    if 0 in unique_ids:
        palette[0] = (0.0, 0.0, 0.0)

    return palette
