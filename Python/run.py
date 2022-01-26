import json
import sys
import time
from pathlib import Path

import click
from clustering_metrics import davies_bouldin, mean_silhouette_coefficient, purity, rand
from dbscan import dbscan
from dbscrn import dbscrn
from plot import plot_out_2d
from utils import Point, load_points

sys.path.extend(str(Path(__file__).parent))


@click.command()
@click.option(
    "-d",
    "--dataset_path",
    type=str,
    required=True,
    help="Path to dataset to use.",
)
@click.option(
    "-o",
    "--output_dir",
    type=Path,
    required=True,
    help="Directory where output files will be saved.",
)
@click.option(
    "-a",
    "--algorithm",
    type=click.Choice(["dbscan", "dbscrn"]),
    required=True,
    help="Type of algorithm to use.",
)
@click.option(
    "--ti",
    type=bool,
    default=False,
    is_flag=True,
    help="If set, will use triangle inequality to optimize runtime of the DBSCRN algorithm.",
)
@click.option("-k", type=int, default=3, help="'k' parameter in DBSCANRN algorithm.")
@click.option(
    "-s", "--min_samples", type=int, default=3, help="'min_samples' DBSCAN parameter."
)
@click.option("-e", "--eps", type=float, default=2.0, help="'eps' DBSCAN parameter.")
@click.option(
    "-p",
    "--m_power",
    type=float,
    default=2.0,
    help="Power used in Minkowsky distance function.",
)
@click.option(
    "--plot",
    type=bool,
    default=False,
    is_flag=True,
    help="If set, will plot results and save them in 'output_dir'.",
)
@click.option(
    "--silhouette",
    type=bool,
    is_flag=True,
    default=False,
    help="If set, will compute silhouette coefficient for STAT file. "
    "By default disabled, as this calculation takes very long time.",
)
def run(
    dataset_path: str,
    output_dir: Path,
    algorithm: str,
    ti: bool,
    plot: bool,
    k: int,
    min_samples: int,
    eps: float,
    m_power: float,
    silhouette: bool,
):
    start_time = time.perf_counter()
    points: list[Point] = load_points(dataset_path)
    runtimes = {"1_read_input_file": time.perf_counter() - start_time}

    dataset_name = Path(dataset_path).stem
    main_info = {
        "#_dimensions": len(points[0].vals),
        "#_points": len(points),
        "input_file": str(dataset_path),
    }

    if algorithm == "dbscan":
        alg_runtimes = dbscan(points, min_samples=min_samples, eps=eps, m=m_power)
        output_dir = (
            output_dir
            / "dbscan"
            / dataset_name
            / f"min_samples_{min_samples}_eps_{eps}_m_{m_power}"
        )
        main_info["algorithm"] = "DBSCAN"
        parameters = {
            "min_samples": min_samples,
            "eps": eps,
            "minkowski_power": m_power,
        }
    elif algorithm == "dbscrn":
        if ti:
            ref_point = Point(id=-1, vals=[min(p.vals[i] for p in points) for i in range(len(points[0].vals))])
        else:
            ref_point = None
        alg_runtimes = dbscrn(points, k=k, m=m_power, ti=ti, ref_point=ref_point)
        alg_dir = "dbscrn" if not ti else "dbscrn_ti"
        output_dir = output_dir / alg_dir / dataset_name / f"k_{k}_m_{m_power}"
        main_info["algorithm"] = "DBSCRN"
        parameters = {"TI_optimized": ti, "k": k, "minkowski_power": m_power}
        if ref_point is not None:
            parameters["TI_reference_point"] = ref_point.vals
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}.")
    runtimes.update(alg_runtimes)

    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / "OUT.csv"
    with out_file.open("w+") as f:
        dims = ",".join([f"x_{i}" for i in range(len(points[0].vals))])
        header = f"point_id,{dims},#_calcs,point_type,c_id\n"
        f.write(header)
        f.writelines([e.serialize_out() for e in points])

    debug_file = output_dir / "DEBUG.tsv"
    with debug_file.open("w+") as f:
        f.write(points[0].get_serialize_debug_header())
        f.writelines([p.serialize_debug() for p in points])

    metrics_computation_start_time = time.perf_counter()
    clustering_stats = {
        "#_clusters": len(set(p.cluster_id for p in points if p.cluster_id > 0)),
        "#_core_points": len([p for p in points if p.point_type == 1]),
        "#_border_points": len([p for p in points if p.point_type == 0]),
        "#_noise_points": len([p for p in points if p.point_type == -1]),
        "avg_#_of_distance_calculation": sum(p.calc_ctr for p in points) / len(points),
    }
    rand_value, tp, tn, n_pairs = rand(points)
    clustering_metrics = {
        "Purity": purity(points),
        "davies_bouldin": davies_bouldin(points, m_power),
        "RAND": rand_value,
        "TN": tp,
        "TP": tn,
        "#_of_pairs": n_pairs,
    }
    if silhouette:
        clustering_metrics["silhouette_coefficient"] = mean_silhouette_coefficient(
            points, m_power
        )
    runtimes["5_stats_calculation"] = (
        time.perf_counter() - metrics_computation_start_time
    )
    runtimes["total_runtime"] = time.perf_counter() - start_time

    stat_file = output_dir / "STAT.json"
    with stat_file.open("w+") as f:
        stat_dict = {
            "main": main_info,
            "parameters": parameters,
            "clustering_metrics": clustering_metrics,
            "clustering_stats": clustering_stats,
            "clustering_time": runtimes,
        }
        json.dump(stat_dict, f, indent=2)

    if plot:
        plot_out_2d(
            out_file,
            output_file=output_dir / "plot.png",
        )


if __name__ == "__main__":
    run()
