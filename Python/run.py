import time
from pathlib import Path

import click

from clustering_metrics import davies_bouldin, mean_silhouette_coefficient, purity, rand
from dbscan import dbscan
from dbscanrn import dbscanrn
from dbscanrn_ti import dbscanrn_ti
from plotting import plot_examples_2d
from utils import Example, load_examples


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
    type=click.Choice(["dbscan", "dbscanrn", "dbscanrn_ti"]),
    required=True,
    help="Type of algorithm to use.",
)
@click.option(
    "--plot",
    type=bool,
    default=False,
    is_flag=True,
    help="If True, will plot results and save them in 'output_dir'.",
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
    "--silhouette",
    type=bool,
    is_flag=True,
    default=False,
    help="If True, will compute silhouette coefficient for STAT file. "
    "By default disabled, as this calculation takes very long time.",
)
def run(
    dataset_path: str,
    output_dir: Path,
    algorithm: str,
    plot: bool,
    k: int,
    min_samples: int,
    eps: float,
    m_power: float,
    silhouette: bool,
):
    start_time = time.perf_counter()
    examples: list[Example] = load_examples(dataset_path)
    read_time = time.perf_counter() - start_time
    dataset_name = Path(dataset_path).stem

    start_time = time.perf_counter()
    if algorithm == "dbscan":
        cluster_ids, runtimes = dbscan(
            examples, min_samples=min_samples, eps=eps, m=m_power
        )
        output_dir = (
            output_dir
            / "dbscan"
            / dataset_name
            / f"min_samples_{min_samples}_eps_{eps}_m_{m_power}"
        )
        stats = {
            "Algorithm": "DBSCAN",
            "Minimum samples": min_samples,
            "Epsilon": eps,
            "Minkowsky distance power": m_power,
        }
    elif algorithm == "dbscanrn":
        cluster_ids, runtimes = dbscanrn(examples, k=k, m=m_power)
        output_dir = output_dir / "dbscanrn" / dataset_name / f"k_{k}_m_{m_power}"
        stats = {"Algorithm": "DBSCANRN", "K": k, "Minkowsky distance power": m_power}
    else:
        cluster_ids, runtimes = dbscanrn_ti(examples, k=k, m=m_power)
        output_dir = output_dir / "dbscanrn_ti" / dataset_name / f"k_{k}_m_{m_power}"
        stats = {
            "Algorithm": "DBSCANRN_TI",
            "K": k,
            "Minkowsky distance power": m_power,
        }
    execution_time = time.perf_counter() - start_time

    for i, example in enumerate(examples):
        cluster_id = cluster_ids[i]
        example.cluster_id = cluster_id

    start_time = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "OUT.csv"
    with out_file.open("w+") as f:
        dims = ",".join([f"x_{i}" for i in range(len(examples[0].vals))])
        header = f"point_id,{dims},#_calcs,point_type,c_id\n"
        f.write(header)
        f.writelines([e.serialize_out() for e in examples])
    stat_file = output_dir / "STAT.txt"
    with stat_file.open("w+") as f:
        f.write(
            f"Input file: {dataset_path}\n"
            f"\t# dimensions: {len(examples[0].vals)}\n"
            f"\t# points: {len(examples)}\n\n"
        )

        f.write("Parameters\n")
        for stat_name, stat_value in stats.items():
            f.write(f"\t{stat_name}: {stat_value}\n")

        f.write("\nClustering stats\n")
        num_clusters = len(
            set(example.cluster_id for example in examples if example.cluster_id > 0)
        )
        f.write(f"\t# clusters: {num_clusters}\n")
        num_noise_points = len(
            [example for example in examples if example.point_type == -1]
        )
        f.write(f"\t# noise points: {num_noise_points}\n")
        num_border_points = len(
            [example for example in examples if example.point_type == 0]
        )
        f.write(f"\t# border_points: {num_border_points}\n")
        avg_calculations = sum(example.calc_ctr for example in examples) / len(examples)
        f.write(
            f"\tAverage # of distance / similarity calculations: {avg_calculations}\n"
        )
        f.write("\nClustering metrics\n")
        rand_value, tp, tn, pair_count = rand(examples)
        f.write(f"\tRAND: {rand_value}\n")
        f.write(f"\t|TP|: {tp}\n")
        f.write(f"\t|TN|: {tn}\n")
        f.write(f"\t# of pairs of points: {pair_count}\n")
        f.write(f"\tPurity: {purity(examples)}\n")
        if silhouette:
            f.write(
                f"\tSilhouette coefficient: {(mean_silhouette_coefficient(examples, m_power))}\n"
            )
        if algorithm == "dbscan":
            f.write(f"\tDavies-Bouldin: {(davies_bouldin(examples, m_power))}\n")

        f.write("\nRuntime\n")
        f.write(f"\tRead runtime: {read_time}\n")
        for runtime_key, runtime in runtimes.items():
            f.write(f"\t{runtime_key}: {runtime}\n")
        f.write(f"\tTotal algorithm execution runtime: {execution_time}\n")

    debug_file = output_dir / "DEBUG.tsv"
    with debug_file.open("w+") as f:
        f.write("\t".join(["point_id"] + list(examples[0].debug_info.keys())) + "\n")
        f.writelines([example.serialize_debug() for example in examples])
        """
        TODO
        - maxEps – Eps value calculated based on first k candidates for k+NN of the point (for TI-optimized versions)
        - minEps is the minimal value of radius Eps within which real k+NN of the point was found (for TI-optimized versions).
        """
    write_time = time.perf_counter() - start_time
    with stat_file.open("a") as f:
        f.write(f"\tWrite runtime: {write_time}\n")
        f.write(f"\tTotal runtime: {read_time + execution_time+write_time}\n")

    if plot:
        plot_examples_2d(
            examples=examples,
            output_file=output_dir / "plot.png",
        )


if __name__ == "__main__":
    run()
