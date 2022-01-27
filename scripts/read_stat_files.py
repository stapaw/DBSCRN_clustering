import json
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option("-i", "--input_glob", type=str, required=True)
@click.option("-o", "--output_csv", type=Path, required=True)
def parse_stat_to_df(input_glob: str, output_csv: Path):
    df_rows = []
    redundant_keys = ("TI_reference_point", "minkowski_power")

    stat_files = list(Path(".").glob(input_glob))
    if len(stat_files) == 0:
        raise ValueError("`input_glob` doesn't match any files")

    for file_path in tqdm(stat_files, desc="Reading files..."):
        stat_data = json.load(file_path.open())

        output_row = {}
        for main_key, sub_dict in stat_data.items():
            for key in redundant_keys:
                if key in sub_dict:
                    sub_dict.pop(key)
            output_row.update(sub_dict)
        df_rows.append(output_row)

    df = pd.DataFrame(df_rows)
    df["input_file"] = df["input_file"].apply(
        lambda s: s.split("/")[-1].split(".")[0]
    )
    df.to_csv(output_csv, sep="\t", index=False)


if __name__ == "__main__":
    parse_stat_to_df()
