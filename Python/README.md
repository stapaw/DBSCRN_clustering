# DBSCRN clustering in python
Folder for python code and (optionally) for streamlit  result dashboard.

## Running the code
To run clustering algorithms, use `run.py`. This should generate OUT, DEBUG and STAT files in dedicated subdirectory in `output_dir`.


```shell
Usage: run.py [OPTIONS]

Options:
  -d, --dataset_path TEXT         Path to dataset to use.  [required]
  -o, --output_dir PATH           Directory where output files will be saved.
                                  [required]
  -a, --algorithm [dbscan|dbscrn]
                                  Type of algorithm to use.  [required]
  --ti                            If set, will use triangle inequality to
                                  optimize runtime of the DBSCRN algorithm.
  -k INTEGER                      'k' parameter in DBSCANRN algorithm.
  -s, --min_samples INTEGER       'min_samples' DBSCAN parameter.
  -e, --eps FLOAT                 'eps' DBSCAN parameter.
  -p, --m_power FLOAT             Power used in Minkowsky distance function.
  --plot                          If set, will plot results and save them in
                                  'output_dir'.
  --silhouette                    If set, will compute silhouette coefficient
                                  for STAT file. By default disabled, as this
                                  calculation takes very long time.
```
