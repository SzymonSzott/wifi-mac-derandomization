# MAC Derandomization Clustering

## Command-Line Arguments

Run the script with various options to customize the clustering analysis:

```
python -m src.cli.main [OPTIONS]
```

### Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input-dir` | `-d` | string | `""` | Subdirectory within `dataset/` to process |
| `--input-type` | `-i` | choice | `json` | Input data format: `json` or `pcap` |
| `--labeled` | `-l` | flag | - | Use labeled data (requires ground truth labels) |
| `--label-from` | - | choice | `folder` | Where to extract labels from: `folder` or `file` |
| `--workers` | - | int | `None` | Number of parallel workers for processing (None = auto) |
| `--scenario` | `-s` | string | `default` | Scenario name for report file naming |
| `--clustering-method` | - | choice | `optics` | Clustering algorithm: `optics`, `dbscan`, or `hdbscan` |
| `--reverted-table` | - | flag | - | Display additional reverted table showing clusters → assigned labels |
| `--experiments-csv` | - | string | `outputs/experiments/optics_experiments.csv` | CSV file where experiment results are appended |


### Algorithm Parameters

| Argument | Used by | Type | Default | Description |
|----------|---------|------|---------|-------------|
| `--min-samples` | OPTICS, DBSCAN, HDBSCAN | int | `2` | Minimum number of samples used as a density/core-point parameter |
| `--xi` | OPTICS | float | `0.05` | OPTICS parameter controlling cluster extraction sensitivity |
| `--min-cluster-size` | OPTICS, HDBSCAN | int | `15` | Minimum size of a detected cluster |
| `--max-eps` | OPTICS | float | `inf` | Maximum neighborhood radius used by OPTICS |
| `--eps` | DBSCAN | float | `0.5` | DBSCAN neighborhood radius |
| `--cluster-selection-epsilon` | HDBSCAN | float | `0.0` | HDBSCAN parameter used to merge nearby clusters |


### Examples

```
python -m src.cli.main -d italian -i pcap -l --label-from folder --workers 8 -s Baseline1
```
- dataset: italian
- file type: pcap
- label source: folder name
- scenario name (for reporting): Baseline1
- clustering algorithm: optics (default)

# Full example with all options
```
python -m src.cli.main \
  --input-dir dataset1 \
  --input-type json \
  --labeled \
  --label-from folder \
  --workers 8 \
  --scenario experiment_001 \
  --clustering-method optics \
  --reverted-table
```

# OPTICS example with parameters
```
python -m src.cli.main \
  -d italian_without_xiaomi \
  -i pcap \
  -l \
  --label-from folder \
  --workers 8 \
  --scenario optics_ms2_xi005_mcs15 \
  --clustering-method optics \
  --reverted-table \
  --min-samples 2 \
  --xi 0.05 \
  --min-cluster-size 15 \
  --experiments-csv outputs/experiments/optics_results.csv
  ```

# Related Project

This repository shares its initial core with [the related project](https://github.com/jkitlinska/MAC_derandomization), including parts of the data loading, feature extraction, distance calculation, and the OPTICS-based clustering pipeline. Both projects originated from a common codebase and were later developed independently in different directions.
