# MAFGN

This repository provides the implementation and reproducibility package for
MAFGN, including the 24 datasets used in the experiments, fixed best
parameters, and the complete grid-search workflow.

The packaged reference run reproduces the reported AUC, MaxF1, and AP values
for all 24 datasets at the four-decimal precision used in the result tables.

## Repository structure

```text
MAFGN-main/
├── Code/
│   ├── mafgn.py
│   ├── run_best_parameters.py
│   ├── run_grid_search.py
│   ├── best_parameters.csv
│   ├── requirements.txt
│   ├── results/
│   └── README.md
└── Datasets/
    └── 24 MATLAB datasets
```

- `Code/mafgn.py`: MAFGN model and data-preprocessing utilities.
- `Code/run_best_parameters.py`: runs MAFGN with the fixed best parameters.
- `Code/run_grid_search.py`: performs the complete fixed-seed grid search.
- `Code/best_parameters.csv`: selected parameters for all 24 datasets.
- `Code/results/fixed_parameter_results_verified_gpu.csv`: verified reference
  results for all 24 datasets.
- `Datasets/`: datasets used in the paper experiments.

## Environment

Python 3.10 or later is recommended.

```bash
cd Code
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA is used automatically when available; otherwise, the scripts run on CPU.

## Quick reproduction

The recommended way to reproduce the reported results is to run each dataset
once with the fixed best parameters:

```bash
cd Code
python run_best_parameters.py
```

The results are saved to:

```text
Code/results/fixed_parameter_results.csv
```

The output includes AUC, MaxF1, AP, runtime, random seeds, and the generated
feature-view splits. An accompanying metadata file records the Python,
package, CUDA, and GPU environment.

The anomaly score is always computed as \(S = 1 - Z\), where \(Z\) is the
model normality output. The score direction is fixed and does not depend on
the labeled AUC.

The verified reference output included in this repository is:

```text
Code/results/fixed_parameter_results_verified_gpu.csv
Code/results/fixed_parameter_results_verified_gpu.metadata.json
```

In this reference run:

- 24/24 datasets match the reported AUC values;
- 24/24 datasets match the reported MaxF1 values;
- 24/24 datasets match the reported AP values.

Matching is checked at four decimal places, which is the precision stored in
the reported result files. The `Matches_Reported_4dp` column records this
check for every dataset.

Run selected datasets only:

```bash
python run_best_parameters.py --datasets wine Lymphography
```

Force a specific device:

```bash
python run_best_parameters.py --device cpu
python run_best_parameters.py --device cuda:0
```

## Complete grid search

To reproduce the full parameter-search process for all 24 datasets:

```bash
cd Code
python run_grid_search.py
```

The search covers:

- granules `K`: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12
- trusted-inlier ratio `q`: 0.75, 0.80, 0.85, 0.90, 0.95, 0.975
- loss weight `gamma`: 0.25, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5
- number of views `V`: determined from the original feature dimensionality

The complete search evaluates 594 or 1188 parameter combinations per
dataset. Intermediate results are saved after every combination, allowing an
interrupted experiment to resume automatically.

Run selected datasets:

```bash
python run_grid_search.py --datasets wine glass Lymphography
```

For multi-GPU or multi-process execution, datasets can be divided into
independent shards:

```bash
python run_grid_search.py --num-shards 4 --shard-index 0 \
  --output-dir results/shard0
```

See [`Code/README.md`](Code/README.md) for the complete command-line options
and experimental details.

## Data preprocessing

Six mixed or categorical datasets use one-hot encoding:

- `adult_morethan50K_3779_variant1`
- `arrhythmia_variant1`
- `audiology_variant1`
- `horse_1_12_variant1`
- `mushroom_p_221_variant1`
- `Lymphography`

The remaining 18 datasets use feature-wise Min-Max scaling. One-hot blocks
originating from the same attribute are retained in the same feature view.
Pseudo anomalies use valid one-hot categorical values and numerical values in
the interval `[0, 1]`.

## Reproducibility

The experiments use feature-view seed 42 and model seed 10. For each
configuration, the model initialization and pseudo-anomaly generation follow
the same random-number sequence as the original experiments.

The reference environment is Python 3.12.7, PyTorch 2.8.0, CUDA 12.8,
NumPy 1.26.4, SciPy 1.16.1, scikit-learn 1.5.1, and an NVIDIA A100 GPU.
The verified result file was generated on this GPU environment. CPU execution
is supported, but CPU and CUDA use different numerical kernels and random
number generators. Therefore, a CPU run can differ from the verified GPU
result even when the same seeds and parameters are used. Files whose names
contain `cpu` should be treated as compatibility checks rather than the
reference paper reproduction.

Minor numerical differences may also occur with other GPU models, CUDA
versions, or library versions. For the closest reproduction, use the package
versions in `Code/requirements.txt` and compare the generated environment
metadata with the included verified metadata file.
