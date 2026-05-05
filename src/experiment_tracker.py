"""
Lightweight experiment logger — appends one row per run to outputs/experiments.csv.

Usage (from run_experiment.py or any script):
    from experiment_tracker import log_experiment
    exp_id = log_experiment(params, results, notes="trying higher depth")
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

EXPERIMENTS_FILE = Path(__file__).parent.parent / "outputs" / "experiments.csv"

_FIELDS = [
    # identity
    "exp_id", "exp_name", "timestamp", "model", "as_of", "train_end",
    # model hyperparameters
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "reg_lambda",
    # feature / portfolio settings
    "features", "top_k",
    # evaluation results
    "val_rank_ic", "val_ic_std", "val_top20_ic", "val_top50_ic", "val_hit_rate", "n_folds",
    "bt_mean_return", "bt_bench_mean", "bt_mean_excess", "bt_std_return", "bt_sharpe",
    # output
    "out_file", "notes",
]


def log_experiment(params: dict, results: dict, notes: str = "") -> str:
    """Append one experiment row and return the exp_id.

    params  — flat dict of hyperparams + settings (keys matching _FIELDS)
    results — flat dict of metrics (val_rank_ic, bt_mean_return, etc.)
    """
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    row = {
        "exp_id": exp_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        **params,
        **results,
    }
    # Serialize list-type fields (e.g. feature list) to JSON strings
    for k, v in row.items():
        if isinstance(v, (list, dict)):
            row[k] = json.dumps(v)

    EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EXPERIMENTS_FILE.exists()
    with open(EXPERIMENTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return exp_id
