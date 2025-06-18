import argparse
import logging
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ASSETS_CSV = "assets.csv"
SENSOR_CSV = "sensor.csv"
FAILURES_CSV = "failures.csv"
MANHOURS_CSV = "manhours.csv"
MODEL_FILE = "pcse_model.pkl"
SCORES_FILE = "pcse_scores.csv"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)


def load_data(data_dir: str):
    assets = _read_csv(os.path.join(data_dir, ASSETS_CSV), parse_dates=["install_date"])
    sensors = _read_csv(
        os.path.join(data_dir, SENSOR_CSV), parse_dates=["timestamp"]
    )
    failures = _read_csv(
        os.path.join(data_dir, FAILURES_CSV), parse_dates=["timestamp"]
    )
    manhours = _read_csv(
        os.path.join(data_dir, MANHOURS_CSV), parse_dates=["timestamp"]
    )
    return assets, sensors, failures, manhours


def _week_start(dt: pd.Timestamp) -> pd.Timestamp:
    return (dt - pd.Timedelta(days=dt.weekday())).normalize()


def build_snapshots(assets: pd.DataFrame, sensors: pd.DataFrame,
                    failures: pd.DataFrame, manhours: pd.DataFrame,
                    latest_only: bool = False) -> pd.DataFrame:
    if sensors.empty:
        end_date = datetime.utcnow()
    else:
        end_date = sensors["timestamp"].max()
    start_sensor = end_date - timedelta(days=90)
    sensors_90 = sensors[sensors["timestamp"] >= start_sensor]

    sensors_90["week"] = sensors_90["timestamp"].apply(_week_start)

    vibration_p95 = sensors_90.groupby("asset_id")["vibration"].quantile(0.95).to_dict()

    sensor_aggs = sensors_90.groupby(["asset_id", "week"]).agg(
        mean_vibration=("vibration", "mean"),
        std_vibration=("vibration", "std"),
        mean_temp=("temperature", "mean"),
        std_temp=("temperature", "std"),
    )
    sensor_aggs = sensor_aggs.reset_index()
    sensor_aggs["readings_above_95th_pct"] = 0
    for idx, row in sensor_aggs.iterrows():
        thr = vibration_p95.get(row["asset_id"], 0)
        mask = (
            (sensors_90["asset_id"] == row["asset_id"]) &
            (sensors_90["week"] == row["week"]) &
            (sensors_90["vibration"] > thr)
        )
        sensor_aggs.at[idx, "readings_above_95th_pct"] = mask.sum()

    all_weeks = pd.date_range(_week_start(start_sensor), _week_start(end_date), freq="W")
    all_assets_weeks = (
        pd.MultiIndex.from_product([assets["asset_id"], all_weeks], names=["asset_id", "week"])
        .to_frame(index=False)
    )
    snapshots = all_assets_weeks.merge(sensor_aggs, on=["asset_id", "week"], how="left")
    snapshots.fillna(0, inplace=True)

    failures_three_years = failures[
        failures["timestamp"] >= end_date - timedelta(days=3 * 365)
    ]

    def compute_failure_features(group):
        times = group.sort_values("timestamp")["timestamp"].tolist()
        return times

    failure_times = failures_three_years.groupby("asset_id").apply(compute_failure_features)

    manhours["week"] = manhours["timestamp"].apply(_week_start)

    for idx, row in snapshots.iterrows():
        asset_id = row["asset_id"]
        week = row["week"]
        # failures_3y
        fail_times = failure_times.get(asset_id, [])
        count_3y = sum((t < week) and (t >= week - timedelta(days=3 * 365)) for t in fail_times)
        last_failure_time = max([t for t in fail_times if t < week], default=None)
        if last_failure_time is None:
            time_since_last = 365 * 3
        else:
            time_since_last = (week - last_failure_time).days
        snapshots.at[idx, "failures_3y"] = count_3y
        snapshots.at[idx, "time_since_last_failure"] = time_since_last
        # asset age
        install_date = assets.loc[assets["asset_id"] == asset_id, "install_date"].values[0]
        snapshots.at[idx, "asset_age_years"] = (week - install_date).days / 365.0
        # man hours
        mh = manhours[(manhours["asset_id"] == asset_id) &
                      (manhours["timestamp"] < week) &
                      (manhours["timestamp"] >= week - timedelta(days=365))]
        snapshots.at[idx, "man_hours_past_yr"] = mh["man_hours"].sum()

        if not latest_only:
            # label: failure within 30 days after snapshot
            label = any(
                (t >= week) and (t < week + timedelta(days=30)) for t in fail_times
            )
            snapshots.at[idx, "label"] = int(label)

    if latest_only:
        latest_week = snapshots["week"].max()
        snapshots = snapshots[snapshots["week"] == latest_week]

    snapshots.fillna(0, inplace=True)
    return snapshots


def train_model(data_dir: str):
    logger.info("Loading data from %s", data_dir)
    assets, sensors, failures, manhours = load_data(data_dir)
    logger.info("Building training snapshots")
    snapshots = build_snapshots(assets, sensors, failures, manhours)

    features = [
        "mean_vibration",
        "std_vibration",
        "mean_temp",
        "std_temp",
        "readings_above_95th_pct",
        "failures_3y",
        "time_since_last_failure",
        "asset_age_years",
        "man_hours_past_yr",
    ]
    X = snapshots[features]
    y = snapshots["label"]

    split_idx = int(0.8 * len(snapshots))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pos_ratio = y_train.mean()
    if pos_ratio < 0.2:
        logger.info("Applying RandomOverSampler")
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred) if len(y_test.unique()) > 1 else float("nan")
    logger.info("Hold-out ROC-AUC: %.4f", roc_auc)

    joblib.dump(model, os.path.join(data_dir, MODEL_FILE))
    logger.info("Model saved to %s", MODEL_FILE)


def predict_scores(data_dir: str):
    model_path = os.path.join(data_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run with --train first.")
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    assets, sensors, failures, manhours = load_data(data_dir)
    logger.info("Building latest snapshots for prediction")
    snapshots = build_snapshots(assets, sensors, failures, manhours, latest_only=True)

    features = [
        "mean_vibration",
        "std_vibration",
        "mean_temp",
        "std_temp",
        "readings_above_95th_pct",
        "failures_3y",
        "time_since_last_failure",
        "asset_age_years",
        "man_hours_past_yr",
    ]

    X = snapshots[features]
    prob_fail = model.predict_proba(X)[:, 1]
    snapshots["prob_fail"] = prob_fail

    # severity factor
    three_years_ago = snapshots["week"].max() - timedelta(days=3 * 365)
    recent_failures = failures[failures["timestamp"] >= three_years_ago]
    severity_mean = recent_failures.groupby("asset_id")["severity"].mean().to_dict()
    snapshots["severity_factor"] = snapshots["asset_id"].map(severity_mean).fillna(1)

    snapshots["criticality_score"] = snapshots["prob_fail"] * snapshots["severity_factor"]
    snapshots["man_hours_required_next_30d"] = snapshots["man_hours_past_yr"] / 12.0

    result = snapshots[["asset_id", "criticality_score", "man_hours_required_next_30d"]]
    result.to_csv(os.path.join(data_dir, SCORES_FILE), index=False)
    logger.info("Scores written to %s", SCORES_FILE)


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def _create_dummy_csvs(directory: str):
    now = datetime.utcnow()
    assets = pd.DataFrame({
        "asset_id": [1, 2],
        "install_date": [now - timedelta(days=365), now - timedelta(days=730)],
    })
    sensors = pd.DataFrame({
        "asset_id": [1, 1, 2, 2],
        "timestamp": [
            now - timedelta(days=7),
            now - timedelta(days=6),
            now - timedelta(days=7),
            now - timedelta(days=6),
        ],
        "vibration": [1.0, 1.2, 0.9, 1.1],
        "temperature": [40, 41, 39, 38],
    })
    failures = pd.DataFrame({
        "asset_id": [1],
        "timestamp": [now - timedelta(days=10)],
        "severity": [2],
    })
    manhours = pd.DataFrame({
        "asset_id": [1, 2],
        "timestamp": [now - timedelta(days=30), now - timedelta(days=30)],
        "man_hours": [12, 8],
    })
    assets.to_csv(os.path.join(directory, ASSETS_CSV), index=False)
    sensors.to_csv(os.path.join(directory, SENSOR_CSV), index=False)
    failures.to_csv(os.path.join(directory, FAILURES_CSV), index=False)
    manhours.to_csv(os.path.join(directory, MANHOURS_CSV), index=False)


def test_workflow(tmp_path):
    data_dir = tmp_path
    _create_dummy_csvs(data_dir)
    train_model(data_dir)
    predict_scores(data_dir)
    output_path = os.path.join(data_dir, SCORES_FILE)
    df = pd.read_csv(output_path)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prabath Criticality Score Engine")
    parser.add_argument("--train", action="store_true", help="Train model from CSVs")
    parser.add_argument("--predict", action="store_true", help="Score latest week")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    parser.add_argument(
        "--data-dir", default=".", help="Directory containing CSV files and model"
    )
    args = parser.parse_args()

    if args.test:
        import pytest

        pytest.main([__file__])
        return

    if args.train:
        train_model(args.data_dir)
    if args.predict:
        predict_scores(args.data_dir)


if __name__ == "__main__":
    main()
