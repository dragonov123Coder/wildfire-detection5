"""Automated daily fire-risk prediction scheduler.

Runs prediction-2/scripts/predict.py every morning for a rolling set of
target dates, records results in the local DB, and fires a desktop
notification if any date's uncalibrated (raw) risk score exceeds 0.90.

Kept import-independent from computer_server.py (imported the other way
around) so this module can also be exercised standalone:

    python scheduler.py
"""
import csv
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

import db

BASE_DIR = Path(__file__).resolve().parent
PREDICTION_ROOT = BASE_DIR / 'prediction-2'
PREDICTION_PYTHON = PREDICTION_ROOT / 'venv' / 'Scripts' / 'python.exe'
PREDICTION_SCRIPT = PREDICTION_ROOT / 'scripts' / 'predict.py'
PREDICTION_OUTPUT_DIR = PREDICTION_ROOT / 'data' / 'output'

NOTIFY_THRESHOLD = 0.90


def daily_target_dates(run_date: date) -> list:
    """The 4 dates predicted every ordinary morning."""
    return [run_date, run_date + timedelta(days=1), run_date + timedelta(days=2), run_date + timedelta(days=7)]


def bootstrap_target_dates(run_date: date) -> list:
    """A full rolling week, computed once on first-ever startup."""
    return [run_date + timedelta(days=i) for i in range(7)]


def summarize_risk_csv(csv_path: Path):
    """Return (row_count, max_risk_score_raw) for a risk_map CSV, using the
    stdlib csv module so the root Flask process doesn't need pandas."""
    row_count = 0
    max_raw = None
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                raw = row.get('risk_score_raw')
                if raw:
                    try:
                        val = float(raw)
                        if max_raw is None or val > max_raw:
                            max_raw = val
                    except ValueError:
                        pass
    except OSError:
        return None, None
    return row_count, max_raw


def _run_predict(target_date_str: str, out_path: Path) -> bool:
    print(f"Predicting {target_date_str} -> {out_path.name}")
    try:
        result = subprocess.run(
            [str(PREDICTION_PYTHON), str(PREDICTION_SCRIPT), '--date', target_date_str, '--out', str(out_path)],
            cwd=str(PREDICTION_ROOT),
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        print(f"  predict.py timed out for {target_date_str}")
        return False

    if result.returncode != 0:
        print(f"  predict.py failed for {target_date_str}:\n{result.stderr.strip()[-1500:]}")
        return False
    return True


def run_daily_predictions(run_date: date = None):
    """Predict the day's target dates (or, on first-ever run, a full
    bootstrap week), farthest-date-first so its wider weather fetch
    populates the shared weather cache for the nearer dates that follow."""
    run_date = run_date or date.today()
    run_date_str = run_date.strftime('%Y-%m-%d')

    bootstrap = db.is_first_prediction_run()
    targets = bootstrap_target_dates(run_date) if bootstrap else daily_target_dates(run_date)
    targets = sorted(set(targets), reverse=True)

    print(f"Running {'bootstrap week' if bootstrap else 'daily'} predictions for run_date={run_date_str}: "
          f"{[t.strftime('%Y-%m-%d') for t in targets]}")

    overall_max = None
    overall_max_date = None

    PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in targets:
        target_str = target.strftime('%Y-%m-%d')
        out_path = PREDICTION_OUTPUT_DIR / f"risk_map_{target_str}_run{run_date_str}.csv"
        if not _run_predict(target_str, out_path):
            continue

        row_count, max_raw = summarize_risk_csv(out_path)
        db.record_prediction_run(run_date_str, target_str, str(out_path), max_raw)

        if max_raw is not None and (overall_max is None or max_raw > overall_max):
            overall_max, overall_max_date = max_raw, target_str

    if overall_max is not None and overall_max > NOTIFY_THRESHOLD:
        _maybe_notify(run_date_str, overall_max, overall_max_date)


def _maybe_notify(run_date_str: str, max_raw: float, target_date_str: str):
    """Fire one desktop toast per morning, regardless of how many target
    dates exceeded the threshold that run."""
    if db.notification_already_sent(run_date_str):
        return
    try:
        from win11toast import notify
        notify(
            title='Wildfire Risk Alert',
            body=f'Uncalibrated risk score {max_raw:.2f} for {target_date_str} (run {run_date_str}) exceeds 0.90.',
        )
    except Exception as e:
        print(f"Failed to send desktop notification: {e}")
    db.mark_notification_sent(run_date_str, max_raw, target_date_str)


def init_scheduler() -> BackgroundScheduler:
    """Register the 5:05 AM daily job. If this is a fresh install (no prior
    prediction runs on record), also kick off the bootstrap week
    immediately in the background so the UI has data without waiting for
    the next 5 AM."""
    db.init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_daily_predictions, 'cron', hour=5, minute=5, id='daily_prediction')
    scheduler.start()

    if db.is_first_prediction_run():
        import threading
        threading.Thread(target=run_daily_predictions, daemon=True).start()

    return scheduler


if __name__ == '__main__':
    db.init_db()
    run_daily_predictions(date.today() if len(sys.argv) < 2 else date.fromisoformat(sys.argv[1]))
