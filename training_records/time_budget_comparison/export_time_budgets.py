"""Match existing training metrics to actual TensorBoard wall-clock timestamps."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
import yaml

REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SPECS = [
    ('LiCN', 'licn_default_1gpu_20k_s42', 42, 1, 20000,
     'mare_online.csv', 'completed_updates', 'mare_kcal_mol', 'kcal/mol'),
    ('H4', 'h4_globe6_scratch_1gpu_16k_s42', 42, 1, 16000,
     'h4_convergence.csv', 'optimizer_updates', 'mean_energy_trailing_200_Ha', 'Ha'),
    ('Ethane', 'ethane_scratch_4gpu_32k_s219', 219, 4, 32000,
     'mare_online.csv', 'optimizer_updates', 'MARE_online_EWM_kcal_per_mol', 'kcal/mol'),
]


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def training_timestamps(files, count):
    """Read standard TFRecord framing and protobufs without sampling events.

    CRC fields are skipped; validate framing, complete/unique steps, monotonic
    times, and independent native progress-log timestamps below.
    """
    times = np.full(count, np.nan)
    for path in files:
        with path.open('rb') as stream:
            while header := stream.read(12):
                assert len(header) == 12, (path, 'truncated record header')
                length = struct.unpack('<Q', header[:8])[0]
                assert length < 100_000_000, (path, length)
                payload = stream.read(length)
                crc = stream.read(4)
                assert len(payload) == length and len(crc) == 4, path
                event = Event.FromString(payload)
                for value in event.summary.value:
                    if value.tag == 'timing/train_step':
                        assert 0 <= event.step < count, (path, event.step)
                        assert np.isnan(times[event.step]), (path, 'duplicate', event.step)
                        times[event.step] = event.wall_time
    assert np.isfinite(times).all(), 'Missing per-step timestamps'
    assert np.all(np.diff(times) > 0), 'Nonmonotonic per-step timestamps'
    return times


def main():
    rows, provenance = [], []
    for name, alias, seed, gpus, count, filename, step_column, metric_column, unit in SPECS:
        run = (REPO / 'runs' / alias).resolve()
        results = run / 'results'
        status = json.loads((results / 'runtime_status.json').read_text())
        config = yaml.safe_load((results / 'training/config.yaml').read_text())
        assert status['stage'] == 'COMPLETE' and status['completed_updates'] == count
        assert config['args']['seed'] == seed
        start = datetime.fromisoformat(status['started_at']).timestamp()
        elapsed = status['elapsed_seconds']
        events = sorted((results / 'training').glob('events.out.tfevents.*'))
        times = training_timestamps(events, count)
        assert start < times[0] < times[-1] <= start + elapsed + 1
        log = (results / 'oneqmc_train.log').read_text()
        checks = []
        for date, step in re.findall(r'\[([\d-]+ [\d:,]+) INFO[^\n]*Progress: (\d+)/', log):
            step = int(step)
            logged = datetime.strptime(date, '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc).timestamp()
            delay = times[step - 1] - logged
            assert -0.01 <= delay < 10, (name, step, delay)
            checks.append({'completed_updates': step, 'event_minus_log_seconds': delay})
        assert checks
        metric_path = results / 'analysis' / filename
        with metric_path.open() as stream:
            metrics = {int(row[step_column]): float(row[metric_column])
                       for row in csv.DictReader(stream) if row[metric_column]}
        time_path = OUTPUT / f'{name.lower()}_step_wall_times.csv'
        with time_path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['completed_updates', 'wall_time_unix', 'elapsed_since_process_start_seconds'])
            writer.writerows((i + 1, t, t - start) for i, t in enumerate(times))
        for hours in [1, 2, 4]:
            deadline = start + hours * 3600
            completed = int(np.searchsorted(times, deadline, side='right'))
            assert completed > 0 and completed in metrics
            actual_time = times[completed - 1]
            assert actual_time <= deadline
            if completed < count:
                assert times[completed] > deadline
            row = dict(
                system=name, seed=seed, gpus=gpus, time_budget_hours=hours,
                completed_updates=completed, metric=metrics[completed], unit=unit,
                metric_name=('six-geometry absolute mean energy, trailing 200 updates'
                             if name == 'H4' else 'online EWM MARE'),
                state='complete_before_budget' if start + elapsed <= deadline else 'training',
                process_started_at_utc=iso(start), deadline_utc=iso(deadline),
                selected_step_timestamp_utc=iso(actual_time),
                selected_step_elapsed_seconds=float(actual_time - start),
                next_step_elapsed_seconds=(float(times[completed] - start) if completed < count else ''),
                total_runtime_seconds=elapsed,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        provenance.append(dict(
            system=name, run_dir=str(run), runtime_status=str(results / 'runtime_status.json'),
            event_files=[str(path) for path in events], metric_csv=str(metric_path),
            metric_csv_sha256=hashlib.sha256(metric_path.read_bytes()).hexdigest(),
            complete_unique_step_timestamps=count, native_progress_log_crosschecks=checks,
        ))
    with (OUTPUT / 'time_budget_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / 'time_budget_metrics_metadata.json').write_text(json.dumps(dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        time_origin='runtime_status.started_at: GPU process start; includes initialization, equilibration and compilation; excludes scheduling wait',
        selection='last completed update with timing/train_step TensorBoard wall_time <= deadline; no time or metric interpolation',
        completed_run_policy='retain final metric and mark complete_before_budget; no extrapolation',
        metrics_are_training_diagnostics=True, provenance=provenance,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
