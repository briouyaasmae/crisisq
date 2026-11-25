"""Check saved device counts/timing without downloading texts or running models.

Device bootstrap intervals retain their original checksummed bytes. To recompute
those intervals as well, reconstruct inputs and run reproduce_results.py.
"""
from pathlib import Path
import collections
import csv
import gzip
import io
import json
import math
import statistics
import zipfile
from review_store import result_root

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def main():
    result = result_root(ROOT)
    raw = result / 'device/raw'
    summary = result / 'device/consolidated'
    seed = json.loads(gzip.decompress((ROOT / 'scripts/full_evaluation_seed.json.gz').read_bytes()))
    membership = {r['source_id']: r for r in seed['labeled_rows']}
    assert len(membership) == 1700
    packet = json.loads((ROOT / 'app/assets/study/full_evaluation_manifest.json').read_text())
    exports = []
    for path in sorted(raw.glob('crisisq_full_device_evaluation_*.zip')):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            done = json.loads(archive.read('completion.json'))
            assert done['complete'] and done['model_outputs'] == 3400
            assert json.loads(archive.read('packet_manifest.json')) == packet
            values = list(csv.DictReader(io.StringIO(archive.read('full_device_predictions.csv').decode())))
        assert len(values) == 3400
        prediction = {}
        for row in values:
            case = membership[row['source_id']]
            for field in ['text_hash', 'group_id', 'partition']:
                assert row[field] == case[field]
            assert int(row['label']) == case['label'] and int(row['gate_v2']) == case['gate_v2']
            p = float(row['probability'])
            assert math.isfinite(p) and 0 <= p <= 1
            assert int(row['prediction_neural']) == int(p >= seed['tau'])
            expected = case['gate_v2'] if case['gate_v2'] >= 0 else int(p >= seed['tau'])
            assert int(row['prediction_hybrid']) == expected
            prediction[row['variant'], row['source_id']] = (p, int(row['prediction_neural']), expected)
        assert len(prediction) == 3400
        exports.append(prediction)
    assert len(exports) == 2 and exports[0] == exports[1]
    prediction = exports[0]
    for row in rows(summary / 'full_device_classification_metrics.csv'):
        cases = [c for c in membership.values() if c['partition'] == row['partition']]
        mode = 1 if row['mode'] == 'neural' else 2
        decisions = [(c['label'], prediction[row['variant'], c['source_id']][mode]) for c in cases]
        counts = collections.Counter(decisions)
        expected = dict(n=len(cases), n_positive=sum(y for y, _ in decisions), n_negative=sum(1-y for y, _ in decisions), tp=counts[1,1], fn=counts[1,0], fp=counts[0,1], tn=counts[0,0])
        for name, value in expected.items():
            assert int(row[name]) == value, (row['partition'], name)
        close(row['recall'], expected['tp']/expected['n_positive']) if expected['n_positive'] else None
        close(row['false_positive_rate'], expected['fp']/expected['n_negative']) if expected['n_negative'] else None
    for row in rows(summary / 'paired_quantization_and_routing_changes.csv'):
        cases = [c for c in membership.values() if c['partition'] == row['partition']]
        def decision(system, case):
            variant, mode = system.split('/')
            return prediction[variant, case['source_id']][1 if mode == 'neural' else 2]
        values = [(c['label'], decision(row['baseline'], c), decision(row['candidate'], c)) for c in cases]
        checks = dict(n=len(cases), changed_decisions=sum(a != b for _,a,b in values), positive_lost=sum(y == 1 and a == 1 and b == 0 for y,a,b in values), positive_gained=sum(y == 1 and a == 0 and b == 1 for y,a,b in values), new_negative_alerts=sum(y == 0 and a == 0 and b == 1 for y,a,b in values), removed_negative_alerts=sum(y == 0 and a == 1 and b == 0 for y,a,b in values))
        for name, value in checks.items():
            assert int(row[name]) == value, (row['comparison'], name)
    grouped = collections.defaultdict(list)
    runs = set()
    for path in sorted(raw.glob('crisisq_device_results_*.zip')):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            done = json.loads(archive.read('completion.json'))
            manifest = json.loads(archive.read('run_manifest.json'))
            assert done['complete'] and done['completed_tasks'] == 16
            assert manifest['configuration']['build_mode'] == 'release' and manifest['configuration']['threads'] == 4
            values = list(csv.DictReader(io.StringIO(archive.read('timing_summary.csv').decode())))
            measured = list(csv.DictReader(io.StringIO(archive.read('device_results_QUINT8_PORTABILITY.csv').decode())))
        assert len(values) == 16 and len(measured) == 5120
        assert manifest['run_id'] not in runs
        runs.add(manifest['run_id'])
        model = manifest['configuration']['device']['model'].lower()
        device = 'Samsung SM-A225F' if 'sm-a225f' in model else 'HONOR HEY2-W09'
        for row in values:
            grouped[device, row['profile'], row['variant'], row['mode']].append(float(row['mean_total_ms']))
    assert len(runs) == 6
    for row in rows(summary / 'timing_three_run_summary.csv'):
        values = grouped[row['device'], row['profile'], row['variant'], row['mode']]
        assert len(values) == int(row['runs']) == 3
        close(row['mean_of_run_mean_ms'], statistics.mean(values))
        close(row['sd_of_run_mean_ms'], statistics.stdev(values))
    print('PASS: two complete device exports; frozen membership/threshold/gate, classification/paired counts, cross-device agreement and six-run timing means/SD. Saved device intervals retained; use full replay to recompute them.')


if __name__ == '__main__':
    main()
