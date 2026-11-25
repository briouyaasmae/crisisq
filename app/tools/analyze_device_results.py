#!/usr/bin/env python3
"""Check frozen device measurements; report QUInt8 drift without calling it equivalence."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise SystemExit('INVALID: ' + message)

def match(actual, expected):
    return math.isfinite(actual) and abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('--out', type=Path, default=Path('device_evaluation'))
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'assets/study/device_protocol.json').read_text())
    cases = {c['source_id']: c for c in map(json.loads, (ROOT / 'assets/study/benchmark_cases.jsonl').read_text().splitlines())}
    device_references = {}
    drift = []
    with zipfile.ZipFile(args.export) as archive:
        done = json.loads(archive.read('completion.json'))
        ctx = json.loads(archive.read('run_manifest.json'))['configuration']
        parity = json.loads(archive.read('parity_report.json'))
        selected = ctx.get('selected_models', list(protocol['models']))
        require(selected == ['fp32'] or sorted(selected) == ['fp32', 'q_quint8'], 'unsupported model selection')
        exploratory = ctx.get('allow_quint8_cross_runtime_drift') is True
        if exploratory:
            require(sorted(selected) == ['fp32', 'q_quint8'] and ctx.get('measurement_protocol') == 'quint8_portability_v1', 'portability protocol not explicitly declared')
        task_count = len(protocol['profiles']) * len(selected) * 2
        measurement_count = len(cases) * len(selected) * 2 * 5
        require(done.get('complete') is True and done.get('completed_tasks') == task_count, 'all selected tasks must complete')
        require(done.get('required_tasks', task_count) == task_count, 'task scope differs')
        require(parity.get('threads') == ctx['threads'] and parity.get('runtime') == ctx['runtime'], 'parity settings differ from run')
        require(parity.get('session_configs', {}) == ctx.get('session_configs', {}), 'native session settings differ')
        tau = ctx['tau']
        require(tau == protocol['tau'], 'threshold differs')
        for variant in selected:
            checked = [r for r in parity.get('models', []) if r['variant'] == variant]
            require(len(checked) == len(cases) and {r['source_id'] for r in checked} == set(cases), 'selected model reference checks incomplete')
            device_references[variant] = {}
            for row in checked:
                sid = row['source_id']
                ref = cases[sid]['reference'][variant]['probability']
                actual = row['actual_probability']
                require(math.isfinite(actual) and 0 <= actual <= 1, 'invalid device reference')
                passed = match(actual, ref) and (actual >= tau) == (ref >= tau)
                require(row['passed'] is passed, 'reference status is inconsistent with stored probabilities')
                if not (exploratory and variant == 'q_quint8'):
                    require(passed, 'selected model reference mismatch')
                device_references[variant][sid] = actual
                drift.append(dict(variant=variant, source_id=sid, profile=cases[sid]['partition'],
                                  kaggle_probability=ref, device_probability=actual,
                                  absolute_error=abs(actual-ref), probability_match=match(actual, ref),
                                  decision_changed=(actual >= tau) != (ref >= tau), kaggle_equivalence_passed=passed))
        require(ctx['evidence_plan_hash'] == protocol['evidence_plan_hash'], 'evidence hash differs')
        require(ctx.get('build_mode') == 'release' and ctx.get('device', {}).get('physical') is True, 'physical release device required')
        expected_hashes = {Path(k).name: v for m in protocol['models'].values() for k, v in m['all_frozen_file_hashes'].items()}
        require(ctx.get('model_hashes') == expected_hashes, 'model hashes differ')
        for name, key in [('device_protocol.json', 'packet_sha256'), ('frozen_rules.json', 'rules_sha256'), ('benchmark_cases.jsonl', 'cases_sha256')]:
            require(ctx[key] == hashlib.sha256((ROOT / 'assets/study' / name).read_bytes()).hexdigest(), 'packet bytes differ: ' + name)
        require(not ctx['full_pipeline'] or parity['tokenizer']['passed'] is True, 'on-device tokenizer parity required')
        filename = 'device_results_QUINT8_PORTABILITY.csv' if exploratory else 'device_results_FP32_ONLY.csv' if selected == ['fp32'] else 'device_results.csv'
        rows = list(csv.DictReader(io.StringIO(archive.read(filename).decode())))
    require(len(rows) == measurement_count, f'expected {measurement_count:,} timed measurements')
    seen, groups = set(), {}
    for row in rows:
        profile, variant, mode, sid = (row[k] for k in ['profile', 'variant', 'mode', 'source_id'])
        require(sid in cases and cases[sid]['partition'] == profile, 'unknown source/profile')
        require(variant in selected and mode in ('neural', 'hybrid'), 'unknown variant/mode')
        repeat = int(row['repeat'])
        require(0 <= repeat < 5, 'repeat outside 0..4')
        key = profile, variant, mode, sid, repeat
        require(key not in seen, 'duplicate measurement')
        seen.add(key)
        require(row['build_mode'] == 'release', 'invalid row build mode')
        for name, value in [('device_id', ctx['device_id']), ('threads', ctx['threads']), ('runtime', ctx['runtime']), ('provider', ctx['provider']), ('evidence_plan_hash', ctx['evidence_plan_hash'])]:
            require(row[name] == str(value), 'mixed contexts: ' + name)
        require(row['tokenizer_mode'] == ('on_device' if ctx['full_pipeline'] else 'pretokenized'), 'mixed tokenization scope')
        gate = cases[sid]['gate_v2']
        invoked = mode == 'neural' or gate < 0
        require(row['invoked'] == str(invoked).lower(), 'incorrect routing')
        ref = cases[sid]['reference'][variant]['probability']
        baseline = device_references[variant][sid]
        portable_row = exploratory and variant == 'q_quint8'
        expected = int((baseline if portable_row else ref) >= tau) if invoked else gate
        require(int(row['decision']) == expected, 'decision mismatch against the declared reference')
        if invoked:
            actual = float(row['probability'])
            require(0 <= actual <= 1 and match(actual, baseline if portable_row else ref), 'probability not repeatable against declared reference')
            equivalence = match(actual, ref) and (actual >= tau) == (ref >= tau)
            require(row['parity_verified'] == str(equivalence).lower(), 'Kaggle parity flag hides or invents drift')
            native_ms = float(row['onnx_ms'])
            require(math.isfinite(native_ms) and native_ms >= 0, 'invalid ONNX timing')
            if portable_row:
                require(row.get('validation_scope') == 'device_repeatability_with_recorded_kaggle_drift', 'missing exploratory row scope')
                require(row.get('device_repeatability_verified') == 'true', 'missing device-repeatability check')
                require(abs(float(row['device_reference_probability']) - baseline) <= 1e-12, 'device baseline differs')
                require(abs(float(row['neural_reference_probability']) - ref) <= 1e-12, 'Kaggle reference changed')
                require(row.get('kaggle_probability_match') == str(match(actual, ref)).lower(), 'probability drift flag differs')
        else:
            require(row['probability'] == '' and row['onnx_ms'] == '' and row['parity_verified'] == 'true', 'invalid gate-exit record')
        for field in ('total_ms', 'gate_ms', 'tokenization_ms'):
            value = float(row[field])
            require(math.isfinite(value) and value >= 0, 'invalid timing')
        groups.setdefault((profile, variant, mode), []).append(row)
    require(len(groups) == task_count and all(len(g) == 320 for g in groups.values()), 'incomplete task matrix')
    args.out.mkdir(parents=True, exist_ok=True)
    def save(name, records):
        with (args.out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summaries, classification = [], []
    for (profile, variant, mode), group in sorted(groups.items()):
        times = sorted(float(row['total_ms']) for row in group)
        summaries.append(dict(profile=profile, variant=variant, mode=mode, messages=64, repeats=5, measurements=len(group), mean_ms=statistics.mean(times), median_ms=statistics.median(times), p95_ms=times[math.ceil(.95 * len(times))-1], stage2_rate=sum(row['invoked'] == 'true' for row in group)/len(group)))
        unique = {row['source_id']: int(row['decision']) for row in group}
        positive = [sid for sid in unique if cases[sid]['label'] == 1]
        negative = [sid for sid in unique if cases[sid]['label'] == 0]
        tp = sum(unique[sid] == 1 for sid in positive)
        fp = sum(unique[sid] == 1 for sid in negative)
        n = len(positive) + len(negative)
        classification.append(dict(profile=profile, variant=variant, mode=mode, unique_messages=len(unique), labeled_messages=n, positive_messages=len(positive), negative_messages=len(negative), source_label_tp=tp if positive else None, source_label_fn=len(positive)-tp if positive else None, source_label_fp=fp if negative else None, source_label_tn=len(negative)-fp if negative else None, recall=tp/len(positive) if positive else None, false_positive_rate=fp/len(negative) if negative else None, accuracy=(tp+len(negative)-fp)/n if n else None, alert_rate=sum(unique.values())/len(unique)))
    save('validated_timing_summary.csv', summaries)
    save('device_source_label_results.csv', classification)
    save('cross_runtime_probability_and_decision_drift.csv', drift)
    lookup = {(r['profile'], r['variant'], r['mode'], r['source_id'], r['repeat']): r for r in rows}
    paired = []
    for profile in protocol['profiles']:
        for variant in selected:
            gains, ratios = [], []
            for sid, case in cases.items():
                if case['partition'] != profile:
                    continue
                neural = statistics.mean(float(lookup[profile, variant, 'neural', sid, str(i)]['total_ms']) for i in range(5))
                hybrid = statistics.mean(float(lookup[profile, variant, 'hybrid', sid, str(i)]['total_ms']) for i in range(5))
                gains.append(neural - hybrid)
                if hybrid > 0:
                    ratios.append(neural / hybrid)
            paired.append(dict(profile=profile, variant=variant, unique_messages=64, mean_saved_ms_per_message=statistics.mean(gains), median_message_speedup=statistics.median(ratios) if ratios else None))
    save('paired_routing_comparison.csv', paired)
    if 'q_quint8' in selected:
        comparison = []
        stats = {(r['profile'], r['variant'], r['mode']): r for r in summaries}
        labels = {(r['profile'], r['variant'], r['mode']): r for r in classification}
        for profile in protocol['profiles']:
            for mode in ['neural', 'hybrid']:
                f, q = stats[profile, 'fp32', mode], stats[profile, 'q_quint8', mode]
                fl, ql = labels[profile, 'fp32', mode], labels[profile, 'q_quint8', mode]
                comparison.append(dict(profile=profile, mode=mode, unique_messages=64,
                    fp32_mean_ms=f['mean_ms'], quint8_mean_ms=q['mean_ms'],
                    quint8_latency_reduction_percent=100*(1-q['mean_ms']/f['mean_ms']) if f['mean_ms'] > 0 else None,
                    ratio_of_mean_latencies=f['mean_ms']/q['mean_ms'] if q['mean_ms'] > 0 else None,
                    fp32_source_label_fn=fl['source_label_fn'], quint8_source_label_fn=ql['source_label_fn'],
                    fp32_source_label_fp=fl['source_label_fp'], quint8_source_label_fp=ql['source_label_fp']))
        save('paired_quantization_comparison.csv', comparison)
    state = 'completed_exploratory_portability_measurements' if exploratory else 'completed_strict_reference_measurements'
    report = dict(measurements_valid=True, kaggle_equivalence_passed=all(r['kaggle_equivalence_passed'] for r in drift), status=state, measurements=measurement_count, selected_models=selected, unique_messages=256, device_context=ctx, cross_runtime_summary={v:dict(probability_mismatches=sum(not r['probability_match'] for r in drift if r['variant']==v), changed_decisions=sum(r['decision_changed'] for r in drift if r['variant']==v), max_probability_difference=max(r['absolute_error'] for r in drift if r['variant']==v)) for v in selected}, notes='Timing sample only; five repeats are not independent texts. Source labels are not clinical judgments. Exploratory QUInt8 measurements do not pass the original equivalence test, establish preserved accuracy, or establish new independent external validity.')
    (args.out / 'validation.json').write_text(json.dumps(report, indent=2))
    print(f'COMPLETE: {task_count} tasks, {measurement_count:,} measurements. Status: {state}. Kaggle equivalence: {report["kaggle_equivalence_passed"]}. Results:', args.out.resolve())
if __name__ == '__main__':
    main()
