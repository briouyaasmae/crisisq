"""One offline entry point for recorded evidence. No inference or training."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print('\nChecking:', ' '.join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    for script in ['verify_release.py', 'check_public_release.py', 'check_repository.py']:
        run('scripts/' + script)
    run('-m', 'unittest', 'discover', '-s', 'tests')
    for script in ['verify_completed_runtime_matrix.py', 'verify_completed_followups.py',
                   'verify_completed_selective_precision.py', 'verify_completed_single_graph_pilot.py',
                   'verify_recorded_device.py', 'analyze_stress_keyword_alerts.py',
                   'summarize_runtime_probes.py', 'audit_additional_evidence.py']:
        run('scripts/' + script)
    run('scripts/audit_selection_policies.py', 'results/replication', '.reproduced/policy_audit')
    print('\nPASS: recorded-evidence checks complete. No model inference or training.', flush=True)


if __name__ == '__main__':
    main()
