#!/usr/bin/env python3
"""Check the GitHub-ready distribution before materializing posts/importing models."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    raw={'benchmark_cases.jsonl','full_evaluation_cases.jsonl','misclassified_texts.csv','lexical_overlap_flags.csv'}
    forbidden=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        if p.suffix in ['.onnx','.safetensors','.pt','.pth'] or p.name in raw or p.name=='crisisq_full_labeled_device_packet.zip':forbidden.append(p)
    assert not forbidden,'Public archive includes materialized posts or private weights. Use the clean GitHub-ready ZIP.'
    print('PASS: public release excludes model weights and materialized source posts.')
if __name__=='__main__':main()
