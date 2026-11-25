"""Resolve single-copy recorded reports into an ignored analysis working tree.

Recorded report bytes remain unchanged. The cache recreates only the original
analysis inputs; new predictions, model downloads and inference are never run.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import shutil
import zipfile


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@lru_cache(maxsize=2)
def result_root(root):
    root = Path(root).resolve()
    reports_path = root / 'manifests/recorded_reports.json'
    aliases_path = root / 'manifests/result_aliases.json'
    reports = json.loads(reports_path.read_text())
    aliases = json.loads(aliases_path.read_text())
    key = sha256_bytes(reports_path.read_bytes() + aliases_path.read_bytes())
    cache = root / '.cache/review_results'
    marker = cache / '.ready'
    if marker.is_file() and marker.read_text() == key:
        return cache
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    # Copy small canonical inputs, never link writable CSV outputs to tracked files.
    stored_reports = {root / r['path'] for r in reports}
    for source in (root / 'results').rglob('*'):
        if source.is_file() and source not in stored_reports:
            _copy(source, cache / source.relative_to(root / 'results'))
    for report in reports:
        source = root / report['path']
        assert sha256_bytes(source.read_bytes()) == report['sha256'], source.name
        archive_path = cache / report['study'] / 'raw' / report['original_name']
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        # Cache links are not shipped or committed. Windows can fall back to a copy.
        try:
            archive_path.symlink_to(source)
        except OSError:
            _copy(source, archive_path)
        with zipfile.ZipFile(source) as archive:
            for name in report['materialize']:
                rel = Path(name)
                assert not rel.is_absolute() and '..' not in rel.parts, name
                target = cache / report['study'] / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    for target, source in aliases.items():
        _copy(cache / source, cache / target)
    marker.write_text(key)
    return cache
