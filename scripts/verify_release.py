"""Check ordinary SHA-256 checksums of the files in this distribution."""
from pathlib import Path
import hashlib


def main():
    root = Path(__file__).resolve().parents[1]
    count = 0
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts, name
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, name
        count += 1
    print(f'PASS: {count} distributed-file checksums.')


if __name__ == '__main__':
    main()
