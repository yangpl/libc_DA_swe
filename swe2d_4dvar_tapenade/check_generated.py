"""Fail if primal, repair script, or generated output has drifted since generation."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root/'tapenade_out/manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'{name} changed: run make generate-adjoint, then make test')
print('Generated adjoint source manifest verified.')
