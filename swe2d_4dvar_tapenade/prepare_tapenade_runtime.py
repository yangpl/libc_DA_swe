#!/usr/bin/env python3
"""Stage the installed Tapenade runtime and patch only the disposable build copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = ('adStack.c', 'adStack.h', 'adBinomial.c', 'adBinomial.h', 'adComplex.h')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adkit', required=True)
    args = parser.parse_args()
    adkit = Path(args.adkit).expanduser().resolve()
    patch = ROOT/'tapenade_alignment.patch'
    out = ROOT/'build/tapenade_runtime'
    manifest_path = out/'manifest.json'
    inputs = {name: adkit/name for name in RUNTIME_FILES}
    # Keep the toolkit's license with the temporary runtime copy.
    inputs['LICENSE.html'] = adkit.parent/'LICENSE.html'
    for path in inputs.values():
        if not path.is_file():
            raise SystemExit(f'Missing Tapenade runtime file: {path}. Set TAPENADE_HOME or ADKIT.')
    provenance = dict(adkit=str(adkit), upstream_sha256={n: digest(p) for n,p in inputs.items()},
                      patch_sha256=digest(patch), preparer_sha256=digest(Path(__file__)))
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if (prior.get('inputs') == provenance and
            all((out/n).is_file() and digest(out/n)==h
                for n,h in prior.get('sha256', {}).items()) and
            set(prior.get('sha256', {})) == set(inputs)):
            return
    out.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='runtime-', dir=out.parent) as tmp:
        stage = Path(tmp)
        for name, path in inputs.items():
            shutil.copy2(path, stage/name)
        # Fail on an incompatible toolkit version rather than silently changing its code.
        result = subprocess.run(['patch','--batch','--forward','--fuzz=0',
                                 '--input',str(patch),'adStack.c'], cwd=stage,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit('Tapenade alignment patch requires review for this installation:\n'+result.stdout)
        manifest = dict(inputs=provenance, sha256={n: digest(stage/n) for n in inputs})
        out.mkdir(exist_ok=True)
        for name in inputs:
            # Replacing changed files ensures Make sees updates, even with old toolkit mtimes.
            if not (out/name).exists() or digest(out/name)!=manifest['sha256'][name]:
                shutil.copyfile(stage/name, out/name)
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Prepared patched runtime from {adkit}')


if __name__ == '__main__':
    main()
