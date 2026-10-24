#!/usr/bin/env python3
"""Generate the reverse misfit; apply checked Tapenade 3.16 C backend repairs."""
import hashlib
import json
import os
from pathlib import Path
from fix_tapenade_c import repair
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    tapenade = os.environ.get('TAPENADE', 'tapenade')
    out = ROOT / 'tapenade_out'
    work = ROOT / 'build/tapenade-generation'
    (ROOT/'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='swe-ad-', dir=ROOT/'build') as tmp:
        stage = Path(tmp)/'ad_primal'
        stage.mkdir()
        for name in ('swe_primal.c','swe_primal.h'):
            shutil.copy2(ROOT/name, stage/name)
        stage_arg = str(stage.relative_to(ROOT))
        cmd = [tapenade, '-reverse', '-head', 'swe_misfit(cost)/(initial)',
               '-I', stage_arg, '-o', 'swe_adjoint', '-O', tmp, stage_arg+'/swe_primal.c']
        result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        raw = Path(tmp) / 'swe_adjoint_b.c'
        if result.returncode or not raw.exists() or 'Uncaught exception' in result.stdout:
            raise RuntimeError(result.stdout[-8000:])
        fixed = repair(raw.read_text())
        out.mkdir(exist_ok=True)
        (work/'raw').mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, work/'ad_primal', dirs_exist_ok=True)
        (work/'generation.log').write_text(result.stdout)
        for name in ('swe_adjoint_b.c', 'swe_primal_b.h', 'swe_adjoint_b.msg'):
            shutil.copy2(Path(tmp)/name, work/'raw'/name)
        (out/'swe_adjoint_b.c').write_text(fixed)
        shutil.copy2(Path(tmp)/'swe_primal_b.h', out/'swe_primal_b.h')
        manifest = dict(command="tapenade -reverse -head 'swe_misfit(cost)/(initial)' -I <staged-input> "
                                "-o swe_adjoint -O <build-output> <staged-input>/swe_primal.c",
                        generator=raw.read_text().splitlines()[1].strip(),
                        sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ('swe_primal.c','swe_primal.h','generate_adjoint.py','fix_tapenade_c.py',
                                             'tapenade_out/swe_adjoint_b.c','tapenade_out/swe_primal_b.h')})
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Generated adjoint, binomial schedule, raw output, and source manifest.')


if __name__ == '__main__':
    main()
