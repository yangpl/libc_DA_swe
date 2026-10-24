#!/usr/bin/env python3
"""Mechanical repairs for the installed Tapenade 3.16 C backend.

No derivative formulas or scheduler calls are replaced. Unexpected output
requires review instead of silently accepting changed replacement counts.
"""
import re

def repair(source):
    # VLA bounds are already declared function arguments, not global variables.
    source, count = re.subn(r'^UnknownType (m|n|nt|stride);\n', '', source, flags=re.M)
    if count != 4:
        raise RuntimeError('Unexpected VLA output; review repairs for this Tapenade version')
    replacements = {
        'adBinomial_next(action, step)': ('adBinomial_next(&action, &step)', 1),
        'pushReal8(*stage);': ('pushReal8Array(stage, 3*n);', 2),
        'pushReal8(*state);': ('pushReal8Array(state, 3*n);', 2),
        'popReal8(stage);': ('popReal8Array(stage, 3*n);', 2),
        'popReal8(state);': ('popReal8Array(state, 3*n);', 2),
    }
    for old, (new, expected) in replacements.items():
        if source.count(old) != expected:
            raise RuntimeError(f'Unexpected count for {old!r}; inspect new generator output')
        source = source.replace(old, new)
    return '/* Reproducible C backend repairs: see fix_tapenade_c.py. */\n' + source
