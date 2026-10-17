"""Fail closed on every `uv pip check` result except the lock's vendor substitutions.

Every problem is collected and written to the receipt before the gate fails.
"""
import json
import pathlib
import re
import subprocess
from importlib import metadata

from packaging.specifiers import SpecifierSet

root = pathlib.Path('/opt/dsv41-platform')
gate = json.loads((root / 'stack.lock.json').read_text())['dependency_gate']
check = subprocess.run(['uv', 'pip', 'check', '--python', '/opt/sglang/bin/python'],
                       check=False, text=True, capture_output=True)
output = check.stdout + check.stderr
(root / 'provenance/pip-check.txt').write_text(output)
print(output)


def installed(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


summary = re.compile(r'Using Python .*|Checked \d+ packages? in .+|Found \d+ incompatibilit(?:y|ies)'
                     r'|All installed packages are compatible')
lines = [line for line in output.splitlines() if line]
conflicts = [line for line in lines if line.startswith('The package ')]
receipt = {
    'allowed': [line for line in conflicts if line in gate['allowed_conflicts']],
    'rejected': [line for line in conflicts if line not in gate['allowed_conflicts']],
    'unrecognized': [line for line in lines
                     if not line.startswith('The package ') and not summary.fullmatch(line)],
    'unexpected_present': {name: installed(name) for name in gate['absent'] if installed(name)},
    'required_mismatch': {name: installed(name) for name, spec in gate['required'].items()
                          if not (installed(name) and SpecifierSet(spec).contains(installed(name)))},
    'returncode': check.returncode,
}
(root / 'provenance/dependency-check.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
assert check.returncode in (0, 1), 'uv pip check did not run'
assert check.returncode == 0 or conflicts, 'uv pip check failed without a classified conflict'
assert not any(receipt[key] for key in
               ('rejected', 'unrecognized', 'unexpected_present', 'required_mismatch')), receipt
