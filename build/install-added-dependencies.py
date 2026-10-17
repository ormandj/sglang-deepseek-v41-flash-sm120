"""Install the lock's runtime additions: exact version and wheel hash, no resolution."""
import json
import pathlib
import subprocess
import sys
import tempfile
from importlib import metadata

root = pathlib.Path('/opt/dsv41-platform')
lock = json.loads((root / 'stack.lock.json').read_text())
receipt = {}
for dep in lock['added_dependencies']:
    try:
        raise AssertionError(f"{dep['name']} already installed: {metadata.version(dep['name'])}")
    except metadata.PackageNotFoundError:
        pass
    assert dep['wheel'].startswith(f"{dep['name']}-{dep['version']}-") and dep['wheel'].endswith('.whl')
    with tempfile.NamedTemporaryFile('w', suffix='.txt') as requirements:
        requirements.write(f"{dep['name']}=={dep['version']} --hash=sha256:{dep['sha256']}\n")
        requirements.flush()
        # Its own requirements must already be satisfied by the installed stack.
        subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--no-deps',
                        '--require-hashes', '--only-binary', ':all:',
                        '--index-url', dep['index'], '-r', requirements.name], check=True)
    assert metadata.version(dep['name']) == dep['version']
    receipt[dep['name']] = {key: dep[key] for key in ('version', 'wheel', 'sha256')}
(root / 'provenance/added-dependencies.json').write_text(json.dumps(receipt, indent=2) + '\n')
