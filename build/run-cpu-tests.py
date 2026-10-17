"""Run the active patches' CPU unit tests against the installed image."""
import json
import os
import pathlib
import subprocess
import sys

root = pathlib.Path('/opt/dsv41-platform')
lock = json.loads((root / 'stack.lock.json').read_text())


def node(path):
    # Lock paths are repository-relative; test/pytest.ini makes test/ the rootdir.
    return path.removeprefix('test/')


paths = list(dict.fromkeys(node(test) for patch in lock['patches'] if 'hold' not in patch
                           for test in patch['cpu_tests']))
command = [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', '-rfEs',
           f"--junitxml={root / 'provenance/cpu-tests.xml'}", *paths]
for entry in lock['cpu_test_deselect']:
    command += ['--deselect', node(entry['test'])]
# The image build has no GPUs. SGLang's test utilities index CUDA_VISIBLE_DEVICES,
# so it must be unset rather than empty.
env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'}
env['PYTHONDONTWRITEBYTECODE'] = '1'
subprocess.run(command, cwd=root / 'sources/sglang/test', env=env, check=True)
