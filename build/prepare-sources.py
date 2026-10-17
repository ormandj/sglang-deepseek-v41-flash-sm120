"""Fetch the locked sources, apply their patch series and packaging edits.

usage: prepare-sources.py ROOT MANIFEST SOURCE...

MANIFEST (relative to ROOT) is stack.lock.json or the stack image's stack.json;
only the named sources are prepared, so the stack image prepares FlashInfer,
DeepGEMM and Triton and the candidate image prepares SGLang on top of it. Fails
unless every commit and tree, every patch hash and each effective tree match.
Writes source receipts and packaging diffs.
"""
import hashlib
import json
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
lock = json.loads((root / sys.argv[2]).read_text())
names = sys.argv[3:]
assert names and set(names) <= set(lock['sources']), names
provenance = root / 'provenance'
provenance.mkdir(exist_ok=True)


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


receipt = {}
for name in names:
    source = lock['sources'][name]
    path = root / 'sources' / name
    path.mkdir(parents=True)
    git(path, 'init', '-q')
    git(path, 'remote', 'add', 'origin', source['repository'])
    git(path, 'fetch', '-q', '--depth=1', 'origin', source['commit'])
    git(path, 'checkout', '-q', '--detach', 'FETCH_HEAD')
    assert git(path, 'rev-parse', 'HEAD') == source['commit'], name
    assert git(path, 'rev-parse', 'HEAD^{tree}') == source['tree'], name
    git(path, 'submodule', 'update', '-q', '--init', '--recursive', '--depth=1')
    receipt[name] = dict(source, submodules=git(path, 'submodule', 'status', '--recursive'))

# Each series applies strictly in lock order: the SGLang series, the FlashInfer
# series, then the Triton backports. Held entries stay in the lock but are not
# part of the image.
series = [('sglang', p) for p in lock.get('patches', []) if 'hold' not in p]
series += [('flashinfer', p) for p in lock.get('flashinfer_patches', [])]
series += [('triton', p) for p in lock.get('triton_patches', [])]
for name, patch in series:
    if name not in names:
        continue
    data = (root / patch['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == patch['sha256'], patch['path']
    subprocess.run(['git', '-C', str(root / 'sources' / name), 'apply', '--index',
                    str(root / patch['path'])], check=True)
    receipt[name].setdefault('applied_patches', []).append(
        {key: patch[key] for key in ('path', 'sha256', 'head', 'upstream_pr')})
for name in names:
    source = lock['sources'][name]
    effective = git(root / 'sources' / name, 'write-tree')
    assert effective == source.get('effective_tree', source['tree']), (name, effective)

# Packaging edits stay unstaged, so the index keeps the effective tree.
packaging = lock['packaging']
if 'sglang' in names:
    pyproject = root / 'sources/sglang/python/pyproject.toml'
    text = pyproject.read_text()
    text, count = re.subn(r'"flashinfer_python\[cu13\]==[^"]+"',
                          f'"flashinfer_python[cu13]=={packaging["flashinfer_python"]}"', text)
    assert count == 1, 'SGLang FlashInfer requirement'
    pyproject.write_text(text)
if 'flashinfer' in names:
    flashinfer = root / 'sources/flashinfer'
    assert (flashinfer / 'version.txt').read_text().strip() == packaging['flashinfer_python']
    artifacts = hashlib.sha256((flashinfer / 'flashinfer/artifacts.py').read_bytes()).hexdigest()
    assert artifacts == packaging['flashinfer_artifacts_sha256'], 'FlashInfer artifact manifest'
if 'deepgemm' in names:
    (root / 'sources/deepgemm/sgl_deep_gemm/VERSION').write_text(packaging['sgl_deep_gemm'] + '\n')

for name in names:
    diff = subprocess.check_output(['git', '-C', str(root / 'sources' / name), 'diff', '--binary'])
    (provenance / f'{name}-packaging.patch').write_bytes(diff)
    receipt[name]['packaging_diff_sha256'] = hashlib.sha256(diff).hexdigest()
(provenance / f"source-receipt-{'-'.join(names)}.json").write_text(json.dumps(receipt, indent=2) + '\n')
for name in names:
    print(f"{name} ready: {len(receipt[name].get('applied_patches', []))} patches, "
          f"tree {lock['sources'][name].get('effective_tree', lock['sources'][name]['tree'])}")
