"""CPU-only identity and import gate for the installed stack."""
import importlib
import json
import pathlib
import subprocess
from importlib import metadata

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = pathlib.Path('/opt/dsv41-platform')
lock = json.loads((root / 'stack.lock.json').read_text())
packaging = lock['packaging']

for name, source in lock['sources'].items():
    path = root / 'sources' / name
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == source['commit'], (name, head)
    # The index still holds exactly the locked series (packaging edits are unstaged).
    tree = subprocess.check_output(['git', '-C', str(path), 'write-tree'], text=True).strip()
    assert tree == source.get('effective_tree', source['tree']), (name, tree)

# SGLang's own pins hold for everything the build did not deliberately replace.
pyproject = tomllib.loads((root / 'sources/sglang/python/pyproject.toml').read_text())
pins = {canonicalize_name(Requirement(r).name): Requirement(r)
        for r in pyproject['project']['dependencies']}
for name in ('torch', 'sglang-kernel', 'nvidia-cutlass-dsl', 'quack-kernels', 'sgl-deep-gemm',
             'flashinfer-python', 'apache-tvm-ffi'):
    assert pins[name].specifier.contains(metadata.version(name), prereleases=True), (
        name, metadata.version(name), str(pins[name]))
assert metadata.version('flashinfer-python') == packaging['flashinfer_python']
assert metadata.version('flashinfer-cubin') == packaging['flashinfer_python']
assert metadata.version('sgl-deep-gemm') == packaging['sgl_deep_gemm']
# Triton is the locked source build (installed from its wheel file), not the vendor's.
triton_url = json.loads(metadata.distribution('triton').read_text('direct_url.json'))['url']
assert triton_url.startswith('file://') and '/triton-wheel/' in triton_url, triton_url
for dep in lock['added_dependencies']:
    assert metadata.version(dep['name']) == dep['version'], dep['name']

receipt = {}
for name in ('torch', 'triton', 'sglang', 'flashinfer', 'flashinfer_cubin', 'deep_gemm', 'tvm_ffi',
             'transformers'):
    module = importlib.import_module(name)
    receipt[name] = {'path': module.__file__, 'version': getattr(module, '__version__', None)}
assert pathlib.Path(receipt['sglang']['path']).is_relative_to(root / 'sources/sglang')
flashinfer_commit = lock['sources']['flashinfer']['commit']
assert importlib.import_module('flashinfer').__git_commit__ == flashinfer_commit
assert importlib.import_module('flashinfer_cubin').__git_version__ == flashinfer_commit
assert receipt['deep_gemm']['version'] == packaging['sgl_deep_gemm']

# Entry points of the serving configuration and of the carried series.
from quack import copy_utils, layout_utils  # noqa: F401
from quack.cute_dsl_utils import ParamsBase  # noqa: F401
from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.function_call.deepseekv41_detector import (
    DeepSeekV41Detector,  # noqa: F401
)

assert DeepseekV41Config.model_type == 'deepseek_v41'
assert LoadFormat('instanttensor') is LoadFormat.INSTANTTENSOR
for module in ('sglang.kernels.ops.gemm.sm120_mxfp8_skinny_gemm',
               'sglang.srt.layers.attention.dsv4.v41_indexer',
               'sglang.srt.mem_cache.unified_cache.tree_core_registry'):
    importlib.import_module(module)
(root / 'provenance/import-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
