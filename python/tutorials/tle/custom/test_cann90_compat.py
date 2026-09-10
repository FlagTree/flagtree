# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for version-gated custom-call compatibility."""
import importlib.util
import platform
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[4] / 'third_party/ascend/backend/custom_op_compat.py'
spec = importlib.util.spec_from_file_location('custom_op_compat_test', path)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


@pytest.fixture(autouse=True)
def toolkit(tmp_path, monkeypatch):
    monkeypatch.setenv('ASCEND_HOME_PATH', str(tmp_path))
    info = tmp_path / (platform.machine() + '-linux')
    info.mkdir()
    compat.is_cann90.cache_clear()
    yield info / 'ascend_toolkit_install.info'
    compat.is_cann90.cache_clear()


@pytest.mark.parametrize('version, expected', [('9.0.0', True), ('9.0.1', True), ('9.1.0', False), ('8.3.0', False)])
def test_version_gate(toolkit, version, expected):
    toolkit.write_text('version=' + version + '\n')
    text = 'hivm.hir.custom {symbol = "triton_cann90_cube_begin"}'
    assert compat.needs_compat(text) is expected
    assert not compat.needs_compat('hivm.hir.custom {symbol = "unadapted_op"}')


def test_unknown_toolkit_uses_native_path(toolkit):
    assert not compat.is_cann90()
    env = {'PATH': '/unchanged'}
    assert compat.compiler_command('module {}', '/compiler', env) == ('/compiler', env)


def test_no_output_preserves_regions():
    text = '''module {
  func.func @kernel(%pid: i32, %cond: i1, %buffer: memref<16xf32>) {
    scf.if %cond {
      hivm.hir.custom ins(%pid : i32) {symbol = "triton_cann90_cube_begin"}
    } else {
      hivm.hir.custom ins(%pid : i32) outs(%buffer : memref<16xf32>) {symbol = "triton_cann90_other", extra_attr = "triton_pass_outputs=true"}
    }
    hivm.hir.custom ins(%pid : i32) {symbol = "triton_cann90_cube_end"}
    return
  }
}
'''
    lowered = compat.lower_custom_op_to_call(text)
    assert '} else {' in lowered
    assert lowered.count('func.call @') == 3
    assert 'func.call @_mlir_ciface_triton_cann90_cube_begin(%pid) : (i32) -> ()' in lowered
    assert 'func.call @_mlir_ciface_triton_cann90_cube_end(%pid) : (i32) -> ()' in lowered
    assert 'memref.extract_aligned_pointer_as_index %buffer' in lowered
    assert 'hivm.hir.custom' not in lowered


def test_unsupported_scratch_rejected():
    with pytest.raises(ValueError, match='non-empty tmps'):
        compat.rewrite_custom_op_segments('operandSegmentSizes = array<i32: 1, 1, 1>')


def test_no_silent_unadapted_abi():
    with pytest.raises(ValueError, match='unadapted custom ABIs'):
        compat.lower_custom_op_to_call('module {\n  hivm.hir.custom ins(%x : i32) {symbol = "external_op"}\n}')


def test_mixed_core_annotations_preserve_vector():
    text = 'mix_mode = "aic" #hivm.tcore_type<CUBE> #hivm.tcore_type<VECTOR>'
    assert 'mix_mode = "mix"' in compat.prepare_linalg(text)
