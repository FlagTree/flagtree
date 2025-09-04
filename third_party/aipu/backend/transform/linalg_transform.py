# ruff: noqa: F403, F405
from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import interpreter
from mlir.dialects.transform.structured import *
from mlir.dialects.transform.extras import sequence, apply_patterns
from mlir.dialects.builtin import module


def _get_transform_ir(ctx):
    with ctx, Location.unknown():

        @module()
        def transform_ir():
            any_op_t = transform.any_op_t()

            @sequence([], transform.FailurePropagationMode.Propagate, [])
            def linalg(target: any_op_t):

                @apply_patterns(target)
                def patterns():
                    ApplyFoldUnitExtentDimsViaReshapesPatternsOp()

                EliminateLinalgOpAnchoredEmptyTensorsOp(target)

    return transform_ir


def linalg_transform(module, ctx):
    transform_ir = _get_transform_ir(ctx)
    first_op = transform_ir.regions[0].blocks[0].operations[0]
    interpreter.apply_named_sequence(module, first_op, transform_ir)
