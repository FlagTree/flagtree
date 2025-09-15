# ruff: noqa: F403, F405
from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import interpreter
from mlir.dialects.transform.vector import *
from mlir.dialects.transform.extras import sequence, apply_patterns
from mlir.dialects.builtin import module


def _get_transform_ir(ctx):
    with ctx, Location.unknown():

        @module()
        def transform_ir():

            @sequence([], transform.FailurePropagationMode.Propagate, [])
            def vector(target: transform.any_op_t()):

                @apply_patterns(target)
                def patterns():
                    ApplyDropUnitDimWithShapeCastPatternsOp()
                    ApplyLowerContractionPatternsOp(lowering_strategy=VectorContractLowering.Matmul)
                    ApplyTransferPermutationPatternsOp()
                    ApplyLowerMultiReductionPatternsOp(lowering_strategy=VectorMultiReductionLowering.InnerParallel)
                    ApplySplitTransferFullPartialPatternsOp(split_transfer_strategy=VectorTransferSplit.LinalgCopy)
                    ApplyTransferToScfPatternsOp(max_transfer_rank=1)
                    ApplyLowerTransferPatternsOp(max_transfer_rank=1)
                    ApplyLowerTransposePatternsOp(lowering_strategy=VectorTransposeLowering.Shuffle1D)

    return transform_ir


def vector_transform(module, ctx):
    transform_ir = _get_transform_ir(ctx)
    first_op = transform_ir.regions[0].blocks[0].operations[0]
    interpreter.apply_named_sequence(module, first_op, transform_ir)
