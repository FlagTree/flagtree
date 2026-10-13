# Copyright 2026- Xcoresigma Technology Co., Ltd

import pytest

from triton.experimental import tle
from triton.experimental.tle.language.dsa.ascend.pipe import (
    pipe_reader,
    pipe_value,
    pipe_wait_result,
    pipe_writer,
    reset_pipe_event_allocator,
)


def make_workspace(capacity=2, name="base"):
    return tle.dsa.workspace(name, capacity=capacity, shape=[4, 8], dtype="float32")


def make_pipe(capacity, scheduler=None, scope="cta", name=None, one_shot=False, **fields):
    return pipe_value(capacity, scope, name, scheduler, fields, one_shot=one_shot)


# one_shot pipes model a single cube->vector edge: capacity is pinned to 1,
# acquire/release are rejected, and the GPU-shaped is_closed flag is constant False.
def test_one_shot_semantics():
    reset_pipe_event_allocator()

    pipe = make_pipe(capacity=1, c=make_workspace(capacity=1), one_shot=True)
    assert pipe.one_shot is True

    with pytest.raises(ValueError, match="one_shot on Ascend requires capacity=1"):
        pipe_value(2, "cta", None, None, {}, one_shot=True)

    with pytest.raises(ValueError, match="one_shot pipes do not use acquire"):
        pipe_writer.acquire.__wrapped__(pipe.writer(), 0)

    with pytest.raises(ValueError, match="one_shot pipes do not use release"):
        pipe_reader.release.__wrapped__(pipe.reader(), 0)

    # No writer.close on Ascend: the GPU-shaped flag is a compile-time constant False.
    assert pipe_wait_result("slot").is_closed is False


# The public tle.pipe entry dispatches by payload kind and keeps the GPU
# frontend's parameter signature; run_pipeline stays an Ascend-only entry point.
def test_public_pipe_entry_matches_gpu_frontend():
    import inspect
    from triton.experimental.tle.language import pipe as public_pipe

    workspace = make_workspace()

    assert public_pipe._pipe_backend({"c": workspace}) == "ascend"

    with pytest.raises(ValueError, match="must be a tle.dsa.workspace payload"):
        public_pipe._pipe_backend({"c": "not-a-workspace"})

    # scheduler= is no longer a tle.pipe parameter: unknown kwargs fall into the
    # payload fields and fail generic payload validation.
    with pytest.raises(ValueError, match="tle.pipe field must be a tle.dsa.workspace payload"):
        public_pipe.pipe.__wrapped__(capacity=2, scheduler="not-a-scheduler", c=workspace)

    parameters = inspect.signature(public_pipe.pipe.__wrapped__).parameters
    assert "scheduler" not in parameters
    assert "readers" in parameters and "one_shot" in parameters

    # run_pipeline stays an Ascend-specific entry point.
    assert callable(tle.dsa.ascend.run_pipeline)
    assert not hasattr(tle.dsa, "run_pipeline")


# Without a semantic context (host-side construction) the pre-arm emission is
# deferred; one_shot pipes have no free direction to pre-arm and init directly.
def test_ensure_initialized_without_semantic_defers_emission():
    reset_pipe_event_allocator()

    pipe = make_pipe(capacity=2, c=make_workspace())
    pipe.ensure_initialized(None)
    assert pipe._inited is False

    one_shot = make_pipe(capacity=1, c=make_workspace(capacity=1), one_shot=True)
    one_shot.ensure_initialized(None)
    assert one_shot._inited is True
