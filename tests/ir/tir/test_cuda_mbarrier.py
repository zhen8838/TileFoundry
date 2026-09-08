"""Cover the CUDA mbarrier definitions and the instructions they emit.

There is no runtime entry to look for in the output: an mbarrier is a word in
shared memory, so nothing here reads a layout and none of it is an op. What the
emission assertions read is the instruction text itself, plus the shared-window
conversion each one takes.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.cuda.sync.mbarrier import (
    MBarrierArriveExpectTx,
    MBarrierInit,
    MBarrierInvalidate,
    MBarrierWaitParity,
)
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType, make_tensor_type
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget

_SMEM_BAR = make_tensor_type((1,), DType.i64, storage="smem")
_GMEM_BAR = make_tensor_type((1,), DType.i64, storage="gmem")
_PHASE = make_tensor_type((), DType.i32, storage="rmem")


def _pf(op, *types) -> PrimFunction:
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(body=(Evaluate(callable=op, args=args), Return())),
    )


def test_init_accepts_a_shared_barrier_and_a_positive_count() -> None:
    verify_prim_function(_pf(MBarrierInit(arrive_count=1), _SMEM_BAR))


@pytest.mark.parametrize("count", [0, -1])
def test_init_refuses_a_non_positive_arrive_count(count: int) -> None:
    """A non-positive arrive count is refused.

    A phase needing zero arrivals is complete before anything is produced, which
    turns every consumer's wait into a no-op.
    """
    with pytest.raises(VerifyError, match="arrive_count must be a positive int"):
        verify_prim_function(_pf(MBarrierInit(arrive_count=count), _SMEM_BAR))


def test_invalidate_accepts_a_shared_barrier() -> None:
    verify_prim_function(_pf(MBarrierInvalidate(), _SMEM_BAR))


@pytest.mark.parametrize(
    "stated",
    [
        pytest.param(_pf(MBarrierInit(arrive_count=1), _GMEM_BAR), id="init"),
        pytest.param(_pf(MBarrierArriveExpectTx(tx_bytes=16), _GMEM_BAR), id="arrive_expect_tx"),
        pytest.param(_pf(MBarrierWaitParity(), _GMEM_BAR, _PHASE), id="wait_parity"),
        pytest.param(_pf(MBarrierInvalidate(), _GMEM_BAR), id="invalidate"),
    ],
)
def test_every_entry_refuses_a_barrier_outside_shared_memory(stated) -> None:
    """A barrier outside shared memory is refused.

    The instructions take a shared-window address, so a barrier in global memory
    is not a slower barrier -- it is not one at all.
    """
    with pytest.raises(VerifyError, match="barrier must be smem"):
        verify_prim_function(stated)


def test_arrive_expect_tx_accepts_a_positive_byte_count() -> None:
    verify_prim_function(_pf(MBarrierArriveExpectTx(tx_bytes=4096), _SMEM_BAR))


@pytest.mark.parametrize("tx", [0, -16])
def test_arrive_expect_tx_refuses_a_non_positive_byte_count(tx: int) -> None:
    with pytest.raises(VerifyError, match="tx_bytes must be a positive int"):
        verify_prim_function(_pf(MBarrierArriveExpectTx(tx_bytes=tx), _SMEM_BAR))


def test_wait_parity_accepts_a_shared_barrier_and_a_phase() -> None:
    verify_prim_function(_pf(MBarrierWaitParity(), _SMEM_BAR, _PHASE))


@module(entry="mbarrier_ring_host")
class MBarrierRing:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def mbarrier_ring_device(a: Tensor[(4,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as m:
            bar = T.alloc_tensor(
                TensorType(shape=(1,), dtype=DType.i64, layout=None, storage=StorageKind.SMEM)
            )
            T.mbarrier_init(bar, arrive_count=1)
            T.sync(m)
            T.mbarrier_arrive_expect_tx(bar, tx_bytes=1024)
            T.mbarrier_wait_parity(bar, 0)
            T.mbarrier_invalidate(bar)

    @prim_func(target=CpuTarget())
    def mbarrier_ring_host(a: Tensor[(4,), "f32"]):
        launch(mbarrier_ring_device, a, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


def _emitted() -> str:
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(MBarrierRing, target=CudaTarget("nvidia.h200_sxm"))
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    return emit_cuda_module(lowered, functions, target).source


@pytest.mark.parametrize(
    "instruction",
    [
        "mbarrier.init.shared::cta.b64 [%0], %1;",
        "mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;",
        "mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], %2;",
        "mbarrier.inval.shared::cta.b64 [%0];",
    ],
)
def test_each_entry_emits_its_instruction(instruction: str) -> None:
    assert instruction in _emitted()


def test_the_attributes_reach_the_instruction_as_immediates() -> None:
    """``arrive_count`` and ``tx_bytes`` are compile-time counts, so they land inline."""
    src = _emitted()
    assert '"r"(1u)' in src
    assert '"r"(1024u)' in src


def test_no_runtime_entry_is_named() -> None:
    """None of this is an op, so nothing here reaches for ``tilefoundry::ops::``.

    The address does go through ``__cvta_generic_to_shared``: the instructions
    name the ``.shared::cta`` window, so the conversion is part of the call and
    not something the assembler is left to redo.
    """
    src = _emitted()
    assert "ops::mbarrier" not in src
    assert src.count("__cvta_generic_to_shared") == 4


def test_the_phase_wait_spins_in_c_plus_plus_not_in_ptx() -> None:
    """A PTX label is emitted once per instantiation and collides on the second.

    ``try_wait`` already parks the warp in hardware for a bounded interval, so a
    single non-blocking test under a C++ loop is what the instruction is for.
    """
    src = _emitted()
    assert "while (!tf_mbar_ready) {" in src
    assert "selp.b32 %0, 1, 0, complete;" in src
