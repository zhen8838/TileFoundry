"""A stride-0 operand of ``ops::elementwise``, which is the whole of broadcast.

``elementwise`` has one loop and no broadcast machinery: an operand that does
not span the destination's domain is read through a ``(shape, stride)`` pair
with a 0 on the axes it does not supply, bound with ``compose`` on the
operand's own projected layout. ``cell`` and ``col`` are one relation written
twice -- ``(M, 1)`` against ``(M, K)`` composes to ``(M, K):(1, 0)`` under
either name. Arity 2 with a bare tag already runs in ``test_sync.py``.

See [runtime §3](docs/spec/runtime.md#3-runtime-ops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


def _bc(shape, strides, m) -> ShardLayout:
    """A layout every instance holds whole, so ``local()`` is the allocation.

    The composed operand has to be read through the layout it actually lies
    under, and a broadcast shared tile is the shortest thing whose projected
    view is rank-2 and static -- which is what makes the ``(M, 1)`` against
    ``(M, K)`` relation the one the emitter sees. Shared and not register
    storage because a full-broadcast rmem allocation drops its writes; see
    ``test_copy_storage.py``'s xfail for that.
    """
    return ShardLayout(layout=Layout(shape=shape, strides=strides), attrs=(Broadcast(),), mesh=m)


@module(entry="col_bcast_host")
class ColumnBroadcast:
    """``dst(m, k) = lhs(m, k) * rhs(m)``, with ``rhs`` a stride-0 column.

    ``col``'s local shape is ``(4, 1)`` against a destination's ``(4, 8)``, so
    the emitter composes ``(4, 8):(1, 0)`` -- the same layout the cell relation
    produces, and the reason there is one branch for both.

    Shared allocations materialise column-major, so the fragments' element
    ``(m, k)`` sits at ``m + 4k`` and the flat global tensors are read and
    written in that order. The reference below is stated in the torch shape
    that makes that mapping an identity rather than a transpose.
    """

    @prim_func(target=_CUDA)
    def col_bcast_device(
        a: Tensor[(32,), "f32"], r: Tensor[(4,), "f32"], out: Tensor[(32,), "f32"]
    ):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as m:
            a_view = T.tensor_view(a, layout=_bc((32,), (1,), m))
            r_view = T.tensor_view(r, layout=_bc((4,), (1,), m))
            out_view = T.tensor_view(out, layout=_bc((32,), (1,), m))
            lhs = T.alloc_tensor(
                TensorType(
                    shape=(4, 8),
                    dtype=DType.f32,
                    layout=_bc((4, 8), (8, 1), m),
                    storage=StorageKind.SMEM,
                )
            )
            col = T.alloc_tensor(
                TensorType(
                    shape=(4, 1),
                    dtype=DType.f32,
                    layout=_bc((4, 1), (1, 1), m),
                    storage=StorageKind.SMEM,
                )
            )
            dst = T.alloc_tensor(
                TensorType(
                    shape=(4, 8),
                    dtype=DType.f32,
                    layout=_bc((4, 8), (8, 1), m),
                    storage=StorageKind.SMEM,
                )
            )
            T.copy(a_view, lhs)
            T.copy(r_view, col)
            T.sync(m)
            T.binary(lhs, col, dst, kind=BinaryKind.MUL)
            T.copy(dst, out_view)

    @prim_func(target=CpuTarget())
    def col_bcast_host(
        a: Tensor[(32,), "f32"], r: Tensor[(4,), "f32"], out: Tensor[(32,), "f32"]
    ):
        launch(col_bcast_device, a, r, out, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821


def test_the_column_operand_is_bound_through_a_stride_zero_layout() -> None:
    """A ``compose`` on the operand, not a second entry and not a fresh tensor.

    Composition asks the operand's own layout for element ``modes(i)``, so a
    strided or sharded operand is read where it lies; a fabricated stride-1
    tensor over its pointer would read where it does not. The stride pair is
    ``(1, 0)``, which is the cell relation's layout as much as the column's.
    """
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(ColumnBroadcast, target=_CUDA)
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    src = emit_cuda_module(lowered, functions, target).source
    assert (
        "cute::make_stride(cute::Int<1>{}, cute::Int<0>{})" in src
        and "tilefoundry::ops::detail::to_local(col_" in src
        and ".compose(" in src
    )
    assert "tilefoundry::ops::mul_op{}" in src
    assert "binary_impl" not in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_column_is_reread_for_every_element_of_its_row() -> None:
    """Each of the 8 elements of row ``m`` multiplies by the one ``r[m]``.

    A stride-0 mode that had been dropped, or read as stride 1, gives ``r``
    walked instead of held -- a different number in 28 of the 32 cells, which
    is what an exact comparison catches. The tiles are column-major over
    ``(4, 8)``, so flat index ``i`` is element ``(i % 4, i // 4)`` and the
    column factor is ``r[i % 4]``.
    """
    rm = tilefoundry.compile(ColumnBroadcast, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(32, dtype=torch.float32, device="cuda")
    r = torch.randn(4, dtype=torch.float32, device="cuda")
    out = torch.zeros(32, dtype=torch.float32, device="cuda")
    rm(a, r, out)
    torch.cuda.synchronize()
    expected = a * r.repeat(8)
    assert torch.allclose(out, expected, rtol=0, atol=0)
