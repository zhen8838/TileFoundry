"""Every reduce tier in one kernel, each forced there by its layouts alone.

``ops::reduce`` picks between four tiers from the (src, dst) shard attrs and
the mesh shape, and nothing at the call site says which. Every tier answers
with a number, so a mis-selected tier is a wrong answer rather than a failure,
which is why all four run on GPU against torch. The four pairs share one
module and one compile because the dispatch reads types only; which tier each
still selects is what the four emission tests below state, one per tier.

See [runtime §3](docs/spec/runtime.md#3-runtime-ops).
"""

from __future__ import annotations

import re

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


def _emit(mod) -> str:
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(mod, target=_CUDA)
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    return emit_cuda_module(lowered, functions, target).source


@module(entry="reduce_tiers_host")
class ReduceTiers:
    """Four operand pairs in one device function, one per reduce tier.

    Each pair carries the shard layouts it had when it was a module of its own,
    and ``reduce_dispatch`` reads nothing but those, so sharing a kernel cannot
    move a pair to another tier. The 32-lane pair runs under a 128-thread block
    on purpose: a mesh narrower than its level wraps, ``(tid / 1) % 32``, so all
    four warps repeat that one butterfly over the same rows and write the same
    answer into the same broadcast cell.
    """

    @prim_func(target=_CUDA)
    def reduce_tiers_device(
        a_plain: Tensor[(128, 8), "f32"],
        out_plain: Tensor[(128,), "f32"],
        a_warp: Tensor[(32, 4), "f32"],
        out_warp: Tensor[(1,), "f32"],
        a_cta: Tensor[(4, 32, 8), "f32"],
        out_cta: Tensor[(1,), "f32"],
        a_cross: Tensor[(4, 32), "f32"],
        out_cross: Tensor[(1, 32), "f32"],
    ):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as mp:
            plain_src = T.tensor_view(
                a_plain,
                layout=ShardLayout(
                    layout=Layout(shape=(128, 8), strides=(8, 1)),
                    attrs=(Split(0),),
                    mesh=mp,
                ),
            )
            plain_dst = T.tensor_view(
                out_plain,
                layout=ShardLayout(
                    layout=Layout(shape=(128,), strides=(1,)),
                    attrs=(Split(0),),
                    mesh=mp,
                ),
            )
            T.reduce(plain_src, plain_dst, axes=(1,), kind=ReduceKind.MEAN)
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mw:
            warp_src = T.tensor_view(
                a_warp,
                layout=ShardLayout(
                    layout=Layout(shape=(32, 4), strides=(4, 1)),
                    attrs=(Split(0),),
                    mesh=mw,
                ),
            )
            warp_dst = T.tensor_view(
                out_warp,
                layout=ShardLayout(
                    layout=Layout(shape=(1,), strides=(1,)),
                    attrs=(Broadcast(),),
                    mesh=mw,
                ),
            )
            T.reduce(warp_src, warp_dst, axes=(1,), kind=ReduceKind.ABS_MAX)
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as mc:
            cta_src = T.tensor_view(
                a_cta,
                layout=ShardLayout(
                    layout=Layout(shape=(4, 32, 8), strides=(256, 8, 1)),
                    attrs=(Split(0), Split(1)),
                    mesh=mc,
                ),
            )
            cta_dst = T.tensor_view(
                out_cta,
                layout=ShardLayout(
                    layout=Layout(shape=(1,), strides=(1,)),
                    attrs=(Broadcast(), Broadcast()),
                    mesh=mc,
                ),
            )
            cta_ws = T.alloc_tensor(
                TensorType(shape=(4,), dtype=DType.f32, layout=None, storage=StorageKind.SMEM)
            )
            T.reduce(cta_src, cta_dst, cta_ws, axes=(2,), kind=ReduceKind.MEAN)
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as mx:
            cross_src = T.tensor_view(
                a_cross,
                layout=ShardLayout(
                    layout=Layout(shape=(4, 32), strides=(32, 1)),
                    attrs=(Split(0), Split(1)),
                    mesh=mx,
                ),
            )
            cross_dst = T.tensor_view(
                out_cross,
                layout=ShardLayout(
                    layout=Layout(shape=(1, 32), strides=(32, 1)),
                    attrs=(Broadcast(), Split(1)),
                    mesh=mx,
                ),
            )
            cross_ws = T.alloc_tensor(
                TensorType(shape=(128,), dtype=DType.f32, layout=None, storage=StorageKind.SMEM)
            )
            T.reduce(cross_src, cross_dst, cross_ws, axes=(0,), kind=ReduceKind.ABS_MAX)

    @prim_func(target=CpuTarget())
    def reduce_tiers_host(
        a_plain: Tensor[(128, 8), "f32"],
        out_plain: Tensor[(128,), "f32"],
        a_warp: Tensor[(32, 4), "f32"],
        out_warp: Tensor[(1,), "f32"],
        a_cta: Tensor[(4, 32, 8), "f32"],
        out_cta: Tensor[(1,), "f32"],
        a_cross: Tensor[(4, 32), "f32"],
        out_cross: Tensor[(1, 32), "f32"],
    ):
        launch(  # noqa: F821
            reduce_tiers_device,  # noqa: F821
            a_plain,
            out_plain,
            a_warp,
            out_warp,
            a_cta,
            out_cta,
            a_cross,
            out_cross,
            grid=(1, 1, 1),
            block=(128, 1, 1),
        )


_ENTRY = r"tilefoundry::ops::reduce<tilefoundry::ops::"


def test_the_plain_pair_reaches_the_entry_with_no_workspace() -> None:
    """No mesh axis is reduced, so there is nothing to cross.

    ``plain_dst`` splits the same mesh axis ``plain_src`` does, so ``reduced[0]``
    is false and ``mesh_reduced`` with it: each thread owns one row of the
    ``(128, 8)`` source outright and folding that row is the whole reduction.
    The emitted line closes after two operands, which is the only compile-side
    trace this tier leaves -- a workspace here would mean a crossing tier.
    """
    src = _emit(ReduceTiers)
    assert re.search(
        _ENTRY + r"mean_op, cute::tuple<cute::Int<1>>>\(plain_src_\d+, plain_dst_\d+\);", src
    )
    assert "reduce_impl" not in src


def test_the_intra_warp_pair_reaches_the_entry_with_no_workspace() -> None:
    """One warp holds every piece of the value, so a butterfly finishes it.

    The mesh is a flat ``(32,)``: its single axis is 32 lanes and one warp, so
    ``warps_per_group`` is 1 and no workspace is wanted or accepted -- which the
    emitted line shows by closing after ``warp_dst``. ``absmax`` is what the
    butterfly's own combine shows, since ``warp_reduce`` is called with
    ``reduce_traits<absmax_op>::combine_op``.
    """
    src = _emit(ReduceTiers)
    assert re.search(
        _ENTRY + r"absmax_op, cute::tuple<cute::Int<1>>>\(warp_src_\d+, warp_dst_\d+\);", src
    )


def test_the_intra_cta_pair_reaches_the_entry_with_its_per_warp_slots() -> None:
    """Both mesh axes are reduced: lanes fold, and then warps fold.

    The mesh is ``(4, 32)`` row-major, so its fast axis is exactly a warp's 32
    lanes and its slow axis is 4 warps. Reducing both makes ``lane_reduced``
    true and ``warps_per_group`` 4, which is the intra-CTA tier and the one that
    needs a slot per warp -- so the workspace appears on the emitted line, and
    the axes pack names the source's own reduced axis, 2.
    """
    src = _emit(ReduceTiers)
    assert re.search(
        _ENTRY + r"mean_op, cute::tuple<cute::Int<2>>>\(cta_src_\d+, cta_dst_\d+, cta_ws_\d+\);",
        src,
    )


def test_the_cross_warp_pair_reaches_the_entry_with_its_per_lane_slots() -> None:
    """Only the warp axis is reduced, so each lane folds its own column.

    ``cross_dst`` broadcasts mesh axis 0 (the 4 warps) and *splits* mesh axis 1
    (the 32 lanes), so only axis 0 is reduced -- and axis 0 of a row-major
    ``(4, 32)`` mesh is whole warps, never lanes. ``lane_reduced`` is therefore
    false with ``warps_per_group == 4``, the one combination that selects the
    cross-warp tier: 32 results, each staged in its own ``(warp, lane)`` slot.
    """
    src = _emit(ReduceTiers)
    assert re.search(
        _ENTRY
        + r"absmax_op, cute::tuple<cute::Int<0>>>\(cross_src_\d+, cross_dst_\d+, cross_ws_\d+\);",
        src,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_every_tier_answers_what_torch_answers() -> None:
    """One compile behind four assertions, each naming the tier that failed.

    Four operand pairs and four destinations, asserted one at a time, so a wrong
    number is read back from the tier that produced it; a single combined
    comparison would say only that one of the four is wrong.
    """
    rm = tilefoundry.compile(ReduceTiers, target=_CUDA)
    torch.manual_seed(0)
    a_plain = torch.randn(128, 8, dtype=torch.float32, device="cuda")
    out_plain = torch.zeros(128, dtype=torch.float32, device="cuda")
    a_warp = torch.randn(32, 4, dtype=torch.float32, device="cuda")
    out_warp = torch.zeros(1, dtype=torch.float32, device="cuda")
    a_cta = torch.randn(4, 32, 8, dtype=torch.float32, device="cuda")
    out_cta = torch.zeros(1, dtype=torch.float32, device="cuda")
    a_cross = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    out_cross = torch.zeros(1, 32, dtype=torch.float32, device="cuda")
    rm(a_plain, out_plain, a_warp, out_warp, a_cta, out_cta, a_cross, out_cross)
    torch.cuda.synchronize()
    assert torch.allclose(out_plain, a_plain.mean(dim=1), rtol=1e-6, atol=1e-6), (
        "plain tier (mesh_reduced false: dst splits the axis src splits): mean "
        "divides by the reduced span alone -- 8 -- which is the only count this "
        "tier has, so a tier that had brought a mesh extent into the divisor "
        "answers 8 or 32 times small"
    )
    assert torch.allclose(out_warp, a_warp.abs().max().reshape(1), rtol=0, atol=0), (
        "intra-warp tier (warps_per_group == 1 on a flat 32-lane mesh): a fold "
        "that added the lanes' maxima instead of maxing them answers roughly 32 "
        "times large"
    )
    assert torch.allclose(out_cta, a_cta.mean().reshape(1), rtol=1e-5, atol=1e-6), (
        "intra-CTA tier (lane_reduced with warps_per_group == 4): mean's divisor "
        "is the product of all three counts the tier knows -- span 8, lanes 32, "
        "warps 4 -- so the 1024 it must divide by is exactly what the greedy "
        "warp walk this dispatch replaced got wrong"
    )
    assert torch.allclose(out_cross[0], a_cross.abs().amax(dim=0), rtol=0, atol=0), (
        "cross-warp tier (warps_per_group == 4 with lane_reduced false): absmax "
        "is what shows the fold combining with the reduction's own operator "
        "rather than adding"
    )
