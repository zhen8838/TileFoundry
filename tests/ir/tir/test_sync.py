"""Exercise every CUDA barrier form emitted for ``T.sync``.

A 128-thread mesh and three slices of it cover the whole block, a single warp,
a multi-warp run based at zero, and a multi-warp run based inside the block --
the four sets ``ops::sync`` picks a different barrier for. Each emits the mesh
and nothing else, so what the assertions read is the size and base the runtime
decides from. Successful completion plus correct output catches a barrier the
wrong threads arrive at, and a deadlock.

See [tir §1.5](docs/spec/tir.md#15-sync).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tests.fixtures.tir.sync import SyncSquare
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget


def test_sync_barrier_forms_emit_expected_cuda() -> None:
    """Each ``T.sync`` lowers to its mesh, and to no barrier named beside it.

    The whole-block sync reaches for the scope's own alias; each slice carries
    its sub-box and the base it starts at, which is the pair ``ops::sync``
    reads to tell a block-wide barrier from a run inside one.
    """
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(SyncSquare, target=CudaTarget("nvidia.h200_sxm"))
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    src = emit_cuda_module(lowered, functions, target).source

    def mesh(shape: str, base: int, resource: str = "") -> str:
        layout = (
            f"cute::Layout<cute::Shape<{shape}>, "
            "cute::Stride<cute::Int<32>, cute::Int<1>>>"
        )
        if base:
            layout = f"cute::ComposedLayout<cute::identity, cute::Int<{base}>, {layout}>"
        return (
            "tilefoundry::ops::sync(tilefoundry::Mesh<"
            "tilefoundry::Topology<tilefoundry::TopologyScope::thread>, "
            f"{layout}>{{}}{resource});"
        )

    assert "tilefoundry::ops::sync(m_1_mesh_t{});" in src
    assert mesh("cute::Int<1>, cute::Int<32>", 0) in src
    assert mesh("cute::Int<2>, cute::Int<32>", 0, ", tilefoundry::ops::bar_id<1>") in src
    assert mesh("cute::Int<2>, cute::Int<32>", 64, ", tilefoundry::ops::bar_id<2>") in src

    assert "SyncKind" not in src


@module(entry="grid_sync_host")
class GridSync:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def grid_sync_device(a: Tensor[(128,), "f32"]):
        with Mesh((Topology("cta", 4),), Layout(shape=(4,), strides=(1,))) as m:
            T.sync(m)

    @prim_func(target=CpuTarget())
    def grid_sync_host(a: Tensor[(128,), "f32"]):
        launch(grid_sync_device, a, grid=(4, 1, 1), block=(128, 1, 1))  # noqa: F821


def test_grid_scope_sync_emits_grid_barrier() -> None:
    """Test grid scope sync emits grid barrier.

    A ``T.sync`` over a full ``cta``-topology mesh lowers to the grid-wide
    software barrier helper (not a within-block ``__syncthreads``), and the
    module defines its own internal-linkage counter for it.
    """
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(GridSync, target=CudaTarget("nvidia.h200_sxm"))
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    src = emit_cuda_module(lowered, functions, target).source
    assert (
        "tilefoundry::ops::sync(m_1_mesh_t{}, tilefoundry::tf_grid_bar_state);"
        in src
    )

    assert "static __device__ unsigned int tf_grid_bar_state[2];" in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sync_kernel_runs_and_squares() -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    expected = x.square()
    runtime = tilefoundry.compile(SyncSquare, target=CudaTarget("nvidia.h200_sxm"))
    runtime(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
