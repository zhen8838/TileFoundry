/// CUDA MMA op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "mma/mma_impl.h"

/// ``c += a @ b``, one entry, the tier read off the operand layouts.
///
/// Rank-2 static shard layouts on ``a`` and ``b`` are a tile, and the entry
/// loops the atom over it; anything else is a lane's gathered fragment and
/// takes the single instruction, and codegen never says which ([runtime
/// §3](docs/spec/runtime.md#3-runtime-ops)). ``a`` is ``(M, K)`` and ``b`` is
/// ``(N, K)`` with no transpose flag: k-major or n-major is a stride. ``c``
/// carries the warp count in its mesh and its fragment map in its layout.
template <class TA, class TB, class TC>
__device__ void mma(TA const &a, TB const &b, TC &c) {
    if constexpr (mma_impl::tile_shaped_v<TA, TB, TC>) {
        mma_impl::Tile{}(a, b, c);
    } else if constexpr (mma_impl::atom_shaped_v<TA, TB, TC>) {
        mma_impl::Atom{}(a, b, c);
    } else {
        static_assert(dependent_false_v<TA>,
                      "ops::mma: the operands are neither a rank-2 static tile "
                      "nor the atom's own (8, 4, 4) lane fragments -- the "
                      "single instruction reads those lengths off the operands "
                      "unconditionally, so anything else it reads past. State "
                      "the tile's shape in the shard layout, or hand over the "
                      "fragment the m16n8k16 atom takes");
    }
}
