/// tilefoundry TMA op — one public entry, the operands checked against it.
///
/// Included IN-CONTEXT from runtime.cuh inside ``namespace tilefoundry::ops``;
/// it opens no namespace and pulls in no system headers.

/// The entry takes tensors, not addresses. A caller that had to pass a byte
/// count and a raw pointer would be deciding the vector width, the row stride
/// and the bounds itself -- which is the layout's job, and the reason
/// [runtime §3](docs/spec/runtime.md#3-runtime-ops) puts the instruction behind
/// one entry rather than in the call site.
#pragma once

#include "tma/tma_impl.h"

/// Stage ``src`` into ``dst``, completing on ``bar``.
///
/// One instruction, with one run-time hand-off inside it: the elected thread
/// issues ``cp.async.bulk``, or the element loop instead for a byte count off
/// the 16-byte grain, which is what the shard leaves behind rather than a fact
/// about a layout type. Either way ``bar`` completes when the data is readable,
/// so the caller waits on the phase and never learns which ran, and either way
/// it is safe to call from every thread in the block. What the operands have to
/// be for both is ``check_tma_operands`` -- an assertion, not a tier.
template <class Src, class Dst>
__device__ inline void tma_copy(Src const &src, Dst &dst, uint64_t *bar) {
    tma_impl::check_tma_operands<Src, Dst>();
    tma_impl::Bulk{}(src, dst, bar);
}
