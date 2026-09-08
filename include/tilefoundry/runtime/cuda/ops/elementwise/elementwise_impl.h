/// CUDA elementwise op implementation. Included in-context from
/// ops/elementwise.cuh inside namespace tilefoundry::ops.
#pragma once

namespace elementwise_impl {

/// One loop, arity 0..N. A fill, a map and a combine are one instantiation
/// rather than three skeletons that have to be kept agreeing.
///
/// Three deliberate absences. No shape or domain check: a mismatched operand
/// is a codegen bug, and shapes are computed on the compile side. No extent
/// parameter: ``cute::size`` of the projected destination is the count, where a
/// passed ``N`` is the same fact stated twice and free to disagree. No
/// ``static_cast``: the only conversion is the one ``d(i) = ...`` implies, and
/// any other is ``fn``'s to state -- as in TIR, where a cast is a node.
struct Elementwise {
    /// A plain loop, and ``to_local`` inside it. Both were measured.
    ///
    /// No ``CUTE_UNROLL``: the extent is often a run-time one here, and forcing
    /// an unrolled body plus a remainder over a dynamic count cost the mega
    /// kernel 120 instructions and 0.5% a token. Hoisting the pack of
    /// ``to_local``s, or the ``size`` out of the condition, changes no
    /// instruction at ``-O3``, so neither buys the machinery it would take.
    template <class Fn, class TOut, class... TIn>
    __device__ void operator()(TOut &dst, Fn fn, TIn const &...src) const {
        auto &&d = detail::to_local(dst);
        for (int i = 0; i < int(cute::size(d)); ++i) {
            d(i) = fn(detail::to_local(src)(i)...);
        }
    }
};

}
