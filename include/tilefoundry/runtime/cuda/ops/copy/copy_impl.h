/// CUDA copy op implementation. Included in-context from ops/copy.cuh inside
/// namespace tilefoundry::ops.
#pragma once

namespace copy_impl {

/// ``copy`` is ``cute::copy`` with the operands resolved to this slice first.
///
/// Nothing else: CuTe reads width and alignment off the two layouts and
/// recasts both *tensors*. Doing it here was worse -- it cast the pointer,
/// not the layout, so a view with a slow mode left read past its fast end.
///
/// The element conversion needs the half types' operators, which torch's
/// ``COMMON_NVCC_FLAGS`` strip; the build restores them
/// (``-U__CUDA_NO_*_CONVERSIONS__``) rather than the runtime carrying a layer.
struct Copy {
    template <class TSrc, class TDst>
    __device__ void operator()(TSrc const &src, TDst &dst) const {
        /// ``cute::copy`` decomposes both sides itself, so neither needs
        /// reorienting; a reversed view also moves the unit-stride mode to
        /// where ``recast`` works harder to find it -- measured 18% slower
        /// for no correctness gain.
        auto &&s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        cute::copy(s, d);
    }
};

/// Whether two projected views hold the same number of elements.
///
/// A dynamic extent on either side answers ``true`` and says nothing: such a
/// pair has already been refused by the width assertion, and asking
/// ``cute::size`` for a compile-time answer it has not got would replace that
/// sentence with an error inside CuTe.
template <class SView, class DView>
CUTE_HOST_DEVICE constexpr bool same_slice_size() {
    using SL = typename cute::remove_cvref_t<SView>::layout_type;
    using DL = typename cute::remove_cvref_t<DView>::layout_type;
    if constexpr (!cute::is_static<SL>::value || !cute::is_static<DL>::value)
        return true;
    else
        return int(decltype(cute::size(SL{}))::value) ==
               int(decltype(cute::size(DL{}))::value);
}

/// The same move, issued asynchronously.
///
/// A separate op rather than a tier of ``copy``: the caller has to wait on it,
/// so which one ran is part of what it is.
///
/// ``cp.async`` takes an address and a byte count, so the width cannot be left
/// to ``cute::copy`` here. It is still CuTe's answer -- the run both sides
/// share, capped by what both are aligned for -- just asked explicitly, and the
/// positions come from a ``recast`` of the tensors rather than from stepping a
/// pointer.
struct CopyAsync {
    template <class TSrc, class TDst>
    __device__ void operator()(TSrc const &src, TDst &dst) const {
        auto &&s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using value_type =
            typename cute::remove_cvref_t<decltype(d)>::value_type;
        constexpr int bytes = detail::async_bytes<decltype(s), decltype(d)>();
        static_assert(
            bytes > int(sizeof(value_type)),
            "ops::copy_async: these two views share no run wide enough for "
            "cp.async -- a dynamic extent on either side, or no 2-element "
            "common vector, leaves async_bytes at one element's width. The "
            "op's name is a promise, so such a pair is refused here rather "
            "than moved element by element while the caller waits on a "
            "pipeline group nothing was ever posted to. Use ops::copy, which "
            "is synchronous and says so");
        static_assert(
            same_slice_size<decltype(s), decltype(d)>(),
            "ops::copy_async: the two projected slices must hold the same "
            "number of elements -- this op moves one instance's share, and "
            "to_local has already placed each side at its own offset. Where "
            "an instance owns part of a wider destination that is the "
            "destination's shard layout to state, not the source's offset to "
            "borrow, which lands right only where both sides have the same "
            "layout geometry");
        constexpr int V = bytes / int(sizeof(value_type));
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
        auto s_v = cute::recast<cute::uint_bit_t<bytes * 8>>(s);
        const int nv = int(cute::size(s_v));
        for (int i = 0; i < nv; ++i)
            __pipeline_memcpy_async(&d(i * V), &s_v(i), bytes);
#else
        static_assert(dependent_false_v<TSrc>,
                      "ops::copy_async: cp.async is sm_80 and up; below it "
                      "there is no asynchronous move to issue and ops::copy is "
                      "the whole of what the hardware offers");
#endif
    }
};

}
