/// ``ops::detail`` view helpers. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
///
/// The projection itself lives beside ``local()`` in shard_tensor.cuh; this
/// only carries the names into ``ops::detail``, where every op reaches for
/// them unqualified.
#pragma once

namespace detail {

/// Bytes per asynchronous move: what both views run contiguously over, capped
/// by what both are aligned for and by the 16-byte hardware maximum.
///
/// CuTe's own formula (``algorithm/copy.hpp``), asked explicitly because
/// ``cp.async`` needs the number rather than a recast tensor. A dynamic layout
/// admits no compile-time answer, so it moves one element at a time.
template <class SView, class DView>
CUTE_HOST_DEVICE constexpr int async_bytes() {
    using SL = typename cute::remove_cvref_t<SView>::layout_type;
    using DL = typename cute::remove_cvref_t<DView>::layout_type;
    using elem = typename cute::remove_cvref_t<DView>::value_type;
    if constexpr (!cute::is_static<SL>::value || !cute::is_static<DL>::value) {
        return int(sizeof(elem));
    } else {
        constexpr int run = int(
            decltype(cute::gcd(cute::max_common_vector(SL{}, DL{}),
                               cute::gcd(cute::max_alignment(SL{}),
                                         cute::max_alignment(DL{}))))::value);
        int b = int(sizeof(elem));
        while (b * 2 <= run * int(sizeof(elem)) && b * 2 <= 16)
            b *= 2;
        return b;
    }
}

using tilefoundry::detail::is_shard_tensor;
using tilefoundry::detail::local_view_t;
using tilefoundry::detail::to_local;
}
