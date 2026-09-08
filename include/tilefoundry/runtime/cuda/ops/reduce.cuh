/// tilefoundry reduce op — single public ``reduce`` entry.
///
/// This file is included IN-CONTEXT from runtime.cuh, at the point inside
/// ``namespace tilefoundry::ops`` where the reduce surface used to live. It
/// therefore does NOT open ``namespace tilefoundry`` / ``ops`` and pulls in no
/// system headers — cute/std and the surrounding names (``detail::to_local``,
/// ``shard::S``/``shard::B``, ``TopologyScope``) are already in scope. The op
/// tags below must precede the impl-header includes because
/// ``reduce_impl::reduce_traits<Op>`` specializes on them.
#pragma once

/// Reduce combine-kind tags — pure compile-time markers. Semantics (init
/// value, elem/combine/finalize) live in one place per tag,
/// ``reduce_impl::reduce_traits<Op>`` (reduce/reduce_common_impl.h), consumed
/// uniformly by all four reduce tiers below.
struct mean_op {};
struct absmax_op {};

/// ── Sharded reduce ───────────────────────────────
/// Per-tier reduce building blocks; the public ``reduce`` entry below selects a
/// tier from the operand shard layouts. MEAN folds as SUM plus a final divide
/// by the total reduced extent.
#include "reduce/reduce_common_impl.h"
#include "reduce/reduce_intra_warp_impl.h"
#include "reduce/reduce_intra_cta_impl.h"
#include "reduce/reduce_cross_warp_impl.h"
#include "reduce/reduce_plain_impl.h"

/// Single public reduce entry: ``dst = reduce_kind(src)`` over ``Axes``.
///
/// Sharded operands select an intra-warp, intra-CTA or cross-warp tier and
/// **the layouts alone say which**: the (src, dst) attrs give the reduced mesh
/// axes, and how much of a warp each is gives ``lanes_reduced`` and
/// ``warps_per_group``. ``ws`` is therefore a resource the layouts *demand*
/// and not the tier selector -- as the selector, a cross-warp reduce called
/// without one took the intra-warp tier and returned one warp's partial as the
/// answer, with no diagnostic. Non-sharded operands take the plain fold.
template <class Op, class Axes, class Src, class Dst,
          class Ws = reduce_impl::no_workspace_t>
__device__ inline void reduce(Src const &src, Dst &dst, Ws &&ws = {}) {
    reduce_impl::check_reduce_domain<Src, Dst>();
    if constexpr (tilefoundry::detail::ShardTensorLike<Src>) {
        using SLs = typename cute::remove_cvref_t<Src>::shard_layout_type;
        using SLd = typename cute::remove_cvref_t<Dst>::shard_layout_type;
        constexpr reduce_impl::reduce_dispatch_info plan =
            reduce_impl::reduce_dispatch<SLs, SLd>();
        constexpr bool has_ws = !std::is_same_v<cute::remove_cvref_t<Ws>,
                                                reduce_impl::no_workspace_t>;
        static_assert(plan.warp_aligned,
                      "ops::reduce: a reduced mesh axis must divide into whole "
                      "lanes and whole warps -- 48 instances of one are 32 "
                      "lanes of a warp and 16 of the next, which is neither a "
                      "butterfly nor a slot per warp");
        if constexpr (!plan.mesh_reduced) {
            /// Nothing to cross: no axis holds a piece of this value, or
            /// the destination asked to stay ``P``. Folding what this
            /// instance holds is then the whole reduction.
            reduce_impl::Plain<Op, Axes>{}(src, dst);
        } else if constexpr (plan.warps_per_group == 1) {
            /// One warp holds every piece of a value, so a butterfly over
            /// ``lanes_reduced`` finishes it and there is nothing to stage.
            reduce_impl::IntraWarp<Op, Axes>{}(src, dst);
        } else if constexpr (!has_ws) {
            static_assert(
                dependent_false_v<Src>,
                "ops::reduce: these shard layouts spread one value over more "
                "than one warp, so the reduction cannot finish inside a warp "
                "-- pass a shared-memory workspace of one float per warp of "
                "the mesh. Without it this used to take the intra-warp tier "
                "and return a single warp's partial as the answer");
        } else if constexpr (plan.lane_reduced) {
            reduce_impl::IntraCta<Op, Axes>{}(src, dst, ws,
                                              plan.warps_per_group);
        } else {
            reduce_impl::CrossWarp<Op, Axes>{}(src, dst, ws,
                                               plan.warps_per_group);
        }
    } else {
        reduce_impl::Plain<Op, Axes>{}(src, dst);
    }
}
