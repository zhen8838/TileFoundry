/// CUDA sync op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "sync/sync_impl.h"

/// One of the fifteen named hardware barriers, as a type.
///
/// A type and not an ``int`` because the id must be a compile-time immediate
/// of ``bar.sync``, refused out of range where it is written. A ``consteval``
/// constructor would let ``sync(mesh, 3)`` do that, but nvcc 13.2 answers an
/// out-of-range argument to one with broken IR instead of a diagnosis.
///
/// Barrier 0 is refused too: ``__syncthreads`` arrives at it, so a subset of
/// the block posting to it releases a whole-block barrier the rest is still
/// walking toward.
template <int Id> struct BarrierId {
    static_assert(Id != 0,
                  "ops::sync: barrier 0 is the one __syncthreads arrives at; a "
                  "run inside the block must take one of 1..15, or it releases "
                  "a whole-block barrier the rest of the block is still "
                  "walking toward");
    static_assert(Id == 0 || (Id >= 1 && Id <= 15),
                  "ops::sync: a named barrier id must be 1..15 -- the hardware "
                  "has sixteen barriers and __syncthreads holds 0. Zero is let "
                  "through this condition so that the one id both assertions "
                  "are about is diagnosed once, by the sentence above that "
                  "explains it");
    static constexpr int value = Id;
};

/// ``bar_id<3>``, so the call site reads as an id and not as a type.
template <int Id> inline constexpr BarrierId<Id> bar_id{};

/// Synchronise every instance of ``mesh``.
///
/// The mesh says *which* barrier; it cannot say what the barrier is *made of*:
///
///     sync(block_mesh)                    // nothing to supply
///     sync(warp_mesh)                     // nothing to supply
///     sync(grid_mesh, tf_grid_bar_state)  // a counter the module owns
///     sync(sub_mesh, bar_id<3>)           // one of the 15 named barriers
///
/// The runtime allocates none: a grid counter belongs to the module, a free
/// named barrier to the whole kernel -- not to one mesh's template.

/// The tiers that need nothing.
///
/// An overload set rather than one defaulted parameter: a default would have
/// to stand for "no resource", and then ``sync(grid_mesh)`` compiles and spins
/// on a null counter. Here it does not compile, and the message names the
/// counter.
template <class TTopo, class TLayout>
__device__ inline void sync(Mesh<TTopo, TLayout> const &) {
    using mesh_t = Mesh<TTopo, TLayout>;
    constexpr auto tier = sync_impl::classify<mesh_t>();
    if constexpr (tier == sync_impl::Tier::warp)
        sync_impl::Warp{}();
    else if constexpr (tier == sync_impl::Tier::block)
        sync_impl::Block{}();
    else if constexpr (tier == sync_impl::Tier::grid)
        static_assert(dependent_false_v<mesh_t>,
                      "ops::sync: a CTA mesh needs the module's grid-barrier "
                      "counter -- sync(mesh, tilefoundry::tf_grid_bar_state)");
    else if constexpr (tier == sync_impl::Tier::named)
        static_assert(dependent_false_v<mesh_t>,
                      "ops::sync: a warp-aligned run inside the block needs a "
                      "named barrier id -- sync(mesh, "
                      "tilefoundry::ops::bar_id<n>) with n in 1..15");
    else
        sync_impl::reject_barrierless<tier, mesh_t>();
}

/// The grid, which needs the module's counter.
///
/// A null counter is the other grid barrier: a cooperative launch's grid group.
/// Which exists is a fact about the launch, so it stays the caller's to state.
template <class TTopo, class TLayout>
__device__ inline void sync(Mesh<TTopo, TLayout> const &, unsigned int *bar) {
    using mesh_t = Mesh<TTopo, TLayout>;
    constexpr auto tier = sync_impl::classify<mesh_t>();
    if constexpr (tier == sync_impl::Tier::grid)
        sync_impl::Grid{}(bar);
    else if constexpr (tier == sync_impl::Tier::warp ||
                       tier == sync_impl::Tier::block ||
                       tier == sync_impl::Tier::named)
        static_assert(
            dependent_false_v<mesh_t>,
            "ops::sync: the grid-barrier counter is the CTA mesh's "
            "resource; a warp-aligned run inside a block takes a named "
            "barrier id instead, and a whole block or warp takes nothing");
    else
        sync_impl::reject_barrierless<tier, mesh_t>();
}

/// A warp-aligned run inside the block, which needs one of the named barriers.
template <class TTopo, class TLayout, int Id>
__device__ inline void sync(Mesh<TTopo, TLayout> const &, BarrierId<Id>) {
    using mesh_t = Mesh<TTopo, TLayout>;
    constexpr auto tier = sync_impl::classify<mesh_t>();
    if constexpr (tier == sync_impl::Tier::named)
        sync_impl::Named<sync_impl::base<mesh_t>(),
                         sync_impl::instances<mesh_t>(), Id>{}();
    else if constexpr (tier == sync_impl::Tier::warp ||
                       tier == sync_impl::Tier::block ||
                       tier == sync_impl::Tier::grid)
        static_assert(
            dependent_false_v<mesh_t>,
            "ops::sync: a named barrier is what a warp-aligned run inside "
            "the block takes; a whole block or warp needs no resource, "
            "and a CTA mesh needs the grid-barrier counter");
    else
        sync_impl::reject_barrierless<tier, mesh_t>();
}

/// A bare integer, which is neither resource.
///
/// It exists to answer ``sync(mesh, 3)`` -- the spelling a reader reaches for
/// -- with what to write instead. Without it the integer would out-rank the
/// pointer overload only when it is ``0``, and a literal ``0`` would then be
/// read as a null grid counter: a cooperative-launch barrier where a named one
/// was meant.
template <class TTopo, class TLayout, class TInt>
requires std::is_integral_v<TInt> __device__ inline void
sync(Mesh<TTopo, TLayout> const &, TInt) {
    static_assert(sizeof(TInt) == 0,
                  "ops::sync: a named barrier id has to be a compile-time one "
                  "-- write sync(mesh, tilefoundry::ops::bar_id<3>), not "
                  "sync(mesh, 3)");
}
