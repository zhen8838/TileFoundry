/// Warp-level hardware primitives. Included in-context from runtime.cuh inside
/// namespace tilefoundry.
///
/// Not ops, and so not in ``ops::``: [runtime
/// §3](docs/spec/runtime.md#3-runtime-ops)'s "What is an op" draws the line
/// once -- an op's operands are ShardTensors or a Mesh and its tier is read off
/// their layouts, and what takes a value and a lane count instead is a utility.
/// Each entry here is one instruction or a short run of them, and its own
/// single implementation.
#pragma once

namespace warp_util {

/// Whether ``__shfl_xor_sync`` takes ``T`` directly. Everything else goes
/// through the word-wise path below.
template <class T>
inline constexpr bool is_shuffle_native_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>;

/// ``__shfl_xor_sync`` for the types the intrinsic accepts, and a word-wise
/// exchange for anything else (a bf16 pair, a small aggregate).
template <class T> struct ShuffleXor {
    __device__ T operator()(T value, int lane_mask,
                            unsigned member_mask) const {
        if constexpr (is_shuffle_native_v<T>) {
            return __shfl_xor_sync(member_mask, value, lane_mask);
        } else {
            static_assert(
                sizeof(T) % sizeof(unsigned) == 0,
                "shuffle_xor: type size must be a multiple of 4 bytes");
            constexpr int kWords = int(sizeof(T) / sizeof(unsigned));
            T out = value;
            unsigned *dst = reinterpret_cast<unsigned *>(&out);
            for (int i = 0; i < kWords; ++i)
                dst[i] = __shfl_xor_sync(member_mask, dst[i], lane_mask);
            return out;
        }
    }
};

/// Elect exactly one thread of the CTA.
///
/// No width parameter: the elected thread is always in the first warp, so a
/// leading run wider than a warp elects what 32 threads would, and electing
/// among fewer is a different instruction. Threads past the first warp answer
/// false without executing the warp-scoped one, whose result is defined only
/// for the lanes in its member mask. On sm_90 that is ``elect.sync``, one
/// instruction where a ballot and a find-first would be three; below it lane 0.
struct Elect {
    __device__ bool operator()() const {
        /// The linearised id and not ``threadIdx.x``: a hardware warp is a run
        /// of it, so ids 0..31 are warp 0 whatever shape the block has, and on
        /// a flat block ``program_id`` folds to ``threadIdx.x`` anyway.
        const size_t tid = program_id<TopologyScope::thread>();
        if (tid >= size_t(kWarpSize))
            return false;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        /// Every lane of the warp participates, which is what makes the one
        /// elected thread the CTA's and not each warp's.
        constexpr unsigned kWholeWarp = 0xFFFFFFFFu;
        unsigned pred = 0u;
        asm volatile("{\n"
                     "  .reg .b32 elected_lane;\n"
                     "  .reg .pred is_elected;\n"
                     "  elect.sync elected_lane|is_elected, %1;\n"
                     "  selp.b32 %0, 1, 0, is_elected;\n"
                     "}\n"
                     : "=r"(pred)
                     : "n"(kWholeWarp));
        return pred != 0u;
#else
        return tid == 0;
#endif
    }
};

}

/// Exchange ``value`` with the lane whose id differs in ``lane_mask``.
template <class T>
__device__ inline T shuffle_xor(T value, int lane_mask,
                                unsigned member_mask = 0xFFFFFFFFu) {
    return warp_util::ShuffleXor<T>{}(value, lane_mask, member_mask);
}

/// One thread of the CTA answers true; every other answers false.
__device__ inline bool shuffle_elect() { return warp_util::Elect{}(); }

/// Fold ``value`` across ``Width`` lanes, every lane left holding the total.
///
/// A butterfly: ``log2(Width)`` shuffles and as many merges, not a loop over
/// the lanes. ``Width`` under 32 leaves the runs independent: a warp carrying
/// several rows folds each row across its own lanes.
///
/// ``Combine`` is any binary functor -- ``ops::add_op``, ``ops::max_op``, or
/// the caller's. It is called as ``Combine{}(a, b)``, the spelling ``binary``
/// and ``reduce`` use, so a merge is defined once for all three.
template <class Combine, int Width = 32, class T>
__device__ inline T warp_reduce(T value) {
    static_assert(Width >= 2 && Width <= 32 && (Width & (Width - 1)) == 0,
                  "warp_reduce: Width must be a power of two up to 32");
    /// After step ``k`` every lane holds the combination of its
    /// ``2^(k+1)``-lane block, so the last step leaves the run's total in every
    /// lane of it and not only in the run's first.
    for (int delta = Width >> 1; delta > 0; delta >>= 1)
        value = Combine{}(value, shuffle_xor(value, delta));
    return value;
}
