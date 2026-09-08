"""Emitter for ``tir.nn.RMSNorm``, as the composition it is.

There is no ``ops::rmsnorm``: an RMS norm is a reduction and a pointwise pass,
and the project's rule is that an op is one or the other. So this emits both --
a ``reduce<add_op>`` of the squares, then one ``elementwise`` applying
``rsqrt`` and the weight -- which is the same arithmetic in the same order the
fused impl ran (checked bit-identical against it), now readable as what it is.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.codegen.cuda.tir.arith import broadcast_view
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.nn.rms_norm import RMSNorm


@register_codegen_cuda(RMSNorm)
def _emit(call, ctx: CodegenContext) -> None:
    """Four statements, and why there are four rather than three.

    The f32 squares scratch exists only because ``reduce`` takes a tensor and
    not a source expression; with one it would fold ``x * x`` on the way past.
    f32 whatever ``src`` is, so the sum accumulates the way the fused impl's
    ``float sum_sq`` did, and ``add_op`` rather than ``mean_op`` puts the
    divide by ``K`` beside ``eps``. Scratch shapes are ``(K, M)``: cute's
    linear index runs the leftmost mode fastest, so naming the contiguous axis
    first is what makes scratch element ``i`` mean ``src`` element ``i``.
    """
    src, dst, weight = call.args[0], call.args[1], call.args[2]
    if (
        not isinstance(src, Var)
        or not isinstance(dst, Var)
        or not isinstance(weight, Var)
    ):
        raise RuntimeError(
            "tir.nn.RMSNorm: demo path expects Var operands for src/dst/weight"
        )
    src_name = ctx.name_for(src)
    dst_name = ctx.name_for(dst)
    weight_name = ctx.name_for(weight)

    M = int(src.type.shape[0])
    K = int(src.type.shape[1])
    eps = call.target.eps

    ctx._counter += 1
    sq = f"rms_sq_{ctx._counter}"
    ss = f"rms_scale_{ctx._counter}"

    ctx.emit(f"float {sq}_buf[{M * K}];")
    ctx.emit(
        f"auto {sq} = cute::make_tensor(cute::make_rmem_ptr(&{sq}_buf[0]), "
        f"cute::make_layout(cute::make_shape(cute::Int<{K}>{{}}, "
        f"cute::Int<{M}>{{}}), cute::make_stride(cute::Int<1>{{}}, "
        f"cute::Int<{K}>{{}})));"
    )
    ctx.emit(f"float {ss}_buf[{M}];")
    ctx.emit(
        f"auto {ss} = cute::make_tensor(cute::make_rmem_ptr(&{ss}_buf[0]), "
        f"cute::make_layout(cute::make_shape(cute::Int<{M}>{{}})));"
    )
    ctx.emit(
        f"tilefoundry::ops::elementwise({sq}, [](auto x) "
        f"{{ return float(x) * float(x); }}, {src_name});"
    )
    ctx.emit(
        f"tilefoundry::ops::reduce<tilefoundry::ops::add_op, "
        f"cute::tuple<cute::Int<0>>>({sq}, {ss});"
    )
    ctx.emit(
        f"tilefoundry::ops::elementwise({ss}, [](float s) "
        f"{{ return rsqrtf(s / {float(K)}f + {eps}f); }}, {ss});"
    )
    scale_bc = broadcast_view(ss, ((K, M), (0, 1)), ctx)
    weight_bc = broadcast_view(weight_name, ((K, M), (1, 0)), ctx)
    ctx.emit(
        f"tilefoundry::ops::elementwise({dst_name}, "
        f"[](auto x, auto scale, auto w) "
        f"{{ return float(x) * scale * float(w); }}, "
        f"{src_name}, {scale_bc}, {weight_bc});"
    )
