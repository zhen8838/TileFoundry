"""Emitters for the mbarrier TIR ops — each writes its instruction inline.

There is no ``tilefoundry::ops::`` entry to call, so what the emitter writes is
the instruction on the word's shared-window address -- the shape
``CpAsyncCommit`` and ``CpAsyncWait`` already take. The text is the one the
hand-written kernels carry, ``__cvta_generic_to_shared`` included.

``try_wait.parity`` is one non-blocking test under a C++ loop: a label inside
inline asm is emitted once per instantiation and collides on the second.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.codegen.cuda.tir.stmts.scalar_expr import render_scalar_expr
from tilefoundry.ir.tir.cuda.sync.mbarrier import (
    MBarrierArriveExpectTx,
    MBarrierInit,
    MBarrierInvalidate,
    MBarrierWaitParity,
)
from tilefoundry.ir.types.shard.shard_layout import ShardLayout


def barrier_word(var, ctx: CodegenContext) -> str:
    """The address of the barrier word itself, as a generic pointer.

    The op's operand is a tensor because that is what TIR has to name a piece
    of shared memory with; the instruction wants the word. A kernel parameter
    wears the ``_tensor`` view the prologue built, and a sharded barrier is
    projected first so the address is this instance's own slot rather than the
    ring's base.
    """
    base = ctx.name_for(var)
    if ctx.is_kernel_param(var):
        base = f"{base}_tensor"
    if isinstance(getattr(getattr(var, "type", None), "layout", None), ShardLayout):
        base = f"tilefoundry::local({base})"
    return f"&{base}(0)"


def _smem_addr(var, ctx: CodegenContext) -> str:
    """The barrier's shared-window address, which every ``mbarrier.*`` takes."""
    return f"static_cast<unsigned>(__cvta_generic_to_shared({barrier_word(var, ctx)}))"


@register_codegen_cuda(MBarrierInit)
def _emit_init(call, ctx: CodegenContext) -> None:
    addr = _smem_addr(call.args[0], ctx)
    count = int(call.target.arrive_count)
    ctx.emit(
        f'asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" '
        f':: "r"({addr}), "r"({count}u));'
    )


@register_codegen_cuda(MBarrierArriveExpectTx)
def _emit_arrive_expect_tx(call, ctx: CodegenContext) -> None:
    addr = _smem_addr(call.args[0], ctx)
    tx = int(call.target.tx_bytes)
    body = (
        "{\\n  .reg .b64 state;\\n"
        "  mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\\n}\\n"
    )
    ctx.emit(f'asm volatile("{body}" :: "r"({addr}), "r"({tx}u));')


@register_codegen_cuda(MBarrierWaitParity)
def _emit_wait_parity(call, ctx: CodegenContext) -> None:
    addr = _smem_addr(call.args[0], ctx)
    phase = render_scalar_expr(call.args[1], ctx)
    ctx.emit("{")
    ctx.indent()
    ctx.emit("unsigned tf_mbar_ready = 0u;")
    ctx.emit("while (!tf_mbar_ready) {")
    ctx.indent()
    body = (
        "{\\n  .reg .pred complete;\\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], %2;\\n"
        "  selp.b32 %0, 1, 0, complete;\\n}\\n"
    )
    ctx.emit(
        f'asm volatile("{body}" : "=r"(tf_mbar_ready) '
        f': "r"({addr}), "r"(unsigned({phase})));'
    )
    ctx.dedent()
    ctx.emit("}")
    ctx.dedent()
    ctx.emit("}")


@register_codegen_cuda(MBarrierInvalidate)
def _emit_invalidate(call, ctx: CodegenContext) -> None:
    addr = _smem_addr(call.args[0], ctx)
    ctx.emit(f'asm volatile("mbarrier.inval.shared::cta.b64 [%0];\\n" :: "r"({addr}));')


__all__ = ["barrier_word"]
