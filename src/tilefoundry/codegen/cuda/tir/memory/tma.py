"""Emitter for ``TmaCopy`` — one line, whichever instruction ends up running.

Which one that is comes off the operand shard layouts inside the entry, so
neither the byte count nor the vector width appears here: declaring the count
at the call site is the drift the entry exists to prevent.

The barrier is the one operand the runtime does not see as a tensor -- it is a
64-bit word -- so it goes over as its own address. ``barrier_word`` is shared
with the ``mbarrier`` emitters, which arm and wait on the same word.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.codegen.cuda.tir.mbarrier import barrier_word
from tilefoundry.ir.tir.cuda.memory.tma import TmaCopy

_TMA_COPY = "tilefoundry::ops::tma_copy"


def _tensor_expr(var, ctx: CodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen_cuda(TmaCopy)
def _emit(call, ctx: CodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    bar = f"reinterpret_cast<uint64_t *>({barrier_word(call.args[2], ctx)})"
    ctx.emit(f"{_TMA_COPY}({src}, {dst}, {bar});")
