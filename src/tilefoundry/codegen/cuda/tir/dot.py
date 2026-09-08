"""Emitter for the ``Dot`` TIR stmt — one uniform runtime call.

The tier is ``ops::dot``'s own answer, read off the operand shard layouts
together with whether a workspace came with them, so the emitted line passes
the workspace or does not and names no strategy either way -- the shape
``reduce``'s emitter already takes.

No axes pack sits beside it: the axes a fold contracts are the ones the
operands' meshes already contract, so nothing here restates them.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.dot import Dot

_DOT = "tilefoundry::ops::dot"


def _tensor_expr(var, ctx: CodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen_cuda(Dot)
def _emit(call, ctx: CodegenContext) -> None:
    if not 3 <= len(call.args) <= 4:
        raise ValueError(
            f"tir.tensor.Dot: three operands plus an optional workspace, "
            f"got {len(call.args)} arguments"
        )
    operands = ", ".join(_tensor_expr(arg, ctx) for arg in call.args)
    ctx.emit(f"{_DOT}({operands});")
