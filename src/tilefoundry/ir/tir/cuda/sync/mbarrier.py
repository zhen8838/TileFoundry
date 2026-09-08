"""Effect-form TIR Ops for the CUDA shared-memory barrier object (mbarrier).

Not ``Sync``, which is a whole-mesh rendezvous every participant reaches: these
let a producer signal completion of work a consumer did not perform, which is
what an asynchronous copy needs.

Each lowers to its own ``mbarrier.*`` instruction rather than to a runtime
entry, and the set is what a ``TmaCopy`` ring needs and no more
([tir §2.3](docs/spec/tir.md#23-tir-ops)).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Scalar, Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

__all__ = [
    "MBarrierInit",
    "MBarrierArriveExpectTx",
    "MBarrierWaitParity",
    "MBarrierInvalidate",
]


def _require_smem_barrier(ctx, call, who: str) -> None:
    """Report unless argument 0 is a shared-memory barrier object.

    The instructions take a shared-window address. A barrier in global memory
    is not a slower barrier, it is not one at all.
    """
    ty = ctx.type_of(call.args[0])
    if ty.storage != StorageKind.SMEM:
        ctx.error(call, f"{who} barrier must be smem, got {ty.storage}")


@register_op(dialect="T", category="sync", name="mbarrier_init")
class MBarrierInit(Op):
    """Arm ``barrier`` so ``arrive_count`` arrivals complete a phase.

    One thread initialises; a ``Sync`` covering every thread that will use the
    barrier must separate this from the first arrival or wait.
    """

    barrier = ParamDef(kind="input", pattern=Tensor)
    arrive_count = ParamDef(kind="attribute", annotation=int)


@register_typeinfer(MBarrierInit)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(MBarrierInit)
def _(call: "Call", ctx: "VerifyContext") -> None:
    count = call.target.arrive_count
    if not isinstance(count, int) or count <= 0:
        ctx.error(call, f"MBarrierInit.arrive_count must be a positive int, got {count!r}")
    _require_smem_barrier(ctx, call, "MBarrierInit")


@register_op(dialect="T", category="sync", name="mbarrier_arrive_expect_tx")
class MBarrierArriveExpectTx(Op):
    """Arrive on ``barrier`` and declare ``tx_bytes`` of asynchronous data.

    One instruction rather than an arrival beside a separate byte declaration,
    so one wait covers a copy the waiting thread did not issue. ``TmaCopy``
    declares its own bytes on the instruction that issues the copy, so this
    belongs to a producer that issues one itself.
    """

    barrier = ParamDef(kind="input", pattern=Tensor)
    tx_bytes = ParamDef(kind="attribute", annotation=int)


@register_typeinfer(MBarrierArriveExpectTx)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(MBarrierArriveExpectTx)
def _(call: "Call", ctx: "VerifyContext") -> None:
    tx = call.target.tx_bytes
    if not isinstance(tx, int) or tx <= 0:
        ctx.error(call, f"MBarrierArriveExpectTx.tx_bytes must be a positive int, got {tx!r}")
    _require_smem_barrier(ctx, call, "MBarrierArriveExpectTx")


@register_op(dialect="T", category="sync", name="mbarrier_wait_parity")
class MBarrierWaitParity(Op):
    """Block until ``barrier``'s phase parity reaches ``phase``.

    The parity alternates 0, 1, 0, ... across successive completions, which is
    what lets a fixed ring of barriers serve a pipeline of any length: stage
    ``t`` of a ring of ``n`` waits on parity ``(t // n) & 1``.
    """

    barrier = ParamDef(kind="input", pattern=Tensor)
    phase = ParamDef(kind="input", pattern=Scalar)


@register_typeinfer(MBarrierWaitParity)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(MBarrierWaitParity)
def _(call: "Call", ctx: "VerifyContext") -> None:
    _require_smem_barrier(ctx, call, "MBarrierWaitParity")


@register_op(dialect="T", category="sync", name="mbarrier_invalidate")
class MBarrierInvalidate(Op):
    """Release ``barrier``'s shared-memory word for other use."""

    barrier = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(MBarrierInvalidate)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(MBarrierInvalidate)
def _(call: "Call", ctx: "VerifyContext") -> None:
    _require_smem_barrier(ctx, call, "MBarrierInvalidate")
