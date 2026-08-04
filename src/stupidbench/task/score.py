"""Score an instruction JSON file on the frozen simulator, in isolation.

Imports ONLY ``frozen_problem`` and reads ``instrs.json`` as data. No
agent-produced Python runs in this process, so ``machine.cycle`` and the
correctness check cannot be monkeypatched from ``perf_takehome.py``. A crafted
instruction stream is checked against the complete ISA before execution, and a
runaway program is cut off by a wall-clock alarm.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "tests" else SCRIPT_DIR
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_problem import (  # noqa: E402  (frozen simulator only; never the kernel)
    DebugInfo,
    Input,
    Machine,
    N_CORES,
    SCRATCH_SIZE,
    SLOT_LIMITS,
    Tree,
    VLEN,
    build_mem_image,
    reference_kernel2,
)

assert "perf_takehome" not in sys.modules, "scoring process must not load agent code"

FOREST_HEIGHT, BATCH_SIZE, ROUNDS = 10, 256, 16
BASELINE = 147734
PENALTY = BASELINE * 2
MAX_BUNDLES = 250_000
RUN_TIMEOUT_S = 120
_ALU_OPS = frozenset(
    {"+", "-", "*", "//", "cdiv", "^", "&", "|", "<<", ">>", "%", "<", "=="}
)
_ENGINES = frozenset({"alu", "valu", "load", "store", "flow"})
_IMMEDIATE_MIN = -(2**31)
_IMMEDIATE_MAX = 2**32 - 1


@dataclass(frozen=True)
class _ScoreResult:
    ok: bool
    score: int


def _integer(
    value: object,
    *,
    maximum: int,
    minimum: int = 0,
) -> int:
    if type(value) is not int:
        raise TypeError("instruction operands must be integers")
    if not minimum <= value <= maximum:
        raise ValueError(f"instruction operand outside [{minimum}, {maximum}]")
    return value


def _scratch(value: object, *, width: int = 1) -> int:
    return _integer(value, maximum=SCRATCH_SIZE - width)


def _validate_slot(engine: str, slot: object, program_size: int) -> None:
    match engine, slot:
        case "alu", [op, dest, left, right] if isinstance(op, str) and op in _ALU_OPS:
            _scratch(dest)
            _scratch(left)
            _scratch(right)
        case "valu", ["vbroadcast", dest, source]:
            _scratch(dest, width=VLEN)
            _scratch(source)
        case "valu", ["multiply_add", dest, left, right, addend]:
            _scratch(dest, width=VLEN)
            _scratch(left, width=VLEN)
            _scratch(right, width=VLEN)
            _scratch(addend, width=VLEN)
        case "valu", [op, dest, left, right] if isinstance(op, str) and op in _ALU_OPS:
            _scratch(dest, width=VLEN)
            _scratch(left, width=VLEN)
            _scratch(right, width=VLEN)
        case "load", ["load", dest, address]:
            _scratch(dest)
            _scratch(address)
        case "load", ["load_offset", dest, address, offset]:
            offset = _integer(offset, maximum=VLEN - 1)
            _scratch(dest + offset)
            _scratch(address + offset)
        case "load", ["vload", dest, address]:
            _scratch(dest, width=VLEN)
            _scratch(address)
        case "load", ["const", dest, value]:
            _scratch(dest)
            _integer(value, minimum=_IMMEDIATE_MIN, maximum=_IMMEDIATE_MAX)
        case "store", ["store", address, source]:
            _scratch(address)
            _scratch(source)
        case "store", ["vstore", address, source]:
            _scratch(address)
            _scratch(source, width=VLEN)
        case "flow", ["select", dest, condition, left, right]:
            _scratch(dest)
            _scratch(condition)
            _scratch(left)
            _scratch(right)
        case "flow", ["add_imm", dest, source, immediate]:
            _scratch(dest)
            _scratch(source)
            _integer(immediate, minimum=_IMMEDIATE_MIN, maximum=_IMMEDIATE_MAX)
        case "flow", ["vselect", dest, condition, left, right]:
            _scratch(dest, width=VLEN)
            _scratch(condition, width=VLEN)
            _scratch(left, width=VLEN)
            _scratch(right, width=VLEN)
        case "flow", ["halt"] | ["pause"]:
            pass
        case "flow", ["trace_write", value]:
            _scratch(value)
        case "flow", ["cond_jump", condition, address]:
            _scratch(condition)
            _integer(address, maximum=program_size)
        case "flow", ["cond_jump_rel", condition, offset]:
            _scratch(condition)
            _integer(offset, minimum=-program_size, maximum=program_size)
        case "flow", ["jump", address]:
            _integer(address, maximum=program_size)
        case "flow", ["jump_indirect", address]:
            _scratch(address)
        case "flow", ["coreid", dest]:
            _scratch(dest)
        case _:
            raise ValueError(f"invalid {engine} slot")


def validate(instrs: object) -> None:
    """Reject programs outside the complete non-debug simulator ISA."""
    if not isinstance(instrs, list) or len(instrs) > MAX_BUNDLES:
        raise ValueError("program must be a list of at most MAX_BUNDLES bundles")
    for bundle in instrs:
        if not isinstance(bundle, dict) or not bundle:
            raise ValueError("each bundle must be a nonempty object")
        for engine, slots in bundle.items():
            if engine not in _ENGINES or not isinstance(slots, list):
                raise ValueError(f"invalid engine or slots: {engine!r}")
            if len(slots) > SLOT_LIMITS[engine]:
                raise ValueError(f"too many {engine} slots")
            for slot in slots:
                _validate_slot(engine, slot, len(instrs))


def run_once(instrs: list[object]) -> int:
    random.seed(os.environ["_STUPIDBENCH_AOPT_SEED"])
    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)

    machine = Machine(copy(mem), instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass
    values_p = ref_mem[6]
    if (
        machine.mem[values_p : values_p + len(inp.values)]
        != ref_mem[values_p : values_p + len(inp.values)]
    ):
        raise AssertionError("incorrect output values")
    return machine.cycle


def score(instructions_path: Path) -> int:
    instrs = json.loads(instructions_path.read_text(encoding="utf-8"))
    validate(instrs)
    assert isinstance(instrs, list)
    return run_once(instrs)


def main() -> int:
    if len(sys.argv) > 2:
        print(f"usage: {Path(sys.argv[0]).name} [instructions.json]", file=sys.stderr)
        return 2
    instructions_path = (
        Path(sys.argv[1]) if len(sys.argv) == 2 else WORK_DIR / "instrs.json"
    )
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.setitimer(signal.ITIMER_REAL, RUN_TIMEOUT_S)
    try:
        cycles = score(instructions_path)
        ok = True
    except BaseException as exc:  # noqa: BLE001  (any failure -> penalty, never trusted)
        cycles = PENALTY
        ok = False
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    print(json.dumps(asdict(_ScoreResult(ok=ok, score=cycles))))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
