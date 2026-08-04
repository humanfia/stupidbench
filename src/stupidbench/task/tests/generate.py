"""Step 1: build the kernel and serialize it to ``instrs.json`` as pure data.

This is the ONLY step that imports the agent-edited ``perf_takehome.py``.
Whatever it does at import or build time stays in this process; the scorer reads
only the JSON it emits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_DIR))

from perf_takehome import KernelBuilder  # noqa: E402  (agent code; runs only here)

# Shapes for the scored problem do_kernel_test(10, 16, 256):
# Tree.generate(10) has 2**(10 + 1) - 1 nodes; batch 256; 16 rounds.
FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS = 10, 2 ** (10 + 1) - 1, 256, 16


def to_plain(value: object) -> object:
    """Reject anything that is not a JSON scalar/container; tuples become lists."""
    if type(value) is bool:
        raise TypeError("bool is not a valid instruction operand")
    if type(value) in (int, str):
        return value
    if type(value) in (list, tuple):
        return [to_plain(item) for item in value]
    if type(value) is dict:
        return {key: to_plain(item) for key, item in value.items()}
    raise TypeError(f"non-serializable value in instrs: {type(value)}")


def main() -> None:
    kb = KernelBuilder()
    kb.build_kernel(FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS)
    instructions = [
        {
            engine: slots
            for engine, slots in bundle.items()
            if engine in ("alu", "valu", "load", "store", "flow")
        }
        for bundle in kb.instrs
    ]
    instructions = [bundle for bundle in instructions if bundle]
    (WORK_DIR / "instrs.json").write_text(
        json.dumps(to_plain(instructions), separators=(",", ":")), encoding="utf-8"
    )
    print(f"generated instrs.json ({len(instructions)} bundles)")


if __name__ == "__main__":
    main()
