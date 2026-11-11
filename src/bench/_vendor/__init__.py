"""Code copied verbatim from other projects, pinned and checked by sha256.

``parity_core`` is bngsim's comparison core (``parity_checks/_core``): the
outcome taxonomy, the deterministic and ensemble verdicts, and fingerprints.
VENDOR.json names the commit it was copied from. Never edit it here; re-run
``tools/vendor_parity_core.py``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

VENDOR_JSON = Path(__file__).with_name("VENDOR.json")


@lru_cache(maxsize=1)
def parity_core_record() -> dict:
    return json.loads(VENDOR_JSON.read_text())


def parity_core_identity() -> dict:
    rec = parity_core_record()
    return {"upstream": rec["upstream"], "ref": rec["ref"], "commit": rec["commit"]}
