"""The registry of capabilities. Each module defines ``jobs(problem)`` and
``run(job, problem, ctx)``; a capability is usable from a spec once it is listed
here and has a protocol validator in ``bench.specs``."""

from __future__ import annotations

from . import ode

CAPABILITIES = {
    "ode": ode,
}
