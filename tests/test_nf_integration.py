"""The network-free engines for real, on the CaMKII pilot: BNG2.pl writes the XML, and
BNGsim's two sessions, BNG2.pl's NFsim and standalone rm_driver each draw a couple of
trajectories on the protocol's grid. Two copies of RuleMonkey fed one seed must give
one trajectory. Needs bngsim, BioNetGen (`pixi run fetch-bionetgen`, plus `fetch-glibc`
on an old-glibc host), the built rm_driver (`bench fetch-rulemonkey`) and the model
file (cached, or the network)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO

from bench import env, specs
from bench.capabilities import nf

pytestmark = pytest.mark.integration

PROBLEM = "camkii_holoenzyme_activation_michalski2012_nfsim"


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    root = os.environ.get("BNGPATH")
    if not (root and (Path(root) / "BNG2.pl").exists()):
        pytest.skip("BioNetGen not installed; run `pixi run fetch-bionetgen`")
    pytest.importorskip("bngsim")
    problem = next(p for p in specs.load_all(REPO / "problems") if p.id == PROBLEM)
    work = tmp_path_factory.mktemp("nf")
    ctx = {"cache_dir": str(REPO / ".cache"), "workdir": str(work), "subprocess_timeout": 300}
    return nf._prepare(problem, ctx), work


def test_the_protocol_is_the_files_own_and_the_xml_names_its_observables(prep):
    p, _ = prep
    assert p["nf"].single and p["nf"].phases[0].n_points == 61
    assert p["names"] == ["Obs_Tot_CaMKII", "Obs_U", "Obs_C", "Obs_P"]


@pytest.mark.parametrize("setting", nf.SETTINGS)
def test_bngsims_sessions_conserve_the_subunits_on_the_grid(prep, setting):
    p, _ = prep
    stack = nf.BngsimNF(setting, p["xml"], p["nf"], p["names"]).ensemble(2, 11)
    assert stack.shape == (2, 61, 4)
    assert np.all(stack[:, :, 0] == stack[0, 0, 0])  # Obs_Tot_CaMKII is conserved
    assert np.allclose(stack[:, :, 1:].sum(axis=2), stack[:, :, 0])


def test_bng2pls_nfsim_runs_under_the_harness_and_writes_the_same_xml(prep):
    p, work = prep
    ref = nf.NfsimBNG2(p, work / "nfsim", timeout=300)
    stack = ref.ensemble(2, 11)
    assert stack.shape == (2, 61, 4) and np.isfinite(stack).all()
    cfg = ref.config()
    # The NFsim the fetched BioNetGen bundles (2.9.5: NFsim 2.0.1).
    want = env.NFSIM_VERSION[env.bionetgen_identity()["version"]]
    assert cfg["version"] == want and cfg["xml_matches_harness"] is True
    assert "-cb" in cfg["nfsim_command"].split()


def test_rm_driver_and_bngsims_rulemonkey_agree_seed_for_seed(prep):
    if env.rulemonkey_root() is None:
        pytest.skip("rm_driver not built; run `bench fetch-rulemonkey`")
    p, _ = prep
    cli = nf.RuleMonkeyCLI(p, timeout=300).ensemble(2, 11)
    embedded = nf.BngsimNF("nf_exact", p["xml"], p["nf"], p["names"]).ensemble(2, 11)
    np.testing.assert_array_equal(cli, embedded)
