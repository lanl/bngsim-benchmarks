# Design

Status: **draft.** This document records the decisions taken so far and the
questions still open. When the code disagrees with it, one of the two is wrong and
gets fixed.

Implemented so far:

- the problem specs (section 5.1) and the job list (5.2);
- one result file per job, with freshness (6.1, 6.2), and `gather` (6.3);
- stored correctness trajectories, and re-scoring them with `rescore` (6.1, 6.2);
- the correctness verdict: bngsim's own, and a per-species check (6.4);
- the serial and local-pool runners with deadlines (8);
- Slurm submission as job arrays, `bench submit` (8);
- the interleaved timing protocol (7);
- one capability: ODE simulation of BNGL and SBML models, comparing BNGsim against
  a registry of reference engines (`run_network`, libRoadRunner, COPASI and AMICI),
  on two core suites, `ode-net` and `ode-sbml` (section 3.2).
- a second capability: exact stochastic simulation (SSA) of BNGL models, comparing
  BNGsim's ensembles against `run_network`, libRoadRunner and COPASI, and against
  exact moments where a model has them (§5.3), on two provisional pilot models.
- a third: partial scaling (PSA) of BNGL models, comparing BNGsim's PSA with
  `run_network`'s at each critical population size N_c, and with exact SSA, its
  N_c = infinity limit (§5.4), on two provisional pilot models.
- a fourth: network-free simulation of BNGL models, BNGsim's two network-free
  engines against standalone NFsim and RuleMonkey and, where a model's network is
  finite, exact SSA on it (§5.5), on two provisional pilot models.
- a fifth: forward sensitivities on the generated network, comparing BNGsim's with
  AMICI's, with finite differences from `run_network` and with COPASI's (§5.6), on
  two provisional pilot models.

Not yet: bundles, the container, the validation of section 9, and every other
capability.

## 1. What the suite is for

The suite answers two kinds of question about [BNGsim](https://github.com/lanl/bngsim):

- **Correctness:** does BNGsim give the same answer as an independent reference? The
  references are other simulators, analytic results, and standard test suites.
- **Cost:** how long does BNGsim take, relative to the tools a modeler would otherwise
  use, on the same problem?

Every comparison in the suite supports one of these claims:

1. BNGsim reproduces the results of the tool it replaces (BioNetGen) and of the
   standard references (libRoadRunner, AMICI).
2. It costs no more than those tools, and less where it matters. The main example is
   re-simulating a loaded model with new parameters, as in a fitting loop.
3. Its internal choices are sound:
   - Automatic linear-solver selection costs about the same as the better of dense
     and KLU.
   - Compiled and interpreted right-hand sides agree.
4. It conforms to the SBML standard.
5. Its approximate method (partial scaling, PSA) is accurate where it is used.

The suite is built so that anyone can run it:

- one job at a time on a laptop;
- many jobs on a single large node;
- unattended on a Slurm cluster.

## 2. Capabilities and comparisons

| Capability | Correctness reference | Cost comparison |
|---|---|---|
| ODE simulation, BNGL models | BioNetGen `run_network`; libRoadRunner (SBML exported from the generated network) | `run_network`; libRoadRunner |
| ODE simulation, SBML models | libRoadRunner | libRoadRunner; AMICI; COPASI (core only) |
| ODE settings: dense, KLU, automatic; compiled vs interpreted | each other | forced dense and forced KLU vs automatic |
| Re-simulation of a loaded model with new parameters | — | `run_network`, libRoadRunner, AMICI |
| Exact stochastic simulation (SSA) | `run_network` SSA, libRoadRunner and COPASI (ensemble statistics, §5.3); exact moments of first-order networks; DSMTS analytic moments | `run_network` SSA; libRoadRunner; COPASI (core only) |
| Partial scaling (PSA) | exact SSA, as the critical population N_c varies; `run_network` PSA | `run_network` PSA; exact SSA |
| Network-free simulation (BNGsim's `nf_reject` and `nf_exact`, two settings) | standalone NFsim driven by BNG2.pl through the protocol's own actions; standalone RuleMonkey (same lineage as `nf_exact`, a secondary reference); exact SSA (`run_network`, BNGsim) on the generated network where it is finite (ensemble statistics, §5.5); BNGsim's two engines against each other (recorded) | standalone NFsim; standalone RuleMonkey |
| Forward sensitivities | AMICI; finite differences | AMICI; COPASI (core only) |
| Steady state | long-time integration; AMICI steady state | integration to steady state; AMICI |
| SBML compliance | SBML Test Suite semantic cases; DSMTS | — |
| Parameter fitting (application) | objective values at fixed parameter points | PyBNF with its legacy BioNetGen backend |

COPASI appears in the core tables and in the BioModels coverage count. It is not part
of breadth timing.

`run_network`'s reference runs use its default (dense) linear solver, as bngsim's
own parity checks do. A model's `sparse` option (BNG2.pl `sparse=>1`, `run_network
-b`, an iterative solver) is not a trustworthy oracle: on at least one pilot model
(Barua et al. 2007) it leaves the solution by 2.5% of a species' peak, even at a
tolerance of 1e-12, while BNGsim and the dense solver agree. When a model asks for
`sparse`, the correctness job records whether the sparse run agrees with the dense
one. The timing job times the authors' sparse setting only when it does, and
otherwise times the dense solver, so a speedup is always measured against a
reference run that gave the right answer.

## 3. Layers

The suite has five layers. Each problem belongs to one layer, and the layer sets the
rule for what may go in it.

Until the core suites are chosen, a sixth, temporary layer, `pilot`, holds provisional
problems that exercise the machinery. A pilot problem may record its verification as
`pending`; no other layer may.

### 3.1 Standards

These are used whole, as published by their maintainers:

- the SBML Test Suite (semantic cases);
- DSMTS (discrete stochastic model test suite);
- AMICI's own forward-sensitivity test cases.

They are conformance sets, not representative models.

### 3.2 Core suites

The core is a set of **suites**. A suite is one bngsim feature on one input format --
ODE on `.net`, ODE on SBML, sensitivities on SBML, and so on -- and holds **exactly 10**
published, verified models (section 4.1), in a fixed order. There are as many suites as
the feature surface needs. A feature and format get a suite only where bngsim supports
that pair: network-free simulation, for example, runs only from BNGL.

Each suite has its own models, rather than one shared set serving every feature:

- **Features need models with opposite properties.** Exact SSA needs low copy numbers.
  Network-free simulation needs aggregation models with no finite network, which have
  no ODE form. Sensitivities need parameter-estimation problems with data. A shared set
  would be a compromise that serves none of them well.
- **Each format is its own code path, with its own references.** bngsim reads `.net`,
  SBML and Antimony through separate readers, and the reference engines differ by
  format: `run_network` reads only `.net`; COPASI and AMICI read SBML; libRoadRunner
  reads SBML and Antimony. One format per suite keeps every row of a suite against the
  same engines, and a slow or wrong reader shows up as that suite's result.

A model may still sit in several suites; a `.net` ODE model is often a steady-state
model too. Each problem spec records the suites it belongs to and its position in each
(`suites: {ode-net: 3}`). A suite that is not complete, with one member at each position
1–10, does not load. [core_suites.md](core_suites.md) lists each suite's members and what
they cover.

**Seeding.** Each suite starts from the models the BNGsim paper already selected and
vetted for that feature, kept whole:

- A set larger than 10 is split along the paper's own grouping, such as network-free
  aggregation models versus signaling models, and each part becomes a suite.
- A set smaller than 10 is topped up to 10, from the same sources, by the paper's own
  selection criterion. For the SBML ODE suite that is the next-largest curated
  BioModels with a figure-reproducing SED-ML horizon.
- An added model is held to the same rule as the seed: published and verified.

**Derived suites.** Where a format has no suitable published models of its own, its
suite is derived from other suites by a pinned conversion. The Antimony ODE suite
interleaves the other two ODE suites:

- its odd positions are the `.net` suite's models 1, 3, 5, 7 and 9, exported to SBML
  and converted to Antimony;
- its even positions are the SBML suite's models 2, 4, 6, 8 and 10, converted to
  Antimony.

The conversion tools are pinned, and the converted file's sha256 is recorded. A
derived model is as published and verified as the model it came from. It also gives a
cross-format check: the same model read from two formats must agree. Handcrafted
Antimony test sets are left to the breadth tier: they are small models, and a suite
drawn only from them would be biased toward small sizes.

**What each suite spans.** Within its 10, a suite should cover what its feature is
sensitive to:

- **ODE:**
  - size from about 10 species to thousands, with models on both sides of the
    automatic dense/KLU threshold and one near it;
  - relaxation to steady state, an oscillator and a bistable switch;
  - at least one strongly stiff model.
- **SSA:** low copy numbers, and one model with known analytic moments.
- **PSA:** species whose populations differ by orders of magnitude.
- **Network-free:**
  - aggregation models with no finite network;
  - at least one model whose network is finite, so network-free results can be
    checked against SSA on that network.
- **Sensitivities:** 10–200 parameters, preferably parameter-estimation problems with
  data.
- **Steady state:** monostable and bistable models, and one with conservation laws,
  whose Jacobian is singular.

### 3.3 Size series

These are variants of verified core models, used **only** for scaling claims:

- ODE cost against network size, for each linear-solver setting;
- SSA cost per reaction event against reaction count;
- the point at which network-free simulation overtakes simulation of the generated
  network.

Rules:

1. The knob is either one the publication itself varies, or a truncation of an
   infinite network (maximum chain length, maximum aggregate size). Nothing else in
   the model changes.
2. The published setting is one point of the series, and that point is the verified
   file.
3. Variants are produced by a committed generator from the verified file. They are
   never edited by hand. The knob value is part of the problem ID, e.g.
   `dushek2011@sites=8`.
4. Variants are labelled as variants wherever results are reported. They never feed
   claims about realistic problems.
5. Correctness checks run on every variant too.

Candidate families, all curated records in
[BNGL-Models](https://github.com/wshlavacek/BNGL-Models) with verification notebooks:

| Family | Knob | Notes |
|---|---|---|
| Dushek et al. 2011, membrane-anchored multisite phosphorylation | number of identical sites | sparse Jacobian; 20 sites give 143 species, 284 reactions |
| Rubenstein et al. 2007, nucleated prion polymerization | maximum chain length (truncation) | much denser coupling; reactions grow roughly as the square of the length |
| Yang et al. 2008, trivalent-ligand bivalent-receptor aggregation | aggregate-size cap for network generation | network-free cost depends on particle count, not network size |

Scaling claims that need no variant model vary a run setting instead:

- N_c for PSA;
- the number of parameters for sensitivities;
- the number of dose points for steady state;
- the number of replicates for stochastic methods.

### 3.4 Breadth

Whole public corpora, used for distributions:

- correctness pass rates;
- the spread of per-engine absolute times, and of any baseline-relative cost derived
  from them (§6.3).

Cost headlines use all public models in the breadth tier. The corpora are:

- **SBML:** the BioModels corpus that bngsim's `rr_parity` suite uses: 1323 models,
  1011 curated (`BIOMD…`) and 312 non-curated (`MODEL…`).
- **BNGL:** the corpus that bngsim's `bng_parity` suite vendors, 895 models from
  BNGL-Models, RuleHub and RuleMonkey.

The admission rule for breadth is section 4.3.

### 3.5 Application

Parameter fitting with [PyBNF](https://github.com/lanl/PyBNF), using published
fitting jobs from BNGL-Models `pybnf-jobs/`. Each job has a recorded validation. Each
job yields three results:

1. **Correctness:**
   - Evaluate the objective with both backends at a fixed set of parameter points:
     the published best fit plus about 20 sampled points.
   - The values must agree within tolerance.
   - This is deterministic, so optimizer randomness doesn't enter.
2. **Paired cost per objective evaluation:** measured on the same points, under the
   timing protocol of section 7.
3. **End-to-end fit:**
   - The job's committed configuration, at its fixed iteration budget, run once per
     backend.
   - Each fit is one job.
   - Both backends' fits of a job run on the same node type.

## 4. Provenance rules

### 4.1 Published and verified (core layer)

A model is **published** if it is described in a peer-reviewed publication. The file
may be:

- the authors' own release, or
- a curated re-implementation, if it is verified.

A file is **verified** if all three hold:

- the exact file reproduces something the publication states;
- the check does not use BNGsim;
- the comparison is committed and can be re-run.

Each core problem records the kind of evidence it has:

| Kind | Meaning |
|---|---|
| `author_file` | the authors' own released model: curated BioModels, the PEtab benchmark collection, or the authors' repository at a pinned commit |
| `equations` | agrees with an independent implementation of the paper's equations |
| `data` | agrees with published numbers (digitized figure or reported values) within a stated tolerance |
| `theory` | agrees with an analytic or limiting-case result |
| `structure` | the network's size or composition matches the paper. Accepted only with a written reason, such as "the paper reports no dynamics". |

A variant file counts as verified only if its verification covers that exact file.

### 4.2 Pinning

Every model file is identified by repository, commit, path and sha256. The suite
refers to files; it does not need to redistribute them. `bench fetch-models`
downloads each file from its repository at the pinned commit into a git-ignored
cache keyed by sha256 and file name. A download, or a
cached copy, whose bytes do not match is never read: the download is refused, and a
mismatched cached copy is downloaded again. Once the cache is filled, no job needs
the network.

### 4.3 Breadth admission and labels

A model may enter the breadth tier if it is publicly available at a pinned commit.
Each model carries a provenance class, for breakdowns:

- `peer_reviewed`: RuleHub `Published/`, curated BNGL-Models records, curated BioModels;
- `example`: RuleHub `Examples/` and `Tutorials/`;
- `test`: engine test cases, such as RuleMonkey `tests/`;
- `working`: unpublished working models;
- `noncurated_biomodels`: BioModels `MODEL…` entries.

Byte-identical files (the same sha256) are counted once.

## 5. Problems and jobs

### 5.1 Problem specification

A problem is a model plus a fixed protocol plus stored reference results. One small
YAML file per problem holds the spec, for example:

```yaml
id: dushek2011
layer: core
source: {repo: wshlavacek/BNGL-Models, commit: <sha>, path: models/<record>/<file>.bngl, sha256: <sha256>}
publication: {citation: "<authors, year, journal>", doi: "<doi>"}
verification: {kind: equations, evidence: models/<record>/<notebook>, metric: max_rel_err, value: <value>}
capabilities: [ode, ode_settings, sens]
protocol:
  ode: {t_end: <t>, n_points: <n>, rtol: <rtol>, atol: <atol>}
  sens: {parameters: independent, fitted: [<names>]}  # section 5.6
references: {ode/run_network: <path to stored reference output>}
```

Stored references let anyone check BNGsim without installing BioNetGen, AMICI or
COPASI. Re-generating the references is a separate job type.

### 5.2 Jobs

A job is the unit of work:

> one problem × one capability × one setting × the engine, or engines timed together

A correctness job compares BNGsim against every reference engine on that problem, and
a timing job interleaves all of them (§7.1) in one drift-cancelling run, so their
absolute times are measured together. Both name the whole set of engines, derived from
the reference-engine registry, rather than a single pair.

Job IDs are deterministic and readable, for example
`ode/dushek2011@sites=8/bngsim-vs-run_network-roadrunner/auto`. Generating the job list
from the problem specs and the suite configuration is itself a command, so the list
never drifts from the specs.

Correctness jobs and timing jobs are separate:

- **Correctness jobs** can run anywhere, at any parallelism.
- **Timing jobs** follow the protocol of section 7.

A timing job runs on problems whose correctness job passed, and on problems it
marked REFERENCE_FAILED -- BNGsim ran and every surviving reference agreed with
it, so there is still a comparison worth timing. A disagreement takes precedence
over a failure in every capability: when a reference that ran disagrees with
BNGsim the outcome is DIFF, whatever else failed (each failed reference is still
recorded as `<name>_error`), so a wrong answer is never timed. The timing job then
gives each reference a trial run and drops any that crash (recording a
`<name>_error`, exactly as correctness does), timing BNGsim against the engines
that survive. Any other correctness outcome (DIFF, EXCEPTION, BAD_TEST, …) blocks
the timing job.

### 5.3 Stochastic verdicts (SSA)

An exact SSA has no single right trajectory: two correct engines sample the same
probability law. So an SSA correctness job compares **ensembles**, with a stated
false-alarm rate, and every number it rests on is reproducible.

**Protocol fields** (`protocol.ssa`, validated in `specs.py`):

| Field | Meaning |
|---|---|
| `netgen` | the `generate_network(...)` call, as for ODE |
| `t_end`, `n_steps` | the output grid: `n_steps + 1` uniform points over `[0, t_end]` |
| `replicates` | trajectories per engine in the correctness job (100 to 10^6) |
| `timing_replicates` | trajectories per timed repetition (default 100) |
| `seed` | the base seed every engine's seeds derive from (default 1) |
| `moments` | `linear`: compare also against the network's exact moments (needs `replicates` >= 10^4) |

`size.ssa_events`, the mean reaction events per trajectory, is a planning estimate
only, like the rest of `size`.

**Seeds.** Each engine draws from a fixed block of its own, `seed + 10^7 k` (BNGsim
k = 0, run_network 1, libRoadRunner 2, COPASI 3), so no two ensembles share a
seed. A reference's trajectory `i` uses the block's first seed plus `i`; BNGsim's
ensemble is `run_replicates(seed=...)`, whose per-replicate seeds bngsim derives
from that base. Every engine reproduces a trajectory from its seed exactly, so a
correctness job re-run on the same build gives the same ensembles, and the same
verdicts to the bit.

**BNGsim against a reference engine: a permutation max-T test.** At every cell
(time point, species) two statistics, the per-time-point tests of DSMTS (Evans,
Gillespie & Wilkinson 2008, *Bioinformatics* 24:285) made two-sample:

- the difference of the ensemble means over its pooled standard error;
- the difference of the ensemble variances over its pooled large-sample standard
  error, `sqrt((m4 - m2^2)(1/n_A + 1/n_B))` -- valid whatever the distribution,
  where a Gaussian formula is not for the skewed, discrete counts of a low-copy
  model.

The test statistic is the largest of their absolute values over all cells. Its null
distribution comes from relabelling (Westfall & Young 1993, *Resampling-Based
Multiple Testing*, single-step max-T): if the two engines sample the same law, the
pooled trajectories are exchangeable, and every split into groups of `n_A` and
`n_B` is equally likely. The p-value is the share of 1999 seeded random splits whose
maximum reaches the observed one, and the comparison fails when it is at most
`alpha` = 10^-3. This makes the false-alarm rate at most `alpha` **exactly**, at any
ensemble size, under any dependence between time points and species (trajectories
are strongly autocorrelated), and for any marginal -- zeros, skew and discreteness
included -- with no normal approximation and none of Bonferroni's conservatism under
dependence. Relabelling stops at the 20th split that beats the observed maximum
(Besag & Clifford 1991, *Biometrika* 78:301); the p-value is then 20 over the
splits drawn, which is valid and can only be a pass, so only a comparison near
failing draws all 1999. Cells constant across both ensembles (the initial
condition) are left out, and so, for the variance test, are cells taking at most
two values (a 0/1 promoter), whose variance is fixed by their mean.

**BNGsim against exact moments: Bonferroni z-tests.** Where `moments: linear` is set,
every reaction must be of order 0 or 1 (in unfixed species), so the mean and
covariance obey closed linear ODEs (Gadgil, Lee & Othmer 2005, *Bull Math Biol*
67:901), which the harness solves exactly, by a matrix exponential, from its own
reading of the `.net` -- no engine's code is involved. Per cell, `sqrt(n)(mean - mu) /
sigma` with the exact `mu` and `sigma`, and `log(s^2 / sigma^2)` over its delta-method
standard error from the sample kurtosis (DSMTS's variance test assumes a Gaussian
kurtosis); the comparison fails when the smallest two-sided normal p-value is at
most `alpha / m` over the `m` tests. There is nothing to relabel against a fixed
law, so this test is asymptotic: it is calibrated at DSMTS's recommended 10^4
trajectories, which a spec with exact moments must ask for. A cell whose exact
variance is zero is deterministic, and every trajectory must hit it. Each
reference's ensemble is checked against the exact moments too, unscored, to say
which side of a disagreement is wrong.

**The job.** It passes only when every comparison passes; failures are attributed as
for ODE (§6.1). A job with three references and exact moments makes four
comparisons, so a correct BNGsim fails it with probability at most 4 x 10^-3, and a
ten-problem suite at most 0.04. A reference that cannot simulate a model's exact
SSA at all is recorded as `UNSUPPORTED` and not run: libRoadRunner's gillespie fires
the SBML rate law as the propensity, which for a repeated reactant (`A + A`) is
`k A^2` where the exact propensity is `k A (A - 1)`.

bngsim's own parity suites judge ensembles by a K-sigma test on the means that
passes when 99% of cells pass (`parity_core.differ.ensemble_verdict`). It has no
stated false-alarm rate, so it does not gate here, but each SSA verdict reports it
beside the gating test, so the two repositories can be compared (§10).

**Timing.** A timed repetition is one ensemble of `timing_replicates` trajectories,
from the same seeds every round, interleaved round-robin as in §7. Since the seeds
repeat, every kept repetition must reproduce the first exactly (`NOT_REPRODUCIBLE`
otherwise), which also shows that each engine's reset between ensembles is
complete. The artifact is each engine's absolute time per ensemble with its
bootstrap CI, and per trajectory as that divided by the ensemble size. The one-time
cost before sampling -- BNGsim's load and propensity compile, libRoadRunner's SBML
load and JIT, COPASI's import -- is measured once, cold, beside it (§6.3);
run_network loads the `.net` in every trajectory, so its load is inside its time.

### 5.4 Partial scaling (PSA)

PSA (Lin, Feng & Hlavacek 2019, *J Chem Phys* 150:244101, doi:10.1063/1.5096774)
accelerates SSA by letting a reaction whose smallest participating population is N
fire `m = max(1, floor(N / N_c))` molecules at once, at `1/m` of its propensity. A
reaction's mean drift is unchanged and its noise becomes that of a population near
N_c: the method trades variance for events, more so the smaller N_c. The system
under test is BNGsim's `Simulator(method="psa", poplevel=N_c)`.

**Protocol fields** (`protocol.psa`, validated in `specs.py`): those of the SSA
protocol (§5.3) except `moments`, plus `poplevels`, the N_c grid -- a strictly
increasing list of up to 10 numbers above 1, by default 10, 30, 100, 300 and 1000,
as in bngsim's `psa` suite and Lin et al. `size.psa_events`, the exact-SSA events per
trajectory under this protocol, is a planning estimate (else `size.ssa_events`).

**Ensembles.** A job draws `2 (K + 1)` ensembles of `replicates` trajectories: BNGsim
and `run_network`, each at exact SSA (N_c = infinity) and at each of the K values of
N_c. `run_network`'s PSA is BioNetGen's own implementation of the method (`-p psa
--poplevel N_c`, what BNG2.pl runs for `simulate({method=>"psa", poplevel=>N_c})`),
a fresh process per trajectory. Each ensemble has a seed block of its own: level k
(0 for exact SSA, i + 1 for the i-th N_c) of engine e (BNGsim 0, run_network 1) draws
from block `2k + e`, so the exact-SSA ensembles are the ones an SSA job on the same
protocol draws (§5.3). They must be disjoint: BNGsim's PSA at an N_c where nothing
scales reproduces its exact SSA trajectory for trajectory from the same seed.

**What is gated.** These comparisons decide PASS or DIFF. Each is the SSA permutation
max-T test on means and variances (§5.3, `alpha` = 10^-3 each), with bngsim's
parity-core verdict recorded beside it:

1. **Like for like**, at every N_c and at N_c = infinity: BNGsim against `run_network`
   (verdicts `run_network@nc=<N_c>` and `run_network`). Both implement one method on
   one network, so they must sample one law.
2. **Exactness where nothing scaled.** At an N_c where no BNGsim trajectory scaled a
   channel, PSA *is* exact SSA, and BNGsim's PSA ensemble must sample the law of its
   exact one (verdict `exact@nc=<N_c>`). A trajectory scaled when some channel fired
   with `m > 1` while it had propensity, that is, when some `qexc_integral` of its
   `Result.psa_diagnostics` is nonzero. `activation_crossed` (some population past 2
   N_c) is not enough: on the Rohrs 2018 pilot, at N_c of 30 and above, every
   trajectory crosses it while every reaction keeps `m = 1`, because a 20-copy kinase
   takes part in all of them.

A job with K = 5 makes at most 11 gated comparisons, so a correct BNGsim fails it
with probability at most 0.011, and a ten-problem suite at most 0.11. A disagreement
is DIFF even when some reference failed, as for steady state, so a wrong answer is
never timed.

**What is reported.** PSA's departure from exact SSA is the method's approximation,
measured and never a failure. At every N_c, each engine's PSA ensemble is compared
with that engine's exact one, and the record keeps:

- a means-only permutation max-T test (`two_sample_verdict(variance=False)`, same
  `alpha`): are the means those of exact SSA?
- the largest mean error as a fraction of the species' exact peak mean (as the ODE
  verdict measures error), the largest mean shift in exact-SSA standard deviations,
  and the median and extremes of the variance ratio PSA / exact, each at its cell and,
  for the means, with that cell's standard error in the same units. They do not shrink
  with the ensemble size, but they carry its noise: where PSA has inflated a cell's
  variance a hundredfold, its mean is ten times noisier than exact SSA's there, which
  is why the shift in SD units is not the headline.
- BNGsim's diagnostics: the share of trajectories that scaled, the event-rate ratio
  `speedup` (exact over scaled events per trajectory), the largest dwell-time-averaged
  multiplier and the reactions that scaled.

Variance inflation is the price the method states, so it is only reported. The means
are what PSA preserves to first order, so a **flag** is raised, beside the outcome and
never instead of it, at every N_c of at least 100 (bngsim documents 100 to 1000 as
PSA's working range) where BNGsim's PSA scaled and its means differ significantly
from exact SSA's. The flag records whether `run_network`'s PSA differs too: if it
does, the method is inaccurate on this model at that N_c; if not, BNGsim is, and the
like-for-like comparison at that N_c will usually have failed too. Error that
shrinks as N_c grows is the expected picture, and the per-N_c rows of the report show
it.

**Failures.** As for SSA: a BNGsim ensemble that fails is EXCEPTION (TIMEOUT); a
`run_network` ensemble that fails is REFERENCE_FAILED with its `<variant>_error`, and
BNGsim is judged against the rest. `run_network`'s PSA aborts at t = 0 on a
zeroth-order reaction (`ERROR: scaling factor of 0 th reaction < 1`, for Lin &
Doering 2016 at every N_c); that is recorded, never worked around, and the record
counts the network's zeroth-order reactions beside it. A `run_network` trajectory is
stopped after 50 times its planned cost (at least 60 s), so a process that hangs
fails its ensemble quickly.

**Timing.** Every ensemble of the correctness job is timed again, warm, as in §5.3:
`timing_replicates` trajectories from the same seeds every round, every variant
interleaved round-robin, each repetition checked against the first. The artifact is
each variant's absolute time per ensemble and per trajectory. The PSA speed-up over
exact SSA of the same engine at each N_c is derived from those times in the report,
beside BNGsim's event-rate ratio, which is what the speed-up would be if an event
cost the same under both. Each BNGsim simulator's load and build is its one-time
cost; `run_network` loads the `.net` inside every trajectory. A `run_network` variant
the correctness job recorded as failed is not timed.

**Format.** BNGsim's PSA also reads SBML, but no independent PSA engine does, so
this capability takes BNGL only. PSA on SBML is to be a setting of the SBML
stochastic suites, checked there against exact SSA alone.
### 5.5 Network-free verdicts

A network-free job (`bench.capabilities.nf`) runs one of BNGsim's two network-free
engines, the system under test of its **setting**: `nf_reject`, the embedded NFsim
(`NfsimSession`), or `nf_exact`, the embedded RuleMonkey (`RuleMonkeySession`). Both
read the BNG-XML that BNG2.pl's `writeXML()` writes from the published BNGL, so a
network-free problem's source is BNGL, and its suites (`nf-xml-*`) are named after
the format the engines read, as `ode-net` is.

**Protocol fields** (`protocol.nf`, validated in `specs.py`):

| Field | Meaning |
|---|---|
| `actions` | `own` (the file's action block) or a list of BNGL actions: the protocol |
| `replicates` | trajectories per engine in the correctness job (100 to 10^6) |
| `timing_replicates` | trajectories per timed repetition (default 20) |
| `seed` | the base seed every engine's seeds derive from (default 1) |
| `netgen` | for a model whose network is finite: the `generate_network(...)` call that makes exact SSA on it a reference |

`size.nf_events`, the reaction events per trajectory, is a planning estimate only.

**The protocol is the authors'.** Standalone NFsim is run the way BioNetGen runs it:
BNG2.pl executes the protocol's actions, writing the XML and starting NFsim for each
network-free `simulate`, and applying any `setParameter` between them, which NFsim's
own command line cannot do mid-run. `bench.nfactions` reads the actions into
network-free steps, each with the parameter changes before it, so every other engine
replays the same protocol: BNGsim's session sets them on the live session, and a step
after one with `get_final_state=>0` starts from the seed species again, as BNG2.pl's
would. Only what every engine can follow is accepted: network-free `simulate` steps,
`setParameter`, and inert actions (`generate_network`, `writeXML`, ...), which are
dropped; anything else is refused by name. BNG2.pl execs `$BNGPATH/bin/NFsim` itself,
so on a host whose glibc is too old the job points BNG2.pl at a mirror of the fetched
distribution whose `NFsim` launches the unchanged binary under the loader swap.

**References.**

- `nfsim`: BNG2.pl and NFsim 1.14.3 (BioNetGen 2.9.3), one BNG2.pl process per
  trajectory, each step's seed replaced. Its NaN stop on a rate law that divides by a
  single-copy count (RuleWorld/nfsim#115) is recorded as such.
- `rulemonkey`: RuleMonkey's `rm_driver` v3.13.0, built from its pinned source with
  the environment's compiler (`bench fetch-rulemonkey`), one process per trajectory on
  the same XML. It is the **same code lineage** as BNGsim's `nf_exact` engine (bngsim
  0.17.2 embeds v3.13.0 plus five vendoring and documentation commits): it checks the
  embedding and is a cost comparator, a secondary reference, not an algorithmically
  independent answer. It cannot change a parameter between runs, so a multi-step
  protocol is `UNSUPPORTED` for it.
- `run_network` and `bngsim_ssa`: where the spec names `netgen`, exact SSA on the
  generated network, by BioNetGen and by BNGsim. Network-free simulation of a finite
  network samples the law of exact SSA on it, so these are the strictest references;
  they are run from the seed species only (one network-free step).

**The verdict** is §5.3's two-sample test, unchanged, on the model's observables by
name: a permutation max-T test on per-cell means and variances, false-alarm rate at
most `alpha` = 10^-3 per comparison, with parity_core's ensemble verdict recorded
beside it. Each engine draws from a seed block of its own (`nf_reject` 0, `nf_exact`
1, NFsim 2, RuleMonkey 3, run_network 4, BNGsim SSA 5, times 10^7). A later
network-free step's seed is 10^6 further on: BNG2.pl starts NFsim afresh for every step,
while BNGsim's session carries its random stream on unless the step restarts from the
seed species. The job passes only when BNGsim agrees with every reference that applies. Recorded beside the verdicts, never scored:

- **the other setting**: BNGsim's other network-free engine, from its own seeds,
  compared the same way, so the two embedded engines are checked against each other
  without one's failure being charged to the other;
- **same seed**: the first five trajectories of the twin reference's own seeds run
  again by BNGsim -- `nf_exact` against `rm_driver`, `nf_reject` in its NFsim v1.14.3
  compatibility mode against BNG2.pl's NFsim -- and compared exactly. Two copies of
  one code fed one seed should give one trajectory.

The ensembles are stored with the record (§6.1), so a changed verdict is re-scored
without re-simulating.

**Timing.** As §5.3: an ensemble of `timing_replicates` trajectories from the same
seeds every round, interleaved round-robin with standalone NFsim and RuleMonkey (the
exact-SSA references are timed by the SSA capability). BNGsim's session is built
once, cold (`load`, after BNG2.pl's `writeXML`, `xml`); each trajectory re-initialises
it from the XML and simulates. The references run as fresh processes, so their
start-up -- for NFsim, BNG2.pl parsing the model and writing the XML every trajectory
-- is inside their time. Every engine also reports its own solve time (BNGsim's
session simulate calls; NFsim's reported simulation CPU time; `rm_driver`'s reported
simulation time), recorded per ensemble beside the wall time, so start-up and solve
are told apart for each. The speedup view divides by standalone NFsim.

### 5.6 Sensitivity verdicts (forward sensitivities)

The `sens` capability (`bench.capabilities.sens`, verdict in `bench.sensitivity`)
compares BNGsim's CVODES forward sensitivities `dx/dp` -- both correctors, staggered
(bngsim's default) and simultaneous (AMICI's) -- with AMICI's forward sensitivities
on the SBML exported from the `.net`, with central finite differences from
`run_network` (which shares no code with BNGsim), and with COPASI's time-course
sensitivities task.

**Protocol fields** (`protocol.sens`, validated in `specs.py`):

| Field | Meaning |
|---|---|
| `netgen`, `t_end`, `n_steps`, `rtol`, `atol` | as for ODE; every engine runs at these tolerances |
| `parameters` | the problem's convention: `independent` (every number in the `.net` that reaches a rate or an initial amount), `rate_constants`, `fitted`, or a list |
| `fitted` | the problem's fitted set; beside another convention it is a job setting of its own (`.../fitted`) |
| `parameter_scales` | a scale for a parameter whose value is zero (a dose not given, a switch off) |
| `preequilibration`, `stimulus` | a two-phase protocol: a phase of `duration` before t = 0 at its own parameter values (`set`), then the `stimulus` values from t = 0 |
| `fd_step`, `fd_rtol`, `fd_atol` | the finite differences' step (relative to each parameter's scale) and their tighter tolerances |

**The verdict.** Columns -- one output's sensitivity to one parameter over time -- are
compared on the semi-scaled sensitivity `s_p dy/dp` (`s_p` the parameter's value, or
its `parameter_scales` entry), so parameters of very different magnitudes are judged
alike. A column's error is its largest difference over time as a fraction of its
peak, floored at `floor_rel` of the output's own peak and at `floor_atol` state
atols; a comparison passes when every column it judges is within `tol` (1e-3). A
reference judges the columns it produced: AMICI its free parameters, the finite
differences only where their two-step error estimate (`h` and `h/2`) is a tenth of
the tolerance. Each further BNGsim corrector must agree with the default, and BNGsim's
observable sensitivities with the ones formed from its species sensitivities.

**Attribution.** A reference that disagrees with BNGsim, where on every failing column
another reference agrees with BNGsim and disagrees with it (two independent engines
against one), is refuted -- its own failure (REFERENCE_FAILED), as a crash or a
timeout is. Any other disagreement is a DIFF. A significant column no agreeing
reference judged is REFERENCE_FAILED too. A reference that cannot run the protocol at
all (COPASI on a two-phase one: its task cannot start from an equilibration's
sensitivities) is `not_applicable`: named, not run, not counted.

**Two phases.** The equilibration runs over `[-duration, 0]`; BNGsim carries its
sensitivities into the measured phase on one Simulator (`carry_sensitivities=True`),
AMICI by its own pre-simulation (`t_presim`, the set parameters compiled as fixed),
the finite differences by a second `run_network` started from the first's state. The
phase has a duration rather than a steady-state criterion: a model need not reach a
steady state (the p38-ATF2 network has an absorbing state).

**Timing.** Each engine's forward-sensitivity solve of the protocol, interleaved
(section 7), beside plain solves of the same protocol by BNGsim and `run_network`, the
unit of a finite-difference gradient (`N + 1` and `2N` solves, derived), and -- timing
only (decision D7) -- AMICI's adjoint gradient of a synthetic Gaussian log-likelihood
of the observables beside BNGsim's gradient from its forward sensitivities. Every
engine's load and compile is reported beside its solve.

## 6. Results

### 6.1 One file per job

Each job writes one JSON file, atomically, under a path derived from its job ID. The
file holds:

- the outcome, using bngsim's outcome vocabulary across all capabilities: PASS,
  DIFF, EXCEPTION (BNGsim failed where the reference ran), REFERENCE_FAILED (a
  reference failed where BNGsim ran, and every reference that ran agrees; one that
  disagrees makes it DIFF, §5.2), BAD_TEST (both failed, or the model could not be
  prepared), TIMEOUT, UNSUPPORTED, SKIP;
- the measured values, and every timing repetition rather than only the summary;
- provenance:
  - the BNGsim version and compiled build commit;
  - the reference engines' versions;
  - this repository's commit;
  - the environment lock or container digest;
  - the node name and hardware (section 7);
  - start and end time;
  - exit status.

A correctness job also stores the trajectories its verdict was computed from, beside
its JSON file as `<job id>.npz` (compressed float64): every engine's compared
trajectory, the output times and the compared species. The JSON file names it, with
its size and sha256, so an `.npz` left by another run of the job is never read as
this one's. On the core suites each is a few MB, and none is above about 10 MB.

### 6.2 Freshness

A result is current when its key matches what would run now. The key combines:

- the job spec;
- this repository's version;
- the BNGsim build;
- the reference engines' versions;
- for timing jobs, the node type.

Only stale or missing jobs run. An interrupted run loses at most the jobs in flight.

A part of the job spec can belong to one kind of job only. The timing protocol's
reproducibility check (`timing_check`) is in the timing job's definition alone, and
the correctness verdict's version (`verdict`, §6.4) in the correctness job's alone.
So a new verdict makes every correctness result stale and leaves every timing job's
definition, id and key as they were.

A correctness result can be stale only in how it is scored: the vendored comparison
core, or the verdict version in its definition (§6.4). Such a result does not need to
re-run. `bench rescore` recomputes its verdicts and outcome from its stored
trajectories and rewrites it with the current key, keeping a history of the outcomes
it replaced. A result that is stale for any other reason, or that stored no
trajectories, is left to re-run.

### 6.3 Gathering and publishing

- The primary published artifact is an **absolute time per (engine, problem, setting)**,
  always tagged with its platform (§7.5). With more than two engines there is no pairwise
  matrix: any speedup is a derived column against a baseline the reader chooses.
- `gather` assembles per-job results into one report per capability and layer.
- **Timing is two numbers per engine, never one:** the one-time *pre-sim* cost (load
  plus any network generation or codegen/compile, measured once and cold, so it has no
  CI) beside the warm *sim* median with its bootstrap 95% CI. A small solve can hide a
  large build -- AMICI's SBML-to-C++ compile, libRoadRunner's SBML read -- and the
  re-simulation claim of §1 (claim 2) needs both; see §7.3. An engine run as a fresh
  subprocess per solve (run_network) has its load inside every rep, and is labelled
  so rather than split.
- **Correctness reads agreed/attempted references** beside the outcome enum:
  `FULL 4/4`, `PARTIAL 3/4 (copasi: crash)`, `DIFF 3/4 (roadrunner: diff)`, so one
  broken reference does not hide the others behind a single REFERENCE_FAILED.
- Both are derived in rendering from fields every result already records; neither
  changes the per-job result schema.
- **Each reference shows both halves of the correctness gate (§6.4):** `ok`, or
  `DIFF (species)`, `DIFF (vendored)` or `DIFF (both)`. Each row names its worst
  species across the references, with its error as a fraction of its own peak.
- A **result bundle** packages one complete set of results for one BNGsim build:
  - per-job results;
  - gathered reports;
  - the environment lock;
  - hardware records.

  Bundles are published as GitHub release assets and linked from
  [lanl/bngsim](https://github.com/lanl/bngsim). Each bundle comes with a rendered
  summary page.
- Raw outputs are never committed to git history.

### 6.4 The correctness verdict

A correctness job compares BNGsim with each reference that ran. A reference agrees
only when two checks both pass.

1. **bngsim's deterministic verdict** (`parity_core.differ`, vendored unmodified, §10).
   It applies a per-cell tolerance, a fail-fraction budget behind hard ceilings, and
   a movement gate. It can fail a species only when that species is significant:
   - its peak is past 1e-3 of the largest peak in the model, or
   - its range is past 1e-3 of the largest range.

   (An absolute ceiling, 1e-2 of the file-wide peak, applies to every species. It
   never binds on a small one.) When a model's species span many decades, most of
   them are never judged. On the ode-sbml suite it judges 1 of Nishio 2008's 59
   compared species, 9 of Ouzounoglou 2014's 89, 10 of Koo 2013's 79 and 16 of Smith
   2013's 105. It passes AMICI on Smith 2013, although AMICI never gives the first of
   the two insulin doses that the model's events administer.
2. **A per-species check**, which judges every compared species on its own scale. A
   species fails when its worst disagreement over the run, max over t of |a − b|, is
   past the larger of two bounds:
   - 1e-2 of its peak, the largest magnitude it reaches in either engine;
   - a floor of 10 times the protocol's absolute tolerance.

   The floor exists because no solver here resolves a difference below its atol. So a
   species that is numerically zero, within a few atol in both engines, cannot fail.
   On Lang 2024, for example, BNGsim holds two TP53 Ser15~p species at exactly 0 and
   run_network and libRoadRunner report round-off of 1e-26 to 1e-25, against an atol
   of 1e-12. That is 100% of the reference's own peak, but it is not a disagreement.

Each reference's verdict records both checks:

- the vendored verdict's counts, and its pass as `vendored_passed`;
- the per-species check's worst species, with its error, its peak, what it was
  allowed, and the time of its worst cell;
- how many species failed, and how many sat at the floor.

`passed` is true only when both checks pass.

**Calibration.** The two constants were set on the trajectories of the ode-net and
ode-sbml correctness jobs.

- **1e-2 of the peak** is the vendored movement gate's own ceiling, now applied to
  every species. Every comparison that should pass is inside it:
  - the worst is libRoadRunner on Koo 2013, at 4.6e-3 of a species' peak. That is
    its own integration error at the SED-ML's rtol of 1e-6: it is 4.4e-3 from the
    converged solution, and it converges at tighter tolerances;
  - next come Koo 2013 at 5.8e-4 and Verma 2016 at 2.2e-4;
  - every ode-net comparison is under 1e-4.

  The known-bad comparisons are past it by 13× or more: AMICI on Smith 2013 (1.0 of
  the insulin's peak, from the dose it never gives), and COPASI (1.0) and
  libRoadRunner (0.13) on Winter 2017.
- **10 × atol** is the vendored movement gate's floor. bngsim's own census of
  BioModels found solvers that agree differing by up to 3.8 atol on a species near
  zero. On the suites here, a species of a passing comparison is never further off
  than 1e-2 of its peak unless it is numerically zero: all such species are within
  2e-8 atol.

**The floor ties the check to the protocol.** A species whose peak sits below the
protocol's atol is judged only by the floor. The protocol must then resolve the
model's species, or a disagreement measures the tolerance rather than the engines.
Nishio 2008 is the one member where it did not:
- 34 of its 59 compared species peak between 1.5e-11 and 1e-9, under an atol of 1e-9;
- the engines that hold atol exactly (AMICI and run_network) were 2–6% of a species'
  peak off the converged solution, so it failed the check;
- at atol 1e-15 every engine is within 7.5e-4.

Its spec records the change ([core_suites.md](core_suites.md)).

**What does not change.** Two other comparisons decide something in both kinds of job:
- the faithfulness check of a network derived from SBML;
- the check of run_network's `sparse` setting against its dense run.

Both keep deciding on the vendored verdict alone, so the timing job decides as before.
The correctness record carries the per-species check of each beside it.

**Versioning.** The check's version is `VERDICT_VERSION`, part of the correctness
job's definition only (§6.2). Changing the check, its constants, how the two
checks combine, or how the verdicts become an outcome (§5.2) bumps it. Version 2
made a disagreement DIFF even when another reference failed; it had been
REFERENCE_FAILED, so a wrong answer could be timed (Winter 2017, Smith 2013). Every correctness result then goes stale and is re-scored
with `bench rescore`, from the trajectories it stored (§6.1).

## 7. Timing protocol

1. **Interleaved, drift-cancelling:**
   - A timing job runs all the engines compared on the problem in the same process
     tree, pinned to one core.
   - Repetitions cycle round-robin through the engines (A B A B … for two, A B C A B C …
     for more), so each engine's runs sit adjacent in time to the others' and the
     common-mode drift of §7.6 cancels out of every comparison.
   - Warm-up repetitions are discarded, and every remaining repetition is kept.
   - **Every kept repetition must reproduce the first run.** A kept repetition's full
     trajectory is compared with the first run's (for BNGsim, its cold first run; for a
     subprocess comparator, its own first kept repetition), and the job is
     `NOT_REPRODUCIBLE` — its speedup not trusted, not published — if any repetition
     disagrees by more than `max_err_over_peak` of `REPRO_TOL` (1e-6). Otherwise the
     ratio can compare BNGsim with a run of something else: `RoadRunner.reset()` restores
     the species but not a parameter an event changed, so a warm re-run after it can be a
     cheaper, different trajectory (issue #8). Each engine's adapter chooses the reset
     after which a re-run is the first run again.
   - The job reports each engine's absolute time with a bootstrap confidence interval
     over its kept repetitions (§7.5). A speedup is a derived view, computed against
     whichever engine the reader takes as baseline; the suite never stores or publishes
     an N×N matrix of pairwise ratios.
2. **Single-threaded:** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and the equivalents
   are set to 1, unless a job measures parallelism on purpose.
3. **Cold vs warm:**
   - Cold cost (loading, code generation, compilation) is a separate phase, measured
     with an empty compile cache.
   - Compile caches live on node-local scratch, never on a shared filesystem.
   - Every job runs with `BNGSIM_CODEGEN_STRICT=1`, whatever the caller set. A
     compiled-code build that fails then fails the job, instead of falling back to the
     interpreter with only a warning, so no job measures or checks the interpreter
     without knowing. The setting is part of each result's freshness key.
4. **One node type per artifact:**
   - Timing results that are compared with each other come from one declared node
     type (Slurm `--constraint`, or a recorded node type for interactive runs).
   - Each result records the node name, CPU model, microcode, frequency governor and
     turbo state, kernel, and how many other jobs shared the node.
5. **Absolute times are the artifact.** Every timing result is an absolute time
   reported with its full platform record (node type, CPU model, microcode, governor,
   turbo, kernel). This is what `gather` publishes (§6.3). Absolute times are more
   sensitive than ratios to everything above, which is exactly why the interleaving of
   §7.1 is kept: it makes a single platform's absolute numbers trustworthy, rather than
   leaving drift to be divided out of a ratio.
6. **Node sharing is settled by measurement, not assumed** (section 9).

Run-to-run drift on a single machine can be as large as the effects being measured.
That is why the engines compared on a problem are timed inside one interleaved job,
never across separate runs — so any comparison drawn between them, absolute or derived,
rests on runs that were adjacent in time on the same core.

## 8. Running

One job list, three ways to run it, all writing the same result files:

| Mode | Use | Notes |
|---|---|---|
| serial | a laptop, one job at a time | `bench run <job-id>`, or `bench run --next N` |
| local pool | a checked-out compute node | `bench run --workers 64`; correctness jobs fill the node, timing jobs follow section 9 |
| Slurm | unattended runs | `bench submit` writes `sbatch` job arrays; many short jobs are packed into each array task; partition, account and time limit come from a config file |

Other commands:

- `bench list`: problems and jobs;
- `bench plan`: what is current, stale, missing, waiting or blocked, and what it
  would cost;
- `bench gather`: per-capability reports and a readable summary;
- `bench rescore`: re-score stored correctness results after a verdict change (§6.4);
- `bench fetch-bionetgen`, `bench fetch-models`, `bench doctor`: set up and check the
  environment;
- `bench submit`: the Slurm mode below;
- `bench bundle`: not yet written.

`bench run` takes `--budget 9h30m`, `--until HH:MM`, or `--until slurm` (the end of
the current allocation, from `squeue`).

An allocation has a maximum wall time, possibly around 10 hours on the cluster in
view (unconfirmed). So:

- **Bounded runs:** a run on a checked-out node takes a deadline, `--until` or
  `--budget`. It does not start a job whose estimated cost would overrun the deadline,
  and it stops cleanly when the deadline arrives.
- **No lost work:** completed jobs are already written, so the next allocation picks
  up the rest.
- **Array tasks fit the limit:** `bench submit` packs jobs into Slurm array tasks by
  estimated cost (first-fit decreasing).
  - Each task's estimated run time is bounded phase by phase, the way the runner runs
    it: Graham's list-scheduling bound, sum/w + (1 − 1/w)·max, for the correctness
    jobs on the task's CPUs, then for the timing jobs on its timing workers.
  - That bound stays within the time limit, less a margin and a start-up allowance.
  - A problem's correctness and timing jobs go to the same task, so a timing job
    never waits on another task.
  - Each task's `bench run` also has a deadline at the end of its allocation, so an
    underestimate costs only the jobs in flight. Those jobs run again at the next
    submit.
- **Long jobs:** a problem whose own estimate exceeds one allocation is flagged and
  not submitted. `--skip-long` submits the rest, and the long ones run on a
  checked-out node.
- **Settings stay local:** partition, account, QOS, constraint and time limit come
  from `slurm.yaml`, which is git-ignored because it names local infrastructure.
  `slurm.example.yaml` documents every key.

The environment is pinned:

- a `pixi` lock for Python and the engines;
- the BioNetGen distribution, pinned by release tarball and sha256;
- a separate pinned interpreter for PyBNF;
- a container image built from the same lock, for clusters where installing is
  impractical.

Corpora are fetched at pinned versions and checked by sha256.

## 9. Validation before trusting cluster timings

Two checks run before any cluster timing is used:

1. **Cross-platform sanity (ARM vs x86):**
   - The published artifact is an absolute time tagged with its platform (§6.3, §7.5),
     so the suite does not claim a timing transfers between machines. This is therefore
     a measurement-hygiene check, not the portability proof it used to be.
   - One corpus is timed on two deliberately different node types on the same cluster:
     an x86 node (Platinum 8260) and an ARM node (Ampere Altra / Neoverse), both under
     the protocol of §7. The candidate corpus is BNGL models exported to SBML, BNGsim
     vs libRoadRunner.
   - It passes if, on each platform independently, the per-problem absolute times are
     repeatable within a stated tolerance and the reproducibility check of §7.1 holds
     (no residual drift). The derived ratios are reported on both platforms and their
     agreement is informative, but it is not a pass condition.
   - An ARM node is used, not a laptop: it is the most different machine reachable on
     the same Slurm, needs no off-cluster handling, and forces the pinned stack to be
     shown building and running on aarch64 — itself worth knowing.
   - The cluster has an ample ARM fleet to serve as the second platform: dedicated
     aarch64 partitions with ThunderX2 nodes, bare-metal Neoverse-N1 nodes, and NVIDIA
     Grace (Neoverse-V2) nodes. The one remaining task is the stack itself: libRoadRunner
     publishes no `linux-aarch64` artifact on PyPI — no wheel, and no sdist — so it is
     built from upstream source (`sys-bio/roadrunner` v2.11.0). The one heavy step is
     LLVM, because the `libroadrunner-deps` prebuilt bundle covers x86-64 Linux,
     macOS-arm64 and Windows but not linux-aarch64, so on ARM Linux LLVM is compiled
     from source — which is exactly what the ARM fleet is for. None of this affects the
     x86-64 wheel the suite runs on today.
2. **Node sharing:**
   - The same timing jobs are run one per node, then packed onto a full node.
   - If the absolute times agree, timing jobs may share nodes.
   - If they don't, each node runs fewer timing jobs at once, down to one.
   - Nodes are not shared with other users, so this is only about the suite's own
     concurrent jobs.

## 10. Relationship to lanl/bngsim

- bngsim keeps its own correctness suites (`parity_checks/`) and its nightly parity
  workflow. They guard bngsim's development; this repository measures releases.
- This repository should use the same comparison protocol as bngsim's
  `parity_checks/_core` (tolerances, ensemble verdicts, outcome vocabulary), so a
  verdict means the same thing in both places. For now it is copied verbatim into
  `src/bench/_vendor/parity_core/` from one pinned bngsim commit, with each file's
  sha256 recorded in `VENDOR.json`; a test fails if the copy is edited or drifts
  from that commit. `tools/vendor_parity_core.py --ref <tag>` refreshes it when the
  pinned bngsim release changes.
- Benchmark harnesses migrate here from bngsim's `benchmarks/suites/` and from
  harnesses maintained elsewhere. They migrate one capability at a time.

## 11. Migration order

1. **Infrastructure:**
   - problem specs;
   - job generation;
   - the serial and local-pool runners, then Slurm;
   - per-job results and `gather`;
   - the environment and container;
   - the validation of section 9.
2. **Breadth correctness:** parallel, with no timing constraints. It reuses bngsim's
   parity comparisons.
3. **Core suites:** seed each suite from the paper's set and top it up to 10, then
   confirm that each model's evidence covers the exact file used. Each suite lands
   with its capability (step 4); the three ODE suites come first.
4. **Capabilities, in order:**
   1. ODE, with the size series;
   2. SSA and PSA;
   3. network-free;
   4. sensitivities;
   5. steady state;
   6. SBML compliance;
   7. fitting.
5. **Breadth timing.**
6. **First result bundle.**

## 12. Open questions

- **Cluster specifics.** Known so far:
  - Slurm;
  - an interactively checked-out compute node is the user's alone;
  - nodes have at least 64 CPUs;
  - GitHub is reachable;
  - allocations have a wall-time limit, possibly about 10 hours.

  Still to establish:
  - node CPU model and whether nodes are homogeneous;
  - glibc version (libRoadRunner's Linux wheel needs glibc 2.28 or newer);
  - node-local scratch size;
  - partitions, account, QOS and the exact time limits;
  - container support;
  - package-index reachability from compute nodes.

  [`tools/survey_cluster.sh`](../tools/survey_cluster.sh) collects all of these.
- **Sharing the comparison protocol with bngsim:** a vendored, pinned copy for now
  (section 10). The cleaner long-term answer is for bngsim to ship it as a public
  module.
- **Where bundles live:** as release assets of this repository linked from
  lanl/bngsim, or as release assets of lanl/bngsim itself.
- **Core suites:** the models that top each suite up to 10 beyond the paper's seed,
  and the knob ranges of each size series.
- **Tolerances:** per-capability correctness tolerances, and the acceptance tolerance
  for section 9.
- **SSA verdict:** whether the SSA suites gate on the permutation test of §5.3 (a
  stated false-alarm rate) or on bngsim's parity-core ensemble verdict (the same
  verdict in both repositories, §10); both are recorded.
- **PSA flag threshold and exact-SSA cost:** the flag of §5.4 starts at N_c = 100,
  from bngsim's documented working range; and the psa-net members whose exact SSA
  takes minutes per trajectory (Kochanczyk 2017, Blinov 2006) cannot afford as many
  exact as PSA trajectories, which may call for a separate exact-SSA ensemble size.
